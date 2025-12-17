# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import transforms
import cv2
from einops import rearrange
import mediapipe as mp
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore, laplacianSmooth, transformation_from_points, transformation_from_points_batch
import kornia as K
from batch_face import RetinaFace, LandmarkPredictor
from .util import timer
import os 
from pathlib import Path
import face_alignment

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def load_fixed_mask(resolution: int) -> torch.Tensor:
    BASE_DIR = Path(__file__).resolve().parent
    mask_path = BASE_DIR / "mask.png"
    # print(f'mask_path:{mask_path}')
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_AREA) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image).float(), "h w c -> c h w")
    return mask_image



class ImageProcessor:
    def __init__(
        self,
        resolution: int = 512,
        mask: str = "fix_mask",
        device: str = "cpu",
        mask_image=None,
        yolo_weights: str = "latentsync/utils/yolov8n-face.pt",   # <- add this
    ):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.mask = mask
        self.device = device  # string is fine for .to(...)

        # ---- masks setup (unchanged) ----
        if mask in ["mouth", "face", "eye"]:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
        if mask == "fix_mask":
            self.face_mesh = None
            self.smoother = laplacianSmooth(device=torch.device("cuda"),)
            self.restorer = AlignRestore()

            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image

            self.mask_image = self.mask_image.to(self.device)

        if device != "cpu":
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
            )
            self.face_mesh = None
        else:
            # self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
            self.face_mesh = None
            self.fa = None

        # ---- batch-face setup: RetinaFace + LandmarkPredictor ----
        if "cuda" in device:
            # device could be "cuda" or "cuda:0"
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
        else:
            gpu_id = -1  # CPU

        # face detector (RetinaFace)
        self.face_detector = RetinaFace(gpu_id=gpu_id)

        # 68-point landmark predictor
        self.landmark_predictor = LandmarkPredictor(gpu_id=gpu_id)

        # you can keep this if you still use it elsewhere
        self.img_size = resolution


    def _detect_face_yolo(self, image_np: np.ndarray):
        """
        image_np: H x W x C (BGR), dtype uint8 or float32 [0,1]/[0,255]
        Returns:
            bbox: np.ndarray [4]  -> [x1, y1, x2, y2] in original image coords
            kps5: np.ndarray [5,2] -> (x,y) points in original image coords
        or (None, None) if no face.
        """
        # Make sure YOLO gets uint8 BGR 0-255
        img_for_yolo = image_np
        # if img_for_yolo.dtype != np.uint8:
        #     img_for_yolo = np.clip(img_for_yolo, 0.0, 1.0)
        #     img_for_yolo = (img_for_yolo * 255.0).astype(np.uint8)
        img_for_yolo = img_for_yolo.astype(np.uint8)

        # Letterbox (same as your test code)
        lb_img, ratio, (dw, dh) = letterbox(img_for_yolo, new_shape=(self.img_size, self.img_size),
                                            auto=False)
        # BGR -> RGB, HWC -> CHW
        lb_img = lb_img[:, :, ::-1].transpose(2, 0, 1)
        lb_img = np.ascontiguousarray(lb_img)

        im = torch.from_numpy(lb_img).to(self.device).float() / 255.0
        im = im.unsqueeze(0)  # [1,3,640,640]

        with torch.no_grad():
            res = self.yolo_model(im)[0]  # first (and only) image

        if res.boxes is None or res.boxes.xyxy.numel() == 0:
            return None, None

        # Choose the highest-confidence detection
        conf = res.boxes.conf
        best_idx = int(conf.argmax().item())

        box = res.boxes.xyxy[best_idx].cpu().numpy()  # [x1,y1,x2,y2] in letterboxed coords

        # Undo letterbox to map back to original image coordinates
        r_w, r_h = ratio
        x1, y1, x2, y2 = box
        x1 = (x1 - dw) / r_w
        x2 = (x2 - dw) / r_w
        y1 = (y1 - dh) / r_h
        y2 = (y2 - dh) / r_h
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        return bbox


    def detect_facial_landmarks(self, image: np.ndarray):
        height, width, _ = image.shape
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            raise RuntimeError("Face not detected")
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image
        landmark_coordinates = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark
        ]  # x means width, y means height
        return landmark_coordinates

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask == "mouth" or self.mask == "face":
            landmark_coordinates = self.detect_facial_landmarks(image)
            if self.mask == "mouth":
                surround_landmarks = mouth_surround_landmarks
            else:
                surround_landmarks = face_surround_landmarks

            points = [landmark_coordinates[landmark] for landmark in surround_landmarks]
            points = np.array(points)
            mask = np.ones((self.resolution, self.resolution))
            mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
        elif self.mask == "half":
            mask = torch.ones((self.resolution, self.resolution))
            height = mask.shape[0]
            mask[height // 2 :, :] = 0
            mask = mask.unsqueeze(0)
        elif self.mask == "eye":
            mask = torch.ones((self.resolution, self.resolution))
            landmark_coordinates = self.detect_facial_landmarks(image)
            y = landmark_coordinates[195][1]
            mask[y:, :] = 0
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Invalid mask type")

        image = image.to(dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * mask
        mask = 1 - mask

        return pixel_values, masked_pixel_values, mask

    def affine_transform_batch_v2(self, images):
        """
        images: sequence of HxWxC uint8/float32 np.ndarrays (all same size, 1 face each)

        Returns:
            faces:          torch.uint8 tensor of shape (B, C, R, R) in [0, 255]
            boxes:          np.ndarray of shape (B, 4) -> [x0, y0, x1, y1] in the warped space
            affine_matrices_out: np.ndarray of shape (B, 2, 3)
            valid_mask:     np.ndarray of shape (B,), bool – True if a face was actually detected
        """
        # -------------------------------------------------------------------------
        # 1) Prepare numpy batch (B, H, W, C) and normalized version for torch
        # -------------------------------------------------------------------------
        B = len(images)

        # -------------------------------------------------------------------------
        # 2) Face detection + landmarks on the whole batch
        # -------------------------------------------------------------------------
        faces = self.face_detector(
            images,                 # list of np.ndarray
            # threshold=0.95,
            # max_size=self.img_size,
        )

        landmarks_list = self.landmark_predictor(faces, images, from_fd=True)
        # landmarks_list: list length B, each element is usually (68,2) or (1,68,2) etc.
        # but may be empty for frames with no face.

        # -------------------------------------------------------------------------
        # 3) Robustly build lm68_batch + valid_mask
        # -------------------------------------------------------------------------
        processed_lms = [None] * B
        valid_mask = np.zeros(B, dtype=bool)
        valid_indicies = []

        # First pass: parse each entry, mark valid if we got real 68 points
        for i, lm in enumerate(landmarks_list):
            if lm is None:
                continue

            lm_arr = np.asarray(lm, dtype=np.float32)
            if lm_arr.size < 68 * 2:
                # empty or incomplete → treat as invalid
                continue

            # Accept common shapes
            if lm_arr.shape == (68, 2):
                lm68 = lm_arr
            elif lm_arr.ndim == 3 and lm_arr.shape[1:] == (68, 2):
                # e.g. (1, 68, 2)
                lm68 = lm_arr[0]
            else:
                # Last resort: try to reshape
                try:
                    lm68 = lm_arr.reshape(-1, 68, 2)[0]
                except Exception:
                    continue

            processed_lms[i] = lm68
            valid_mask[i] = True
            valid_indicies.append(i)

        # If we have at least one valid frame, propagate its landmarks to invalid ones
        if valid_mask.any():
            # Forward pass: fill gaps with last seen valid
            last_valid = None
            first_valid_idx = None
            for i in range(B):
                if processed_lms[i] is not None:
                    last_valid = processed_lms[i]
                    if first_valid_idx is None:
                        first_valid_idx = i
                elif last_valid is not None:
                    processed_lms[i] = last_valid

            # Backward pass: fill leading invalids with first valid
            if first_valid_idx is not None:
                first_valid_lm = processed_lms[first_valid_idx]
                for i in range(first_valid_idx - 1, -1, -1):
                    if processed_lms[i] is None:
                        processed_lms[i] = first_valid_lm
        else:
            # ---------------------------------------------------------------------
            # Entire batch has no faces → create a safe dummy layout.
            # We don't care what it looks like since you'll ignore these frames,
            # but we must avoid degenerate (all-equal) points that break the affine.
            # ---------------------------------------------------------------------
            tmpl = self.restorer.face_template.astype(np.float32)  # (3, 2) in template space

            # Build a fake 68-point set whose group means roughly equal tmpl
            lm68_default = np.zeros((68, 2), dtype=np.float32)
            # Just copy template to the 3 groups we use later
            lm68_default[42:48, :] = tmpl[0]
            lm68_default[36:42, :] = tmpl[1]
            lm68_default[27:36, :] = tmpl[2]
            # Fill the rest with something non-degenerate (e.g. middle point)
            lm68_default[:27, :] = tmpl[1]
            lm68_default[48:, :] = tmpl[1]

            processed_lms = [lm68_default for _ in range(B)]
            # valid_mask stays all False

        lm68_batch = np.stack(processed_lms, axis=0)  # (B, 68, 2)

        # -------------------------------------------------------------------------
        # 4) Smooth landmarks in batch
        # -------------------------------------------------------------------------
        points_list = self.smoother.smooth_batch(lm68_batch)  # list length B, each (68, 2)
        points = np.stack([p.detach().cpu().numpy() for p in points_list], axis=0)

        # -------------------------------------------------------------------------
        # 5) Compute the 3 mean points per frame (what AlignRestore expects)
        # -------------------------------------------------------------------------
        part0 = points[:, 42:48, :].mean(axis=1)  # (B, 2)
        part1 = points[:, 36:42, :].mean(axis=1)  # (B, 2)
        part2 = points[:, 27:36, :].mean(axis=1)  # (B, 2)

        landmarks3_batch = np.stack([part0, part1, part2], axis=1)  # (B, 3, 2)

        assert landmarks3_batch.shape[0] == B, (landmarks3_batch.shape, B)

        # -------------------------------------------------------------------------
        # 6) Align & warp with your AlignRestore (batched)
        # -------------------------------------------------------------------------
        affine_matrices = self.restorer.compute_affine(landmarks3_batch, smooth=True)
        out_w, out_h = self.restorer.face_size  # note order

        # Boxes in warped coordinates – here simply the full warped image
        boxes = np.tile(np.array([0, 0, out_w, out_h], dtype=np.int32), (B, 1))


        # -------------------------------------------------------------------------
        # 7) Kornia warp_affine in batch
        # -------------------------------------------------------------------------
        # Stack the original numpy images into a single array.
        # Assumes all images are same shape & dtype (e.g. uint8).
        batch_np = np.stack(images, axis=0)  # (B, H, W, C), dtype stays as-is (ideally uint8)

        # Move to torch and normalize on GPU.
        img_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2)  # (B, C, H, W)
        img_t = img_t.to(device=self.device, dtype=torch.float32) / 255.0

        # affine_matrices -> torch (B, 2, 3)
        if isinstance(affine_matrices, torch.Tensor):
            M = affine_matrices.to(device=img_t.device, dtype=torch.float32)
            if M.dim() == 2:
                M = M.unsqueeze(0)
        else:
            M = torch.as_tensor(affine_matrices, device=img_t.device, dtype=torch.float32)
            if M.dim() == 2:
                M = M.unsqueeze(0)

        warped = K.geometry.transform.warp_affine(
            img_t,
            M,
            dsize=(out_h, out_w),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (B, C, out_h, out_w)

        # -------------------------------------------------------------------------
        # 8) Resize to self.resolution in batch
        # -------------------------------------------------------------------------
        faces = torch.nn.functional.interpolate(
            warped,
            size=(self.resolution, self.resolution),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)  # (B, C, R, R)

        faces = (faces * 255.0 + 0.5).to(torch.uint8)
        affine_matrices_out = M.detach().cpu().numpy()  # (B, 2, 3)

        return faces, boxes, affine_matrices_out, images, valid_indicies



    def affine_transform_batch(self, images):

        device = torch.device("cuda")

        img_list = list(images)

        B = len(img_list)

        lmk3_list = []      
        valid_indices = []  

        with timer("# --- 1) Landmark detection per image (can stay in a simple for-loop) ---"):
            for idx, img in enumerate(img_list):

                if self.fa is None:

                    landmark_coordinates = np.array(self.detect_facial_landmarks(img))
                    if landmark_coordinates is None or len(landmark_coordinates) == 0:
                        self.smoother.reset()
                        continue
                    lm68 = mediapipe_lm478_to_face_alignment_lm68(landmark_coordinates)
                else:
                    detected_faces = self.fa.get_landmarks(img)
                    if detected_faces is None or len(detected_faces) == 0:
                        self.smoother.reset()
                        continue
                    lm68 = detected_faces[0]

                points = self.smoother.smooth_batch(lm68)[0]  
                points = points.to(device=device, dtype=torch.float32)

                lmk3 = points.new_zeros((3, 2))    
                lmk3[0] = points[17:22].mean(0)
                lmk3[1] = points[22:27].mean(0)
                lmk3[2] = points[27:36].mean(0)

                lmk3_list.append(lmk3)
                valid_indices.append(idx)

            lmk3_batch = torch.stack(lmk3_list, dim=0)

        affine_matrix_t, _ = transformation_from_points_batch(
            lmk3_batch,                        
            self.restorer.face_template,       
            smooth=True,
            device=device,
            dtype=torch.float32,
        )  

        # --- 3) Prepare image batch for valid images only ---
        valid_imgs = [img_list[i] for i in valid_indices]  
        img_batch_np = np.stack(valid_imgs, axis=0).astype(np.float32)

        img_t = torch.from_numpy(img_batch_np)              
        img_t = rearrange(img_t, "b h w c -> b c h w").to(device)  

        # --- 4) Batched warp_affine ---
        dsize = (self.restorer.face_size[1], self.restorer.face_size[0])  

        face_t = K.geometry.transform.warp_affine(
            img_t,
            affine_matrix_t,  
            dsize=dsize,
            mode="bilinear",
            padding_mode="fill",
            fill_value=torch.tensor([127.0, 127.0, 127.0], device=device),
            align_corners=True,
        )  

        _, _, h_face, w_face = face_t.shape
        box = [0, 0, w_face, h_face]
        boxes = [box for _ in range(len(valid_indices))]

        # --- 5) Resize all warped faces to (resolution, resolution) ---
        face_t = torch.nn.functional.interpolate(
            face_t,
            size=(self.resolution, self.resolution),
            mode="bicubic",
            align_corners=False,
        )  

        # --- 6) Convert affine matrices to numpy for external use ---
        affine_matrix_np = affine_matrix_t.detach().cpu().numpy() 

        return face_t, boxes, affine_matrix_np, valid_imgs, valid_indices




    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        # image = rearrange(image, "c h w-> h w c").numpy()
        if self.fa is None:
            landmark_coordinates = np.array(self.detect_facial_landmarks(image))
            lm68 = mediapipe_lm478_to_face_alignment_lm68(landmark_coordinates)
        else:
            detected_faces = self.fa.get_landmarks(image)
            if detected_faces is None:
                raise RuntimeError("Face not detected")
            lm68 = detected_faces[0]

        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)
        # print(lmk3_)
        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
        )
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix



    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        
        image = image.to(self.device, dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]



    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")
        if self.mask == "fix_mask":
            results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]
        else:
            results = [self.preprocess_one_masked_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """
    lm478: [B, 478, 3] or [478,3]
    """
    # lm478[..., 0] *= W
    # lm478[..., 1] *= H
    landmarks_extracted = []
    for index in landmark_points_68:
        x = lm478[index][0]
        y = lm478[index][1]
        landmarks_extracted.append((x, y))
    return np.array(landmarks_extracted)


landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


# Refer to https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
mouth_surround_landmarks = [
    164,
    165,
    167,
    92,
    186,
    57,
    43,
    106,
    182,
    83,
    18,
    313,
    406,
    335,
    273,
    287,
    410,
    322,
    391,
    393,
]

face_surround_landmarks = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    435,
    433,
    411,
    425,
    423,
    327,
    326,
    94,
    97,
    98,
    203,
    205,
    187,
    213,
    215,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
]

if __name__ == "__main__":
    image_processor = ImageProcessor(512, mask="fix_mask")
    video = cv2.VideoCapture("/mnt/bn/maliva-gen-ai-v2/chunyu.li/HDTF/original/val/RD_Radio57_000.mp4")
    while True:
        ret, frame = video.read()
        # if not ret:
        #     break

        # cv2.imwrite("image.jpg", frame)

        frame = rearrange(torch.Tensor(frame).type(torch.uint8), "h w c ->  c h w")
        # face, masked_face, _ = image_processor.preprocess_fixed_mask_image(frame, affine_transform=True)
        face, _, _ = image_processor.affine_transform(frame)

        break

    face = (rearrange(face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    cv2.imwrite("face.jpg", face)

    # masked_face = (rearrange(masked_face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    # cv2.imwrite("masked_face.jpg", masked_face)
