# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
from einops import rearrange
import kornia
import torch 
import torch.nn.functional as F

def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    """
    Batched similarity transform from points1 -> points0.

    points1: (N, 2) or (B, N, 2)
    points0: (N, 2)           (template, shared across batch)

    Returns:
        M:  (2, 3)      if input was (N, 2)
            (B, 2, 3)   if input was (B, N, 2)
        p_bias: always None (kept for API compatibility)
    """
    # Convert to arrays
    points1 = np.asarray(points1, dtype=np.float64)
    points0 = np.asarray(points0, dtype=np.float64)

    original_ndim = points1.ndim  # 2 -> single, 3 -> batched

    B, N, D = points1.shape

    points2 = np.broadcast_to(points0[None, ...], (B, N, 2)).copy()

    # --- Center ---
    c1 = points1.mean(axis=1, keepdims=True)  # (B, 1, 2)
    c2 = points2.mean(axis=1, keepdims=True)  # (B, 1, 2)
    points1c = points1 - c1                   # (B, N, 2)
    points2c = points2 - c2                   # (B, N, 2)

    # --- Scale ---
    s1 = points1c.std(axis=(1, 2), keepdims=True)  # (B,1,1)
    s2 = points2c.std(axis=(1, 2), keepdims=True)  # (B,1,1)

    points1n = points1c / s1                       # (B, N, 2)
    points2n = points2c / s2                       # (B, N, 2)

    # --- Cross-covariance (batched) ---
    # H_b = points1n[b].T @ points2n[b]
    H = np.einsum("bni, bnj -> bij", points1n, points2n)  # (B, 2, 2)

    # --- Batched SVD ---
    U, S, Vt = np.linalg.svd(H)                     # each (B, 2, 2)
    R = np.matmul(U, Vt).transpose(0, 2, 1)         # (B, 2, 2), same as (U@Vt).T

    # --- Scale rotation ---
    scale = (s2 / s1)                               # (B,1,1)
    sR = scale * R                                  # (B, 2, 2)

    # --- Translation ---
    c1v = np.transpose(c1, (0, 2, 1))               # (B, 2, 1)
    c2v = np.transpose(c2, (0, 2, 1))               # (B, 2, 1)
    T = c2v - scale * np.matmul(R, c1v)             # (B, 2, 1)

    # --- Assemble affine ---
    M = np.concatenate([sR, T], axis=2)             # (B, 2, 3)

    # --- Optional smooth bias (no p_bias reuse, as requested) ---
    if smooth:
        if N <= 2:
            raise ValueError("Need at least 3 points for smooth bias (index 2).")
        # points1n / points2n are normalized as in your original code
        bias = points2n[:, 2, :] - points1n[:, 2, :]   # (B, 2)
        M[:, :, 2] = M[:, :, 2] + bias                 # add to translation column

    # Preserve original return shape for non-batched input
    if original_ndim == 2:
        M_out = M[0]           # (2,3)
    else:
        M_out = M              # (B,2,3)

    # p_bias is always None now
    return M_out, None



class AlignRestore(object):
    def __init__(self, align_points=3, resolution=256, device="cuda", dtype=torch.float32):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = resolution / 256 * 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.device = device    
            self.dtype = dtype
            self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
            self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)
    
    torch.no_grad()
    def restore_imgs(self, input_imgs: np.ndarray, faces: torch.Tensor, affine_matrices):
        """
        Batched version of restore_img.

        input_imgs: (B, H, W, C) numpy
        faces:      (B, C, Hf, Wf) tensor (already resized as needed)
        affine_matrices: (B, 2, 3) or (2, 3) np/tensor
        """
        device, dtype = self.device, self.dtype

        # Shapes
        B, H, W, C = input_imgs.shape

        # --- Affine matrices ---
        if isinstance(affine_matrices, np.ndarray):
            affine_matrices = torch.from_numpy(affine_matrices)
        affine_matrices = affine_matrices.to(device=device, dtype=dtype)
        if affine_matrices.dim() == 2:  # (2, 3) -> (1, 2, 3)
            affine_matrices = affine_matrices.unsqueeze(0)
        if affine_matrices.shape[0] == 1 and B > 1:
            affine_matrices = affine_matrices.expand(B, -1, -1)

        # --- Faces ---
        if faces.dim() == 3:
            faces = faces.unsqueeze(0)
        faces = faces.to(device=device, dtype=dtype)  # (B, C, Hf, Wf)

        # --- Inverse affine ---
        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrices)  # (B, 2, 3)

        # --- Warp faces back to frame size, batched ---
        inv_faces = kornia.geometry.transform.warp_affine(
            faces,
            inv_affine_matrix,
            (H, W),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )  # (B, C, H, W)
        inv_faces = (inv_faces / 2 + 0.5).clamp(0, 1) * 255  # back to [0,255]

        # --- Input images to tensor, batched ---
        input_imgs_t = torch.from_numpy(input_imgs).to(device=device, dtype=dtype)  # (B, H, W, C)
        input_imgs_t = rearrange(input_imgs_t, "b h w c -> b c h w")                # (B, C, H, W)

        # --- Mask, batched ---
        mask = self.mask.to(device=device, dtype=dtype)  # (1, 1, Hm, Wm) or (B, 1, Hm, Wm)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1)  # (B, 1, Hm, Wm)

        inv_mask = kornia.geometry.transform.warp_affine(
            mask, inv_affine_matrix, (H, W), padding_mode="zeros"
        )  # (B, 1, H, W)

        # --- Erode (first pass) on GPU, same kernel for whole batch ---
        kernel_size = int(2 * self.upscale_factor)

        # ensure odd kernel to preserve spatial size (optional but recommended)
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2

        inv_mask_erosion = -F.max_pool2d(
            -inv_mask,               # (B,1,H,W)
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
        )

        # --- Compute per-image parameters (still batched) ---
        total_face_area = inv_mask_erosion.flatten(2).sum(dim=2)  # (B, 1)
        w_edge = (total_face_area.sqrt() // 20).to(torch.int64).clamp(min=1)  # (B, 1)
        erosion_radius = (w_edge * 2).clamp(min=1)  # (B, 1)
        blur_sizes = (w_edge * 2 + 1).clamp(min=3)  # (B, 1) odd kernel sizes

        # --- Second erosion (GPU, single radius) ---
        r = int(erosion_radius.max().item())
        r = max(r, 1)

        if r % 2 == 0:
            r += 1

        pad = r // 2

        inv_mask_center = -F.max_pool2d(
            -inv_mask_erosion,  # (B,1,H,W)
            kernel_size=r,
            stride=1,
            padding=pad,
        )


        # --- Gaussian blur, batched (we pick a single blur size for the batch) ---
        # You *could* loop per-image here too; this is a trade-off.
        blur_size = int(blur_sizes.max().item())
        if blur_size % 2 == 0:
            blur_size += 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8

        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center,
            (blur_size, blur_size),
            (sigma, sigma),
        )  # (B, 1, H, W)

        inv_soft_mask_3d = inv_soft_mask.expand(-1, inv_faces.shape[1], -1, -1)  # (B, C, H, W)
        inv_mask_erosion_t = inv_mask_erosion.expand_as(inv_faces)               # (B, C, H, W)

        pasted_face = inv_mask_erosion_t * inv_faces
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_imgs_t  # (B, C, H, W)

        img_back = rearrange(img_back, "b c h w -> b h w c").contiguous().to(dtype=torch.uint8)

        # IMPORTANT: now return a tensor, not numpy
        return img_back  # (B, H, W, C), torch on device



    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape

        # Ensure affine_matrix is a tensor on the correct device
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix)

        affine_matrix = affine_matrix.to(device=self.device, dtype=self.dtype)
        if affine_matrix.dim() == 2:  # (2, 3) -> (1, 2, 3)
            affine_matrix = affine_matrix.unsqueeze(0)

        # Ensure face is on the same device
        face = face.to(device=self.device, dtype=self.dtype).unsqueeze(0)

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)

        inv_face = kornia.geometry.transform.warp_affine(
            face,
            inv_affine_matrix,
            (h, w),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_img = rearrange(
            torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w"
        )

        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, inv_affine_matrix, (h, w), padding_mode="zeros"
        )
        
        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
            ),
        )

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2

        # This step will consume a large amount of GPU memory.
        # inv_mask_center = kornia.morphology.erosion(
        #     inv_mask_erosion, torch.ones((erosion_radius, erosion_radius), device=self.device, dtype=self.dtype)
        # )

        # Run on CPU to avoid consuming a large amount of GPU memory.
        inv_mask_erosion = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        inv_mask_center = torch.from_numpy(inv_mask_center).to(device=self.device, dtype=self.dtype)[None, None, ...]

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img

        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = img_back.cpu().numpy()
        return img_back

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img
    
    def compute_affine(self, landmarks3, smooth=True):
        affine_np, self.p_bias = transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )
        return affine_np


    def align_warp_face(self, img, landmarks3, smooth=True):
        """
        img:
            - (B, H, W, C) np.ndarray    -> batch of images

        landmarks3:
            - (B, N, 2) or list[(N, 2), ...]  -> batched landmarks

        Returns:
            cropped_face(s):
                - (B, Hf, Wf, C) for batch
            affine_matrix(ces):
                - (B, 2, 3) for batch
        """
        # ---- Convert inputs to numpy ----
        img_np = np.asarray(img)
        lm_np = np.asarray(landmarks3, dtype=np.float64)

        img_np_b = img_np
        lm_np_b = lm_np

        # ---- Compute batched affine matrices from landmarks ----
        affine_np, self.p_bias = transformation_from_points(
            lm_np_b, self.face_template, smooth, self.p_bias
        )   # affine_np: (B, 2, 3)

        # Ensure we always have a batch dimension here
        if affine_np.ndim == 2:
            affine_np = affine_np[None, ...]   # (1, 2, 3)

        # ---- Warp with Kornia in batch ----
        img_t = torch.from_numpy(img_np_b).to(device=self.device, dtype=self.dtype)
        img_t = rearrange(img_t, "b h w c -> b c h w")   # (B, C, H, W)

        affine_t = torch.from_numpy(affine_np).to(device=self.device, dtype=self.dtype)  # (B, 2, 3)

        cropped_t = kornia.geometry.transform.warp_affine(
            img_t,
            affine_t,
            (self.face_size[1], self.face_size[0]),  # (H_out, W_out)
            mode="bilinear",
            padding_mode="zeros",   # follow your original logic
        )  # (B, C, Hf, Wf)

        cropped_np = rearrange(cropped_t, "b c h w -> b h w c").cpu().numpy().astype(np.uint8)

        return cropped_np, affine_t


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        # last smoothed frame: (N, 2) or None
        self.pts_last = None

    def smooth(self, pts_cur):
        """
        pts_cur:
            - (N, 2) single frame of landmarks
            - (T, N, 2) sequence of T consecutive frames

        Returns:
            list of length T (or 1), each entry (N, 2) smoothed landmarks
        """
        pts_cur = np.asarray(pts_cur, dtype=np.float32)

        # Normalize to (T, N, 2)
        if pts_cur.ndim == 2:
            pts_cur = pts_cur[None, ...]  # (1, N, 2)
        if pts_cur.ndim != 3 or pts_cur.shape[-1] != 2:
            raise ValueError(
                f"Expected pts_cur to be (N,2) or (T,N,2); got {pts_cur.shape}"
            )

        T, N, D = pts_cur.shape
        if D != 2:
            raise ValueError(f"Last dim must be 2, got {D}")

        pts_update = np.empty_like(pts_cur)

        for t in range(T):
            cur = pts_cur[t]  # (N, 2)

            if self.pts_last is None:
                # First ever call: no history → no smoothing
                prev = cur
            else:
                if self.pts_last.shape != (N, 2):
                    raise ValueError(
                        f"Shape mismatch: last points {self.pts_last.shape}, "
                        f"current frame {(N, 2)}"
                    )
                prev = self.pts_last

            # Compute width from current frame
            x_coords = cur[:, 0]                    # (N,)
            x1 = x_coords.min()
            x2 = x_coords.max()
            width = x2 - x1
            if width <= 0:
                width = 1e-6

            # Squared distance between new and old points
            diff = cur - prev                       # (N, 2)
            tmp = np.sum(diff ** 2, axis=-1)        # (N,)

            # Weight per point
            w = np.exp(-tmp / (width * self.smoothAlpha))  # (N,)

            # Expand to (N, 1) for broadcasting
            w_expanded = w[:, None]                # (N, 1)

            # Update: convex combination of prev and cur
            upd = prev * w_expanded + cur * (1.0 - w_expanded)  # (N, 2)
            pts_update[t] = upd

            # This frame becomes "previous" for:
            # - the next frame in this batch
            # - the first frame in the next call
            self.pts_last = upd.copy()

        # Return as list[(N,2)] to match your existing usage
        return [pts_update[t] for t in range(T)]
