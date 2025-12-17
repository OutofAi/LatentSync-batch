# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
from einops import rearrange
import kornia
import torch 
import torch.nn.functional as F

import torch

def transformation_from_points_batch(points1, points0, smooth=True, p_bias=None,
                                           device=None, dtype=torch.float32):
    """
    Torch version of transformation_from_points_batch.

    points1: (B, N, 2) or (N, 2)  – source
    points0: (N, 2)               – target template (e.g. self.restorer.face_template)

    Returns:
        M_out: (B, 2, 3) or (2, 3) torch tensor
        None  (to mirror original API)
    """
    # Convert inputs to tensors if they are numpy or list
    if not isinstance(points1, torch.Tensor):
        points1 = torch.as_tensor(points1)
    if not isinstance(points0, torch.Tensor):
        points0 = torch.as_tensor(points0)

    # Set device / dtype
    if device is None:
        device = points1.device
    points1 = points1.to(device=device, dtype=dtype)
    points0 = points0.to(device=device, dtype=dtype)

    original_ndim = points1.dim()  # 2 -> single, 3 -> batched

    # Ensure batched shape
    if original_ndim == 2:
        points1 = points1.unsqueeze(0)  # (1, N, 2)

    B, N, D = points1.shape  # D should be 2

    # points0 is (N, 2) template -> broadcast to (B, N, 2)
    if points0.dim() == 2:
        points2 = points0.unsqueeze(0).expand(B, N, 2).clone()
    else:
        raise ValueError("points0 is expected to be (N, 2) in this setup.")

    # --- Center ---
    c1 = points1.mean(dim=1, keepdim=True)  # (B, 1, 2)
    c2 = points2.mean(dim=1, keepdim=True)  # (B, 1, 2)
    points1c = points1 - c1                 # (B, N, 2)
    points2c = points2 - c2                 # (B, N, 2)

    # --- Scale ---
    # np.std uses ddof=0 -> torch.std(..., unbiased=False)
    s1 = points1c.std(dim=(1, 2), keepdim=True, unbiased=False)  # (B,1,1)
    s2 = points2c.std(dim=(1, 2), keepdim=True, unbiased=False)  # (B,1,1)

    points1n = points1c / s1
    points2n = points2c / s2

    # --- Cross-covariance ---
    H = torch.einsum("bni,bnj->bij", points1n, points2n)  # (B, 2, 2)

    # --- Batched SVD ---
    # torch.linalg.svd returns U, S, Vh with Vh = V^H
    U, S, Vh = torch.linalg.svd(H)        # each (B, 2, 2)
    R = (U @ Vh).transpose(1, 2)          # (B, 2, 2), equivalent to (U @ Vt).T

    # --- Scale rotation ---
    scale = s2 / s1                       # (B,1,1)
    sR = scale * R                        # (B, 2, 2)

    # --- Translation ---
    c1v = c1.transpose(1, 2)              # (B, 2, 1)
    c2v = c2.transpose(1, 2)              # (B, 2, 1)
    T = c2v - scale * (R @ c1v)           # (B, 2, 1)

    # --- Assemble affine ---
    M = torch.cat([sR, T], dim=2)         # (B, 2, 3)

    # --- Optional smooth bias ---
    if smooth:
        if N <= 2:
            raise ValueError("Need at least 3 points for smooth bias (index 2).")
        bias = points2n[:, 2, :] - points1n[:, 2, :]   # (B, 2)
        M[:, :, 2] = M[:, :, 2] + bias                 # add to translation

    # Restore original shape
    if original_ndim == 2:
        M_out = M[0]   # (2, 3)
    else:
        M_out = M      # (B, 2, 3)

    return M_out, None

def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias

class AlignRestore(object):
    def __init__(self, align_points=3, resolution=256, device="cuda", dtype=torch.float32):
        if align_points == 3:
            self.upscale_factor = 1
            self.crop_ratio = (2.8, 2.8)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * 2.8
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
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up))
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8)
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        if np.max(upsample_img) > 256:
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        return upsample_img

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img
    
    def compute_affine(self, landmarks3, smooth=True):

        affine_nps = []
        for item in landmarks3:
            affine_np, _ = transformation_from_points(
                item, self.face_template, smooth, self.p_bias
            )
            affine_nps.append(affine_np)
        return affine_nps


    def align_warp_face(self, img, lmks3, smooth=True, border_mode="constant"):
        affine_matrix, self.p_bias = transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=[127, 127, 127]
        )
        return cropped_face, affine_matrix


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.5, device=None, dtype=torch.float32):
        self.smoothAlpha = smoothAlpha
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # last smoothed frame: (N, 2) torch tensor on device, or None
        self.pts_last = None

    def set_smooth_apha(self, smoothAlpha):
        self.smoothAlpha = smoothAlpha

    def reset(self):
        self.pts_last = None

    def _to_tensor(self, pts):
        """Helper: convert numpy/list/tensor to torch on the right device/dtype."""
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts)
        elif not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts)
        return pts.to(device=self.device, dtype=self.dtype)

    def smooth_batch(self, pts_cur):
        """
        pts_cur: (N, 2) or (T, N, 2), numpy or torch.
        Returns: list of T elements, each (N, 2) numpy array (same as your current API).
        """
        pts_cur_t = self._to_tensor(pts_cur)  # -> torch, (N,2) or (T,N,2)

        # Normalize to (T, N, 2)
        if pts_cur_t.dim() == 2:
            pts_cur_t = pts_cur_t.unsqueeze(0)  # (1, N, 2)
        if pts_cur_t.dim() != 3 or pts_cur_t.shape[-1] != 2:
            raise ValueError(
                f"Expected pts_cur to be (N,2) or (T,N,2); got {tuple(pts_cur_t.shape)}"
            )

        T, N, D = pts_cur_t.shape
        if D != 2:
            raise ValueError(f"Last dim must be 2, got {D}")

        pts_update = torch.empty_like(pts_cur_t, device=self.device, dtype=self.dtype)

        pts_last = self.pts_last  # may be None or (N,2) tensor

        for t in range(T):
            cur = pts_cur_t[t]  # (N, 2)

            if pts_last is None:
                # First ever call: no history → no smoothing
                prev = cur
            else:
                if pts_last.shape != (N, 2):
                    raise ValueError(
                        f"Shape mismatch: last points {tuple(pts_last.shape)}, "
                        f"current frame {(N, 2)}"
                    )
                prev = pts_last

            # Compute width from current frame
            x_coords = cur[:, 0]                 # (N,)
            x1 = x_coords.min()
            x2 = x_coords.max()
            width = (x2 - x1).clamp_min(1e-6)    # avoid div by 0

            # Squared distance between new and old points
            diff = cur - prev                    # (N, 2)
            tmp = (diff ** 2).sum(dim=-1)        # (N,)

            # Weight per point
            w = torch.exp(-tmp / (width * self.smoothAlpha))  # (N,)

            # Expand to (N, 1) for broadcasting
            w_expanded = w.unsqueeze(-1)         # (N, 1)

            # Update: convex combination of prev and cur
            upd = prev * w_expanded + cur * (1.0 - w_expanded)  # (N, 2)
            pts_update[t] = upd

            # This frame becomes "previous"
            pts_last = upd

        # Store last frame for next call
        self.pts_last = pts_last.detach()

        # Return as list[(N,2)] of numpy arrays to match your existing usage
        return [pts_update[t] for t in range(T)]

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
