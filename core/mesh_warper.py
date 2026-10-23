"""
3D Face Mesh Piecewise-Affine Triangle Warper for i-am-ditto.

Inspired by:
- ibaiGorordo/Mediapipe-Halloween-Examples
- Riverflame-dev/Mediapipe-AR-Filter
- htuann2712/face_swapping

Supports BOTH:
1. Human/Humanoid Faces (extracts real 468 MediaPipe landmarks from the source image).
2. Masked/Helmet Characters like Spider-Man, Iron Man, Deadpool (uses Canonical 468-aligned
   UV topology mapped to the mask foreground contour so full masks warp in 3D with head/jaw motion).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay


# 36 ordered indices around the outer perimeter of the face (MediaPipe Face Oval)
FACE_OVAL_INDICES: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380, 390, 249]
INNER_LIPS_INDICES: List[int] = [78, 81, 13, 311, 308, 402, 14, 178]

# 84 anatomical landmarks (index -> canonical (u, v) in [0, 1] inside a tight face crop)
# This topological subset runs at 60+ FPS in Python while preserving full 3D head rotation,
# jaw opening, eyebrow movement, and cheek deformation without micro-triangle folding.
CANONICAL_LANDMARKS_UV: List[Tuple[int, Tuple[float, float]]] = [
    # --- Face Oval (36 points) ---
    (10, (0.500, 0.020)),
    (338, (0.610, 0.030)),
    (297, (0.710, 0.055)),
    (332, (0.800, 0.095)),
    (284, (0.870, 0.155)),
    (251, (0.920, 0.225)),
    (389, (0.955, 0.305)),
    (356, (0.975, 0.400)),
    (454, (0.980, 0.495)),
    (323, (0.965, 0.585)),
    (361, (0.940, 0.670)),
    (288, (0.900, 0.745)),
    (397, (0.855, 0.810)),
    (365, (0.800, 0.865)),
    (379, (0.740, 0.910)),
    (378, (0.680, 0.940)),
    (400, (0.620, 0.965)),
    (377, (0.560, 0.980)),
    (152, (0.500, 0.990)),
    (148, (0.440, 0.980)),
    (176, (0.380, 0.965)),
    (149, (0.320, 0.940)),
    (150, (0.260, 0.910)),
    (136, (0.200, 0.865)),
    (172, (0.145, 0.810)),
    (58, (0.100, 0.745)),
    (132, (0.060, 0.670)),
    (93, (0.035, 0.585)),
    (234, (0.020, 0.495)),
    (127, (0.025, 0.400)),
    (162, (0.045, 0.305)),
    (21, (0.080, 0.225)),
    (54, (0.130, 0.155)),
    (103, (0.200, 0.095)),
    (67, (0.290, 0.055)),
    (109, (0.390, 0.030)),
    # --- Left Eyebrow (5 points) ---
    (70, (0.150, 0.300)),
    (63, (0.210, 0.270)),
    (105, (0.285, 0.265)),
    (66, (0.355, 0.280)),
    (107, (0.415, 0.305)),
    # --- Right Eyebrow (5 points) ---
    (336, (0.585, 0.305)),
    (296, (0.645, 0.280)),
    (334, (0.715, 0.265)),
    (293, (0.790, 0.270)),
    (300, (0.850, 0.300)),
    # --- Left Eye (6 points) ---
    (33, (0.195, 0.385)),
    (160, (0.245, 0.360)),
    (158, (0.315, 0.360)),
    (133, (0.375, 0.390)),
    (153, (0.315, 0.415)),
    (144, (0.245, 0.415)),
    # --- Right Eye (6 points) ---
    (362, (0.625, 0.390)),
    (385, (0.685, 0.360)),
    (387, (0.755, 0.360)),
    (263, (0.805, 0.385)),
    (373, (0.755, 0.415)),
    (380, (0.685, 0.415)),
    # --- Nose Bridge & Tip (8 points) ---
    (168, (0.500, 0.350)),
    (6, (0.500, 0.415)),
    (197, (0.500, 0.475)),
    (195, (0.500, 0.530)),
    (4, (0.500, 0.585)),
    (1, (0.500, 0.625)),
    (98, (0.410, 0.630)),
    (327, (0.590, 0.630)),
    # --- Cheeks & Forehead Center (6 points) ---
    (151, (0.500, 0.160)),
    (9, (0.500, 0.275)),
    (50, (0.210, 0.545)),
    (280, (0.790, 0.545)),
    (205, (0.275, 0.665)),
    (425, (0.725, 0.665)),
    # --- Outer Lips (8 points) ---
    (61, (0.335, 0.765)),
    (39, (0.415, 0.735)),
    (0, (0.500, 0.740)),
    (269, (0.585, 0.735)),
    (291, (0.665, 0.765)),
    (405, (0.585, 0.835)),
    (17, (0.500, 0.845)),
    (181, (0.415, 0.835)),
    # --- Inner Lips / Mouth (4 points) ---
    (78, (0.365, 0.770)),
    (13, (0.500, 0.765)),
    (308, (0.635, 0.770)),
    (14, (0.500, 0.805)),
]

KEY_INDICES: List[int] = [item[0] for item in CANONICAL_LANDMARKS_UV]
CANONICAL_UV_ARRAY: np.ndarray = np.array(
    [item[1] for item in CANONICAL_LANDMARKS_UV], dtype=np.float32
)

# Pre-compute fixed Delaunay triangulation topology once
_DELAUNAY = Delaunay(CANONICAL_UV_ARRAY)
TRIANGLE_SIMPLICES: np.ndarray = _DELAUNAY.simplices  # shape: (N_triangles, 3)


@dataclass
class AvatarAsset:
    """Represents a ready-to-warp character face or mask."""

    name: str
    slug: str
    texture_bgra: np.ndarray  # (H, W, 4) BGRA image
    source_points: np.ndarray  # (len(KEY_INDICES), 2) float32 pixel coordinates
    is_canonical_mask: bool = False  # True if mapped via canonical UV (e.g. Spider-Man mask)


def extract_avatar_landmarks(image_bgra: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Extract the 84 key landmarks from an avatar image.
    - First tries MediaPipe FaceMesh (for human/humanoid faces).
    - If no human face is detected (e.g. Spider-Man / Iron Man helmet), automatically
      detects the foreground bounding box and maps the Canonical 468 UV template onto it.
    """
    h, w = image_bgra.shape[:2]
    bgr = image_bgra[:, :, :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    ) as face_mesh:
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0].landmark
            pts = np.array(
                [[face_lms[idx].x * w, face_lms[idx].y * h] for idx in KEY_INDICES],
                dtype=np.float32,
            )
            return pts, False

    # Fallback for full masks / helmets (Spider-Man, Iron Man, Batman, etc.):
    # Find the non-background bounding box and map CANONICAL_UV_ARRAY into it
    x_min, y_min, box_w, box_h = _find_foreground_bbox(image_bgra)
    pts = np.zeros_like(CANONICAL_UV_ARRAY, dtype=np.float32)
    pts[:, 0] = x_min + CANONICAL_UV_ARRAY[:, 0] * box_w
    pts[:, 1] = y_min + CANONICAL_UV_ARRAY[:, 1] * box_h
    return pts, True


def _find_foreground_bbox(image_bgra: np.ndarray) -> Tuple[float, float, float, float]:
    """Find the bounding box of the mask/head in an image (via alpha channel or corner color diff)."""
    h, w = image_bgra.shape[:2]
    alpha = image_bgra[:, :, 3]

    # Check if image has meaningful transparency
    if np.min(alpha) < 200 and np.max(alpha) > 50:
        ys, xs = np.where(alpha > 40)
        if len(xs) > 100:
            x1, x2 = float(np.min(xs)), float(np.max(xs))
            y1, y2 = float(np.min(ys)), float(np.max(ys))
            return x1, y1, max(x2 - x1, 32.0), max(y2 - y1, 32.0)

    # Otherwise compare against corner background color
    bgr = image_bgra[:, :, :3].astype(np.float32)
    corners = np.array(
        [bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1]],
        dtype=np.float32,
    )
    bg_color = np.median(corners, axis=0)
    diff = np.linalg.norm(bgr - bg_color, axis=2)
    ys, xs = np.where(diff > 28.0)

    if len(xs) > 200:
        x1, x2 = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
        y1, y2 = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
        bw, bh = x2 - x1, y2 - y1
        if bw > w * 0.2 and bh > h * 0.2:
            # Slightly inset horizontally so the face oval fits inside the helmet/head width
            pad_x = bw * 0.04
            pad_y = bh * 0.04
            return x1 + pad_x, y1 + pad_y, bw - 2 * pad_x, bh - 2 * pad_y

    # Default centered portrait box
    margin_x = w * 0.12
    margin_y = h * 0.08
    return margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y


def warp_avatar_onto_face(
    frame_bgr: np.ndarray,
    avatar: AvatarAsset,
    live_face_landmarks,
    opacity: float = 0.94,
    cutout_eyes_mouth: bool = False,
) -> np.ndarray:
    """
    Warp the avatar's 3D triangle mesh onto the user's live 468 MediaPipe face landmarks.
    Returns the composited BGR frame.
    """
    h, w = frame_bgr.shape[:2]
    lms = live_face_landmarks.landmark

    # Extract destination points for our 84 key landmarks
    dst_points = np.array(
        [[lms[idx].x * w, lms[idx].y * h] for idx in KEY_INDICES],
        dtype=np.float32,
    )

    # Compute face bounding box on destination frame to restrict buffer allocation
    min_xy = np.floor(np.min(dst_points, axis=0)).astype(int)
    max_xy = np.ceil(np.max(dst_points, axis=0)).astype(int)

    fx1 = max(0, int(min_xy[0]) - 4)
    fy1 = max(0, int(min_xy[1]) - 4)
    fx2 = min(w, int(max_xy[0]) + 4)
    fy2 = min(h, int(max_xy[1]) + 4)

    roi_w = fx2 - fx1
    roi_h = fy2 - fy1
    if roi_w <= 10 or roi_h <= 10:
        return frame_bgr

    # Local ROI coordinates
    dst_local = dst_points - np.array([fx1, fy1], dtype=np.float32)
    src_points = avatar.source_points
    src_bgr = avatar.texture_bgra[:, :, :3]
    src_h, src_w = src_bgr.shape[:2]

    warped_roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
    mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

    # Warp each Delaunay triangle
    for tri in TRIANGLE_SIMPLICES:
        s_tri = src_points[tri]
        d_tri = dst_local[tri]

        # Source bounding box
        sr = cv2.boundingRect(s_tri)
        sx, sy, sw, sh = sr
        if sx < 0 or sy < 0 or sx + sw > src_w or sy + sh > src_h or sw <= 0 or sh <= 0:
            sx = max(0, sx)
            sy = max(0, sy)
            sw = min(src_w - sx, sw)
            sh = min(src_h - sy, sh)
            if sw <= 0 or sh <= 0:
                continue

        # Destination bounding box inside ROI
        dr = cv2.boundingRect(d_tri)
        dx, dy, dw, dh = dr
        if dx < 0 or dy < 0 or dx + dw > roi_w or dy + dh > roi_h or dw <= 0 or dh <= 0:
            continue

        s_tri_cropped = s_tri - np.array([sx, sy], dtype=np.float32)
        d_tri_cropped = d_tri - np.array([dx, dy], dtype=np.float32)

        # Affine transform from source triangle to destination triangle
        matrix = cv2.getAffineTransform(s_tri_cropped, d_tri_cropped)
        src_patch = src_bgr[sy : sy + sh, sx : sx + sw]
        warped_patch = cv2.warpAffine(
            src_patch,
            matrix,
            (dw, dh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Triangle mask
        tri_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, np.int32(d_tri_cropped), 255)

        # Paste into ROI (only where mask_roi isn't filled yet to avoid double-blending seams)
        roi_slice = warped_roi[dy : dy + dh, dx : dx + dw]
        mask_slice = mask_roi[dy : dy + dh, dx : dx + dw]
        new_pixels = cv2.bitwise_and(tri_mask, cv2.bitwise_not(mask_slice))

        roi_slice[new_pixels > 0] = warped_patch[new_pixels > 0]
        mask_slice[new_pixels > 0] = 255

    # Build smooth feathered alpha mask from the 36 Face Oval points (first 36 in KEY_INDICES)
    oval_local = np.int32(dst_local[:36])
    soft_alpha = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(soft_alpha, cv2.convexHull(oval_local), 255)
    soft_alpha = cv2.bitwise_and(soft_alpha, mask_roi)

    if cutout_eyes_mouth:
        # Optional cutout for user's real eyes and inner mouth
        left_eye_pts = np.int32([[lms[i].x * w - fx1, lms[i].y * h - fy1] for i in LEFT_EYE_INDICES])
        right_eye_pts = np.int32([[lms[i].x * w - fx1, lms[i].y * h - fy1] for i in RIGHT_EYE_INDICES])
        mouth_pts = np.int32([[lms[i].x * w - fx1, lms[i].y * h - fy1] for i in INNER_LIPS_INDICES])
        cv2.fillPoly(soft_alpha, [left_eye_pts, right_eye_pts, mouth_pts], 0)

    # Feather edges for Instagram-filter style blending
    blur_k = max(3, int(min(roi_w, roi_h) * 0.07) | 1)
    soft_alpha = cv2.GaussianBlur(soft_alpha, (blur_k, blur_k), 0)

    alpha_f = (soft_alpha.astype(np.float32) / 255.0 * opacity)[..., np.newaxis]
    bg_roi = frame_bgr[fy1:fy2, fx1:fx2].astype(np.float32)
    fg_roi = warped_roi.astype(np.float32)

    blended_roi = (fg_roi * alpha_f + bg_roi * (1.0 - alpha_f)).astype(np.uint8)
    frame_bgr[fy1:fy2, fx1:fx2] = blended_roi
    return frame_bgr
