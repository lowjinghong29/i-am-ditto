"""
Full-Body Skeleton AR Overlay for i-am-ditto.
Adapted from ibaiGorordo/Mediapipe-Halloween-Examples (utils/skeleton_pose_utils.py).

Warps skull.png, torso.png, and bone.png onto the 33 MediaPipe Pose body landmarks in real time.
"""

from pathlib import Path
import cv2
import numpy as np


ASSETS_SKELETON_DIR = Path(__file__).resolve().parent.parent / "assets" / "skeleton"

SKULL_INDICES = [6, 3, 10, 9]
TORSO_INDICES = [12, 11, 24, 23]
LIMB_BONE_PAIRS = [
    [12, 14],  # Left upper arm
    [14, 16],  # Left lower arm
    [11, 13],  # Right upper arm
    [13, 15],  # Right lower arm
    [24, 26],  # Left upper leg
    [26, 28],  # Left lower leg
    [23, 25],  # Right upper leg
    [25, 27],  # Right lower leg
]

SKULL_SRC_COORDS = np.array([[192, 330], [446, 349], [324, 673]], dtype=np.float32)


class BodySkeletonOverlay:
    """Warps full-body skeleton bones, ribcage torso, and skull onto MediaPipe Pose landmarks."""

    def __init__(self):
        self.bone_img = cv2.imread(str(ASSETS_SKELETON_DIR / "bone.png"))
        self.torso_img = cv2.imread(str(ASSETS_SKELETON_DIR / "torso.png"))
        self.skull_img = cv2.imread(str(ASSETS_SKELETON_DIR / "skull.png"))
        self.available = (
            self.bone_img is not None
            and self.torso_img is not None
            and self.skull_img is not None
        )

    def draw(self, frame_bgr: np.ndarray, pose_landmarks, include_skull: bool = False) -> np.ndarray:
        """Overlay ribcage torso, arm/leg bones, and optionally skull onto frame_bgr."""
        if not self.available or pose_landmarks is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        kpts = np.array(
            [
                [
                    int(min(max(lm.x * w, 0), w - 1)),
                    int(min(max(lm.y * h, 0), h - 1)),
                ]
                for lm in pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

        if include_skull:
            self._draw_skull(frame_bgr, kpts[SKULL_INDICES], w, h)

        self._draw_torso(frame_bgr, kpts[TORSO_INDICES], w, h)
        for pair in LIMB_BONE_PAIRS:
            self._draw_bone(frame_bgr, kpts[pair], w, h)

        return frame_bgr

    def _draw_skull(self, image: np.ndarray, skull_kpts: np.ndarray, w: int, h: int) -> None:
        mouth = (skull_kpts[2] + skull_kpts[3]) / 2.0
        dst_pts = np.vstack((skull_kpts[:2], mouth)).astype(np.float32)
        matrix = cv2.getAffineTransform(SKULL_SRC_COORDS, dst_pts)
        warped = cv2.warpAffine(self.skull_img, matrix, (w, h))
        mask = warped[:, :, 2] > 15
        image[mask] = warped[mask]

    def _draw_torso(self, image: np.ndarray, torso_kpts: np.ndarray, w: int, h: int) -> None:
        mid_hip = (torso_kpts[2] + torso_kpts[3]) / 2.0
        dst_pts = np.vstack((torso_kpts[:2], mid_hip)).astype(np.float32)
        th, tw = self.torso_img.shape[:2]
        src_pts = np.array([[0, 0], [tw - 1, 0], [tw // 2, th - 1]], dtype=np.float32)
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(self.torso_img, matrix, (w, h))
        mask = warped[:, :, 2] > 15
        image[mask] = warped[mask]

    def _draw_bone(self, image: np.ndarray, bone_kpts: np.ndarray, w: int, h: int) -> None:
        bh, bw = self.bone_img.shape[:2]
        src_pts = np.array([[bw // 2, 0], [bw // 2, bh - 1]], dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, bone_kpts.astype(np.float32))
        if matrix is None:
            return
        warped = cv2.warpAffine(self.bone_img, matrix, (w, h))
        mask = warped[:, :, 2] > 15
        image[mask] = warped[mask]
