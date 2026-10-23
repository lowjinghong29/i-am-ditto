"""
MediaPipe Holistic Tracker & Neon-Green Cyber Stick Renderer for i-am-ditto.
"""

from typing import Optional
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Futuristic Neon Green styling for MediaPipe sticks & joints
NEON_GREEN = (100, 255, 0)       # BGR for bright neon green
NEON_CYAN = (255, 255, 0)        # BGR for glowing cyan joint nodes
DARK_GREEN_MESH = (60, 190, 0)   # Subtle green for face wireframe

POSE_STICK_SPEC = mp_drawing.DrawingSpec(color=NEON_GREEN, thickness=3, circle_radius=4)
POSE_JOINT_SPEC = mp_drawing.DrawingSpec(color=NEON_CYAN, thickness=2, circle_radius=5)

HAND_STICK_SPEC = mp_drawing.DrawingSpec(color=NEON_GREEN, thickness=2, circle_radius=3)
HAND_JOINT_SPEC = mp_drawing.DrawingSpec(color=NEON_CYAN, thickness=2, circle_radius=3)

FACE_CONTOUR_SPEC = mp_drawing.DrawingSpec(color=NEON_GREEN, thickness=1, circle_radius=1)
FACE_MESH_SPEC = mp_drawing.DrawingSpec(color=DARK_GREEN_MESH, thickness=1, circle_radius=0)


class HolisticScanner:
    """Wraps MediaPipe Holistic for simultaneous Face (478), Pose (33), and Hands (42) scanning."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=True,
        )

    def process(self, frame_bgr: np.ndarray):
        """Process a BGR frame and return MediaPipe Holistic results."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        return results

    def draw_green_sticks(
        self,
        frame_bgr: np.ndarray,
        results,
        draw_face_mesh: bool = True,
        draw_body_and_hands: bool = True,
    ) -> None:
        """Draw futuristic neon-green sticks on the body, hands, and optionally the face."""
        if draw_body_and_hands:
            # 1. Body Pose Skeleton Sticks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=POSE_JOINT_SPEC,
                    connection_drawing_spec=POSE_STICK_SPEC,
                )

            # 2. Left & Right Hand Sticks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=HAND_JOINT_SPEC,
                    connection_drawing_spec=HAND_STICK_SPEC,
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=HAND_JOINT_SPEC,
                    connection_drawing_spec=HAND_STICK_SPEC,
                )

        if draw_face_mesh and results.face_landmarks:
            # 3. Cybernetic Green Face Mesh
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=FACE_MESH_SPEC,
            )
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=FACE_CONTOUR_SPEC,
            )

    def close(self):
        self.holistic.close()


def draw_futuristic_hud(
    frame_bgr: np.ndarray,
    fps: float,
    active_avatar_name: Optional[str],
    ai_status: str,
    last_voice_text: str,
    sticks_enabled: bool,
    face_wireframe_enabled: bool,
    typing_mode: bool = False,
    typed_buffer: str = "",
) -> None:
    """Render the i-am-ditto translucent cybernetic HUD."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Top bar backdrop
    cv2.rectangle(overlay, (0, 0), (w, 68), (12, 18, 12), -1)
    # Bottom bar backdrop
    cv2.rectangle(overlay, (0, h - 64), (w, h), (12, 18, 12), -1)
    cv2.addWeighted(overlay, 0.68, frame_bgr, 0.32, 0, frame_bgr)

    # Top accent line
    cv2.line(frame_bgr, (0, 68), (w, 68), NEON_GREEN, 1)
    cv2.line(frame_bgr, (0, h - 64), (w, h - 64), NEON_GREEN, 1)

    # Brand & FPS
    mode_label = f"DITTO // {active_avatar_name.upper()}" if active_avatar_name else "DITTO // SCANNER MODE"
    cv2.putText(
        frame_bgr,
        mode_label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        NEON_GREEN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"FPS: {fps:.0f} | Sticks: {'ON' if sticks_enabled else 'OFF'} | FaceGrid: {'ON' if face_wireframe_enabled else 'OFF'}",
        (16, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 230, 200),
        1,
        cv2.LINE_AA,
    )

    # AI Resolver Status (Top Right)
    if ai_status:
        status_color = NEON_CYAN if "Searching" in ai_status or "Generating" in ai_status else NEON_GREEN
        text_size = cv2.getTextSize(ai_status, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        cv2.putText(
            frame_bgr,
            ai_status,
            (max(16, w - text_size[0] - 18), 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            1,
            cv2.LINE_AA,
        )

    # Bottom bar: Voice command or text input prompt
    if typing_mode:
        prompt_str = f"TYPE CHARACTER & PRESS ENTER > {typed_buffer}_"
        cv2.putText(
            frame_bgr,
            prompt_str,
            (16, h - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            NEON_CYAN,
            2,
            cv2.LINE_AA,
        )
    else:
        voice_display = (
            f'VOICE: "{last_voice_text}"'
            if last_voice_text
            else 'Say: "Hey I want to become Spider-Man" (or press [T] to type any character)'
        )
        cv2.putText(
            frame_bgr,
            voice_display,
            (16, h - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (240, 255, 240),
            1,
            cv2.LINE_AA,
        )

    controls_hint = "[T] Type Avatar  [1-4] Presets  [S] Body Sticks  [F] Face Grid  [E] Eye/Mouth Cutout  [C] Clear  [Q] Quit"
    cv2.putText(
        frame_bgr,
        controls_hint,
        (16, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 200, 150),
        1,
        cv2.LINE_AA,
    )
