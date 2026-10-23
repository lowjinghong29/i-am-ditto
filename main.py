"""
i-am-ditto — Voice & AI Powered Real-Time Human Face & Body AR Transformer.

Features:
1. MediaPipe Holistic neon-green body pose (33), hand (42), and face (478) wireframe sticks.
2. Voice command listener ("Hey I want to become Spider-Man", "Become <anything>").
3. 3-Tier AI Avatar Resolver (Local Library -> Online Search -> AI Generator Fallback).
4. Real-Time 3D Delaunay Triangle Mesh Warping (Instagram-style AR face filter).
"""

import argparse
import time
from typing import Optional
import cv2

from ai.avatar_resolver import AvatarResolver
from ai.voice_listener import VoiceCommandListener
from core.holistic_tracker import HolisticScanner, draw_futuristic_hud
from core.mesh_warper import AvatarAsset, warp_avatar_onto_face


class DittoApp:
    """Main controller orchestrating Holistic tracking, 3D mesh warping, and AI voice resolution."""

    def __init__(self, camera_index: int = 0, initial_avatar: Optional[str] = None):
        self.camera_index = camera_index
        self.active_avatar: Optional[AvatarAsset] = None
        self.ai_status: str = "System Ready"
        self.last_voice_text: str = ""

        # Visual toggles
        self.sticks_enabled: bool = True
        self.face_wireframe_enabled: bool = True
        self.cutout_eyes_mouth: bool = False

        # In-window typing prompt state
        self.typing_mode: bool = False
        self.typed_buffer: str = ""

        self.scanner = HolisticScanner()
        self.resolver = AvatarResolver(on_status_update=self._on_status)
        self.voice = VoiceCommandListener(
            on_command=self._on_voice_command,
            on_transcript=self._on_voice_transcript,
            on_status=self._on_status,
        )

        if initial_avatar:
            self.request_avatar(initial_avatar)

    def _on_status(self, status: str) -> None:
        self.ai_status = status

    def _on_voice_transcript(self, text: str) -> None:
        self.last_voice_text = text

    def _on_voice_command(self, action: str, target: str) -> None:
        if action == "clear":
            self.active_avatar = None
            self.face_wireframe_enabled = True
            self.ai_status = "Scanner Mode (Filter Cleared)"
        elif action == "become" and target:
            self.request_avatar(target)

    def request_avatar(self, character_name: str) -> None:
        """Trigger asynchronous AI resolution for any character name."""
        def _apply_avatar(asset: AvatarAsset) -> None:
            self.active_avatar = asset
            # Hide green face wireframe when wearing a character mask so the mask is crystal clear
            self.face_wireframe_enabled = False

        self.resolver.resolve_async(character_name, on_ready=_apply_avatar)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")

        # Set 720p resolution if supported by webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.voice.start()
        prev_time = time.time()

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                # Mirror frame for natural selfie interaction
                frame = cv2.flip(frame, 1)

                # 1. Scan Face + Body + Hands with MediaPipe Holistic
                results = self.scanner.process(frame)

                # 2. If an avatar is active and face is detected, warp 3D avatar mesh onto face
                if self.active_avatar is not None and results.face_landmarks:
                    warp_avatar_onto_face(
                        frame,
                        self.active_avatar,
                        results.face_landmarks,
                        opacity=0.94,
                        cutout_eyes_mouth=self.cutout_eyes_mouth,
                    )

                # 3. Draw neon-green MediaPipe body & hand sticks (+ optional face mesh)
                self.scanner.draw_green_sticks(
                    frame,
                    results,
                    draw_face_mesh=self.face_wireframe_enabled,
                    draw_body_and_hands=self.sticks_enabled,
                )

                # 4. Calculate FPS & draw Futuristic HUD
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                draw_futuristic_hud(
                    frame,
                    fps=fps,
                    active_avatar_name=self.active_avatar.name if self.active_avatar else None,
                    ai_status=self.ai_status,
                    last_voice_text=self.last_voice_text,
                    sticks_enabled=self.sticks_enabled,
                    face_wireframe_enabled=self.face_wireframe_enabled,
                    typing_mode=self.typing_mode,
                    typed_buffer=self.typed_buffer,
                )

                cv2.imshow("i-am-ditto // AI Face & Body AR Scanner", frame)

                key = cv2.waitKey(5) & 0xFF
                if key == 255:
                    continue

                if self._handle_key(key):
                    break
        finally:
            self.voice.stop()
            self.scanner.close()
            cap.release()
            cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        """Handle keyboard shortcuts and on-screen character typing. Returns True to quit."""
        if self.typing_mode:
            if key in (13, 10):  # Enter
                query = self.typed_buffer.strip()
                self.typing_mode = False
                self.typed_buffer = ""
                if query:
                    self.last_voice_text = f"become {query}"
                    self.request_avatar(query)
            elif key == 27:  # ESC cancels typing mode
                self.typing_mode = False
                self.typed_buffer = ""
            elif key in (8, 127):  # Backspace
                self.typed_buffer = self.typed_buffer[:-1]
            elif 32 <= key <= 126:
                self.typed_buffer += chr(key)
            return False

        # Standard shortcuts when not typing
        if key in (ord("q"), 27):
            return True
        if key in (ord("t"), ord("T")):
            self.typing_mode = True
            self.typed_buffer = ""
        elif key in (ord("s"), ord("S")):
            self.sticks_enabled = not self.sticks_enabled
        elif key in (ord("f"), ord("F")):
            self.face_wireframe_enabled = not self.face_wireframe_enabled
        elif key in (ord("e"), ord("E")):
            self.cutout_eyes_mouth = not self.cutout_eyes_mouth
        elif key in (ord("c"), ord("C")):
            self.active_avatar = None
            self.face_wireframe_enabled = True
            self.ai_status = "Scanner Mode (Filter Cleared)"
        elif key == ord("1"):
            self.request_avatar("spiderman")
        elif key == ord("2"):
            self.request_avatar("iron man")
        elif key == ord("3"):
            self.request_avatar("batman")
        elif key == ord("4"):
            self.request_avatar("joker")
        return False


def process_static_image(
    image_path: str,
    character_name: Optional[str],
    output_path: str = "output_scanned.jpg",
) -> None:
    """Run i-am-ditto on a static image file and save the result."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    scanner = HolisticScanner()
    resolver = AvatarResolver()
    results = scanner.process(frame)

    resolved_holder = []
    if character_name:
        resolver._resolve_worker(character_name, lambda asset: resolved_holder.append(asset))

    if resolved_holder and results.face_landmarks:
        warp_avatar_onto_face(frame, resolved_holder[0], results.face_landmarks)
        scanner.draw_green_sticks(frame, results, draw_face_mesh=False, draw_body_and_hands=True)
    else:
        scanner.draw_green_sticks(frame, results, draw_face_mesh=True, draw_body_and_hands=True)

    cv2.imwrite(output_path, frame)
    scanner.close()
    print(f"Saved output to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="i-am-ditto: Voice & AI Powered Human Face & Body AR Transformer."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--become", type=str, default=None, help="Initial character to transform into")
    parser.add_argument("--image", type=str, default=None, help="Optional static image path to process")
    parser.add_argument("--output", type=str, default="output_scanned.jpg", help="Output path for --image mode")
    args = parser.parse_args()

    if args.image:
        process_static_image(args.image, args.become, args.output)
    else:
        app = DittoApp(camera_index=args.camera, initial_avatar=args.become)
        app.run()


if __name__ == "__main__":
    main()
