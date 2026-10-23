import argparse
import time
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def draw_landmarks(image, results):
    """Draw face mesh, body pose, and hand landmarks on the frame."""
    # 1. Face Mesh (468 / 478 landmarks)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    # 2. Body Pose (33 landmarks)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # 3. Left & Right Hands (21 landmarks each)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def draw_hud(image, results, fps: float):
    """Draw status HUD showing detected regions and FPS."""
    face_detected = results.face_landmarks is not None
    pose_detected = results.pose_landmarks is not None
    lh_detected = results.left_hand_landmarks is not None
    rh_detected = results.right_hand_landmarks is not None

    status_lines = [
        f"FPS: {fps:.1f}",
        f"Face: {'YES' if face_detected else 'NO'}",
        f"Body: {'YES' if pose_detected else 'NO'}",
        f"Hands: L={'YES' if lh_detected else 'NO'} | R={'YES' if rh_detected else 'NO'}",
    ]

    for idx, line in enumerate(status_lines):
        y = 30 + idx * 28
        cv2.putText(
            image,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )


def run_webcam(camera_index: int = 0):
    """Real-time human face and body scanning via webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera at index {camera_index}")

    prev_time = time.time()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
    ) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            results = holistic.process(rgb_frame)

            rgb_frame.flags.writeable = True
            annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            draw_landmarks(annotated_frame, results)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            draw_hud(annotated_frame, results, fps)

            cv2.imshow("Face & Body Scanner (Press Q to quit)", annotated_frame)
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


def scan_image(image_path: str, output_path: str = "output_scanned.jpg"):
    """Scan human face and body from a static image file."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        refine_face_landmarks=True,
    ) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated = image.copy()
        draw_landmarks(annotated, results)
        cv2.imwrite(output_path, annotated)
        print(f"Saved scanned image to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Human Face & Body Scanner using MediaPipe Holistic and OpenCV."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an input image file. If omitted, runs real-time webcam mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_scanned.jpg",
        help="Output file path when scanning an image (default: output_scanned.jpg).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    args = parser.parse_args()

    if args.image:
        scan_image(args.image, args.output)
    else:
        run_webcam(args.camera)


if __name__ == "__main__":
    main()
