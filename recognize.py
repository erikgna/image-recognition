import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = REPO_ROOT / "weights"
MODEL_PATH = REPO_ROOT / "model" / "embeddings.npz"
DETECTOR_PATH = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = WEIGHTS_DIR / "face_recognition_sface_2021dec.onnx"
OBJECT_MODEL_PATH = WEIGHTS_DIR / "yolov8s-worldv2.pt"
DETECTION_SCORE_THRESHOLD = 0.6
DETECTION_NMS_THRESHOLD = 0.3
COSINE_DISTANCE_TYPE = 0
COSINE_MATCH_THRESHOLD = 0.363
OBJECT_CLASSES = [
    "phone",
    "pen",
    "pencil",
    "cup",
    "bottle",
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "watch",
]
OBJECT_CONFIDENCE_THRESHOLD = 0.2
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5


def require_weights():
    missing = [p for p in (DETECTOR_PATH, RECOGNIZER_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        sys.exit(f"Missing model weights: {names} in {WEIGHTS_DIR}")


def load_embeddings():
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run train.py first.")
    data = np.load(MODEL_PATH)
    return data["embeddings"], data["labels"]


def identify(recognizer, query_embedding, embeddings, labels):
    best_score = -1.0
    best_label = "Unknown"
    for embedding, label in zip(embeddings, labels):
        score = recognizer.match(query_embedding, embedding.reshape(1, -1), COSINE_DISTANCE_TYPE)
        if score > best_score:
            best_score = score
            best_label = str(label)
    if best_score < COSINE_MATCH_THRESHOLD:
        return "Unknown", best_score
    return best_label, best_score


def main():
    require_weights()
    embeddings, labels = load_embeddings()

    detector = cv2.FaceDetectorYN.create(
        model=str(DETECTOR_PATH),
        config="",
        input_size=(320, 240),
        score_threshold=DETECTION_SCORE_THRESHOLD,
        nms_threshold=DETECTION_NMS_THRESHOLD,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(model=str(RECOGNIZER_PATH), config="")

    object_model = YOLO(str(OBJECT_MODEL_PATH))
    object_model.set_classes(OBJECT_CLASSES)

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )
    pose = mp_pose.Pose(
        min_detection_confidence=POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(
            "Could not open webcam. On macOS check System Settings > Privacy & "
            "Security > Camera and grant access to your terminal, then re-run."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stream ended.")
            break

        _, faces = detector.detect(frame)
        if faces is not None:
            for face_row in faces:
                aligned = recognizer.alignCrop(frame, face_row)
                query_embedding = recognizer.feature(aligned)
                name, score = identify(recognizer, query_embedding, embeddings, labels)
                x, y, w, h = face_row[:4].astype(int)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({score:.2f})",
                    (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        pose_result = pose.process(rgb)
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        object_results = object_model.predict(frame, conf=OBJECT_CONFIDENCE_THRESHOLD, verbose=False)[0]
        for box in object_results.boxes:
            obj_name = object_model.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{obj_name} ({conf:.2f})",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        cv2.imshow("recognize", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
