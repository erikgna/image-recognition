import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = REPO_ROOT / "weights"
MODEL_DIR = REPO_ROOT / "model"
DETECTOR_PATH = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = WEIGHTS_DIR / "face_recognition_sface_2021dec.onnx"
DETECTION_SCORE_THRESHOLD = 0.6
DETECTION_NMS_THRESHOLD = 0.3


def require_weights():
    missing = [p for p in (DETECTOR_PATH, RECOGNIZER_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        sys.exit(f"Missing model weights: {names} in {WEIGHTS_DIR}")


def largest_face(faces):
    areas = faces[:, 2] * faces[:, 3]
    return faces[np.argmax(areas)]


def embed_face(recognizer, image, face_row):
    aligned = recognizer.alignCrop(image, face_row)
    return recognizer.feature(aligned).flatten()


def main():
    require_weights()

    person_dirs = sorted(p for p in REPO_ROOT.glob("*-photos") if p.is_dir())
    if not person_dirs:
        sys.exit(f"No *-photos folders found in {REPO_ROOT}. Create one like erik-photos/ and re-run.")

    detector = cv2.FaceDetectorYN.create(
        model=str(DETECTOR_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=DETECTION_SCORE_THRESHOLD,
        nms_threshold=DETECTION_NMS_THRESHOLD,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(model=str(RECOGNIZER_PATH), config="")

    embeddings = []
    labels = []

    for person_dir in person_dirs:
        label = person_dir.name.removesuffix("-photos")
        used = 0
        skipped = 0
        for image_path in sorted(person_dir.iterdir()):
            if image_path.name.startswith("."):
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping (unreadable): {image_path}")
                skipped += 1
                continue
            height, width = image.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(image)
            if faces is None or len(faces) == 0:
                print(f"Skipping (no face found): {image_path}")
                skipped += 1
                continue
            if len(faces) > 1:
                print(f"Multiple faces detected, using largest: {image_path}")
            face_row = largest_face(faces)
            embeddings.append(embed_face(recognizer, image, face_row))
            labels.append(label)
            used += 1
        print(f"{label}: {used} used, {skipped} skipped")

    if not embeddings:
        sys.exit("No usable training faces found across any *-photos folder.")

    MODEL_DIR.mkdir(exist_ok=True)
    output_path = MODEL_DIR / "embeddings.npz"
    np.savez(
        output_path,
        embeddings=np.array(embeddings, dtype=np.float32),
        labels=np.array(labels),
    )
    print(f"Trained on {len(embeddings)} faces across {len(person_dirs)} people.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
