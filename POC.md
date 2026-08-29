# Face Recognition POC — Conclusion

## What it does

Trains on a folder of photos of a person, then opens the webcam and shows a
combined live overlay: a box around every face it sees, labeling any face that
matches a trained person by name (and "Unknown" otherwise); a box around
common objects it recognizes (phone, pen, cup, laptop, etc.); and a skeleton
over any visible hands and body pose. Fully local — no cloud APIs, no network
calls at runtime once the models are downloaded.

## Tools used

- **Python 3.9** in a repo-local `venv/`, no global installs.
- **OpenCV** (`opencv-contrib-python`) — provides the two face networks below
  plus webcam capture (`cv2.VideoCapture`) and drawing (`cv2.rectangle`,
  `cv2.putText`).
- **YuNet** — a small pretrained face *detector* network (ONNX file,
  `weights/face_detection_yunet_2023mar.onnx`, from OpenCV's model zoo).
  Given an image, it returns a bounding box, 5 facial landmarks (eyes, nose,
  mouth corners), and a confidence score for every face it finds.
- **SFace** — a small pretrained face *recognition* network (ONNX file,
  `weights/face_recognition_sface_2021dec.onnx`, same source). Given an aligned
  face crop, it returns a 128-number vector (an "embedding") that represents
  that face — similar faces produce similar vectors.
- **YOLO-World** (via `ultralytics`, `weights/yolov8s-worldv2.pt`) — an
  open-vocabulary object detector. Unlike classic YOLO (fixed to COCO's 80
  classes), YOLO-World is given a list of plain-English class names at
  startup (`model.set_classes([...])`) and detects those specific classes.
- **MediaPipe** (`mediapipe`, pinned to `0.10.21`) — hand (21-point) and body
  pose (33-point) landmark detection, using the classic `mp.solutions.hands`
  / `mp.solutions.pose` API. Ships its own bundled model files inside the pip
  package, so no separate weights download is needed for this part.

The face and object models are frozen, pretrained networks — nothing gets
trained from scratch here. Face weights (~37MB) were a one-time download.
YOLO-World's checkpoint plus the CLIP text encoder it depends on for
`set_classes()` add another ~360MB on first run (see "Limitations" below);
after that, everything runs offline.

## How it works

**Enrollment (`train.py`)**

1. Scans the repo root for folders named `<name>-photos` (e.g. `erik-photos`).
2. For each photo: YuNet finds the face(s); if more than one is found, the
   largest is assumed to be the subject; if none is found, the photo is
   skipped.
3. SFace aligns that face (rotates/crops it to a standard 112x112 pose using
   YuNet's landmarks) and turns it into a 128-number embedding.
4. All embeddings, tagged with the person's name, are saved to
   `model/embeddings.npz`.

**Live recognition (`recognize.py`)**

1. Loads the saved embeddings.
2. For every webcam frame: YuNet finds all faces, SFace turns each into an
   embedding.
3. Each live embedding is compared against every stored embedding using
   cosine similarity (how close two vectors point in the same direction — 1.0
   is identical, 0 is unrelated). The closest match above a threshold (0.363)
   wins; below that, the face is labeled "Unknown".
4. A box and label are drawn on the frame in real time.

Adding a new person later is just adding a new `<name>-photos/` folder and
rerunning `train.py` — no code changes.

**Object, hand, and pose overlays (`recognize.py`, per frame)**

1. `object_model.predict(frame, conf=0.2)` runs YOLO-World directly on the
   BGR frame (no color conversion needed — Ultralytics handles that
   internally) and returns a box, class name, and confidence for every match
   against the fixed `OBJECT_CLASSES` list set once at startup via
   `set_classes()`.
2. The frame is converted to RGB once (`cv2.cvtColor(..., COLOR_BGR2RGB)`) —
   MediaPipe expects RGB, unlike the OpenCV/YOLO calls above — and passed to
   `hands.process()` and `pose.process()`.
3. `mp_drawing.draw_landmarks()` draws the connected skeleton for each result
   straight onto the original BGR frame.

All three (face, object, hand/pose) run every frame in the same loop, so the
live view is a straight sum of their individual costs — see "Limitations".

## Why this stack

- **YuNet over Haar cascades**: Haar cascades (OpenCV's classic 2001-era face
  detector) are fast but noticeably less accurate — more missed faces and
  false positives, especially at odd angles or lighting. YuNet is a modern,
  still lightweight DNN that's meaningfully more accurate and ships inside
  OpenCV, so there's no added dependency for the upgrade.
- **SFace over LBPH**: LBPH (OpenCV's built-in classic recognizer) compares
  raw pixel patterns and is fairly sensitive to lighting, angle, and crop
  alignment. SFace produces a learned embedding that's far more robust to
  those variations, at the cost of a larger model download and slightly more
  setup — worth it here since recognition accuracy is the point of the app.
- **Not dlib/`face_recognition`**: also a deep-embedding approach and roughly
  comparable in accuracy to SFace, but it requires compiling `dlib` from
  source on macOS (needs `cmake`, can take 10-20+ minutes). YuNet/SFace ship
  as prebuilt ONNX files with zero compile step.
- **YOLO-World over fixed-vocabulary YOLO/MobileNet-SSD**: standard COCO
  detectors only know 80 fixed classes and don't include things like "pen" or
  "watch". YOLO-World trades a heavier dependency (`ultralytics`, which pulls
  in `torch`) for the ability to name arbitrary classes via `set_classes()`
  without retraining.
- **MediaPipe over a hand/pose model in OpenCV's zoo**: OpenCV doesn't ship a
  ready hand-landmark or body-pose model. MediaPipe is the standard offline
  option and, at the pinned `0.10.21` version, ships working prebuilt models
  inside the pip package with no extra download step.

## Limitations found while building this

- 2 of the 17 training photos were skipped because no face was detected
  (full-body shots where the face is too small or turned away) — expected
  and handled gracefully rather than crashing.
- Photos with more than one face in frame use "largest face wins" as a
  heuristic — fine for portraits/selfies, not reliable for true group photos.
- The 0.363 cosine-similarity threshold is a tunable starting point, not a
  guarantee — it may need adjusting after testing under your actual webcam
  and lighting.
- **`mediapipe` had to be pinned to `0.10.21`, not left unpinned.** The
  current `1.0.x` releases (`1.0.0`/`1.0.1`, the versions `pip install
  mediapipe` picks up by default as of writing) removed the classic
  `mp.solutions.hands` / `mp.solutions.pose` API entirely — `mp.solutions`
  doesn't exist on that version. The replacement `mediapipe.tasks.python`
  API exists and its Python surface (`HandLandmarker`, `PoseLandmarker`,
  `drawing_utils`) looks correct, but calling `.detect()` on this machine
  crashed the process outright with a native `SIGABRT` (`Check failed:
  service_ Service is unavailable`, thrown from `DrishtiMetalHelper` /
  `TensorsToDetectionsCalculator::Open()` deep in MediaPipe's C++ graph
  runtime) — reproducible even after forcing the CPU delegate explicitly, so
  it isn't a GPU/Metal opt-in issue that can be configured away from Python.
  `0.10.21` (last release before the `1.0` API cut) still has `mp.solutions`
  and was verified end-to-end (hand + pose detection and drawing on a real
  photo, then a live 8-second webcam run producing no errors). If this
  dependency is ever bumped, re-verify against a real image before trusting
  it — the failure here would not have been caught by reading MediaPipe's
  own docs, which still document `mp.solutions` as if it were current.
- **`ultralytics` silently installs another package at runtime the first time
  `set_classes()` is called on a YOLO-World model.** It shells out to `pip
  install git+https://github.com/ultralytics/CLIP.git` (an "AutoUpdate")
  because YOLO-World uses CLIP's text encoder to turn class name strings into
  embeddings, then downloads CLIP's own `ViT-B-32.pt` checkpoint (~338MB,
  landed in `weights/clip/` here) on top of the ~25MB YOLO-World checkpoint
  itself — noticeably more than the plan's original ~50MB estimate. This repo
  pins `git+https://github.com/ultralytics/CLIP.git` directly in
  `requirements.txt` so that install happens up front during `pip install -r
  requirements.txt`, not as a surprise mid-run `pip install` subprocess the
  first time the script executes.
- **`opencv-python` and `opencv-contrib-python` end up installed side by side**,
  because `ultralytics` depends on plain `opencv-python` while this repo needs
  `opencv-contrib-python` for `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`
  (contrib-only APIs). Both packages install files into the same `cv2/`
  directory in `site-packages`. Installing everything from `requirements.txt`
  in one `pip install` resolves both to the same version (verified: both
  landed on `4.11.0.86`, `pip check` raises no conflict, and
  `cv2.FaceDetectorYN`/`cv2.FaceRecognizerSF` remain available), but this is
  fragile — don't `pip install`/`pip uninstall` either OpenCV package
  individually later, since their installed-files records overlap and doing
  so can silently corrupt or delete the other's files. If `cv2` ever breaks
  after touching dependencies, the reliable fix is deleting `venv/` and
  reinstalling from `requirements.txt` in one shot.
- Running five models per frame on CPU (YuNet, SFace, YOLO-World, MediaPipe
  Hands, MediaPipe Pose) is noticeably heavier than the original face-only
  loop. It stayed responsive in local testing on this Apple Silicon Mac; if
  the live view feels laggy on other hardware, the straightforward next step
  is throttling object detection to every Nth frame (it's the most expensive
  of the three additions).
