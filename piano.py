import sys

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

MAX_NUM_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5

MIRROR_FRAME = True

HAND_LABELS = ("Left", "Right")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

FINGER_LANDMARKS = {
    "thumb": (4, 3),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
THUMB_REFERENCE_LANDMARK = 5

FINGER_CURL_DOWN_MARGIN = 0.05
FINGER_CURL_UP_MARGIN = 0.02
THUMB_CURL_DOWN_DISTANCE = 0.08
THUMB_CURL_UP_DISTANCE = 0.11

SAMPLE_RATE = 44100
TONE_DURATION_SECONDS = 0.25
ATTACK_SECONDS = 0.015
RELEASE_SECONDS = 0.08
TONE_AMPLITUDE = 0.4

NOTE_MAP = {
    ("Left", "thumb"): ("C4", 261.63),
    ("Left", "index"): ("D4", 293.66),
    ("Left", "middle"): ("E4", 329.63),
    ("Left", "ring"): ("F4", 349.23),
    ("Left", "pinky"): ("G4", 392.00),
    ("Right", "thumb"): ("A4", 440.00),
    ("Right", "index"): ("B4", 493.88),
    ("Right", "middle"): ("C5", 523.25),
    ("Right", "ring"): ("D5", 587.33),
    ("Right", "pinky"): ("E5", 659.25),
}

HUD_ORIGIN_X = 20
HUD_ORIGIN_Y = 20
HUD_KEY_WIDTH = 90
HUD_KEY_HEIGHT = 70
HUD_KEY_GAP = 10
HUD_ROW_GAP = 20
HUD_IDLE_COLOR = (180, 180, 180)
HUD_PRESSED_COLOR = (0, 220, 0)
HUD_TEXT_IDLE_COLOR = (255, 255, 255)
HUD_TEXT_PRESSED_COLOR = (0, 0, 0)
HUD_ROW_LABEL_COLOR = (200, 200, 200)


def make_tone(frequency):
    sample_count = int(SAMPLE_RATE * TONE_DURATION_SECONDS)
    t = np.linspace(0, TONE_DURATION_SECONDS, sample_count, endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)
    envelope = np.ones(sample_count)
    attack_samples = int(SAMPLE_RATE * ATTACK_SECONDS)
    release_samples = int(SAMPLE_RATE * RELEASE_SECONDS)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    return (waveform * envelope * TONE_AMPLITUDE).astype(np.float32)


def build_tone_table():
    return {note_name: make_tone(frequency) for note_name, frequency in NOTE_MAP.values()}


def landmark_xy(hand_landmarks, index):
    landmark = hand_landmarks.landmark[index]
    return landmark.x, landmark.y


def finger_curl_score(hand_landmarks, finger_name, aspect_ratio):
    if finger_name == "thumb":
        tip_x, tip_y = landmark_xy(hand_landmarks, 4)
        ref_x, ref_y = landmark_xy(hand_landmarks, THUMB_REFERENCE_LANDMARK)
        dx = (tip_x - ref_x) * aspect_ratio
        dy = tip_y - ref_y
        distance = (dx * dx + dy * dy) ** 0.5
        return -distance
    tip_index, pip_index = FINGER_LANDMARKS[finger_name]
    _, tip_y = landmark_xy(hand_landmarks, tip_index)
    _, pip_y = landmark_xy(hand_landmarks, pip_index)
    return tip_y - pip_y


def update_finger_state(states, key, curl_score, down_margin, up_margin):
    state = states[key]
    if state == "up":
        if curl_score > down_margin:
            states[key] = "down"
            return True
    elif state == "down":
        if curl_score < up_margin:
            states[key] = "up"
    return False


def process_hand(hand_landmarks, handedness_label, states, aspect_ratio):
    fired = []
    for finger_name in FINGER_NAMES:
        curl_score = finger_curl_score(hand_landmarks, finger_name, aspect_ratio)
        if finger_name == "thumb":
            down_margin, up_margin = -THUMB_CURL_DOWN_DISTANCE, -THUMB_CURL_UP_DISTANCE
        else:
            down_margin, up_margin = FINGER_CURL_DOWN_MARGIN, FINGER_CURL_UP_MARGIN
        key = (handedness_label, finger_name)
        if update_finger_state(states, key, curl_score, down_margin, up_margin):
            fired.append(finger_name)
    return fired


def play_note(tone_table, note_name):
    sd.play(tone_table[note_name], samplerate=SAMPLE_RATE)


def draw_hud(frame, states):
    for row_index, hand_label in enumerate(HAND_LABELS):
        row_y = HUD_ORIGIN_Y + row_index * (HUD_KEY_HEIGHT + HUD_ROW_GAP)
        cv2.putText(
            frame,
            hand_label[0],
            (HUD_ORIGIN_X - 15, row_y + HUD_KEY_HEIGHT // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HUD_ROW_LABEL_COLOR,
            2,
        )
        for col_index, finger_name in enumerate(FINGER_NAMES):
            x = HUD_ORIGIN_X + col_index * (HUD_KEY_WIDTH + HUD_KEY_GAP)
            note_name, _ = NOTE_MAP[(hand_label, finger_name)]
            pressed = states[(hand_label, finger_name)] == "down"
            color = HUD_PRESSED_COLOR if pressed else HUD_IDLE_COLOR
            text_color = HUD_TEXT_PRESSED_COLOR if pressed else HUD_TEXT_IDLE_COLOR
            thickness = -1 if pressed else 2
            cv2.rectangle(frame, (x, row_y), (x + HUD_KEY_WIDTH, row_y + HUD_KEY_HEIGHT), color, thickness)
            cv2.putText(
                frame,
                note_name,
                (x + 10, row_y + HUD_KEY_HEIGHT // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )


def main():
    tone_table = build_tone_table()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(
            "Could not open webcam. On macOS check System Settings > Privacy & "
            "Security > Camera and grant access to your terminal, then re-run."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = width / height

    states = {(hand_label, finger_name): "up" for hand_label in HAND_LABELS for finger_name in FINGER_NAMES}

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stream ended.")
            break

        if MIRROR_FRAME:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                handedness_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fired = process_hand(hand_landmarks, handedness_label, states, aspect_ratio)
                for finger_name in fired:
                    note_name, _ = NOTE_MAP[(handedness_label, finger_name)]
                    play_note(tone_table, note_name)

        draw_hud(frame, states)
        cv2.imshow("piano", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
