import sys
import time
from pathlib import Path

import librosa
import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = REPO_ROOT / "recordings"

SAMPLE_RATE = 44100
CHANNELS = 1

COMB_DELAYS_MS = [29.7, 37.1, 41.1, 43.7]
COMB_GAINS = [0.805, 0.827, 0.783, 0.764]
ALLPASS_DELAYS_MS = [5.0, 1.7]
ALLPASS_GAIN = 0.7

DEFAULT_SETTINGS = {
    "pitch": 0.0,
    "speed": 1.0,
    "robot": False,
    "robot_freq": 30.0,
    "telephone": False,
    "distortion": False,
    "distortion_bits": 4.0,
    "echo": False,
    "echo_delay": 0.25,
    "echo_decay": 0.4,
    "echo_repeats": 4,
    "reverb": False,
    "reverb_mix": 0.3,
}

PRESETS = {
    "1": ("Robot", {"pitch": -2.0, "robot": True, "robot_freq": 30.0}),
    "2": ("Chipmunk", {"pitch": 7.0, "speed": 1.3}),
    "3": ("Deep / demon", {"pitch": -7.0, "speed": 0.9, "reverb": True, "reverb_mix": 0.25}),
    "4": ("Telephone call", {"telephone": True, "distortion": True, "distortion_bits": 6.0}),
    "5": ("AM radio", {"telephone": True, "distortion": True, "distortion_bits": 4.0}),
    "6": ("Alien", {"pitch": 4.0, "robot": True, "robot_freq": 45.0, "reverb": True, "reverb_mix": 0.2}),
    "7": ("Cathedral", {"reverb": True, "reverb_mix": 0.6, "echo": True, "echo_decay": 0.3, "echo_repeats": 3}),
}


def record_audio():
    print("Recording... press Enter to stop")
    blocks = []

    def callback(indata, frames, time_info, status):
        blocks.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
        ):
            input()
    except sd.PortAudioError as error:
        sys.exit(
            f"Could not open microphone ({error}). On macOS check System Settings > "
            "Privacy & Security > Microphone and grant access to your terminal, then re-run."
        )

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks)[:, 0]


def load_audio(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y


def shift_pitch(y, semitones):
    if semitones == 0.0:
        return y
    return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=semitones)


def stretch_speed(y, rate):
    if rate == 1.0:
        return y
    return librosa.effects.time_stretch(y, rate=rate)


def apply_robot(y, freq):
    t = np.arange(len(y)) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * freq * t)
    return (y * carrier).astype(np.float32)


def apply_telephone(y):
    sos = signal.butter(4, [300.0, 3400.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)


def apply_distortion(y, bits):
    levels = 2.0**bits
    return (np.round(y * levels) / levels).astype(np.float32)


def comb_filter(y, delay_samples, feedback, floor=1e-4):
    out = y.astype(np.float64).copy()
    gain = feedback
    shift = delay_samples
    while gain > floor and shift < len(y):
        out[shift:] += gain * y[: len(y) - shift]
        gain *= feedback
        shift += delay_samples
    return out


def allpass_filter(y, delay_samples, feedback):
    b = np.zeros(delay_samples + 1)
    b[0] = -feedback
    b[delay_samples] = 1.0
    a = np.zeros(delay_samples + 1)
    a[0] = 1.0
    a[delay_samples] = -feedback
    return signal.lfilter(b, a, y)


def apply_reverb(y, mix):
    wet = np.zeros_like(y, dtype=np.float64)
    for ms, gain in zip(COMB_DELAYS_MS, COMB_GAINS):
        wet += comb_filter(y, int(ms / 1000 * SAMPLE_RATE), gain)
    wet /= len(COMB_DELAYS_MS)
    for ms in ALLPASS_DELAYS_MS:
        wet = allpass_filter(wet, int(ms / 1000 * SAMPLE_RATE), ALLPASS_GAIN)
    return ((1 - mix) * y + mix * wet).astype(np.float32)


def apply_echo(y, delay_seconds, decay, repeats):
    delay_samples = int(delay_seconds * SAMPLE_RATE)
    if delay_samples <= 0 or repeats <= 0:
        return y
    out = np.zeros(len(y) + delay_samples * repeats, dtype=np.float32)
    out[: len(y)] += y
    for i in range(1, repeats + 1):
        start = delay_samples * i
        out[start : start + len(y)] += y * (decay**i)
    return out


def limit(y, threshold=0.95):
    if y.size == 0:
        return y
    peak = np.abs(y).max()
    if peak > threshold:
        y = y * (threshold / peak)
    return y


def apply_effects(y, settings):
    if y.size == 0:
        return y
    y = shift_pitch(y, settings["pitch"])
    y = stretch_speed(y, settings["speed"])
    if settings["robot"]:
        y = apply_robot(y, settings["robot_freq"])
    if settings["telephone"]:
        y = apply_telephone(y)
    if settings["distortion"]:
        y = apply_distortion(y, settings["distortion_bits"])
    if settings["echo"]:
        y = apply_echo(y, settings["echo_delay"], settings["echo_decay"], settings["echo_repeats"])
    if settings["reverb"]:
        y = apply_reverb(y, settings["reverb_mix"])
    return limit(y)


def play_audio(y):
    if y.size == 0:
        print("Nothing to play.")
        return
    sd.play(y, samplerate=SAMPLE_RATE, blocking=True)


def prompt_save(y):
    answer = input("Save? [y/n] ").strip().lower()
    if answer != "y":
        return
    RECORDINGS_DIR.mkdir(exist_ok=True)
    default_name = f"voice_{int(time.time())}.wav"
    default_path = RECORDINGS_DIR / default_name
    entered = input(f"Output path [{default_path}]: ").strip()
    out_path = Path(entered) if entered else default_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, SAMPLE_RATE)
    print(f"Saved to {out_path}")


def read_float(prompt, current):
    entered = input(f"{prompt} [{current}]: ").strip()
    if not entered:
        return current
    try:
        return float(entered)
    except ValueError:
        print("Not a number, keeping current value.")
        return current


def read_bool(prompt, current):
    entered = input(f"{prompt} (y/n) [{'y' if current else 'n'}]: ").strip().lower()
    if not entered:
        return current
    return entered == "y"


def configure_robot(settings):
    settings["robot"] = read_bool("Robot (ring modulation)", settings["robot"])
    if settings["robot"]:
        freq = read_float("Carrier frequency in Hz, >0 (~20-80 typical)", settings["robot_freq"])
        if freq <= 0:
            print("Frequency must be positive (0Hz silences all audio), keeping current value.")
        else:
            settings["robot_freq"] = freq


def configure_telephone(settings):
    settings["telephone"] = read_bool("Telephone (300-3400Hz bandpass)", settings["telephone"])


def configure_distortion(settings):
    settings["distortion"] = read_bool("Distortion (bitcrush)", settings["distortion"])
    if settings["distortion"]:
        bits = read_float("Bit depth, >=1 (lower=crunchier, 2-8 typical)", settings["distortion_bits"])
        settings["distortion_bits"] = max(1.0, bits)


def configure_echo(settings):
    settings["echo"] = read_bool("Echo", settings["echo"])
    if settings["echo"]:
        settings["echo_delay"] = max(0.01, read_float("Delay in seconds", settings["echo_delay"]))
        settings["echo_decay"] = min(0.95, max(0.0, read_float("Decay per repeat, 0-0.95", settings["echo_decay"])))
        settings["echo_repeats"] = max(1, int(read_float("Repeats", settings["echo_repeats"])))


def configure_reverb(settings):
    settings["reverb"] = read_bool("Reverb", settings["reverb"])
    if settings["reverb"]:
        settings["reverb_mix"] = min(1.0, max(0.0, read_float("Wet mix, 0-1", settings["reverb_mix"])))


def choose_preset(settings):
    print()
    for key, (name, _) in PRESETS.items():
        print(f"  [{key}] {name}")
    print("  [0] Reset to defaults (no effects)")
    choice = input("Preset > ").strip()
    if choice == "0":
        settings.clear()
        settings.update(DEFAULT_SETTINGS)
        print("Reset to defaults.")
        return
    preset = PRESETS.get(choice)
    if not preset:
        print("Unknown preset.")
        return
    name, values = preset
    settings.clear()
    settings.update(DEFAULT_SETTINGS)
    settings.update(values)
    print(f"Applied preset: {name}")


def print_menu(settings):
    print()
    print("Voice Modifier")
    print(f"  pitch:      {settings['pitch']} semitones   (thick <-> thin)")
    print(f"  speed:      {settings['speed']}x           (slower <-> faster)")
    robot = f"on (freq={settings['robot_freq']}Hz)" if settings["robot"] else "off"
    telephone = "on" if settings["telephone"] else "off"
    distortion = f"on (bits={settings['distortion_bits']})" if settings["distortion"] else "off"
    echo = (
        f"on (delay={settings['echo_delay']}s decay={settings['echo_decay']} x{settings['echo_repeats']})"
        if settings["echo"]
        else "off"
    )
    reverb = f"on (mix={settings['reverb_mix']})" if settings["reverb"] else "off"
    print(f"  robot:      {robot}")
    print(f"  telephone:  {telephone}")
    print(f"  distortion: {distortion}")
    print(f"  echo:       {echo}")
    print(f"  reverb:     {reverb}")
    print("[p] pitch  [s] speed  [1] robot  [2] telephone  [3] distortion  [4] echo  [5] reverb")
    print("[m] presets  [r] record + preview + save  [f] apply to file  [q] quit")


def main():
    settings = dict(DEFAULT_SETTINGS)

    while True:
        print_menu(settings)
        choice = input("> ").strip().lower()

        if choice == "p":
            settings["pitch"] = read_float("Pitch (semitones, negative=thick, positive=thin)", settings["pitch"])
        elif choice == "s":
            new_rate = read_float("Speed rate (<1=slower, >1=faster)", settings["speed"])
            if new_rate <= 0:
                print("Speed rate must be positive, keeping current value.")
            else:
                settings["speed"] = new_rate
        elif choice == "1":
            configure_robot(settings)
        elif choice == "2":
            configure_telephone(settings)
        elif choice == "3":
            configure_distortion(settings)
        elif choice == "4":
            configure_echo(settings)
        elif choice == "5":
            configure_reverb(settings)
        elif choice == "m":
            choose_preset(settings)
        elif choice == "r":
            y = record_audio()
            y = apply_effects(y, settings)
            play_audio(y)
            prompt_save(y)
        elif choice == "f":
            in_path = input("Input file path: ").strip()
            if not in_path or not Path(in_path).exists():
                print("File not found.")
                continue
            y = load_audio(in_path)
            y = apply_effects(y, settings)
            play_audio(y)
            prompt_save(y)
        elif choice == "q":
            break
        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
