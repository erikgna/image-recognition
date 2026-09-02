# Voice Modifier POC — `voice.py`

## What it does

An interactive terminal menu that runs a chain of up to seven voice effects
— pitch, speed, robot (ring modulation), telephone (bandpass), distortion
(bitcrush), echo, and reverb — then previews the result and optionally
saves it. Seven built-in presets (`m`) set a whole combination at once
(Robot, Chipmunk, Deep/demon, Telephone call, AM radio, Alien, Cathedral).
Two ways to get audio in: record fresh from the mic, or apply the current
settings to an existing audio file on disk. Fully offline, no cloud APIs,
no new dependencies beyond what was already installed transitively
(`scipy` ships with `librosa`).

## How it works

1. `p` / `s` / `1`-`5` update a single `settings` dict (pitch semitones,
   speed rate, and on/off + params for robot/telephone/distortion/echo/
   reverb) via `input()`. `m` overwrites the whole dict at once from a
   preset. Toggling an effect off leaves its tuned params in the dict so
   re-enabling it restores the last values rather than resetting to
   defaults.
2. `r` opens a `sounddevice.InputStream`, buffers float32 blocks in a
   callback until Enter is pressed, then concatenates them into one mono
   array. `f` instead loads a file via `soundfile.read`, averaging to mono
   and resampling to `SAMPLE_RATE` with `librosa.resample` if the file's
   native rate differs.
3. Both paths run the same `apply_effects()`, a fixed-order pipeline:
   pitch → speed → robot → telephone → distortion → echo → reverb → limiter.
   Each stage is skipped when off/identity (pitch=0, speed=1.0, or the
   effect's toggle is `False`) to avoid a no-op DSP pass. The limiter always
   runs last, unconditionally — it's the fix for the clipping gap the
   original version had (see Verified/Limitations below).
   - Pitch/speed: `librosa.effects.pitch_shift` (semitones, duration-
     preserving) / `time_stretch` (rate, pitch-preserving), same as before.
   - Robot: ring modulation — multiply the signal by a sine carrier
     (`y * sin(2*pi*freq*t)`), which shifts energy to sum/difference
     frequencies around each partial and nulls the original pitch. This
     is *not* the same thing as pitch-shifting; it produces the classic
     inharmonic "robot" timbre instead of a clean pitch move.
   - Telephone: a 4th-order Butterworth bandpass (300-3400Hz, the classic
     telephone passband) via `scipy.signal.butter(..., output="sos")` +
     `sosfilt` (causal — see Limitations for why not `sosfiltfilt`).
   - Distortion: bitcrush — quantize to `2**bits` levels per unit amplitude
     (`round(y * levels) / levels`).
   - Echo: a finite sum of decayed, delayed copies of the *dry* signal
     (`out[n*delay:] += y * decay**n`) for `repeats` taps — not a feedback
     filter, so it's stable by construction and its cost is
     `O(len(y) * repeats)`.
   - Reverb: a Schroeder reverberator — 4 parallel comb filters (delays
     29.7/37.1/41.1/43.7ms, feedback gains 0.805/0.827/0.783/0.764,
     classic textbook values) summed and averaged, then 2 series allpass
     filters (5ms/1.7ms, gain 0.7), dry/wet blended by `mix`. Each comb
     filter is implemented as a truncated tap-sum (see Why below), not
     `scipy.signal.lfilter` directly.
   - Limiter: peak-normalize down to 0.95 only if the peak exceeds it
     (never boosts a quiet signal).

There is no real-time/streaming pitch or speed shifting here — every effect
is applied to a full in-memory clip, then played back. This was an explicit
design choice, not a shortcut: streaming pitch/time shift needs a
phase-vocoder running on overlapping duplex I/O buffers, which is real DSP
engineering, not something to bolt onto a POC menu loop. "Preview" instead
means record-or-load → process → play → decide.

## Why this design

- **`librosa.effects.pitch_shift`/`time_stretch` over a hand-rolled phase
  vocoder**: librosa's implementations are STFT-based phase vocoders that are
  well-tested and tunable (`res_type`, `bins_per_octave`), and correctly
  decouple pitch from duration (verified below) — reimplementing this
  correctly is its own multi-week project, out of scope for a POC about
  voice effects rather than DSP internals.
- **`soundfile` for I/O, not `librosa.load` as the primary path**: `sf.read`
  is a thin wrapper over libsndfile (fast, no resampling by default), while
  `librosa.load` always resamples and mixes to mono by decoding through
  `audioread`/`soundfile` internally anyway. Reading directly with
  `soundfile` and only calling `librosa.resample` when the file's rate
  actually differs from `SAMPLE_RATE` avoids a redundant resample on the
  common case (a file already at 44100 Hz).
- **Callback-based `InputStream` over `sd.rec()`**: `sd.rec()` needs an
  upfront fixed duration; a callback appending to a list supports
  press-Enter-to-stop recording of unknown length, matching the interactive
  menu style of the rest of the repo (`piano.py`'s keypress loop).
- **Menu loop over argparse/CLI flags**: matches `piano.py`'s interactive
  convention rather than `recognize.py`/`train.py`'s no-argument scripts —
  there's no OpenCV window to capture keystrokes from here, so a plain
  `input()` loop is the direct equivalent.
- **`scipy.signal` directly, not a new DSP dependency**: `scipy` is already
  pulled in transitively by `librosa` (confirmed importable in the repo's
  venv without touching `requirements.txt`). Robot/telephone/echo/reverb/
  distortion all reduce to filtering, FFT, and array arithmetic that `scipy`
  and `numpy` already cover — reaching for e.g. `pedalboard` or a dedicated
  reverb library would add a new native dependency for something four
  functions and ~40 lines already solve.
- **A hand-rolled tap-sum comb filter, not `scipy.signal.lfilter`, for
  reverb**: tried `lfilter` with a sparse feedback-only coefficient array
  first (`a[0]=1, a[delay]=-gain`, `b=[1]` — the textbook way to express
  `y[n] = x[n] + g*y[n-D]`). It's *correct* — verified by diffing against
  the tap-sum version (max abs diff 5.3e-5 on a 1s test tone) — but slow:
  `lfilter` doesn't exploit the coefficient array being 99.9% zero, so it
  costs `O(len(y) * delay)`, and at a ~1300-1900 sample delay (30-44ms @
  44100Hz) one comb filter alone took ~0.2s on an 8s clip. Four of those
  plus two allpasses made the full effect chain take **1.26s on an 8s
  clip** — noticeably laggy for an interactive preview loop. Rewriting the
  comb filter as an explicit sum of `decay**k`-scaled, `k*delay`-shifted
  copies of the input (mathematically the same IIR, just truncated once
  `decay**k` drops below `1e-4`, ~25-42 taps depending on gain) dropped the
  same 8s/7-effect chain to **41-74ms** — a ~17-30x speedup — because it's
  `O(len(y) * taps)` instead of `O(len(y) * delay)`, and `taps` (≈25-42) is
  far smaller than `delay` (≈1300-1900 samples) at these settings. The
  allpass filters were left on `lfilter` since their delays are short
  (5ms/1.7ms ≈ 75-220 samples) and cost ~0.03s each — not worth the same
  rewrite.
- **`scipy.signal.sosfilt` (causal) over `sosfiltfilt` (zero-phase) for the
  telephone bandpass**: tried `sosfiltfilt` first since it's the usual
  scipy recommendation for filtering without phase distortion. Testing an
  8000Hz tone (well above the 3400Hz passband) through it showed the
  *steady-state* output was correctly attenuated (down to ~2% of input
  energy, matching the filter's designed -35.8dB response at 8000Hz per
  `sosfreqz`), but the **last ~6 samples of the array spiked to 0.46 peak
  amplitude** — 9x louder than the true passband content — because
  `sosfiltfilt`'s default edge-padding/extrapolation produces a transient
  right at the signal boundary for content far outside the passband.
  Measuring "peak amplitude" naively (as the first version of this
  effect's test did) reported the filter as barely attenuating out-of-band
  content at all, which was wrong — it was measuring a boundary artifact,
  not the filter's actual behavior. Switching to causal `sosfilt` (no
  padding, no backward pass) removed the artifact entirely and is also
  cheaper (one pass instead of two); zero-phase preservation isn't needed
  for a voice effect, only linear-phase-sensitive applications care.
- **Echo as dry-signal tap-sum, not a feedback delay line**: real analog
  echo/delay units feed the *output* back into the delay line (so echoes
  build on each other's coloration), but summing decayed copies of the
  original dry signal is simpler, unconditionally stable for any
  `decay < 1` (no risk of the runaway growth a `decay >= 1` feedback loop
  would produce — see Limitations), and audibly indistinguishable from a
  feedback delay line for the handful of repeats (1-8) exposed here.

## Limitations found while building this

- **`librosa` pulls in `numba`+`llvmlite` (JIT compiler) and `scikit-learn`**
  as transitive dependencies just for `pitch_shift`/`time_stretch` — a much
  heavier install (~14 packages, ~45MB of wheels) than the two features
  used would suggest. `numba`/`llvmlite` in particular are native-compiled
  and version-sensitive (pinned `llvmlite<0.44,>=0.43.0dev0` by `numba`
  itself); if this dependency is ever bumped, re-verify the install resolves
  cleanly on the target Python version rather than assuming it will, the
  same caution `POC-IMAGE.md` calls out for `mediapipe`.
- **Passing `rate <= 0` to `time_stretch` raises
  `librosa.util.exceptions.ParameterError` ("rate must be a positive
  number")** — not caught by librosa internally. Confirmed by direct testing
  (`speed_rate=0.0` and `speed_rate=-1.0` both raise). `voice.py` validates
  this at the `s` prompt (rejects <= 0 and keeps the previous value) so the
  menu loop can't be crashed this way; `pitch_shift` has no equivalent
  invalid range — any float `n_steps` runs without error, including extreme
  values like ±48 semitones (4 octaves), which stayed non-silent in testing
  but were not checked for how the pitch actually sounds at that extreme.
- **Stereo input files are downmixed by simple channel averaging**
  (`y.mean(axis=1)`), not a loudness-aware downmix — verified this doesn't
  crash and produces a plausible peak amplitude on a synthetic stereo test
  file, but it will change relative channel balance for real hard-panned
  stereo recordings.
- **`time_stretch` at very large rates (tested 20x) still runs and returns
  a shortened non-silent array** with no lower bound on how few samples
  remain — no crash, but a multi-second clip stretched to 20x becomes a
  fraction of a second, and there's no check for a result so short it's
  effectively unplayable.
- Only `.wav` was exercised end-to-end (`soundfile` also supports FLAC/OGG
  read, and `librosa.resample` handles the rate-mismatch case regardless of
  container) — other formats were not tested against this code path.
- **This limiter gap is now fixed** — `limit()` runs unconditionally as the
  last stage of `apply_effects()`, peak-normalizing to 0.95 if exceeded.
  Confirmed effective: a synthetic signal deliberately amplified to 3x (peak
  3.0) came out at exactly 0.95 after the full chain; a quiet real recording
  (peak 0.079) was left untouched (limiter never engages below threshold,
  by design — it's a ceiling, not a compressor). One caveat: it operates on
  the *whole processed array* including reverb/echo tails, so a loud dry
  voice plus a long reverb tail gets normalized as one block — a transient
  peak anywhere in the clip (dry or tail) turns down the entire clip's
  level, not just that moment. A real limiter would work sample-by-sample
  or in short windows; this one doesn't, because per-sample/windowed
  limiting is a bigger feature than a POC menu effect chain needs.
- **`apply_echo` with negative `repeats` crashed** (`ValueError: operands
  could not be broadcast together`) — `delay_samples * repeats` went
  negative, making the output array shorter than the input before the
  first in-place add. Real bug, found by boundary-value testing (`repeats
  <= 0`), not something a happy-path test would catch. Fixed by clamping
  `echo_repeats` to `>= 1` at the input-validation layer
  (`configure_echo`), the same pattern the original `speed_rate <= 0`
  check used — `apply_echo` itself still has a defensive `repeats <= 0:
  return y` early-return as a second guard.
- **`robot_freq = 0` silences the entire clip**, not a partial effect —
  `sin(2*pi*0*t) == 0` for all `t`, so the "carrier" is a constant zero and
  `y * carrier` zeroes everything. Not a crash, but a real gotcha a user
  could hit by typing `0` expecting "no ring mod." `configure_robot` now
  rejects `freq <= 0` and keeps the previous value, mirroring how `s`
  already rejected `speed_rate <= 0`.
- **`echo_decay >= 1.0` makes each repeat louder than the last** (tested
  `decay=1.5`: peak grew from 0.5 to 4.2 over 4 repeats) — not a crash,
  and the final limiter catches the result before playback/save, but it's
  semantically not an "echo" (decaying repetition) anymore at that point.
  `configure_echo` clamps the prompt to `[0, 0.95]` to keep the label
  honest; the underlying `apply_echo` function itself doesn't enforce this,
  since it's a legitimate (if oddly named) effect and the limiter is a
  real backstop either way.
- **Distortion's `bits` parameter isn't literally "output bit depth"** —
  `levels = 2**bits` is levels *per unit amplitude*, so the actual number
  of discrete steps a signal hits depends on the signal's own peak
  amplitude too (a signal peaking at 0.9 with `bits=4` produces ~29
  distinct levels, not 16, because it spans ±0.9 not ±1.0 of a 16-level
  scale). This tripped up the first version of the automated test, which
  assumed unique-level-count `<= 2**bits`. Not a code bug — `bits` behaves
  correctly as a "crunchiness" knob — but the naming invites the wrong
  mental model if you're expecting literal PCM bit-depth semantics.
- No mixing/normalization step for *input* levels: if a recording clips
  (amplitude > 1.0) going in, pitch/time-shifting can push it further out
  of range; `soundfile.write` will still write the out-of-range floats
  without erroring. The output-side limiter (above) catches this before
  save/playback now, but there's still no warning if the *source* material
  was already clipped before any effect ran.
- Reverb's comb-filter tap-sum is an approximation, not an exact match to
  the ideal infinite feedback IIR — truncated once `decay**k < 1e-4`
  (confirmed max abs diff 5.3e-5 vs. the exact `lfilter` version on a 1s
  test tone). Inaudible in practice, but worth naming: this reverb is
  "close enough," not bit-exact to a textbook Schroeder reverberator.
- No way to chain/reorder effects arbitrarily — the pipeline order
  (pitch → speed → robot → telephone → distortion → echo → reverb) is
  fixed in code, not user-selectable. This was a deliberate simplicity
  choice (the alternative is a much more complex "build your own chain"
  UI) but it means, e.g., you can't put reverb before distortion to hear
  a "distorted reverb tail" instead of a "reverbed distortion."

## Verified so far

- `librosa==0.11.0` and `soundfile==0.13.1` install cleanly into the
  existing repo-local `venv/` (Python 3.9.6) alongside the existing pinned
  deps, no version conflicts (`pip install` reused existing `numpy 1.26.4`
  and `scipy 1.13.1` rather than upgrading them).
- On this version, `librosa.effects.pitch_shift` and `time_stretch` take
  `sr`/`n_steps`/`rate` as **keyword-only** arguments (confirmed via
  `inspect.signature`) — `voice.py` calls them accordingly.
- Automated check against a synthetic 220Hz sine WAV (no mic needed):
  - `load_audio` round-trips duration correctly, including the
    rate-mismatch resample path (fed a file with a mismatched header sample
    rate and confirmed `librosa.resample` preserves true duration, not
    sample count).
  - `pitch_shift(-6)` and `pitch_shift(+6)` both produce non-silent output
    of the **same length** as the input (duration-preserving, as designed).
  - `time_stretch(rate=0.7)` and `time_stretch(rate=1.4)` both produced
    output within 0.01% of the expected `duration_in / rate` — essentially
    exact on this synthetic input, not just "close."
  - Combined pitch+speed, WAV write/read round-trip, and a zero-length
    input (empty recording) all behave correctly with no exceptions.
  - `voice.py` byte-compiles cleanly (`py_compile`).
- `sd.query_devices()` on this Mac lists a working input device ("MacBook
  Pro Microphone") and `sd.PortAudioError` is a real, catchable exception
  class in `sounddevice==0.5.6`, so the mic-permission-denied path added to
  `record_audio()` has a real exception type to catch — though the actual
  permission-denied *scenario* itself was not triggered (would require
  revoking mic access first), only confirmed the exception class exists and
  the happy-path mic query succeeds.

### New effects (robot, telephone, distortion, echo, reverb, limiter)

All of the following used a 220Hz synthetic sine (matching the original
verification approach) plus FFT analysis and direct `scipy.signal.sosfreqz`
frequency-response checks — not just "ran without crashing":

- **Robot (ring mod)**: FFT of the output of `apply_robot(sine(220), 30Hz)`
  shows peaks at 190Hz and 250Hz (the expected sum/difference frequencies,
  220±30) and confirms energy at the original 220Hz carrier drops to <5% of
  the output's peak FFT bin — i.e. it's genuinely ring modulation (frequency
  translation), not a disguised pitch shift.
- **Telephone (bandpass)**: `sosfreqz` confirms the filter design itself
  matches spec (0dB at 1000Hz, -3dB at the 300/3400Hz band edges, -35.8dB
  at 8000Hz). End-to-end test against 100Hz/1000Hz/8000Hz tones (measuring
  *steady-state* samples, i.e. excluding the causal filter's startup
  transient) confirms in-band passes at ~100% and both out-of-band tones
  are attenuated to <2% of input peak.
- **Distortion (bitcrush)**: confirmed quantization to a small, finite set
  of discrete output levels (29 unique values for a 0.9-amplitude tone at
  `bits=4`, matching the `2*amp*2**bits` levels-per-unit-amplitude math —
  see Limitations) and confirmed `bits=16` is audibly/numerically
  indistinguishable from the unprocessed signal (<0.001 max diff).
- **Echo**: confirmed output length grows by exactly `delay*repeats`
  samples, the appended tail (past the original clip's end) is non-silent
  audio (the echo itself, not silence), and empty input produces empty
  output.
- **Reverb**: confirmed output length is unchanged (Schroeder reverb, unlike
  echo, doesn't extend the clip — the comb/allpass tails overlap the
  original duration rather than appending), silence stays silent, `mix=0`
  returns the dry signal unchanged (within float rounding), and `mix=1.0`
  (full wet) stays finite with no blow-up despite feedback gains up to
  0.827 (confirmed stable, not just "didn't crash in this one test" — see
  the near-instability check below).
- **Limiter**: confirmed it reduces a 3x-amplified signal to exactly the
  0.95 threshold, leaves an already-quiet signal completely unchanged
  (bit-for-bit via `np.allclose`), and handles empty input.

### Stress and boundary testing (beyond the original POC's scope)

- **Full 7-effect chain on an 8-second clip**: 41-74ms depending on
  settings — confirmed fast enough to feel interactive, after the comb
  filter performance fix (see Why This Design). Before the fix: 1.26s,
  which is a materially different, laggier experience for a preview loop.
- **Feedback stability at extreme settings**: comb filter feedback gains up
  to 0.827 with `mix=1.0` (full wet reverb) stay finite — no divergence.
  This matters because IIR feedback filters *can* diverge if `|feedback| >=
  1`; the hardcoded Schroeder gains here are all comfortably under 1, and
  this was verified rather than assumed.
- **Audio shorter than the filter/reverb delays**: a 44-sample clip (~1ms,
  shorter than every comb/allpass delay, which range ~75-1900 samples) run
  through every effect individually produces finite, non-crashing output —
  `scipy.signal.lfilter`/`sosfilt` and the tap-sum comb filter all degrade
  gracefully (fewer/no taps land inside such a short array) rather than
  erroring on a size mismatch.
- **Empty (0-sample) and single-sample clips** through every individual
  effect and every preset via the full `apply_effects()` pipeline: all
  finite, no crashes, no exceptions.
- **Every one of the 7 presets run end-to-end through the real
  `apply_effects()` pipeline** (not just the individual effect functions in
  isolation) on both a synthetic tone and the one real recorded voice clip
  present in `recordings/` (6.74s, peak 0.079, quiet) — all finite, none
  exceed the 0.95 limiter ceiling. On the quiet real recording specifically,
  none of the presets pushed the signal anywhere near the limiter threshold
  (peaks stayed 0.04-0.08), which is expected for quiet source material but
  means the limiter's actual clipping-prevention behavior was only
  exercised by the synthetic amplified-signal test, not by this real clip.
- **Degenerate/negative parameters swept deliberately**: `echo repeats <=
  0` (found the crash bug above), `echo decay` at 0.0/-0.5/1.5, `robot
  freq=0`, `distortion bits` at -2 and 0.5 — all either fixed with input
  validation or confirmed non-crashing (finite output) at the function
  level even where the UI doesn't fully constrain them.

## Not yet verified — needs a live run with a real mic and speakers

These require a human speaking, listening, and judging quality, not just
code execution, so they are explicitly **not** claimed as working yet:

- Whether real recorded speech, run through `r`, is actually audible on
  playback at a reasonable volume.
- Whether a negative pitch value sounds noticeably "thick"/deeper on real
  speech, and a positive value noticeably "thin"/higher — the automated
  check only confirms the output is non-silent and the correct length, not
  that it sounds like the intended effect on a human voice (a sine tone
  doesn't reveal formant artifacts the way speech does).
- Whether `speed_rate = 0.7` and `1.4` sound slower/faster with pitch
  audibly unchanged on real speech (formant-preserving quality, not just
  correct output duration).
- Whether the press-Enter-to-stop recording UX actually captures a clean
  start/end with no truncated first/last word.
- Whether the real mic-permission-denied case on macOS actually raises
  `sd.PortAudioError` (vs. e.g. hanging or raising something else) — only
  the exception class's existence was confirmed, not the real denial path.
- Audible artifacts from the phase-vocoder approach — e.g., "robotic"/
  "phasey" texture, or transient smearing — at the pitch/speed ranges
  actually used for a real voice, versus the extreme values (±48 semitones,
  20x speed) which were only checked structurally (non-silent, correct
  length) above.
- Whether the `y`-to-save flow at both prompts (`r` and `f`) produces a
  file that actually plays back correctly in an external player, not just
  round-trips through `soundfile` in-process.
- **Whether any of the 5 new effects (robot, telephone, distortion, echo,
  reverb) or 7 presets actually sound like their names on real speech** —
  the FFT/frequency-response checks above confirm the DSP is mathematically
  doing what it's supposed to (frequency translation, band attenuation,
  quantization, delayed decay, comb/allpass reverberation), but "this
  measurably shifts energy to 190Hz/250Hz" is not the same claim as "this
  sounds like a robot," and none of that was judged by ear yet.
- Whether the reverb's fixed Schroeder parameters (comb/allpass delays and
  gains) sound like a plausible room/hall at the exposed `mix` range, or
  whether they need retuning once heard — these are standard textbook
  values, not tuned against this specific voice/mic chain.
- Whether the fixed effect order (pitch → speed → robot → telephone →
  distortion → echo → reverb) sounds right, or whether a different order
  (e.g. reverb before distortion) would be preferred — see Limitations.
- Whether the telephone bandpass's causal-filter phase distortion
  (`sosfilt` vs. the zero-phase `sosfiltfilt` it replaced) is audible on
  real speech — theoretically phase distortion isn't perceptually
  significant for this kind of effect, but that's an assumption, not a
  listening-test result.
