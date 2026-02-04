"""
mp_terminal.py

A terminal-only, real-time audio “alert” prototype using MediaPipe Tasks AudioClassifier.

WHAT THIS DOES
--------------
1) Listens to your microphone continuously.
2) Every INFER_EVERY_S seconds, it takes the most recent INFER_WINDOW_S seconds of audio.
3) It feeds that audio into a pretrained audio classifier (YAMNet packaged for MediaPipe).
4) The model outputs labels like “Doorbell”, “Ringtone”, “Water”, “Knock”, etc. with scores.
5) We map those labels into YOUR six categories:
      - Fire alarm
      - Knock
      - Microwave beep
      - Phone ringtone
      - Doorbell
      - Water running
6) To reduce false triggers, we use:
      - MIN_SCORE threshold (ignore low-confidence labels)
      - Voting over a rolling window (VOTE_WINDOW)
      - Cooldown to prevent spamming (COOLDOWN_S)

HOW TO RUN
----------
Activate your Python 3.11 venv (your .venv311):
    source .venv311/bin/activate

Then run:
    python mp_terminal.py

Stop:
    Ctrl + C

TROUBLESHOOTING
---------------
- If you see mostly "(none)", you may need to grant microphone permission:
  System Settings → Privacy & Security → Microphone → enable Terminal (or VS Code).
"""

# ----------------------------
# Standard library imports
# ----------------------------
import os                 # For checking if model file exists on disk
import time               # For timestamps + timing inference intervals
import queue              # Thread-safe queue for audio callback -> main loop
from collections import deque  # Efficient fixed-length ring buffers / vote windows
from typing import Dict, List, Optional, Tuple  # Type hints (for readability)

# ----------------------------
# Third-party imports
# ----------------------------
import numpy as np        # Numerical arrays (audio waveform)
import sounddevice as sd  # Capturing audio from microphone

# MediaPipe Tasks (audio classifier)
# - python.BaseOptions: points to model file
# - audio.AudioClassifier: classifier wrapper
# - containers.AudioData: wraps raw waveform into format expected by classifier
import mediapipe as mp  # noqa: F401 (import helps verify mediapipe is available)
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers


# =============================================================================
# 1) USER CONFIGURATION
# =============================================================================

# EVENT_RULES maps YOUR “events” to sets of keywords.
# The model produces labels (strings). We lowercase them and look for these keywords.
#
# Example:
#   Model label might be "Doorbell" or "Chime"
#   If it contains "doorbell" or "chime", we count it as Doorbell.
EVENT_RULES: Dict[str, List[str]] = {
    "Fire alarm": ["siren", "alarm", "smoke", "fire"],
    "Knock": ["knock", "tap", "bang", "thump"],
    "Microwave beep": ["microwave", "beep", "buzzer", "timer"],
    "Phone ringtone": ["ringtone", "telephone", "phone", "ringing"],
    "Doorbell": ["doorbell", "chime"],
    "Water running": ["water", "faucet", "sink", "bathtub", "toilet", "shower"],
}

# ---- Noise reduction / stability controls ----

# Ignore any category score below MIN_SCORE.
# Lower this if you're missing events; raise it if you get false triggers.
MIN_SCORE = 0.25

# How many recent inference results we remember for voting.
# Larger = more stable but slower to trigger.
VOTE_WINDOW = 10

# How many times an event must appear within the last VOTE_WINDOW
# before we “trigger” it.
VOTES_REQUIRED = 4

# After triggering an event, we won’t trigger that SAME event again
# until this many seconds have passed (prevents spamming).
COOLDOWN_S = 6.0

# ---- Audio capture settings ----

# Sample rate for microphone capture.
# 16 kHz is common for many audio models and is a safe default.
SR = 16000

# Microphone callback chunk size.
# Smaller = lower latency but more overhead; 0.25s is a good balance.
BLOCK_S = 0.25

# How often we run classification.
# For example, every 0.5 seconds.
INFER_EVERY_S = 0.50

# How much recent audio we classify each time.
# For example, classify the last 1.0 seconds.
INFER_WINDOW_S = 1.00

# Number of top categories to print each inference for debugging.
# Increase to see more model labels.
PRINT_TOP_K = 5

# ---- Model file settings ----

# MediaPipe AudioClassifier requires a .tflite file WITH MediaPipe metadata.
# This model URL is a MediaPipe-packaged YAMNet.
MODEL_FILE = "sound_classifier.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"


# =============================================================================
# 2) HELPER FUNCTIONS
# =============================================================================

def ensure_model_file() -> None:
    """
    Ensure the MediaPipe-compatible model file exists locally.

    If it doesn't exist, we download it from MODEL_URL and save it as MODEL_FILE.
    This prevents you from manually downloading the file.
    """
    # If the model is already present in the current folder, do nothing.
    if os.path.exists(MODEL_FILE):
        return

    # Otherwise download it.
    print(f"Model '{MODEL_FILE}' not found. Downloading...")
    import urllib.request  # Standard library downloader

    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print(f"Downloaded '{MODEL_FILE}' successfully.")


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """
    Convert audio to float32 and clamp values to [-1.0, 1.0].

    Why:
    - Many audio pipelines expect float32 waveforms.
    - Clamping avoids weird out-of-range values.
    """
    x = x.astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0)


def extract_categories(result) -> List:
    """
    Extract a list of "categories" (label+score objects) from MediaPipe output.

    MediaPipe API version differences:
    - Some versions return an AudioClassifierResult object:
        result.classifications[0].categories
    - Some versions return a list of AudioClassifierResult:
        result[0].classifications[0].categories

    This function makes the rest of the program independent of that difference.

    Returns:
        A Python list of ClassificationCategory objects.
        Each category typically has:
          - category_name (string label)
          - score (float confidence)
    """
    # If MediaPipe returned a list, take the first result item.
    if isinstance(result, list):
        result_obj = result[0] if len(result) > 0 else None
    else:
        result_obj = result

    # If nothing is there, no categories.
    if result_obj is None:
        return []

    # Typical structure: result_obj.classifications is a list of "heads"
    # and each head has .categories.
    if hasattr(result_obj, "classifications") and result_obj.classifications:
        head = result_obj.classifications[0]  # take the first head
        if hasattr(head, "categories") and head.categories:
            return list(head.categories)

    # If we can't find the expected attributes, return empty.
    return []


def map_to_event(categories, min_score: float) -> Optional[Tuple[str, float]]:
    """
    Convert model categories → your custom event label.

    Strategy:
    - The model gives categories like:
         "Doorbell": 0.62
         "Chime": 0.40
         "Water": 0.55
    - We lowercase the label and see if it contains any keyword
      from EVENT_RULES[event].
    - We keep the highest score matched for each event.
    - We return the best-scoring event overall.

    Args:
        categories: list of ClassificationCategory objects
        min_score: ignore categories below this confidence

    Returns:
        (event_name, best_score) if something matched, else None
    """
    # Convert categories into a simple list of (label_lowercase, score_float)
    label_scores: List[Tuple[str, float]] = []
    for c in categories:
        name = getattr(c, "category_name", "")
        score = float(getattr(c, "score", 0.0))
        label_scores.append((name.lower(), score))

    best_event: Optional[str] = None
    best_score: float = 0.0

    # For each event, check if any of its keywords appear in any label.
    for event, keywords in EVENT_RULES.items():
        ev_best = 0.0  # best score found for THIS event

        for label, score in label_scores:
            if score < min_score:
                continue  # ignore weak predictions

            for kw in keywords:
                # Simple substring match (works well enough for a prototype)
                if kw in label:
                    ev_best = max(ev_best, score)

        # If this event is the strongest so far, remember it.
        if ev_best > best_score:
            best_score = ev_best
            best_event = event

    # If no event matched any keywords, return None.
    if best_event is None:
        return None

    return best_event, best_score


def top_k_string(categories, k: int) -> str:
    """
    Convert top-k categories into a readable one-line string for printing.

    Example output:
        "Doorbell:0.62 | Chime:0.40 | Ringtone:0.10 | ..."

    Args:
        categories: list of ClassificationCategory objects
        k: how many to show
    """
    out = []
    for c in categories[:k]:
        out.append(f"{c.category_name}:{c.score:.2f}")
    return " | ".join(out)


# =============================================================================
# 3) MAIN LOOP
# =============================================================================

def main() -> None:
    """
    Main application logic.

    Steps:
    1) Ensure the model file exists locally.
    2) Construct MediaPipe AudioClassifier.
    3) Start microphone stream using sounddevice.
    4) Continuously buffer microphone audio into a ring buffer.
    5) Periodically run inference on the latest audio window.
    6) Print top labels + trigger events using voting/cooldown.
    """
    # Make sure we have the correct model file on disk.
    ensure_model_file()

    # Create the MediaPipe classifier.
    # base_options points to the .tflite file (with metadata).
    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)

    # options: how many results (labels) we want back each inference
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        max_results=PRINT_TOP_K
    )

    # Build classifier object from options
    classifier = audio.AudioClassifier.create_from_options(options)

    # ----------------------------
    # Audio buffering objects
    # ----------------------------

    # Audio callback will push short chunks here.
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    # Ring buffer stores the last ~10 seconds of audio.
    # (We only classify ~1 second at a time, but keeping more is harmless.)
    ring = deque(maxlen=int(SR * 10))

    # Voting: store last VOTE_WINDOW best_event outputs
    recent_votes: deque[Optional[str]] = deque(maxlen=VOTE_WINDOW)

    # Cooldown tracking: last time we triggered each event type
    last_trigger_time: Dict[str, float] = {}

    # ----------------------------
    # Microphone callback
    # ----------------------------

    def callback(indata, frames, time_info, status):
        """
        Called by sounddevice in a separate audio thread whenever
        a new block of microphone samples is available.

        IMPORTANT:
        - Keep this function FAST.
        - Do not run ML here.
        - Just copy audio and push it into a queue for the main loop.
        """
        # indata shape: (frames, channels). We use mono: channel 0.
        audio_q.put(indata[:, 0].copy())

    print("\nStarting mic... (Ctrl+C to stop)\n")

    # Open a microphone stream. When inside this 'with' block:
    # - sounddevice continuously calls callback(...)
    # - we do processing in the while loop below
    with sd.InputStream(
        channels=1,                         # mono
        samplerate=SR,                      # sample rate
        blocksize=int(SR * BLOCK_S),        # frames per callback
        callback=callback                   # function called on each block
    ):
        last_infer = 0.0  # tracks the last time we ran inference

        # Infinite loop: keep running until user stops (Ctrl+C)
        while True:
            # -----------------------------------------------------
            # 1) Move any queued audio chunks into the ring buffer
            # -----------------------------------------------------
            try:
                while True:
                    # Get next chunk without blocking. If empty, queue.Empty is thrown.
                    chunk = audio_q.get_nowait()

                    # Append chunk samples into ring buffer (as Python floats)
                    ring.extend(chunk.tolist())
            except queue.Empty:
                # No more queued chunks right now.
                pass

            now = time.time()

            # -----------------------------------------------------
            # 2) Time to run inference?
            # -----------------------------------------------------
            # Only run classifier if enough time has passed AND we have enough audio buffered.
            if now - last_infer >= INFER_EVERY_S and len(ring) > int(SR * 0.5):
                last_infer = now

                # -------------------------------------------------
                # 3) Prepare the input audio window for the model
                # -------------------------------------------------
                # Extract the last INFER_WINDOW_S seconds from ring buffer.
                n = int(SR * INFER_WINDOW_S)

                # Convert ring buffer to numpy array (last n samples)
                wave = np.array(list(ring)[-n:], dtype=np.float32)

                # Normalize/clamp
                wave = normalize_audio(wave)

                # Wrap it as MediaPipe AudioData
                audio_data = containers.AudioData.create_from_array(
                    wave,
                    sample_rate=SR
                )

                # -------------------------------------------------
                # 4) Run the audio classifier
                # -------------------------------------------------
                result = classifier.classify(audio_data)

                # Extract categories robustly (fixes your “list has no attribute” issue)
                categories = extract_categories(result)

                # -------------------------------------------------
                # 5) Print top-K labels for debugging
                # -------------------------------------------------
                timestamp = time.strftime("%H:%M:%S")
                if categories:
                    debug_line = top_k_string(categories, PRINT_TOP_K)
                    print(f"[{timestamp}] top-k: {debug_line}")
                else:
                    print(f"[{timestamp}] top-k: (none)")

                # -------------------------------------------------
                # 6) Map model labels to YOUR event names
                # -------------------------------------------------
                mapped = map_to_event(categories, MIN_SCORE)

                # Best event name (or None if nothing matched)
                best_event = mapped[0] if mapped else None

                # Score for the best event (0 if none)
                best_score = mapped[1] if mapped else 0.0

                # -------------------------------------------------
                # 7) Vote smoothing (reduce flicker / false triggers)
                # -------------------------------------------------
                recent_votes.append(best_event)

                # Count how many times each event appeared in the vote window
                vote_counts: Dict[str, int] = {}
                for v in recent_votes:
                    if v is None:
                        continue
                    vote_counts[v] = vote_counts.get(v, 0) + 1

                # -------------------------------------------------
                # 8) Decide whether to trigger an event
                # -------------------------------------------------
                triggered = None
                for ev, cnt in vote_counts.items():
                    # Need enough votes
                    if cnt >= VOTES_REQUIRED:
                        # Also respect cooldown
                        last_t = last_trigger_time.get(ev, 0.0)
                        if now - last_t >= COOLDOWN_S:
                            triggered = ev
                            break

                # If triggered, log it and print a big message
                if triggered:
                    last_trigger_time[triggered] = now
                    print(f"   ✅ EVENT: {triggered} (score ~ {best_score:.2f})\n")

            # -----------------------------------------------------
            # 9) Sleep a tiny amount to reduce CPU usage
            # -----------------------------------------------------
            time.sleep(0.02)


# =============================================================================
# 4) ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # This block runs when you execute: python mp_terminal.py
    # We catch Ctrl+C cleanly so you don’t get an ugly stack trace.
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
