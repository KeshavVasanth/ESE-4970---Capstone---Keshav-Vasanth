import time
import queue
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import sounddevice as sd
import scipy.signal
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub


# ----------------------------
# Configuration
# ----------------------------

TARGET_SR = 16000  # YAMNet expects 16 kHz mono waveform in [-1, 1]  :contentReference[oaicite:2]{index=2}
BLOCK_SECONDS = 0.25            # audio callback chunk size (sec)
INFER_SECONDS = 1.00            # how much recent audio to classify each time
INFER_EVERY_SECONDS = 0.50      # how often we run inference (sec)
TOP_K = 8                       # how many YAMNet labels to inspect each inference

MIN_CONFIDENCE = 0.30           # ignore weak predictions
SMOOTHING_WINDOW = 8            # number of recent inference steps to vote over
VOTES_REQUIRED = 3              # votes required within window to trigger event
EVENT_COOLDOWN_SECONDS = 5.0    # suppress repeated triggers for same event


# Map your human-friendly classes -> lists of YAMNet label keywords.
# You will tune these after you test your actual sounds.
EVENT_RULES: Dict[str, List[str]] = {
    "Fire alarm / Siren": ["Siren", "Smoke detector", "Alarm", "Fire alarm"],
    "Door knock": ["Knock", "Tap", "Thump", "Bang", "Wood"],
    "Microwave / Timer beep": ["Microwave oven", "Beep", "Buzzer", "Alarm clock"],
    "Doorbell": ["Doorbell", "Chime"],
    "Phone ringtone": ["Ringtone", "Telephone bell ringing", "Telephone"],
    "Dog bark": ["Bark", "Dog"],
}


# ----------------------------
# Helpers
# ----------------------------

def resample_to_16k(x: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample mono audio to 16kHz float32."""
    if orig_sr == TARGET_SR:
        return x.astype(np.float32, copy=False)
    # scipy.signal.resample_poly is fast and good quality
    gcd = np.gcd(orig_sr, TARGET_SR)
    up = TARGET_SR // gcd
    down = orig_sr // gcd
    y = scipy.signal.resample_poly(x, up, down).astype(np.float32)
    return y


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Ensure float32 waveform in [-1, 1]. sounddevice usually gives float32 already."""
    x = x.astype(np.float32, copy=False)
    # If input is somehow outside range, clip safely
    x = np.clip(x, -1.0, 1.0)
    return x


def load_yamnet_and_labels() -> Tuple[hub.KerasLayer, List[str]]:
    # TF Hub YAMNet tutorial shows loading from TF Hub. :contentReference[oaicite:3]{index=3}
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    # Class map CSV is in the TF Models repo. :contentReference[oaicite:4]{index=4}
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    class_map = pd.read_csv(class_map_url)
    class_names = class_map["display_name"].tolist()
    return model, class_names


def match_events_from_labels(
    labels_with_scores: List[Tuple[str, float]],
    rules: Dict[str, List[str]],
    min_conf: float,
) -> List[Tuple[str, float]]:
    """
    Given YAMNet (label, score) pairs, return matched high-level events (event, score).
    """
    matched = []
    for event_name, keywords in rules.items():
        best = 0.0
        for label, score in labels_with_scores:
            if score < min_conf:
                continue
            # Keyword match (simple, fast). You can improve this later.
            for kw in keywords:
                if kw.lower() in label.lower():
                    best = max(best, float(score))
        if best >= min_conf:
            matched.append((event_name, best))
    # Sort by score descending
    matched.sort(key=lambda t: t[1], reverse=True)
    return matched


@dataclass
class EventLogItem:
    timestamp: float
    event: str
    score: float


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Audio Alert Prototype", layout="wide")
st.title("Real-Time ML Audio Alerts (YAMNet Prototype)")

with st.sidebar:
    st.header("Controls")
    input_device = st.number_input("Input device index (leave as 0 if unsure)", min_value=0, value=0, step=1)

    st.subheader("Thresholds")
    min_conf = st.slider("Min confidence", 0.05, 0.90, float(MIN_CONFIDENCE), 0.05)
    votes_required = st.slider("Votes required", 1, 10, int(VOTES_REQUIRED), 1)
    cooldown_s = st.slider("Cooldown (sec)", 0.0, 15.0, float(EVENT_COOLDOWN_SECONDS), 0.5)

    st.subheader("Timing")
    infer_every = st.slider("Inference every (sec)", 0.25, 2.0, float(INFER_EVERY_SECONDS), 0.05)
    infer_window = st.slider("Inference window (sec)", 0.5, 3.0, float(INFER_SECONDS), 0.1)

    st.subheader("Event rules (editable in code)")
    st.caption("Edit EVENT_RULES in app.py to tune label matching.")


status_placeholder = st.empty()
col1, col2 = st.columns([1, 1])
detected_box = col1.container()
log_box = col2.container()

detected_box.subheader("Current detection")
current_event_text = detected_box.markdown("**Listening...**")
current_topk_text = detected_box.text("")

log_box.subheader("Event log")
log_table_placeholder = log_box.empty()

# Load model once
@st.cache_resource
def _cached_model():
    return load_yamnet_and_labels()

yamnet_model, class_names = _cached_model()

# Audio ring buffer
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
ring = deque(maxlen=int(TARGET_SR * 10))  # keep last ~10 seconds at 16 kHz

# Smoothing votes
recent_votes = deque(maxlen=int(SMOOTHING_WINDOW))
last_trigger_time: Dict[str, float] = {}

event_log: List[EventLogItem] = []

def audio_callback(indata, frames, time_info, status):
    if status:
        # Drop status into the queue for visibility (non-fatal)
        pass
    # indata: shape (frames, channels)
    x = indata[:, 0]  # mono from first channel
    audio_q.put(x.copy())

def get_recent_audio(seconds: float) -> np.ndarray:
    n = int(TARGET_SR * seconds)
    if len(ring) < n:
        # pad with zeros if we don't have enough yet
        x = np.zeros(n, dtype=np.float32)
        r = np.array(ring, dtype=np.float32)
        x[-len(r):] = r
        return x
    else:
        # take last n samples
        return np.array(list(ring)[-n:], dtype=np.float32)

def run_yamnet_on_waveform(wave_16k: np.ndarray):
    """
    Returns:
      scores: (num_frames, 521)
      embeddings: (num_frames, 1024)
      spectrogram: ...
    """
    # YAMNet accepts 1-D float32 waveform at 16 kHz. :contentReference[oaicite:5]{index=5}
    wave_16k = normalize_audio(wave_16k)
    scores, embeddings, spectrogram = yamnet_model(wave_16k)
    return scores.numpy()

def topk_labels(scores: np.ndarray, k: int) -> List[Tuple[str, float]]:
    # scores is (time_frames, 521). Average across time for the clip. :contentReference[oaicite:6]{index=6}
    mean_scores = scores.mean(axis=0)
    top_idx = np.argsort(mean_scores)[::-1][:k]
    return [(class_names[i], float(mean_scores[i])) for i in top_idx]

# Start audio stream
try:
    stream = sd.InputStream(
        device=int(input_device),
        channels=1,
        samplerate=TARGET_SR,  # easiest: capture directly at 16k
        blocksize=int(TARGET_SR * BLOCK_SECONDS),
        callback=audio_callback
    )
    stream.start()
    status_placeholder.success("Microphone stream started.")
except Exception as e:
    status_placeholder.error(f"Failed to start mic stream: {e}")
    st.stop()

# Main loop
last_infer = 0.0

# We use Streamlit's "while True" pattern with st.empty updates.
# Stop button
stop = st.button("Stop")

while not stop:
    # Pull any new audio chunks into ring buffer
    try:
        while True:
            chunk = audio_q.get_nowait()
            ring.extend(chunk.tolist())
    except queue.Empty:
        pass

    now = time.time()
    if now - last_infer >= infer_every and len(ring) > int(TARGET_SR * 0.5):
        last_infer = now

        wave = get_recent_audio(infer_window)
        scores = run_yamnet_on_waveform(wave)
        top = topk_labels(scores, TOP_K)

        # Display top-k (debugging / tuning)
        current_topk_text.text("\n".join([f"{lbl:35s}  {sc:0.3f}" for lbl, sc in top]))

        # Map to your events
        matched_events = match_events_from_labels(top, EVENT_RULES, min_conf)

        # Vote for the best matched event (or None)
        best_event: Optional[str] = matched_events[0][0] if matched_events else None
        recent_votes.append(best_event)

        # Determine if any event has enough votes
        vote_counts: Dict[str, int] = {}
        for v in recent_votes:
            if v is None:
                continue
            vote_counts[v] = vote_counts.get(v, 0) + 1

        triggered = None
        triggered_score = 0.0
        for ev, cnt in vote_counts.items():
            if cnt >= votes_required:
                # find score from matched list (if present this frame)
                score_now = 0.0
                for me, sc in matched_events:
                    if me == ev:
                        score_now = sc
                # cooldown check
                last_t = last_trigger_time.get(ev, 0.0)
                if now - last_t >= cooldown_s:
                    triggered = ev
                    triggered_score = score_now
                    break

        if triggered:
            last_trigger_time[triggered] = now
            event_log.insert(0, EventLogItem(timestamp=now, event=triggered, score=triggered_score))
            current_event_text.markdown(f"## ✅ **{triggered}**  \nConfidence: `{triggered_score:0.2f}`")
        else:
            if best_event:
                current_event_text.markdown(f"**Listening…** likely: `{best_event}`")
            else:
                current_event_text.markdown("**Listening…**")

        # Update log table
        if event_log:
            df = pd.DataFrame([{
                "Time": time.strftime("%H:%M:%S", time.localtime(it.timestamp)),
                "Event": it.event,
                "Score": round(it.score, 2)
            } for it in event_log[:25]])
            log_table_placeholder.dataframe(df, use_container_width=True)
        else:
            log_table_placeholder.info("No events logged yet.")

    time.sleep(0.02)

# Cleanup
stream.stop()
stream.close()
st.write("Stopped.")
