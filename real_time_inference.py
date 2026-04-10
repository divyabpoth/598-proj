"""
Real-Time Cough Inference
-------------------------
Records audio from your microphone, detects cough segments,
extracts features, and predicts: healthy / COVID-19 / symptomatic.

Requirements:
    pip install sounddevice numpy scipy librosa pycaret[full] joblib

Usage:
    1. Train and save your model first (see bottom of this file).
    2. Run:  python realtime_cough_inference.py
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import sounddevice as sd
import joblib

# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050   # Hz  (matches training)
RECORD_SECONDS   = 5       # seconds to record per session
AUDIO_LENGTH     = 22050   # samples per cough segment (matches training)
MODEL_PATH       = "xgboost_model.pkl"   # saved PyCaret pipeline


# ── Feature extraction (must match training exactly) ─────────────────────────

def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    features = []
    stft = np.abs(librosa.stft(audio_data))

    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    features.extend(mfcc)  # 40

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features.extend(chroma)  # 12 → 52

    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
    features.extend(mel)  # 128 → 180

    fmin_val = 0.5 * sample_rate * 2 ** (-6)
    contrast = np.mean(
        librosa.feature.spectral_contrast(S=stft, sr=sample_rate, fmin=fmin_val).T, axis=0
    )
    features.extend(contrast)  # 7 → 187

    return np.array(features)


# ── Cough segmentation (identical hysteresis logic from training) ─────────────

def segment_cough(
    x: np.ndarray,
    fs: int,
    cough_padding: float = 0.1,
    min_cough_len: float  = 0.1,
    th_l_multiplier: float = 0.1,
    th_h_multiplier: float = 2.0,
):
    rms       = np.sqrt(np.mean(np.square(x)))
    seg_th_l  = th_l_multiplier * rms
    seg_th_h  = th_h_multiplier * rms

    cough_segments = []
    padding            = round(fs * cough_padding)
    min_cough_samples  = round(fs * min_cough_len)
    cough_start = cough_end = 0
    cough_in_progress   = False
    tolerance           = round(0.01 * fs)
    below_th_counter    = 0

    for i, sample in enumerate(x ** 2):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = min(i + padding, len(x) - 1)
                    cough_in_progress = False
                    if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                        cough_segments.append(x[cough_start : cough_end + 1])
            elif i == len(x) - 1:
                cough_end = i
                cough_in_progress = False
                if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                    cough_segments.append(x[cough_start : cough_end + 1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = max(i - padding, 0)
                cough_in_progress = True

    return cough_segments


# ── Pad / trim segment to fixed length ───────────────────────────────────────

def normalize_segment(audio: np.ndarray, target_len: int = AUDIO_LENGTH) -> np.ndarray:
    if len(audio) < target_len:
        return librosa.util.pad_center(audio, size=target_len)
    return audio[:target_len]


# ── Recording ─────────────────────────────────────────────────────────────────

def record_audio(duration: int = RECORD_SECONDS, fs: int = SAMPLE_RATE) -> np.ndarray:
    print(f"\n🎙️  Recording for {duration} seconds... Cough now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    print("✅  Recording complete.\n")
    return audio.flatten()


# ── Main inference loop ───────────────────────────────────────────────────────

def predict(model, audio: np.ndarray, fs: int = SAMPLE_RATE):
    import pandas as pd

    segments = segment_cough(audio, fs)

    if not segments:
        print("⚠️  No cough detected. Try again and cough more clearly.")
        return

    print(f"🔍  Detected {len(segments)} cough segment(s). Extracting features...")

    all_features = []
    for seg in segments:
        if len(seg) < 8000:
            continue  # skip very short fragments
        normed  = normalize_segment(seg)
        feat    = extract_features(normed, fs)
        all_features.append(feat)

    if not all_features:
        print("⚠️  Segments too short for reliable prediction.")
        return

    # Build a DataFrame with the same column names used during training
    feat_arr   = np.array(all_features)
    col_names  = (
        [f"mfcc{i}"  for i in range(1, 41)]
      + [f"mel{i}"   for i in range(1, 129)]
      + [f"chr{i}"   for i in range(1, 13)]
      + [f"con{i}"   for i in range(1, 8)]
    )
    # uuid column is ignored by the pipeline (ignore_features=['uuid'] in setup)
    df = pd.DataFrame(feat_arr, columns=col_names)
    df.insert(0, "uuid", "live_recording")

    predictions = model.predict(df.drop(columns=["uuid"]))

    # Majority vote across segments
    unique, counts = np.unique(predictions, return_counts=True)
    final = unique[np.argmax(counts)]

    print("━" * 40)
    print(f"  Segment predictions : {list(predictions)}")
    print(f"  ➜  Final prediction : {final.upper()}")
    print("━" * 40)

    return final


def run_inference_loop(model_path: str = MODEL_PATH):
    print("=" * 50)
    print("   Real-Time Cough COVID Classifier")
    print("=" * 50)

    # Load saved PyCaret pipeline
    try:
        from pycaret.classification import load_model
        model = load_model(model_path.replace(".pkl", ""))
        print(f"✅  Model loaded from '{model_path}'")
    except Exception as e:
        print(f"❌  Could not load model: {e}")
        print("    Train & save it first — see the bottom of this file.")
        sys.exit(1)

    while True:
        print("\nPress ENTER to start recording (or type 'q' + ENTER to quit):")
        user_input = input().strip().lower()
        if user_input == "q":
            print("Goodbye! 👋")
            break

        audio = record_audio()
        predict(model, audio)


# ── How to save the trained model from your training script ──────────────────
# Add these two lines at the end of your training notebook/script:
#
#   from pycaret.classification import save_model
#   save_model(xgboost, 'cough_model')   # saves as cough_model.pkl
#
# Then run this file in the same directory.

if __name__ == "__main__":
    run_inference_loop()
