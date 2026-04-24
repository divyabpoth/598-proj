"""
Real-Time Cough Inference Pipeline
-------------------------
Record audio from mic, detect cough segments, extract features, 
predict class (healthy / COVID-19 / symptomatic)

Requirements (or refer to requirements.txt):
    pip install sounddevice numpy scipy librosa pycaret pandas

Usage:
    1. Train + save model (using train_xgboost.py)
    2. Run: python realtime_cough_inference.py
"""

# imports
import sys
import numpy as np
import librosa
import sounddevice as sd
import pandas as pd


# audio configuration settings
SAMPLE_RATE = 22050   # Hz  (matches training)
RECORD_SECONDS = 5 # sec to record
AUDIO_LENGTH = 22050
MODEL_PATH = "xgboost_model.pkl" # saved model from train_xgboost.py


# feature extraction 
def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    features = []
    stft = np.abs(librosa.stft(audio_data))

    # computing mfcc, chroma, mel, and spectral contrast features (which is same as training)
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    features.extend(mfcc)

    # computing chroma features
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features.extend(chroma)

    # computing mel spectrogram features
    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
    features.extend(mel)  # 128 → 180

    # computing spectral contrast features
    fmin_val = 0.5 * sample_rate * 2 ** (-6)
    contrast = np.mean(
        librosa.feature.spectral_contrast(S=stft, sr=sample_rate, fmin=fmin_val).T, axis=0
    )
    features.extend(contrast)

    return np.array(features)


# cought segmentation to isloate coughs from background noise
def segment_cough(
    x: np.ndarray,
    fs: int,
    cough_padding: float = 0.1,
    min_cough_len: float  = 0.1,
    th_l_multiplier: float = 0.1,
    th_h_multiplier: float = 2.0,
):
    # compute rms and set thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    cough_segments = []
    padding = round(fs * cough_padding) # samples to pad on either side of detected cough segment
    min_cough_samples = round(fs * min_cough_len)
    cough_start = cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0

    # iterate through samples to find cough segments based on thresholds
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


# pad / train segment to fixed length of 1 sec
def normalize_segment(audio: np.ndarray, target_len: int = AUDIO_LENGTH) -> np.ndarray:
    if len(audio) < target_len:
        return librosa.util.pad_center(audio, size=target_len)
    return audio[:target_len]


# record, then flatten to 1D array for processing
def record_audio(duration: int = RECORD_SECONDS, fs: int = SAMPLE_RATE) -> np.ndarray:
    print(f"\n  Recording for {duration} seconds... Cough now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.\n")
    return audio.flatten()


# main inference loop: record audio, segment coughs, extract features, predict with model, print results
def predict(model, audio: np.ndarray, fs: int = SAMPLE_RATE):

    segments = segment_cough(audio, fs)

    if not segments:
        print("No cough detected.")
        return

    print(f"Detected {len(segments)} cough segment(s). Extracting features...")

    # extracting features and store in list
    all_features = []
    for seg in segments:
        if len(seg) < 8000:
            continue # skipping very short segments
        normed = normalize_segment(seg)
        feat = extract_features(normed, fs)
        all_features.append(feat)

    if not all_features:
        print("Segments too short for reliable prediction.")
        return

    # building dataframe that matches training format
    feat_arr = np.array(all_features)
    col_names = (
        [f"mfcc{i}"  for i in range(1, 41)]
      + [f"mel{i}"   for i in range(1, 129)]
      + [f"chr{i}"   for i in range(1, 13)]
      + [f"con{i}"   for i in range(1, 8)]
    )

    # note that uuid column is ignored by pipeline
    df = pd.DataFrame(feat_arr, columns=col_names)
    df.insert(0, "uuid", "live_recording")

    # make segment-level predictions with model
    predictions = model.predict(df.drop(columns=["uuid"]))

    # majority vote across segments
    unique, counts = np.unique(predictions, return_counts=True)
    final = unique[np.argmax(counts)]

    print(f"Segment predictions : {list(predictions)}")
    print(f"Final prediction : {final.upper()}")

    return final

# main loop to run inference continuously until user quits
def run_inference_loop(model_path: str = MODEL_PATH):
    print("Real-Time Cough Classifier")

    # load saved PyCaret pipeline
    try:
        from pycaret.classification import load_model
        model = load_model(model_path.replace(".pkl", ""))
        print(f"Model loaded from '{model_path}'")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Train & save model first.")
        sys.exit(1)

    # continuously prompt user to record audio and make predictions until they choose to quit
    while True:
        print("\nPress ENTER to start recording (or type 'q' + ENTER to quit):")
        user_input = input().strip().lower()
        if user_input == "q":
            break

        audio = record_audio()
        predict(model, audio)

if __name__ == "__main__":
    run_inference_loop()
