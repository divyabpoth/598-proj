"""
Real-time cough inference — pairs with train_coughvid.py
Requires: sounddevice, torchaudio, torch, pickle

Usage:  python infer_realtime.py
"""

import pickle, time
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import sounddevice as sd

from train_coughvid import CoughCNN, TARGET_SR, CLIP_SECONDS, N_MELS, N_FFT, HOP_LENGTH

MODEL_PATH   = "coughvid_model.pth"
ENCODER_PATH = "label_encoder.pkl"

device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")

# ── load model ──
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)
model = CoughCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded  |  classes: {list(le.classes_)}")

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=50, f_max=8000,
)

def preprocess(audio_np):
    """numpy float32 array → (1, 1, 64, 64) tensor."""
    waveform = torch.tensor(audio_np, dtype=torch.float32)
    target_len = TARGET_SR * CLIP_SECONDS
    if waveform.shape[0] >= target_len:
        waveform = waveform[:target_len]
    else:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[0]))

    mel = mel_transform(waveform)
    mel = torch.log1p(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = F.interpolate(
        mel.unsqueeze(0).unsqueeze(0), size=(64, 64),
        mode="bilinear", align_corners=False,
    )
    return mel


def predict(audio_np):
    x = preprocess(audio_np).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx   = int(torch.argmax(torch.tensor(probs)))
    pred_label = le.classes_[pred_idx]
    confidence = probs[pred_idx]

    return pred_label, confidence, probs, latency_ms


def main():
    duration = CLIP_SECONDS
    print(f"\nRecording {duration}s — cough now ...")
    audio = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print("Done.\n")

    label, conf, probs, lat = predict(audio)

    print(f"Prediction  : {label}")
    print(f"Confidence  : {conf:.1%}")
    print(f"Latency     : {lat:.1f} ms\n")
    print("All class probabilities:")
    for cls, p in zip(le.classes_, probs):
        bar = "█" * int(p * 30)
        print(f"  {cls:15s} {p:.1%}  {bar}")


if __name__ == "__main__":
    main()
