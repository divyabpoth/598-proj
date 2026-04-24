# main steps: 
# 1 record audio from microphone
# 2 convert to mono
# 3 compute spectrogram
# 4 reshape to CNN input
# 5 load trained model
# 6 run inference
# 7 print predicted cough class

# imports
import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import time

from old_versions.cough_classifier_v2 import BaseCNN

# device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# using same MEL spectrogram as done in training
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

labels = ["COVID-19", "healthy", "symptomatic"]  # adjust if label encoder differs


# load in model
model = BaseCNN().to(device)
model.load_state_dict(torch.load("cough_classifier_v2.pth", map_location=device))
model.eval()


# record audio
samplerate = 16000
duration = 3 # in sec

print("Recording...")

audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=1
)

sd.wait()
print("Recording complete")

# flatten and convert to tensor
audio = audio.flatten()
audio_tensor = torch.tensor(audio, dtype=torch.float32)


# create mel spectrogram
mel_spec = mel_transform(audio_tensor)
mel_spec = torch.log1p(mel_spec)
mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)


# resize to 64x64
mel_spec = torch.nn.functional.interpolate(
    mel_spec.unsqueeze(0).unsqueeze(0),
    size=(64, 64),
    mode="bilinear",
    align_corners=False
).squeeze()


# cnn input shape reqs
X = mel_spec.unsqueeze(0).unsqueeze(0).to(device)


# inference
start = time.time()
with torch.no_grad():
    output = model(X)
    pred = torch.argmax(output, dim=1).item()
end = time.time()

print("Prediction:", labels[pred])
print("Inference latency:", end - start, "seconds")