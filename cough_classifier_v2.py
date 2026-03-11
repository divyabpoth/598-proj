# imports
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

# device setup (Mac GPU support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

## DATASET PREP
DATASET_PATH = "public_dataset"

VALID_LABELS = ["healthy", "symptomatic", "COVID-19"]

X = []
y = []

max_samples = 3000
count = 0

healthy_count = 0
symptomatic_count = 0
covid_count = 0

max_per_class = 1000

# Mel Spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

for file in os.listdir(DATASET_PATH):

    if not file.endswith(".json"):
        continue

    json_path = os.path.join(DATASET_PATH, file)

    with open(json_path, "r") as f:
        metadata = json.load(f)

    label = metadata.get("status")

    # skip missing or invalid labels
    if label not in VALID_LABELS:
        continue

    # skip low cough detection confidence
    if float(metadata.get("cough_detected", 0)) < 0.8:
        continue

    # class balancing
    if label == "healthy":
        healthy_count += 1
        if healthy_count > max_per_class:
            continue

    elif label == "symptomatic":
        symptomatic_count += 1
        if symptomatic_count > max_per_class:
            continue

    elif label == "COVID-19":
        covid_count += 1
        if covid_count > max_per_class:
            continue

    wav_file = file.replace(".json", ".wav")
    wav_path = os.path.join(DATASET_PATH, wav_file)

    if not os.path.exists(wav_path):
        continue

    samplerate, audio = wavfile.read(wav_path)

    # stereo → mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # remove very short clips (<0.5 sec)
    if len(audio) < samplerate * 0.5:
        continue

    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    # normalize audio length to 3 seconds
    target_length = 16000 * 3   # 3 seconds

    if len(audio_tensor) < target_length:
        pad = target_length - len(audio_tensor)
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))
    else:
        audio_tensor = audio_tensor[:target_length]

    # MEL SPECTROGRAM
    mel_spec = mel_transform(audio_tensor)

    mel_spec = torch.log1p(mel_spec)

    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    mel_spec = torch.nn.functional.interpolate(
    mel_spec.unsqueeze(0).unsqueeze(0),
    size=(64, 64),
    mode="bilinear",
    align_corners=False).squeeze()

    X.append(mel_spec.numpy())
    y.append(label)

    count += 1
    if count >= max_samples:
        break


print("Healthy samples:", healthy_count)
print("Symptomatic samples:", symptomatic_count)
print("COVID samples:", covid_count)

le = LabelEncoder()
y = le.fit_transform(y)

X = np.array(X)
y = np.array(y)

# add CNN channel dimension
X = X[:, np.newaxis, :, :]

print("Dataset shape:", X.shape)


## MODEL
class BaseCNN(nn.Module):

    def __init__(self):
        super(BaseCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # 3 classes
        )

    def forward(self, x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


## TENSOR CONVERSION
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor,
    y_tensor,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)


## TRAINING
model = BaseCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


## EVALUATION
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())


print(classification_report(
    all_labels,
    all_preds,
    target_names=le.classes_
))