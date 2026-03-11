# imports
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

## DATASET PREP
DATASET_PATH =  "public_dataset"

X = []
y = []

max_samples = 2000   # limit dataset for now - taking too long to load entire data
count = 0

# to allevaite class imbalance
healthy_count = 0
symptomatic_count = 0
max_per_class = 250

for file in os.listdir(DATASET_PATH):

    if file.endswith(".json"):

        json_path = os.path.join(DATASET_PATH, file)

        with open(json_path, "r") as f:
            metadata = json.load(f)

        # skip samples that aren't labeled
        if "status" not in metadata:
            continue

        # skip samples that have low cough-detected (thresh == 0.8 for now)
        if float(metadata["cough_detected"]) < 0.8:
            continue

        label = metadata["status"]

        # if label not in ["healthy", "symptomatic"]:
        #     continue
        
        if label == "healthy":
            healthy_count += 1
            if healthy_count >= max_per_class:
                continue

        elif label == "symptomatic": 
            symptomatic_count += 1
            if symptomatic_count >= max_per_class:
                continue

        else:
            continue # skip if diff label (for now)

        # get corresponding audio file
        wav_file = file.replace(".json", ".wav")
        wav_path = os.path.join(DATASET_PATH, wav_file)

        # if .wav doesn't exis for that .json
        if not os.path.exists(wav_path):
            continue

        samplerate, audio = wavfile.read(wav_path)

        # convert from stereo to mono
        if len(audio.shape) > 1:
            audio = audio[:,0]

        # generate spectrogram - referencing pa1_1
        f, t_spec, Sxx = spectrogram(
            audio,
            fs=samplerate,
            nperseg=1024
        )

        Sxx = np.log1p(Sxx)
        Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-8) # adding 1e-8 so no divide by 0 issues

        # reduce size
        Sxx = Sxx[:64, :64]
        X.append(Sxx)
        y.append(label)


        count += 1
        if count >= max_samples:
            break

le = LabelEncoder()
y = le.fit_transform(y)
X = np.array(X)
y = np.array(y)

# add channel dimension - for cnn input to be: (num_smaples, 1, 64, 64)
X = X[:, np.newaxis, :, :]

print("Dataset shape:", X.shape)


## PRETRAINING
# cnn base model
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# converting data to tensors for cnn
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

## TRAINING LOOP
model = BaseCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# evaluate model performance
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=le.classes_
))