# Cough Audio Classifier: Healthy / COVID-19 / Symptomatic

A real-time cough classification system that records audio from a microphone, detects cough segments, extracts acoustic features, and predicts one of three classes: **healthy**, **COVID-19**, or **symptomatic (non-COVID)**, using a trained XGBoost model.

---

## Overview

This project trains and deploys a 3-way cough classifier using the [CoughVID](https://coughvid.epfl.ch/) public dataset. The pipeline covers:

1. Loading and filtering labeled cough audio (`.wav`) and metadata (`.csv`)
2. Segmenting cough events from raw audio using power-based hysteresis
3. Extracting hand-crafted audio features (MFCCs, mel spectrogram, chroma, spectral contrast)
4. Training a classifier via [PyCaret](https://pycaret.org/) + XGBoost
5. Running real-time inference from a microphone with majority-vote prediction over segments

---

## Project Structure

```
proj/
├── train_xgboost.py          # Training pipeline: data, features, PyCaret/XGBoost
├── real_time_inference.py    # Mic recording, segmentation, features, prediction
├── xgboost_model.pkl         # Saved PyCaret model pipeline (produced by training)
├── requirements.txt          # Pinned Python dependencies
├── public_dataset/
│   ├── metadata_compiled.csv # Labels and metadata for ~27,550 recordings
│   ├── *.wav                 # Raw cough audio files (UUID-named)
│   └── *.json                # Per-file metadata (UUID-named)
└── old_versions/             # Earlier PyTorch-based classifiers and notebooks
```

---

## Features

- **Audio feature extraction (187 dimensions per 1-second segment):**
  - 40 MFCCs
  - 128 mel spectrogram coefficients
  - 12 chroma features
  - 7 spectral contrast bands
- **Cough segmentation** via power-envelope hysteresis (no ML-based VAD required)
- **Balanced training**: 2,185 samples per class (healthy / COVID-19 / symptomatic)
- **PyCaret preprocessing**: normalization, power transformation, 10-fold cross-validation
- **Real-time inference loop**: record, predict, and repeat from the command line

---

## Setup

### Prerequisites

- Python 3.8+
- A working microphone (for real-time inference)
- The [CoughVID public dataset](https://zenodo.org/record/4498364) placed under `public_dataset/`

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` was generated from a conda environment and contains a `file://` reference for `packaging`. If installation fails on that line, replace it with `packaging` (no version pin) before running.

---

## Usage

### 1. Train the model

```bash
python train_xgboost.py
```

This reads `public_dataset/metadata_compiled.csv`, loads the corresponding `.wav` files, extracts features, runs `pycaret.setup()` + `compare_models()`, trains an XGBoost classifier, and saves the pipeline to `xgboost_model.pkl`.

Reported cross-validated performance (10-fold):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.72  |
| F1 Score  | ~0.705 |

### 2. Run real-time inference

```bash
python real_time_inference.py
```

The script records 5 seconds of audio from the default microphone, segments cough events, extracts features, and prints the predicted class. Press Enter to record again or type `q` to quit.

---

## Dataset

The project uses the **CoughVID** dataset, a large-scale crowdsourced collection of cough recordings with physician-validated COVID-19 status labels. Only recordings that meet quality filters are used:

- `cough_detected >= 0.8`
- `quality == 'good'`
- Non-null `status` label

---

## Model Details

| Component      | Details                          |
|----------------|----------------------------------|
| Model          | XGBoost (`xgboost==3.2.0`)       |
| Preprocessing  | PyCaret (`pycaret==3.3.2`)       |
| Feature dim.   | 187 per segment                  |
| Classes        | healthy, COVID-19, symptomatic   |
| Sampling rate  | 22,050 Hz                        |
| Segment length | 1 second (padded/trimmed)        |

---

## Old Versions

The `old_versions/` directory contains earlier experimental approaches:

- `cough_classifier_v1.py`: PyTorch-based classifier using per-file features
- `cough_classifier_v2.py`: PyTorch + `torchaudio` MelSpectrogram (MPS/CPU support)
- `train_diff_classifiers.ipynb`: Jupyter notebook exploring multiple classifiers
- `infer_realtime.py`: Earlier real-time inference prototype

These are not maintained and require PyTorch installed separately.

---