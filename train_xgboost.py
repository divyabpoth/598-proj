# utilities
from math import nan
import os
import sys
from tqdm import tqdm
import random

# data manipulation
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# pycaret
from pycaret.classification import *

# scipy
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter,filtfilt
from scipy.signal import cwt
from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal.windows import get_window

# Set seed for reproducibility
seed_value= 32 
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# set variables
ROOT = 'public_dataset/'
class_names = ['healthy','COVID-19','symptomatic']
audio_length = 22050

# load coughvid meta
data_raw = pd.read_csv(ROOT+'metadata_compiled.csv')


def split_by_physicians(df):
    column_names = ['uuid', 'datetime', 'cough_detected', 'SNR', 'latitude', 'longitude', 
                    'age', 'gender', 'respiratory_condition', 'fever_muscle_pain', 'status', 
                    'quality', 'cough_type', 'dyspnea', 'wheezing', 'stridor', 'choking', 
                    'congestion', 'nothing', 'diagnosis', 'severity' ]
    physician_01 = df.iloc[:, 0:21]
    physician_01 = physician_01[physician_01.quality_1.notna()].reset_index(drop=True)
    physician_01.columns = column_names

    physician_02 = pd.concat([df.iloc[:, 0:11], df.iloc[:, 21:31]], axis=1)
    physician_02 = physician_02[physician_02.quality_2.notna()].reset_index(drop=True)
    physician_02.columns = column_names

    physician_03 = pd.concat([df.iloc[:, 0:11], df.iloc[:, 31:41]], axis=1)
    physician_03 = physician_03[physician_03.quality_3.notna()].reset_index(drop=True)
    physician_03.columns = column_names

    physician_04 = pd.concat([df.iloc[:, 0:11], df.iloc[:, 41:51]], axis=1)
    physician_04 = physician_04[physician_04.quality_4.notna()].reset_index(drop=True)
    physician_04.columns = column_names
    return physician_01, physician_02, physician_03, physician_04
    
def process_csv(df):
    #split by physicians
    physician_01, physician_02, physician_03, physician_04 = split_by_physicians(df)
    # combine into one dataframe
    df = pd.concat([physician_01,physician_02,physician_03,physician_04]).reset_index(drop=True)  
    # drop null status
    df = df[df.status.notna()]
    # drop cough_detected < 0.8
    df = df[df.cough_detected >= 0.8 ]
    # select good and ok quality
    df = df[df.quality == 'good']
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True) 
    df = df[['uuid', 'status','cough_type', 'dyspnea', 'wheezing', 'stridor', 'choking', 'congestion', 'severity']]
    return df

processed_df = process_csv(data_raw)

def segment_cough(x,fs, cough_padding=0.2,min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
    #Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power                
    cough_mask = np.array([False]*len(x))
    
    #Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h =  th_h_multiplier*rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    return coughSegments, cough_mask

def extract_features(audio_data, sample_rate):

    features = []
    stft = np.abs(librosa.stft(audio_data))

    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T,axis=0)
    features.extend(mfcc) # 40 = 40

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    features.extend(chroma) # 12 = 52

    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
    features.extend(mel) # 128 = 180

    fmin_val = 0.5 * sample_rate * 2**(-6)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate, fmin=fmin_val).T,axis=0)
    features.extend(contrast) # 7 = 187

    return np.array(features)


def load_features(df):
    all_data, all_fname = [], []
    for idx in tqdm(range(len(df))):
        fname = df.uuid.iloc[idx]
        path = ROOT+fname+'.wav' 

        # load sound sample
        audio, sample_rate = librosa.load(path, mono=True)

        # Segment each audio into individual coughs using a hysteresis comparator on the signal power
        cough_segments, cough_mask = segment_cough(audio, sample_rate, min_cough_len=0.1, cough_padding=0.1, th_l_multiplier = 0.1, th_h_multiplier = 2)

        # For each segment, resize to the same length(11025)
        if len(cough_segments) > 0 :
            i = 0
            for audio in cough_segments:
                i+=1
                if len(audio) > 8000:
                    if len(audio) < audio_length:
                        audio_pad = librosa.util.pad_center(audio, size=audio_length)
                    else:
                        audio_pad = audio[:audio_length]  

                feature = extract_features(audio_pad, sample_rate)
                #print(len(feature))
                all_data.append(feature)
                all_fname.append(fname)
    
    return np.array(all_fname), np.array(all_data)


uuid, X = load_features(processed_df)

# Store each features in different dataframe so you can choose to train all features or individual
X_mfcc = X[:, 0:40]
X_chroma = X[:, 40:52]
X_mel = X[:, 52:180]
X_contrast = X[:, 180:]

# mfcc only
uuid_df = pd.DataFrame({'uuid':uuid})
mfcc_df = pd.DataFrame(X_mfcc)
mfcc_df.columns=["mfcc"+str(i) for i in range(1, X_mfcc.shape[1]+1)]
all_mfcc_df = pd.concat([uuid_df, mfcc_df], axis=1)

# mel spectogram only
mel_df = pd.DataFrame(X_mel)
mel_df.columns=["mel"+str(i) for i in range(1, X_mel.shape[1]+1)]
all_mel_df = pd.concat([uuid_df, mel_df], axis=1)

# chroma only
chroma_df = pd.DataFrame(X_chroma)
chroma_df.columns=["chr"+str(i) for i in range(1, X_chroma.shape[1]+1)]
all_chroma_df = pd.concat([uuid_df, chroma_df], axis=1)

# contrast only
contrast_df = pd.DataFrame(X_contrast)
contrast_df.columns=["con"+str(i) for i in range(1, X_contrast.shape[1]+1)]
all_contrast_df = pd.concat([uuid_df, contrast_df], axis=1)

# all features
all_df = pd.concat([uuid_df, mfcc_df, mel_df, chroma_df, contrast_df ], axis=1)

# Instead of predicting the status (healthy/covid/symptomatic), we train a model to to identify the cough type (dry/wet)

# Select what you would like to predict ('status', 'cough_type', 'dyspnea', 'wheezing', 'stridor', 'choking', 'congestion', 'severity')
label_df = processed_df[['uuid', 'status']].reset_index(drop=True)

# merge features and label to train
dataset = pd.merge(all_df, label_df, on='uuid')

# # remove null columns
# dataset = dataset[dataset.cough_type != 'unknown']

dataset = dataset.groupby('status').sample(n=2185)

exp_clf102 = setup(
    data = dataset, 
    target = 'status',
    normalize = True, 
    transformation = True, 
    ignore_features=['uuid']
)
best_model = compare_models()
xgboost = create_model('xgboost')
plot_model(xgboost)
plot_model(xgboost, plot = 'confusion_matrix')

save_model(xgboost, 'xgboost_model')  # saves as cough_model.pkl