import kagglehub

# Download latest version
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
print("Path to dataset files:", path)

#========================================================================================
#word-level split

import os
import pandas as pd

data = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split('_')
            speaker  = parts[0]           # OAF or YAF
            word     = parts[1]           # target word
            emotion  = parts[2].replace(".wav", "").lower()
            transcript = f"say the word {word}"
            data.append({
                "path": os.path.join(root, file),
                "speaker": speaker,
                "word": word,
                "emotion": emotion,
                "transcript": transcript
            })

df = pd.DataFrame(data)

# Encode labels
emotion_map = {
    'angry':0, 'disgust':1, 'fear':2,
    'happy':3, 'neutral':4, 'ps':5, 'sad':6
}
df['label'] = df['emotion'].map(emotion_map)

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.2,
                    stratify=df['label'], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5,
                    stratify=temp_df['label'], random_state=42)


import librosa
import numpy as np

def load_audio(path, sr=16000, max_len=3):
    audio, _ = librosa.load(path, sr=sr)

    # remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # fixed length
    max_samples = sr * max_len

    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]

    # normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    return audio

#CNN1D encoder
class CNN1DEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            # Block 2
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pool → fixed size
        )
        self.out_dim = 256

    def forward(self, x):
        # x: (batch, T, features) → need (batch, features, T) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.encoder(x)       # (batch, 256, 1)
        return out.squeeze(-1)      # (batch, 256)
    
#BiLSTM encoder

import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.out_dim = hidden_dim * 2  # 256

    def forward(self, x):
        out, _ = self.bilstm(x)     # (batch, T, 256)
        # Mean pool across time
        return out.mean(dim=1)      # (batch, 256)
    
#multi-head self-attention encoder
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm    = nn.LayerNorm(input_dim)
        self.out_dim = input_dim

    def forward(self, x):
        # Self-attention: each frame attends to all others
        attn_out, _ = self.attention(x, x, x)  # (batch, T, dim)
        attn_out = self.norm(attn_out + x)      # residual connection
        return attn_out.mean(dim=1)             # (batch, dim)


#FOR MFCCs
#-----------------------------------------------------------------------------------------------------

import os
import numpy as np
import librosa
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split


# STEP 1 — Clean the dataframe

df = df.drop_duplicates(subset=['word', 'speaker', 'emotion']).reset_index(drop=True)
print(f"Clean df size: {len(df)}")  # expect 2800

# Fix OA label
df.loc[df['speaker'] == 'OA', 'speaker'] = 'OAF'
print("Speakers:", df['speaker'].unique())  # expect ['YAF', 'OAF']


# STEP 2 — Word-level split

all_words = df['word'].unique()
print(f"Unique words: {len(all_words)}")  # expect 200

train_words, temp_words = train_test_split(all_words, test_size=0.2, random_state=42)
val_words, test_words   = train_test_split(temp_words, test_size=0.5, random_state=42)

train_df = df[df['word'].isin(train_words)].reset_index(drop=True)
val_df   = df[df['word'].isin(val_words)].reset_index(drop=True)
test_df  = df[df['word'].isin(test_words)].reset_index(drop=True)

# Sanity checks — all must be 0
print("Train ∩ Val:",  len(set(train_words) & set(val_words)))
print("Train ∩ Test:", len(set(train_words) & set(test_words)))
print("Val ∩ Test:",   len(set(val_words)   & set(test_words)))

print(f"\nTrain: {len(train_df)}")
print(f"Val:   {len(val_df)}") 
print(f"Test:  {len(test_df)}")    

# Both speakers must appear in every split
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"{name} speakers: {sorted(split['speaker'].unique())}")


# STEP 3 — Feature extraction

def extract_mfcc(audio, sr=16000, n_mfcc=40, max_len=128):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2]).T  # (time_frames, 120)

    if features.shape[0] < max_len:
        pad = np.zeros((max_len - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:max_len]

    # Per-sample normalisation
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0,  keepdims=True) + 1e-9
    features = (features - mean) / std

    return features  # (128, 120)


save_dir = "/content/drive/MyDrive/tess_features_mfcc"
os.makedirs(save_dir, exist_ok=True)


def extract_split(df, split_name):
    X, y    = [], []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        try:
            audio    = load_audio(row['path'])
            features = extract_mfcc(audio)
            X.append(features)
            y.append(row['label'])
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
            skipped += 1

    X = np.array(X)
    y = np.array(y)

    print(f"{split_name} — X: {X.shape}, y: {y.shape}, skipped: {skipped}")

    np.save(f"{save_dir}/X_{split_name}.npy", X)
    np.save(f"{save_dir}/y_{split_name}.npy", y)

    del X
    gc.collect()


extract_split(train_df, "train")
extract_split(val_df,   "val")
extract_split(test_df,  "test")

print("\nSaved files:", os.listdir(save_dir))

# MODELLING-MFCCs

import torch
from torch.utils.data import TensorDataset, DataLoader

X_train = np.load(
    "/content/drive/MyDrive/tess_features_mfcc/X_train.npy"
)

y_train = np.load(
    "/content/drive/MyDrive/tess_features_mfcc/y_train.npy"
)

X_val = np.load(
    "/content/drive/MyDrive/tess_features_mfcc/X_val.npy"
)

y_val = np.load(
    "/content/drive/MyDrive/tess_features_mfcc/y_val.npy"
)

X_test = np.load(
    "/content/drive/MyDrive/tess_features_mfcc/X_test.npy"
)

y_test = np.load(
    "/content/drive/MyDrive/tess_features_mfcc/y_test.npy"
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=32
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32
)

import torch.nn as nn
#classifier head for mfcc
class EmotionClassifier(nn.Module):

    def __init__(self, encoder, num_classes=7):

        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.encoder(x)

        return self.classifier(x)
    
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=1e-3,
    save_path="model.pth"
):

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_acc = 0

    for epoch in range(epochs):

        # TRAIN
        model.train()

        train_correct = 0
        train_total = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            preds = outputs.argmax(dim=1)

            train_correct += (
                preds == y
            ).sum().item()

            train_total += y.size(0)

        train_acc = train_correct / train_total


        # VALIDATION
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device)

                outputs = model(X)

                preds = outputs.argmax(dim=1)

                val_correct += (
                    preds == y
                ).sum().item()

                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}"
            f" | Train Acc: {train_acc:.4f}"
            f" | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                save_path
            )

            print("Best model saved!")

    print(
        "\nBest Validation Accuracy:",
        best_val_acc
    )

import os

model_dir = "/content/drive/MyDrive/tess_models_mfcc"
os.makedirs(model_dir, exist_ok=True)

#cnn

encoder = CNN1DEncoder(input_dim=120)

cnn_model = EmotionClassifier(encoder)

train_model(
    cnn_model,
    train_loader,
    val_loader,
    epochs=20,
    save_path=f"{model_dir}/cnn_mfcc.pth"
)

#bilstm
encoder = BiLSTMEncoder(input_dim=120)

lstm_model = EmotionClassifier(encoder)

train_model(
    lstm_model,
    train_loader,
    val_loader,
    epochs=20,
    save_path=f"{model_dir}/bilstm_mfcc.pth"
)

#self-attention 
encoder = AttentionPooling(input_dim=120)

attn_model = EmotionClassifier(encoder)

train_model(
    attn_model,
    train_loader,
    val_loader,
    epochs=20,
    save_path=f"{model_dir}/attention_mfcc.pth"
)

#----------------------------------------------------------------------------------------------------------------

#FOR HuBERT (ACTUAL USE CASE)
import os
import numpy as np
import torch
import librosa                       
import gc
from tqdm import tqdm
from transformers import HubertModel, AutoFeatureExtractor
from sklearn.model_selection import train_test_split
from google.colab import drive 
#dataset is large so we will use google drive to store features and models and train on colab gpu

drive.mount('/content/drive')

save_dir = "/content/drive/MyDrive/tess_hubert_features"
os.makedirs(save_dir, exist_ok=True)

# ── Clean df + word-level split ──
df = df.drop_duplicates(subset=['word', 'speaker', 'emotion']).reset_index(drop=True)
df.loc[df['speaker'] == 'OA', 'speaker'] = 'OAF'
print(f"Clean df size: {len(df)}")

all_words = df['word'].unique()
train_words, temp_words = train_test_split(all_words, test_size=0.2, random_state=42)
val_words,   test_words = train_test_split(temp_words, test_size=0.5, random_state=42)

train_df = df[df['word'].isin(train_words)].reset_index(drop=True)
val_df   = df[df['word'].isin(val_words)].reset_index(drop=True)
test_df  = df[df['word'].isin(test_words)].reset_index(drop=True)

print("Train ∩ Val:",  len(set(train_words) & set(val_words)))
print("Train ∩ Test:", len(set(train_words) & set(test_words)))
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Load HuBERT ──
processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert    = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hubert.to(device)
print("Using device:", device)

# ── load_audio ──
def load_audio(path, sr=16000, max_len=3):
    audio, _ = librosa.load(path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    max_samples = sr * max_len
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    return audio / (np.max(np.abs(audio)) + 1e-9)

# ── HuBERT extraction ──
def extract_hubert(audio, sr=16000, max_len=200):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hubert(**inputs)
    features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    if features.shape[0] < max_len:
        pad = np.zeros((max_len - features.shape[0], features.shape[1]), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[:max_len]
    return features.astype(np.float32)

# ── Split extraction ──
def process_split(df, split_name):
    X_path = f"{save_dir}/X_{split_name}_hubert.npy"
    y_path = f"{save_dir}/y_{split_name}_hubert.npy"

    X = np.memmap(X_path, dtype='float32', mode='w+', shape=(len(df), 200, 768))
    y = np.memmap(y_path, dtype='int32',   mode='w+', shape=(len(df),))

    skipped = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=split_name)):
        try:
            audio    = load_audio(row['path'])
            features = extract_hubert(audio)
            X[i] = features
            y[i] = row['label']
        except Exception as e:
            print(f"Skipping index {i}: {e}")
            skipped.append(i)
        if i % 50 == 0:
            X.flush(); y.flush()

    X.flush(); y.flush()
    torch.cuda.empty_cache()
    gc.collect()

    # Verify — use memmap not np.load for raw binary files
    X_check = np.memmap(X_path, dtype='float32', mode='r', shape=(len(df), 200, 768))
    print(f"{split_name} — shape: {X_check.shape}, skipped: {len(skipped)}")
    del X, y, X_check
    gc.collect()

# ── Run ──
process_split(train_df, "train")
process_split(val_df,   "val")
process_split(test_df,  "test")

print("\nSaved files:", os.listdir(save_dir))

#training code for HuBERT features will be similar to the MFCC section, just with updated paths and input dimensions
import numpy as np
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

# ---------- TRAIN ----------
X_train = np.memmap(
    "/content/drive/MyDrive/tess_hubert_features/X_train_hubert.npy",
    dtype='float32',
    mode='r',
    shape=(2240, 200, 768)
)

y_train = np.memmap(
    "/content/drive/MyDrive/tess_hubert_features/y_train_hubert.npy",
    dtype='int32',
    mode='r',
    shape=(2240,)
)


# ---------- VAL ----------
X_val = np.memmap(
    "/content/drive/MyDrive/tess_hubert_features/X_val_hubert.npy",
    dtype='float32',
    mode='r',
    shape=(280, 200, 768)
)

y_val = np.memmap(
    "/content/drive/MyDrive/tess_hubert_features/y_val_hubert.npy",
    dtype='int32',
    mode='r',
    shape=(280,)
)


# ---------- TEST ----------
X_test = np.memmap(
    "/content/drive/MyDrive/tess_hubert_features/X_test_hubert.npy",
    dtype='float32',
    mode='r',
    shape=(280, 200, 768)
)

y_test = np.memmap(
    "/content/drive/MyDrive/tess_hubert_features/y_test_hubert.npy",
    dtype='int32',
    mode='r',
    shape=(280,)
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=16
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=16
)

import torch.nn as nn
#classifier head for hubert features
class EmotionClassifier(nn.Module):

    def __init__(self, encoder, num_classes=7):

        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.encoder(x)

        return self.classifier(x)
    

#training func 
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=1e-3,
    save_path="model.pth"
):

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_acc = 0

    for epoch in range(epochs):

        # ---------- TRAIN ----------
        model.train()

        train_correct = 0
        train_total = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            preds = outputs.argmax(dim=1)

            train_correct += (
                preds == y
            ).sum().item()

            train_total += y.size(0)

        train_acc = train_correct / train_total


        # ---------- VALIDATION ----------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device)

                outputs = model(X)

                preds = outputs.argmax(dim=1)

                val_correct += (
                    preds == y
                ).sum().item()

                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}"
            f" | Train Acc: {train_acc:.4f}"
            f" | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                save_path
            )

            print("Best model saved!")

    print(
        "\nBest Validation Accuracy:",
        best_val_acc
    )

import os
model_dir = "/content/drive/MyDrive/tess_models_hubert"
os.makedirs(model_dir, exist_ok=True)

#CNN 1D for HuBERT features

import os

model_dir = "/content/drive/MyDrive/tess_models_hubert"
os.makedirs(model_dir, exist_ok=True)
encoder = CNN1DEncoder(input_dim=768)
cnn_hubert = EmotionClassifier(encoder)

train_model(
    cnn_hubert,
    train_loader,
    val_loader,
    epochs=20,
    save_path=f"{model_dir}/cnnhubert.pth"
)

#BiLSTM for HuBERT features

encoder = BiLSTMEncoder(input_dim=768)

bilstm_hubert = EmotionClassifier(encoder)

train_model(
    bilstm_hubert,
    train_loader,
    val_loader,
    epochs=20,
    save_path=f"{model_dir}/bilstm_hubert.pth"
)

#Self-attention for HuBERT features

encoder = AttentionPooling(input_dim=768)

attention_hubert = EmotionClassifier(encoder)

train_model(
    attention_hubert,
    train_loader,
    val_loader,
    epochs=20,
    save_path=f"{model_dir}/attention_hubert.pth"
)


#=======================================================================================

#testing out speaker level split for MFCC and HuBERT features, code structure will be similar to the word-level split but with updated paths and input dimensions. The main change is in the initial data splitting step where we will split based on the 2 speakers instead of words. The rest of the training and evaluation code will remain largely unchanged, just with updated dataset paths and possibly some adjustments to the model architecture if needed.

# ── Build DataFrame ── (for mfcc due to trying out augmentation on it) Hubert, BERT and fusion use the same initial DataFrame construction as before.
data = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".wav"):
            parts    = file.split('_')
            speaker  = parts[0].upper()   # normalise to OAF / YAF
            word     = parts[1]
            emotion  = parts[2].replace(".wav", "").lower()
            data.append({
                "path":       os.path.join(root, file),
                "speaker":    speaker,
                "word":       word,
                "emotion":    emotion,
                "transcript": f"say the word {word}"
            })

df = pd.DataFrame(data)
print(f"Total files: {len(df)}")
print(f"Speakers: {df['speaker'].unique()}")
print(f"Emotions: {sorted(df['emotion'].unique())}")

# ── Label encoding ──
emotion_map = {'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'ps':5,'sad':6}
df['label'] = df['emotion'].map(emotion_map)

# drop any rows where emotion wasn't in map
before = len(df)
df = df.dropna(subset=['label']).reset_index(drop=True)
df['label'] = df['label'].astype(int)
if len(df) < before:
    print(f"Dropped {before - len(df)} unrecognised emotion rows")

# ── Speaker-level split ──
oaf_df = df[df['speaker'] == 'OAF'].copy()
yaf_df = df[df['speaker'] == 'YAF'].copy()

train_df = oaf_df.reset_index(drop=True)

val_df, test_df = train_test_split(
    yaf_df,
    test_size=0.5,
    stratify=yaf_df['label'],
    random_state=42
)
val_df  = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ── Sanity checks ──
train_paths = set(train_df['path'])
val_paths   = set(val_df['path'])
test_paths  = set(test_df['path'])

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train speakers: {train_df['speaker'].unique()}")
print(f"Val speakers:   {val_df['speaker'].unique()}")
print(f"Test speakers:  {test_df['speaker'].unique()}")
print(f"Train ∩ Val:    {len(train_paths & val_paths)}")   # must be 0
print(f"Train ∩ Test:   {len(train_paths & test_paths)}")  # must be 0
print(f"Val   ∩ Test:   {len(val_paths & test_paths)}")    # must be 0

print(f"\nTrain emotion dist:\n{train_df['emotion'].value_counts()}")
print(f"\nTest emotion dist:\n{test_df['emotion'].value_counts()}")

# ── Audio loader ──
def load_audio(path, sr=16000, max_len=3):
    audio, _ = librosa.load(path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    max_samples = sr * max_len
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    return audio / (np.max(np.abs(audio)) + 1e-9)


# ── Imports ──
import os, gc
import numpy as np
import torch
import librosa
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive')

# ── Dirs ──
raw_mfcc_dir = "/content/drive/MyDrive/tess_speaker_mfcc"
aug_mfcc_dir = "/content/drive/MyDrive/tess_speaker_mfcc_aug"
os.makedirs(aug_mfcc_dir, exist_ok=True)


# AUGMENTATION FUNCTIONS

def load_audio(path, sr=16000, max_len=3):
    audio, _ = librosa.load(path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    max_samples = sr * max_len
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    return audio / (np.max(np.abs(audio)) + 1e-9)

def pitch_shift(audio, sr=16000):
    """Shift pitch by random semitones — simulates different vocal registers."""
    n_steps = np.random.uniform(-3, 3)          # ±3 semitones
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, sr=16000, max_len=3):
    """Speed up or slow down utterance — simulates speaking rate variation."""
    rate = np.random.uniform(0.85, 1.15)         # ±15% speed
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    max_samples = sr * max_len
    if len(stretched) < max_samples:
        stretched = np.pad(stretched, (0, max_samples - len(stretched)))
    else:
        stretched = stretched[:max_samples]
    return stretched

def add_noise(audio, snr_db=None):
    """Add Gaussian noise at random SNR — simulates mic/env variation."""
    if snr_db is None:
        snr_db = np.random.uniform(20, 40)       # 20–40 dB SNR (subtle noise)
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def vtlp(audio, sr=16000, n_mfcc=40, max_len=128):
    """
    Vocal Tract Length Perturbation — most effective cross-speaker augmentation.
    Warps the frequency axis to simulate different vocal tract lengths.
    Applied directly in the spectral domain.
    """
    alpha = np.random.uniform(0.9, 1.1)          # warp factor
    stft  = librosa.stft(audio)
    freqs = np.linspace(0, sr/2, stft.shape[0])
    new_freqs = np.clip(freqs * alpha, 0, sr/2)
    warped = np.zeros_like(stft)
    for i, nf in enumerate(new_freqs):
        src_idx = np.argmin(np.abs(freqs - nf))
        warped[i] = stft[src_idx]
    audio_warped = librosa.istft(warped, length=len(audio))
    return audio_warped / (np.max(np.abs(audio_warped)) + 1e-9)

def spec_augment(mfcc, time_mask_param=20, freq_mask_param=15, n_time_masks=2, n_freq_masks=2):
    """
    SpecAugment — mask random time and frequency bands in the MFCC.
    Applied AFTER feature extraction (augments in feature space).
    """
    aug = mfcc.copy()
    T, F = aug.shape                             # (128 time, 120 freq)

    for _ in range(n_time_masks):
        t  = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, max(1, T - t))
        aug[t0:t0+t, :] = 0

    for _ in range(n_freq_masks):
        f  = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, max(1, F - f))
        aug[:, f0:f0+f] = 0

    return aug

def extract_mfcc(audio, sr=16000, n_mfcc=40, max_len=128):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T   # (T, 120)
    if features.shape[0] < max_len:
        pad = np.zeros((max_len - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:max_len]
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0,  keepdims=True) + 1e-9
    return ((features - mean) / std).astype(np.float32)

# ────────────────────────────────────────
# AUGMENTATION STRATEGY
# Each training sample gets 4 augmented copies:
#   1. pitch shift
#   2. time stretch
#   3. noise + pitch shift (combined)
#   4. VTLP (most important for cross-speaker)
# SpecAugment applied on top of all copies in feature space.
# Val/test sets are NOT augmented — clean evaluation only.
# ────────────────────────────────────────

AUG_FUNCS = [
    ("pitch",    lambda a: pitch_shift(a)),
    ("stretch",  lambda a: time_stretch(a)),
    ("noise+pitch", lambda a: pitch_shift(add_noise(a))),
    ("vtlp",     lambda a: vtlp(a)),
]

splits = [("train", train_df), ("val", val_df), ("test", test_df)]

for split_name, split_df in splits:
    n = len(split_df)

    if split_name == "train":
        # Train: original + 4 augmented copies = 5× dataset
        n_total = n * (1 + len(AUG_FUNCS))
        X = np.zeros((n_total, 128, 120), dtype=np.float32)
        y = np.zeros(n_total, dtype=np.int32)
        skipped = 0

        for i, (_, row) in enumerate(tqdm(split_df.iterrows(), total=n, desc="train (aug)")):
            try:
                audio = load_audio(row['path'])

                # Original
                X[i] = extract_mfcc(audio)
                y[i] = row['label']

                # Augmented copies
                for aug_idx, (aug_name, aug_fn) in enumerate(AUG_FUNCS):
                    slot = n + (aug_idx * n) + i
                    try:
                        aug_audio = aug_fn(audio)
                        mfcc_aug  = extract_mfcc(aug_audio)
                        # Apply SpecAugment on top in feature space
                        mfcc_aug  = spec_augment(mfcc_aug)
                        X[slot]   = mfcc_aug
                        y[slot]   = row['label']
                    except Exception as e:
                        # Fall back to original if augmentation fails
                        X[slot] = X[i]
                        y[slot] = row['label']

            except Exception as e:
                print(f"Skip {i}: {e}")
                skipped += 1

        # Shuffle augmented training set
        idx = np.random.permutation(n_total)
        X, y = X[idx], y[idx]

        np.save(f"{aug_mfcc_dir}/X_train.npy", X)
        np.save(f"{aug_mfcc_dir}/y_train.npy", y)
        print(f"Train (aug): {X.shape}  |  skipped: {skipped}")
        print(f"  Original: {n} samples  |  Augmented total: {n_total} samples")

    else:
        # Val / Test: NO augmentation
        X = np.zeros((n, 128, 120), dtype=np.float32)
        y = np.zeros(n, dtype=np.int32)
        skipped = 0
        for i, (_, row) in enumerate(tqdm(split_df.iterrows(), total=n, desc=split_name)):
            try:
                audio = load_audio(row['path'])
                X[i]  = extract_mfcc(audio)
                y[i]  = row['label']
            except Exception as e:
                print(f"Skip {i}: {e}")
                skipped += 1
        np.save(f"{aug_mfcc_dir}/X_{split_name}.npy", X)
        np.save(f"{aug_mfcc_dir}/y_{split_name}.npy", y)
        print(f"{split_name}: {X.shape}  |  skipped: {skipped}")

    del X; gc.collect()

print("\n=== Augmented features saved ===")
print(f"Training set expanded {1 + len(AUG_FUNCS)}x — use aug_mfcc_dir for speaker-split training")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Clean df (same as always)
df = df.drop_duplicates(subset=['word', 'speaker', 'emotion']).reset_index(drop=True)
df.loc[df['speaker'] == 'OA', 'speaker'] = 'OAF'
df['transcript'] = df['word'].apply(lambda w: f"say the word {w}")
df['label'] = df['emotion'].map({
    'angry':0, 'disgust':1, 'fear':2,
    'happy':3, 'neutral':4, 'ps':5, 'sad':6
})

# Speaker-level split (IMPORTANT)
train_df = df[df['speaker'] == 'OAF'].reset_index(drop=True)  # 1400 samples
yaf_df   = df[df['speaker'] == 'YAF'].reset_index(drop=True)  # 1400 samples

# Split YAF 50/50 into val and test
val_df, test_df = train_test_split(yaf_df, test_size=0.5,
                                   stratify=yaf_df['label'],
                                   random_state=42)
val_df  = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Verify
print(f"Train (OAF): {len(train_df)}")   # expect 1400
print(f"Val   (YAF): {len(val_df)}")     # expect 700
print(f"Test  (YAF): {len(test_df)}")    # expect 700

print(f"\nTrain speakers: {train_df['speaker'].unique()}")
print(f"Val speakers:   {val_df['speaker'].unique()}")
print(f"Test speakers:  {test_df['speaker'].unique()}")

print(f"\nTrain emotion dist:\n{train_df['emotion'].value_counts()}")
print(f"Test emotion dist:\n{test_df['emotion'].value_counts()}")

#Extraction

# ── Imports & Setup ──
import os, gc
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import HubertModel, AutoFeatureExtractor
from google.colab import drive

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

mfcc_dir   = "/content/drive/MyDrive/tess_speaker_mfcc"
hubert_dir = "/content/drive/MyDrive/tess_speaker_hubert"
os.makedirs(mfcc_dir,   exist_ok=True)
os.makedirs(hubert_dir, exist_ok=True)

splits = [("train", train_df), ("val", val_df), ("test", test_df)]

# ── Shared Audio Loader ──
def load_audio(path, sr=16000, max_len=3):
    audio, _ = librosa.load(path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    max_samples = sr * max_len
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    return audio / (np.max(np.abs(audio)) + 1e-9)

# ────────────────────────────────────────
# MFCC EXTRACTION
# ────────────────────────────────────────
print("\n=== MFCC Extraction ===")

def extract_mfcc(audio, sr=16000, n_mfcc=40, max_len=128):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T
    if features.shape[0] < max_len:
        pad = np.zeros((max_len - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:max_len]
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0,  keepdims=True) + 1e-9
    return ((features - mean) / std).astype(np.float32)

for split_name, split_df in splits:
    X = np.zeros((len(split_df), 128, 120), dtype=np.float32)
    y = np.zeros(len(split_df), dtype=np.int32)
    skipped = 0
    for i, (_, row) in enumerate(tqdm(split_df.iterrows(), total=len(split_df), desc=split_name)):
        try:
            audio = load_audio(row['path'])
            X[i]  = extract_mfcc(audio)
            y[i]  = row['label']
        except Exception as e:
            print(f"Skip {i}: {e}"); skipped += 1
    np.save(f"{mfcc_dir}/X_{split_name}.npy", X)
    np.save(f"{mfcc_dir}/y_{split_name}.npy", y)
    print(f"MFCC {split_name}: {X.shape}, skipped: {skipped}")
    del X; gc.collect()

# ────────────────────────────────────────
# HUBERT EXTRACTION
# ────────────────────────────────────────
print("\n=== HuBERT Extraction ===")

processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert    = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval().to(device)

def extract_hubert(audio, sr=16000, max_len=200):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hubert(**inputs)
    features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    if features.shape[0] < max_len:
        pad = np.zeros((max_len - features.shape[0], features.shape[1]), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[:max_len]
    return features.astype(np.float32)

for split_name, split_df in splits:
    n = len(split_df)
    X = np.memmap(f"{hubert_dir}/X_{split_name}.npy", dtype='float32', mode='w+', shape=(n, 200, 768))
    y = np.memmap(f"{hubert_dir}/y_{split_name}.npy", dtype='int32',   mode='w+', shape=(n,))
    skipped = 0
    for i, (_, row) in enumerate(tqdm(split_df.iterrows(), total=n, desc=split_name)):
        try:
            audio = load_audio(row['path'])
            X[i]  = extract_hubert(audio)
            y[i]  = row['label']
        except Exception as e:
            print(f"Skip {i}: {e}"); skipped += 1
        if i % 50 == 0:
            X.flush(); y.flush()
    X.flush(); y.flush()
    torch.cuda.empty_cache(); gc.collect()
    X_check = np.memmap(f"{hubert_dir}/X_{split_name}.npy", dtype='float32', mode='r', shape=(n, 200, 768))
    print(f"HuBERT {split_name}: {X_check.shape}, skipped: {skipped}")
    del X, y, X_check; gc.collect()

del hubert; torch.cuda.empty_cache(); gc.collect()
print("\n=== Speech features saved to Drive ===")


#Training.
# ── Imports & Setup ──
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']

mfcc_dir   = "/content/drive/MyDrive/tess_speaker_mfcc_aug"
hubert_dir = "/content/drive/MyDrive/tess_speaker_hubert"
bert_dir   = "/content/drive/MyDrive/tess_speaker_bert"
model_dir  = "/content/drive/MyDrive/tess_speaker_models"

import os; 
os.makedirs(model_dir, exist_ok=True)

def run_training(model, train_loader, val_loader, save_path, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_val  = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for batch in train_loader:
            inputs, y_b = batch[:-1], batch[-1].to(device)
            inputs = [x.to(device) for x in inputs]
            optimizer.zero_grad()
            out  = model(*inputs)
            loss = criterion(out, y_b)
            loss.backward(); optimizer.step()
            correct += out.argmax(1).eq(y_b).sum().item()
            total   += y_b.size(0)
        train_acc = correct / total
        model.eval(); correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, y_b = batch[:-1], batch[-1].to(device)
                inputs = [x.to(device) for x in inputs]
                correct += model(*inputs).argmax(1).eq(y_b).sum().item()
                total   += y_b.size(0)
        val_acc = correct / total
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            marker = "  ← Best saved"
        print(f"Epoch {epoch:2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f}{marker}")
    print(f"Best Val: {best_val:.4f}\n")
    return best_val

def make_loader(tensors, batch_size=32, shuffle=False):
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

# ════════════════════════════════════════
# MODEL 1 — MFCC + CNN
# ════════════════════════════════════════

print("\n" + "="*50)
print("MODEL 1: MFCC + CNN")
print("="*50)

X_tr = torch.tensor(np.load(f"{mfcc_dir}/X_train.npy"), dtype=torch.float32)
y_tr = torch.tensor(np.load(f"{mfcc_dir}/y_train.npy"), dtype=torch.long)
X_vl = torch.tensor(np.load(f"{mfcc_dir}/X_val.npy"),   dtype=torch.float32)
y_vl = torch.tensor(np.load(f"{mfcc_dir}/y_val.npy"),   dtype=torch.long)
X_te = torch.tensor(np.load(f"{mfcc_dir}/X_test.npy"),  dtype=torch.float32)
y_te = torch.tensor(np.load(f"{mfcc_dir}/y_test.npy"),  dtype=torch.long)

train_loader = make_loader([X_tr, y_tr], shuffle=True)
val_loader   = make_loader([X_vl, y_vl])
test_loader  = make_loader([X_te, y_te])

class CNN1D(nn.Module):
    def __init__(self, input_dim=120, num_classes=7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.encoder(x.permute(0,2,1)).squeeze(-1))

model_cnn = CNN1D().to(device)
run_training(model_cnn, train_loader, val_loader,
             f"{model_dir}/best_speaker_cnn.pt", epochs=30)


# ════════════════════════════════════════
# MODEL 2 — HuBERT + BiLSTM
# ════════════════════════════════════════
print("\n" + "="*50)
print("MODEL 2: HuBERT + BiLSTM")
print("="*50)

X_tr = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_train.npy", dtype='float32', mode='r', shape=(1400,200,768))), dtype=torch.float32)
y_tr = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_train.npy", dtype='int32',   mode='r', shape=(1400,))),        dtype=torch.long)
X_vl = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_val.npy",   dtype='float32', mode='r', shape=(700,200,768))),  dtype=torch.float32)
y_vl = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_val.npy",   dtype='int32',   mode='r', shape=(700,))),         dtype=torch.long)
X_te = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_test.npy",  dtype='float32', mode='r', shape=(700,200,768))),  dtype=torch.float32)
y_te = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_test.npy",  dtype='int32',   mode='r', shape=(700,))),         dtype=torch.long)

train_loader = make_loader([X_tr, y_tr], shuffle=True)
val_loader   = make_loader([X_vl, y_vl])
test_loader  = make_loader([X_te, y_te])

class HuBERTBiLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=7):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.classifier(out.mean(dim=1))

model_hubert = HuBERTBiLSTM().to(device)
run_training(model_hubert, train_loader, val_loader,
             f"{model_dir}/best_speaker_hubert.pt", epochs=30)