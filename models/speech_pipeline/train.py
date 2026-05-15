import kagglehub

# Download latest version
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
print("Path to dataset files:", path)

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
