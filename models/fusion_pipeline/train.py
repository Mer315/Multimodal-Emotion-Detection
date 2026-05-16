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
#--------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# ── Load precomputed features ──
# Speech: use HuBERT features (best performing)
X_train_speech = np.memmap("/content/drive/MyDrive/tess_hubert_features/X_train_hubert.npy",
                            dtype='float32', mode='r', shape=(2240, 200, 768))
X_val_speech   = np.memmap("/content/drive/MyDrive/tess_hubert_features/X_val_hubert.npy",
                            dtype='float32', mode='r', shape=(280, 200, 768))
X_test_speech  = np.memmap("/content/drive/MyDrive/tess_hubert_features/X_test_hubert.npy",
                            dtype='float32', mode='r', shape=(280, 200, 768))

# Text: BERT CLS vectors
X_train_text = np.memmap("/content/drive/MyDrive/tess_bert_features/X_train_bert.npy",
                          dtype='float32', mode='r', shape=(2240, 768))
X_val_text   = np.memmap("/content/drive/MyDrive/tess_bert_features/X_val_bert.npy",
                          dtype='float32', mode='r', shape=(280, 768))
X_test_text  = np.memmap("/content/drive/MyDrive/tess_bert_features/X_test_bert.npy",
                          dtype='float32', mode='r', shape=(280, 768))

# Labels
y_train = np.memmap("/content/drive/MyDrive/tess_hubert_features/y_train_hubert.npy",
                    dtype='int32', mode='r', shape=(2240,))
y_val   = np.memmap("/content/drive/MyDrive/tess_hubert_features/y_val_hubert.npy",
                    dtype='int32', mode='r', shape=(280,))
y_test  = np.memmap("/content/drive/MyDrive/tess_hubert_features/y_test_hubert.npy",
                    dtype='int32', mode='r', shape=(280,))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DataLoader ──
def make_fusion_loader(X_speech, X_text, y, batch_size=32, shuffle=False):
    Xs = torch.tensor(np.array(X_speech), dtype=torch.float32)
    Xt = torch.tensor(np.array(X_text),   dtype=torch.float32)
    Yy = torch.tensor(np.array(y),         dtype=torch.long)
    return DataLoader(TensorDataset(Xs, Xt, Yy),
                      batch_size=batch_size, shuffle=shuffle)

train_loader = make_fusion_loader(X_train_speech, X_train_text, y_train, shuffle=True)
val_loader   = make_fusion_loader(X_val_speech,   X_val_text,   y_val)
test_loader  = make_fusion_loader(X_test_speech,  X_test_text,  y_test)

# ── Speech temporal encoder (BiLSTM) ──
class SpeechBiLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.3)
        self.out_dim = hidden_dim * 2  # 256

    def forward(self, x):                    # x: (batch, 200, 768)
        out, _ = self.bilstm(x)              # (batch, 200, 256)
        return out.mean(dim=1)               # (batch, 256)

# ── Late Fusion Model ──
class LateFusionModel(nn.Module):
    def __init__(self, speech_dim=256, text_dim=768, num_classes=7):
        super().__init__()
        self.speech_encoder = SpeechBiLSTM()
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion_fc = nn.Linear(speech_dim + 256, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, speech_feat, text_feat):
        s = self.speech_encoder(speech_feat)     # (batch, 256)
        t = self.text_projection(text_feat)      # (batch, 256)
        fused = torch.cat([s, t], dim=-1)        # (batch, 512)
        fused = self.fusion_fc(fused)            # (batch, 256)
        return self.classifier(fused)            # (batch, 7)

# ── Training ──
model     = LateFusionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_val_acc  = 0.0
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']

for epoch in range(1, 21):
    model.train()
    correct = total = 0
    for X_s, X_t, y_b in train_loader:
        X_s, X_t, y_b = X_s.to(device), X_t.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_s, X_t)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        correct += out.argmax(dim=1).eq(y_b).sum().item()
        total   += y_b.size(0)
    train_acc = correct / total

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_s, X_t, y_b in val_loader:
            X_s, X_t, y_b = X_s.to(device), X_t.to(device), y_b.to(device)
            correct += model(X_s, X_t).argmax(dim=1).eq(y_b).sum().item()
            total   += y_b.size(0)
    val_acc = correct / total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_fusion_model.pt")
        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}  ← Best saved")
    else:
        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nBest Val Acc: {best_val_acc:.4f}")
