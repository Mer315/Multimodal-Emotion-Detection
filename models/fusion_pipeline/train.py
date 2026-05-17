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

import os
model_dir = "/content/drive/MyDrive/tess_models_fusion"
os.makedirs(model_dir, exist_ok=True)

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
        torch.save(
            model.state_dict(),
            f"{model_dir}/hubert_bilstm_bert_latefusion_tess.pth"
        )
        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}  ← Best saved")
    else:
        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nBest Val Acc: {best_val_acc:.4f}")


#==============================================================================================================

#speaker level split
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

# Speaker-level split
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

#FUSION
# ── Imports ──
import os
import numpy as np

mfcc_dir   = "/content/drive/MyDrive/tess_speaker_mfcc_aug"
hubert_dir = "/content/drive/MyDrive/tess_speaker_hubert"
bert_dir   = "/content/drive/MyDrive/tess_speaker_bert"
fusion_dir = "/content/drive/MyDrive/tess_speaker_fusion"
os.makedirs(fusion_dir, exist_ok=True)

split_sizes = {"train": 1400, "val": 700, "test": 700}


# LOAD + FUSE: pool HuBERT → mean vector,
# then concat [mfcc_flat | hubert_pooled | bert_cls]
print("\n=== Fusion ===")

for split_name, n in split_sizes.items():

    # -- Load MFCC (128 × 120) → flatten to 15360
    X_mfcc = np.load(f"{mfcc_dir}/X_{split_name}.npy")           # (n, 128, 120)
    X_mfcc_flat = X_mfcc.reshape(n, -1)                           # (n, 15360)

    # -- Load HuBERT (200 × 768) → mean-pool to 768
    X_hub = np.memmap(f"{hubert_dir}/X_{split_name}.npy",
                      dtype='float32', mode='r', shape=(n, 200, 768))
    X_hub_pooled = X_hub.mean(axis=1)                             # (n, 768)

    # -- Load BERT CLS (768)
    X_bert = np.memmap(f"{bert_dir}/X_{split_name}.npy",
                       dtype='float32', mode='r', shape=(n, 768))  # (n, 768)

    # -- Labels (use HuBERT's; all three are identical)
    y = np.memmap(f"{hubert_dir}/y_{split_name}.npy",
                  dtype='int32', mode='r', shape=(n,))

    # -- Concatenate → (n, 15360 + 768 + 768) = (n, 16896)
    X_fused = np.concatenate([X_mfcc_flat, X_hub_pooled, X_bert], axis=1)

    # -- Save
    np.save(f"{fusion_dir}/X_{split_name}.npy", X_fused)
    np.save(f"{fusion_dir}/y_{split_name}.npy", np.array(y))

    print(f"Fused {split_name}: {X_fused.shape} | labels: {np.array(y).shape}")
    del X_mfcc, X_mfcc_flat, X_hub, X_hub_pooled, X_bert, X_fused, y

#Training.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

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

def run_test(model, test_loader, save_path, label):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, y_b = batch[:-1], batch[-1]
            inputs = [x.to(device) for x in inputs]
            preds.extend(model(*inputs).argmax(1).cpu().numpy())
            labels.extend(y_b.numpy())
    print(f"\n{'='*50}\nTEST RESULTS — {label}\n{'='*50}")
    print(classification_report(labels, preds, target_names=emotion_names))

def make_loader(tensors, batch_size=32, shuffle=False):
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']
hubert_dir    = "/content/drive/MyDrive/tess_speaker_hubert"
bert_dir      = "/content/drive/MyDrive/tess_speaker_bert"
model_dir     = "/content/drive/MyDrive/tess_speaker_models"


# MODEL 4 — FUSION (HuBERT BiLSTM + BERT)
print("\n" + "="*50)
print("MODEL 4: FUSION (HuBERT BiLSTM + BERT)")
print("="*50)

Xs_tr = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_train.npy", dtype='float32', mode='r', shape=(1400,200,768))), dtype=torch.float32)
Xt_tr = torch.tensor(np.array(np.memmap(f"{bert_dir}/X_train.npy",   dtype='float32', mode='r', shape=(1400,768))),     dtype=torch.float32)
y_tr  = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_train.npy", dtype='int32',   mode='r', shape=(1400,))),        dtype=torch.long)

train_loader = make_loader([Xs_tr, Xt_tr, y_tr], shuffle=True)


class FusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.speech_encoder = nn.LSTM(768, 128, num_layers=2,
                                       batch_first=True, bidirectional=True, dropout=0.3)
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion_fc  = nn.Linear(256 + 256, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, speech, text):
        s, _ = self.speech_encoder(speech)
        s    = s.mean(dim=1)
        t    = self.text_projection(text)
        return self.classifier(self.fusion_fc(torch.cat([s, t], dim=-1)))

model_fusion = FusionModel().to(device)
run_training(model_fusion, train_loader, val_loader,
             f"{model_dir}/best_speaker_fusion.pt", epochs=30)
