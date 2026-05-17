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
#BERT feature extraction and model training will be done

#preprocessing and feature extraction 

import os
import numpy as np
import torch
import gc
from tqdm import tqdm
from google.colab import drive
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# ---------- DRIVE ----------
drive.mount('/content/drive')

save_dir = "/content/drive/MyDrive/tess_bert_features"
os.makedirs(save_dir, exist_ok=True)

# ──────────────────────────────────────────
# STEP 1 — Clean df + word-level split
# ──────────────────────────────────────────
df = df.drop_duplicates(subset=['word', 'speaker', 'emotion']).reset_index(drop=True)
df.loc[df['speaker'] == 'OA', 'speaker'] = 'OAF'
df['transcript'] = df['word'].apply(lambda w: f"say the word {w}")  # ensure column exists

print(f"Clean df size: {len(df)}")
print(f"Sample transcript: {df['transcript'].iloc[0]}")

all_words = df['word'].unique()
train_words, temp_words = train_test_split(all_words, test_size=0.2, random_state=42)
val_words,   test_words = train_test_split(temp_words, test_size=0.5, random_state=42)

train_df = df[df['word'].isin(train_words)].reset_index(drop=True)
val_df   = df[df['word'].isin(val_words)].reset_index(drop=True)
test_df  = df[df['word'].isin(test_words)].reset_index(drop=True)

print("Train ∩ Val:",  len(set(train_words) & set(val_words)))
print("Train ∩ Test:", len(set(train_words) & set(test_words)))
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ──────────────────────────────────────────
# STEP 2 — Load BERT
# ──────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert      = BertModel.from_pretrained('bert-base-uncased')
bert.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)
print("Using device:", device)

# ──────────────────────────────────────────
# STEP 3 — Feature extraction functions
# ──────────────────────────────────────────
def preprocess_text(transcript, max_len=10):
    encoded = tokenizer(
        transcript.lower(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    return (
        encoded['input_ids'].to(device),
        encoded['attention_mask'].to(device)
    )

def extract_bert(input_ids, attention_mask):
    with torch.no_grad():
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
    cls_vector = outputs.last_hidden_state[:, 0, :]
    return cls_vector.squeeze(0).cpu().numpy().astype(np.float32)  # (768,)

# ──────────────────────────────────────────
# STEP 4 — Process splits
# ──────────────────────────────────────────
def process_split(df, split_name):
    X_path = f"{save_dir}/X_{split_name}_bert.npy"
    y_path = f"{save_dir}/y_{split_name}_bert.npy"

    X = np.memmap(X_path, dtype='float32', mode='w+', shape=(len(df), 768))
    y = np.memmap(y_path, dtype='int32',   mode='w+', shape=(len(df),))

    skipped = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=split_name)
    ):
        try:
            input_ids, attention_mask = preprocess_text(row['transcript'])
            features = extract_bert(input_ids, attention_mask)
            X[i] = features
            y[i] = row['label']
        except Exception as e:
            print(f"Skipping index {i}: {e}")
            skipped.append(i)

        if i % 50 == 0:
            X.flush()
            y.flush()

    X.flush()
    y.flush()
    gc.collect()

    # Verify
    X_check = np.memmap(X_path, dtype='float32', mode='r', shape=(len(df), 768))
    print(f"{split_name} — shape: {X_check.shape}, skipped: {len(skipped)}")
    del X, y, X_check
    gc.collect()

# ──────────────────────────────────────────
# STEP 5 — Run
# ──────────────────────────────────────────
process_split(train_df, "train")
process_split(val_df,   "val")
process_split(test_df,  "test")

print("\nSaved files:", os.listdir(save_dir))


#Text modelling

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# ── Load features ──
X_train = np.memmap("/content/drive/MyDrive/tess_bert_features/X_train_bert.npy",
                    dtype='float32', mode='r', shape=(2240, 768))
y_train = np.memmap("/content/drive/MyDrive/tess_bert_features/y_train_bert.npy",
                    dtype='int32',   mode='r', shape=(2240,))

X_val   = np.memmap("/content/drive/MyDrive/tess_bert_features/X_val_bert.npy",
                    dtype='float32', mode='r', shape=(280, 768))
y_val   = np.memmap("/content/drive/MyDrive/tess_bert_features/y_val_bert.npy",
                    dtype='int32',   mode='r', shape=(280,))

X_test  = np.memmap("/content/drive/MyDrive/tess_bert_features/X_test_bert.npy",
                    dtype='float32', mode='r', shape=(280, 768))
y_test  = np.memmap("/content/drive/MyDrive/tess_bert_features/y_test_bert.npy",
                    dtype='int32',   mode='r', shape=(280,))

# ── DataLoaders ──
def make_loader(X, y, batch_size=32, shuffle=False):
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    y_t = torch.tensor(np.array(y), dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t),
                      batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ──
class TextClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

model     = TextClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

import os
model_dir = "/content/drive/MyDrive/tess_models_text"
os.makedirs(model_dir, exist_ok=True)

# ── Training loop ──
best_val_acc = 0.0
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']

for epoch in range(1, 21):
    # Train
    model.train()
    correct = total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        preds = model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)
    train_acc = correct / total

    # Val
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)
    val_acc = correct / total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            f"{model_dir}/bert_fc_text_only_tess.pth"
        )
        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}  ← Best saved")
    else:
        print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nBest Val Acc: {best_val_acc:.4f}")


#===============================================================================================================

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

# ── Imports (if running this cell independently) ──
import os, gc
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
# drive already mounted; if not: from google.colab import drive; drive.mount('/content/drive')

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_dir  = "/content/drive/MyDrive/tess_speaker_bert"
os.makedirs(bert_dir, exist_ok=True)

splits = [("train", train_df), ("val", val_df), ("test", test_df)]


# BERT EXTRACTION
print("\n=== BERT Extraction ===")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert      = BertModel.from_pretrained('bert-base-uncased')
bert.eval().to(device)

def preprocess_text(transcript, max_len=10):
    encoded = tokenizer(
        transcript.lower(), padding='max_length',
        truncation=True, max_length=max_len, return_tensors='pt'
    )
    return encoded['input_ids'].to(device), encoded['attention_mask'].to(device)

def extract_bert_cls(input_ids, attention_mask):
    with torch.no_grad():
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)

for split_name, split_df in splits:
    n = len(split_df)
    X = np.memmap(f"{bert_dir}/X_{split_name}.npy", dtype='float32', mode='w+', shape=(n, 768))
    y = np.memmap(f"{bert_dir}/y_{split_name}.npy", dtype='int32',   mode='w+', shape=(n,))
    skipped = 0
    for i, (_, row) in enumerate(tqdm(split_df.iterrows(), total=n, desc=split_name)):
        try:
            ids, mask = preprocess_text(row['transcript'])
            X[i] = extract_bert_cls(ids, mask)
            y[i] = row['label']
        except Exception as e:
            print(f"Skip {i}: {e}"); skipped += 1
        if i % 50 == 0:
            X.flush(); y.flush()
    X.flush(); y.flush()
    gc.collect()
    X_check = np.memmap(f"{bert_dir}/X_{split_name}.npy", dtype='float32', mode='r', shape=(n, 768))
    print(f"BERT {split_name}: {X_check.shape}, skipped: {skipped}")
    del X, y, X_check; gc.collect()

del bert; torch.cuda.empty_cache(); gc.collect()
print("\n=== Text features saved to Drive ===")

#Training
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
# run_training / run_test / make_loader must be defined (run Cell 1 first, or redefine here)
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
bert_dir      = "/content/drive/MyDrive/tess_speaker_bert"
model_dir     = "/content/drive/MyDrive/tess_speaker_models"

# MODEL 3 — TEXT BERT
print("\n" + "="*50)
print("MODEL 3: TEXT BERT")
print("="*50)

X_tr = torch.tensor(np.array(np.memmap(f"{bert_dir}/X_train.npy", dtype='float32', mode='r', shape=(1400,768))), dtype=torch.float32)
y_tr = torch.tensor(np.array(np.memmap(f"{bert_dir}/y_train.npy", dtype='int32',   mode='r', shape=(1400,))),   dtype=torch.long)

train_loader = make_loader([X_tr, y_tr], shuffle=True)

class TextClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

model_text = TextClassifier().to(device)
run_training(model_text, train_loader, val_loader,
             f"{model_dir}/best_speaker_text.pt", epochs=30)
