
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report


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
model     = LateFusionModel().to(device)
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']

#data loader
def make_fusion_loader(X_speech, X_text, y, batch_size=32, shuffle=False):
    Xs = torch.tensor(np.array(X_speech), dtype=torch.float32)
    Xt = torch.tensor(np.array(X_text),   dtype=torch.float32)
    Yy = torch.tensor(np.array(y),         dtype=torch.long)
    return DataLoader(TensorDataset(Xs, Xt, Yy),
                      batch_size=batch_size, shuffle=shuffle)

test_loader  = make_fusion_loader(X_test_speech,  X_test_text,  y_test)
# ── Test evaluation ──
model_dir = "/content/drive/MyDrive/tess_models_fusion"
model.load_state_dict(
    torch.load(
        f"{model_dir}/hubert_bilstm_bert_latefusion_tess.pth"
    )
)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_s, X_t, y_b in test_loader:
        preds = model(X_s.to(device), X_t.to(device)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

print("\nFusion Test Results:")
print(classification_report(all_labels, all_preds, target_names=emotion_names))

#=============================================================================================

#speaker level split evaluation

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

Xs_vl = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_val.npy",   dtype='float32', mode='r', shape=(700,200,768))),  dtype=torch.float32)
Xt_vl = torch.tensor(np.array(np.memmap(f"{bert_dir}/X_val.npy",     dtype='float32', mode='r', shape=(700,768))),      dtype=torch.float32)
y_vl  = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_val.npy",   dtype='int32',   mode='r', shape=(700,))),         dtype=torch.long)

Xs_te = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_test.npy",  dtype='float32', mode='r', shape=(700,200,768))),  dtype=torch.float32)
Xt_te = torch.tensor(np.array(np.memmap(f"{bert_dir}/X_test.npy",    dtype='float32', mode='r', shape=(700,768))),      dtype=torch.float32)
y_te  = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_test.npy",  dtype='int32',   mode='r', shape=(700,))),         dtype=torch.long)

val_loader   = make_loader([Xs_vl, Xt_vl, y_vl])
test_loader  = make_loader([Xs_te, Xt_te, y_te])

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

run_test(model_fusion, test_loader,
         f"{model_dir}/best_speaker_fusion.pt", "Fusion HuBERT+BERT (Speaker-level)")
