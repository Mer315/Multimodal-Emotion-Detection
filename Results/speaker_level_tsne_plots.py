# ── Imports ──
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir  = "/content/drive/MyDrive/tess_speaker_models"
hubert_dir = "/content/drive/MyDrive/tess_speaker_hubert"
bert_dir   = "/content/drive/MyDrive/tess_speaker_bert"
mfcc_dir   = "/content/drive/MyDrive/tess_speaker_mfcc_aug"
tsne_dir   = "/content/drive/MyDrive/tess_speaker_tsne"

import os; os.makedirs(tsne_dir, exist_ok=True)

emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']

# ── Redefine models (must match your training code exactly) ──
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
        return self.encoder(x.permute(0,2,1)).squeeze(-1)  # return encoder features

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
        return out.mean(dim=1)  # return BiLSTM pooled features

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
        return self.fusion_fc(torch.cat([s, t], dim=-1))  # return fused features

# ── Load test data ──
X_mfcc = torch.tensor(np.load(f"{mfcc_dir}/X_test.npy"),  dtype=torch.float32)
y_mfcc = torch.tensor(np.load(f"{mfcc_dir}/y_test.npy"),  dtype=torch.long)

X_hub  = torch.tensor(np.array(np.memmap(f"{hubert_dir}/X_test.npy",
         dtype='float32', mode='r', shape=(700,200,768))), dtype=torch.float32)
y_hub  = torch.tensor(np.array(np.memmap(f"{hubert_dir}/y_test.npy",
         dtype='int32', mode='r', shape=(700,))), dtype=torch.long)

X_bert = torch.tensor(np.array(np.memmap(f"{bert_dir}/X_test.npy",
         dtype='float32', mode='r', shape=(700,768))), dtype=torch.float32)
y_bert = torch.tensor(np.array(np.memmap(f"{bert_dir}/y_test.npy",
         dtype='int32', mode='r', shape=(700,))), dtype=torch.long)

# ── Extract representations ──
def get_reps(model, loader, device):
    model.eval()
    reps, labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, y_b = batch[:-1], batch[-1]
            inputs = [x.to(device) for x in inputs]
            reps.append(model(*inputs).cpu().numpy())
            labels.append(y_b.numpy())
    return np.vstack(reps), np.concatenate(labels)

# 1. MFCC CNN — temporal block representations (256-dim encoder output)
cnn = CNN1D().to(device)
cnn.load_state_dict(torch.load(f"{model_dir}/best_speaker_cnn.pt",map_location=device), strict=False)
loader_mfcc = DataLoader(TensorDataset(X_mfcc, y_mfcc), batch_size=64)
reps_cnn, labels_cnn = get_reps(cnn, loader_mfcc, device)
np.save(f"{tsne_dir}/reps_cnn.npy",    reps_cnn)
np.save(f"{tsne_dir}/labels_cnn.npy",  labels_cnn)
print(f"CNN reps: {reps_cnn.shape}")

# 2. HuBERT BiLSTM — contextual block representations (256-dim BiLSTM output)
bilstm = HuBERTBiLSTM().to(device)
bilstm.load_state_dict(torch.load(f"{model_dir}/best_speaker_hubert.pt",map_location=device), strict=False)
loader_hub = DataLoader(TensorDataset(X_hub, y_hub), batch_size=32)
reps_hub, labels_hub = get_reps(bilstm, loader_hub, device)
np.save(f"{tsne_dir}/reps_hub.npy",   reps_hub)
np.save(f"{tsne_dir}/labels_hub.npy", labels_hub)
print(f"HuBERT BiLSTM reps: {reps_hub.shape}")

# 3. Fusion block — fused representations (256-dim fusion FC output)
fusion = FusionModel().to(device)
fusion.load_state_dict(torch.load(f"{model_dir}/best_speaker_fusion.pt",map_location=device), strict=False)
loader_fus = DataLoader(TensorDataset(X_hub, X_bert, y_hub), batch_size=32)
reps_fus, labels_fus = get_reps(fusion, loader_fus, device)
np.save(f"{tsne_dir}/reps_fus.npy",   reps_fus)
np.save(f"{tsne_dir}/labels_fus.npy", labels_fus)
print(f"Fusion reps: {reps_fus.shape}")

print("\nAll representations saved.")



from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np, os

tsne_dir = "/content/drive/MyDrive/tess_speaker_tsne"

configs = [
    ("reps_cnn",  "labels_cnn",  "Temporal Block (MFCC + CNN)"),
    ("reps_hub",  "labels_hub",  "Contextual Block (HuBERT + BiLSTM)"),
    ("reps_fus",  "labels_fus",  "Fusion Block (HuBERT + BERT)"),
]

for rep_name, lbl_name, title in configs:
    reps   = np.load(f"{tsne_dir}/{rep_name}.npy")
    labels = np.load(f"{tsne_dir}/{lbl_name}.npy")

    # Standardize before t-SNE — important for fair comparison
    reps_scaled = StandardScaler().fit_transform(reps)

    print(f"Running t-SNE for {title}  ({reps_scaled.shape}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,       # good for ~700 samples
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        init='pca'           # PCA init is more stable than random
    )
    embedding = tsne.fit_transform(reps_scaled)

    np.save(f"{tsne_dir}/tsne_{rep_name}.npy", embedding)
    print(f"  Saved tsne_{rep_name}.npy  shape: {embedding.shape}")

print("\nAll t-SNE embeddings done.")

#OUTPUT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

tsne_dir      = "/content/drive/MyDrive/tess_speaker_tsne"
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']
COLORS        = ['#E24B4A','#7F77DD','#378ADD','#639922','#888780','#D85A30','#1D9E75']

configs = [
    ("tsne_reps_cnn", "labels_cnn",  "Temporal Block\n(MFCC + CNN)"),
    ("tsne_reps_hub", "labels_hub",  "Contextual Block\n(HuBERT + BiLSTM)"),
    ("tsne_reps_fus", "labels_fus",  "Fusion Block\n(HuBERT + BERT)"),
]

fig = plt.figure(figsize=(18, 6))
fig.suptitle(
    "t-SNE Emotion Cluster Separability — Speaker-Level Split (Test Set: YAF)",
    fontsize=15, fontweight='bold', y=1.02
)
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

for idx, (tsne_name, lbl_name, title) in enumerate(configs):
    embedding = np.load(f"{tsne_dir}/{tsne_name}.npy")
    labels    = np.load(f"{tsne_dir}/{lbl_name}.npy")

    ax = fig.add_subplot(gs[0, idx])

    for emo_idx, (emo, col) in enumerate(zip(emotion_names, COLORS)):
        mask = labels == emo_idx
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=col, s=28, alpha=0.80, linewidths=0,
            label=emo, zorder=2
        )

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')

# Shared legend below all plots
handles = [mpatches.Patch(color=c, label=e) for c, e in zip(COLORS, emotion_names)]
fig.legend(
    handles=handles, loc='lower center', ncol=7,
    fontsize=10, frameon=False,
    bbox_to_anchor=(0.5, -0.08)
)

plt.tight_layout()
plt.savefig(f"{tsne_dir}/tsne_speaker_split.png", dpi=180, bbox_inches='tight')
plt.show()
print("Saved: tsne_speaker_split.png")

# ── Individual high-res plots ──
for idx, (tsne_name, lbl_name, title) in enumerate(configs):
    embedding = np.load(f"{tsne_dir}/{tsne_name}.npy")
    labels    = np.load(f"{tsne_dir}/{lbl_name}.npy")

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for emo_idx, (emo, col) in enumerate(zip(emotion_names, COLORS)):
        mask = labels == emo_idx
        ax2.scatter(embedding[mask,0], embedding[mask,1],
                    c=col, s=35, alpha=0.85, linewidths=0, label=emo, zorder=2)

    ax2.set_title(f"{title.replace(chr(10),' ')} — Speaker Split", fontsize=13, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=11)
    ax2.set_ylabel('t-SNE 2', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('#FAFAFA')
    ax2.legend(loc='best', fontsize=9, framealpha=0.7, markerscale=1.4)

    fname = f"{tsne_dir}/tsne_{['cnn','hub','fus'][idx]}_hires.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")