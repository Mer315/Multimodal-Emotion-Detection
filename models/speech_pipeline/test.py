import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

#multihead attention encoder
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

#MFCCs
#-------------------------------------------------------------------------------------------------
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
    
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
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

#CNN 1D 

encoder = CNN1DEncoder(input_dim=120)
model = EmotionClassifier(encoder)
model.to(device)

model.load_state_dict(torch.load("/content/drive/MyDrive/tess_models_mfcc/cnn_mfcc.pth"))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch.to(device))

        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad'
        ]
    )
)

#BiLSTM 
encoder = BiLSTMEncoder(input_dim=120)

model = EmotionClassifier(encoder)

model.to(device)

model.load_state_dict(torch.load("/content/drive/MyDrive/tess_models_mfcc/bilstm_mfcc.pth"))

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch.to(device))

        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad'
        ]
    )
)


#SELF ATTENTION
encoder = AttentionPooling(input_dim=120)

model = EmotionClassifier(encoder)

model.to(device)

model.load_state_dict(torch.load("/content/drive/MyDrive/tess_models_mfcc/attention_mfcc.pth"))

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch.to(device))

        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad'
        ]
    )
)

#--------------------------------------------------------------------------------------------------

#HuBERT evaluation

#CNN

encoder = CNN1DEncoder(input_dim=768)

model = EmotionClassifier(encoder)

model.to(device)
# Load best saved model and evaluate on test set
model.load_state_dict(torch.load("/content/drive/MyDrive/tess_models_hubert/cnnhubert.pth"))

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch.to(device))

        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad'
        ]
    )
)

#BiLSTM

encoder = BiLSTMEncoder(input_dim=768)

model = EmotionClassifier(encoder)

model.to(device)

model.load_state_dict(torch.load("/content/drive/MyDrive/tess_models_hubert/bilstm_hubert.pth"))

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch.to(device))

        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad'
        ]
    )
)

#SELF ATTENTION

encoder = AttentionPooling(input_dim=768)

model = EmotionClassifier(encoder)

model.to(device)

model.load_state_dict(torch.load("/content/drive/MyDrive/tess_models_hubert/attention_hubert.pth"))

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch.to(device))

        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'ps',
            'sad'
        ]
    )
)