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

model     = TextClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
emotion_names = ['angry','disgust','fear','happy','neutral','ps','sad']

# ── Test evaluation ──
model.load_state_dict(torch.load("best_text_model.pt"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print("\nText-only Test Results:")
print(classification_report(all_labels, all_preds, target_names=emotion_names))