# train_cnn_gru_preproc.py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import scipy.signal as signal
import pywt

# ---------------- Config ----------------
WINDOW_LEN = 5 * 250  # 5s @ 250 Hz
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
NUM_CLASSES = 4
FS = 250

# ---------------- Filters ----------------
def bandpass_filter(sig, fs=FS, low=5, high=15):
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    return signal.filtfilt(b, a, sig)

def notch_filter(sig, fs=FS, freq=50):
    b, a = signal.iirnotch(freq, 30, fs)
    return signal.filtfilt(b, a, sig)

def wavelet_denoise(sig, wavelet="db4", level=1):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    coeffs = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

def preprocess_window(win):
    """ Apply Bandpass + Notch + Wavelet denoising """
    win = bandpass_filter(win)
    win = notch_filter(win)
    win = wavelet_denoise(win)
    return win

# ---------------- Model ----------------
class ECGModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        return self.fc(h[-1])

# ---------------- Load Data ----------------
def load_xy():
    X = np.load("X.npy")
    y = np.load("y.npy")
    return X, y

# ---------------- Main ----------------
def main():
    X, y = load_xy()
    print("Loaded", X.shape, y.shape)

    # Apply preprocessing filters to every window
    print("Applying preprocessing filters...")
    X_proc = []
    for i in tqdm(range(len(X))):
        X_proc.append(preprocess_window(X[i]))
    X = np.array(X_proc)

    # Normalize each window
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=0
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                           torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = ECGModel().to(device)

    # Class balancing
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        avg_loss = running / len(train_loader)

        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    # Save
    os.makedirs("model_out", exist_ok=True)
    torch.save(model.state_dict(), "model_out/ecg_classifier.pth")
    model.cpu().eval()
    example = torch.randn(1,1,WINDOW_LEN)
    traced = torch.jit.trace(model, example)
    traced.save("model_out/ecg_classifier_ts.pt")
    print("✅ Saved model_out/ecg_classifier_ts.pt")

if __name__ == "__main__":
    main()
