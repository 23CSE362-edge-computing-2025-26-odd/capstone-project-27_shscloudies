'''
import serial
import numpy as np
import time
import torch
import torch.nn as nn
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# ========================
# 1. UART Setup (Raspberry Pi <-> STM32)
# ========================
ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)

# ========================
# 2. Deep Learning Model (CNN + GRU)
# ========================
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)  # 2 classes: Normal / Abnormal

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)   # (batch, seq_len, features)
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out

# Load trained model (dummy random weights if no file)
model = ECGModel()
# model.load_state_dict(torch.load("ecg_model.pth", map_location="cpu"))
model.eval()

# ========================
# 3. Fuzzy Logic for Decision Making
# ========================
hr = ctrl.Antecedent(np.arange(30, 200, 1), 'hr')
confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
decision = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'decision')

# Membership functions
hr['low'] = fuzz.trimf(hr.universe, [30, 30, 60])
hr['normal'] = fuzz.trimf(hr.universe, [60, 75, 100])
hr['high'] = fuzz.trimf(hr.universe, [100, 150, 200])

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

decision['normal'] = fuzz.trimf(decision.universe, [0, 0, 0.5])
decision['alert'] = fuzz.trimf(decision.universe, [0.5, 1, 1])

# Rules
rule1 = ctrl.Rule(hr['normal'] & confidence['high'], decision['normal'])
rule2 = ctrl.Rule(hr['low'] | hr['high'] | confidence['low'], decision['alert'])

decision_ctrl = ctrl.ControlSystem([rule1, rule2])
decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)

# ========================
# 4. Data Loop
# ========================
logfile = open("ecg_log.txt", "a")

def classify_ecg(ecg_window):
    """Runs CNN+GRU model on ECG window"""
    x = torch.tensor(ecg_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        return pred, prob[0][pred].item()

print("Raspberry Pi ECG Monitoring Started...")

buffer = []
window_size = 1250   # Default = 5s window @ 250 Hz

while True:
    line = ser.readline().decode().strip()
    if line:
        try:
            val = float(line)
            buffer.append(val)
            logfile.write(f"{val}\n")

            if len(buffer) >= window_size:
                # Run classification
                pred, conf = classify_ecg(buffer)
                hr_value = 75 if pred == 0 else 150  # Fake HR estimate (replace with STM32 bpm if sent)

                # Fuzzy logic evaluation
                decision_sim.input['hr'] = hr_value
                decision_sim.input['confidence'] = conf
                decision_sim.compute()

                if decision_sim.output['decision'] > 0.5:
                    print("🚨 ALERT: Abnormal ECG detected")
                else:
                    print("✅ Normal ECG")

                buffer = []  # reset window

        except ValueError:
            continue
'''
# above is the initial code


'''
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, iirnotch
import pywt
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# ---------------- Config ----------------
FS = 250
WINDOW_LEN = 5 * FS
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 4   # Normal, Bradycardia, Tachycardia, Arrhythmia

# ---------------- Preprocessing ----------------
def bandpass_filter(sig, low=0.5, high=40, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def notch_filter(sig, freq=50.0, fs=FS, Q=30.0):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, sig)

def wavelet_denoise(sig, wavelet="db4", level=1, thr=0.02):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def preprocess_dataset(X):
    print("🩺 Applying preprocessing filters (bandpass + notch + wavelet)...")
    X_filt = []
    for win in tqdm(X):
        sig = bandpass_filter(win)
        sig = notch_filter(sig)
        sig = wavelet_denoise(sig)
        X_filt.append(sig[:WINDOW_LEN])
    return np.array(X_filt)

# ---------------- CNN + GRU Model ----------------
class ECGModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)                
        x = x.permute(0, 2, 1)        
        _, h = self.gru(x)            
        return self.fc(h[-1])         

# ---------------- Fuzzy Logic ----------------
def build_fuzzy_system():
    hr = ctrl.Antecedent(np.arange(30, 201, 1), 'hr')
    confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
    decision = ctrl.Consequent(np.arange(0, 4, 1), 'decision')

    hr['brady'] = fuzz.trimf(hr.universe, [30, 30, 60])
    hr['normal'] = fuzz.trimf(hr.universe, [60, 75, 100])
    hr['tachy'] = fuzz.trimf(hr.universe, [100, 150, 200])

    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

    decision['brady'] = fuzz.trimf(decision.universe, [0, 0, 1])
    decision['normal'] = fuzz.trimf(decision.universe, [1, 1, 2])
    decision['tachy'] = fuzz.trimf(decision.universe, [2, 2, 3])
    decision['arrhythmia'] = fuzz.trimf(decision.universe, [3, 3, 3])

    rules = [
        ctrl.Rule(hr['brady'] & confidence['high'], decision['brady']),
        ctrl.Rule(hr['normal'] & confidence['high'], decision['normal']),
        ctrl.Rule(hr['tachy'] & confidence['high'], decision['tachy']),
        ctrl.Rule(confidence['low'], decision['arrhythmia'])
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# ---------------- Training ----------------
def main():
    # Load dataset
    # Base directory = folder where stage_2.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths
    X_path = os.path.join(BASE_DIR, "X.npy")
    y_path = os.path.join(BASE_DIR, "y.npy")

    print("🔎 Looking for dataset files in:", BASE_DIR)

    # Load
    X = np.load(X_path)
    y = np.load(y_path)


    print("📂 Loaded dataset:", X.shape, y.shape)
    
    # Check class distribution BEFORE any processing
    unique, counts = np.unique(y, return_counts=True)
    print("🔎 Original class distribution:", dict(zip(unique, counts)))

    # Remove classes with insufficient samples (need at least 2 for stratified split)
    # This needs to happen FIRST, before any preprocessing
    min_samples = 2
    valid_classes = unique[counts >= min_samples]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    
    # Update class distribution after filtering
    unique_after, counts_after = np.unique(y, return_counts=True)
    print(f"✅ After filtering classes with <{min_samples} samples:")
    print("   Class distribution:", dict(zip(unique_after, counts_after)))
    print("   Total samples remaining:", len(X))
    
    if len(unique_after) < 2:
        raise ValueError("Not enough classes with sufficient samples for training!")

    # Relabel classes to be contiguous (0, 1, 2, ...)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_after)}
    y_relabeled = np.array([label_map[label] for label in y])
    print("🏷️ Relabeled classes:", label_map)
    
    # Update NUM_CLASSES to actual number of classes
    global NUM_CLASSES
    NUM_CLASSES = len(unique_after)
    print(f"🎯 Training with {NUM_CLASSES} classes")

    # Double-check that we can safely stratify
    unique_final, counts_final = np.unique(y_relabeled, return_counts=True)
    print("🔍 Final class check before split:", dict(zip(unique_final, counts_final)))
    
    # Ensure we can split with test_size=0.15
    
    test_size = 0.15
    for class_label, count in zip(unique_final, counts_final):
        test_samples = int(count * test_size)
        train_samples = count - test_samples
        if test_samples < 1 or train_samples < 1:
            print(f"⚠️ Warning: Class {class_label} has only {count} samples.")
            print(f"   This would result in {test_samples} test and {train_samples} train samples.")
            # Use a smaller test size or remove stratification for very small datasets
            if count < 4:  # If less than 4 samples, we can't reliably stratify
                print("🚫 Removing stratification due to small dataset")
                stratify_param = None
                break
    else:
        stratify_param = y_relabeled
    """
    # Ensure we can split with test_size=0.15
    test_size = 0.15
    stratify_param = y_relabeled  # Initialize stratify parameter

    for class_label, count in zip(unique_final, counts_final):
        test_samples = int(count * test_size)
        train_samples = count - test_samples
        if test_samples < 1 or train_samples < 1:
            print(f"⚠️ Warning: Class {class_label} has only {count} samples.")
            print(f"   This would result in {test_samples} test and {train_samples} train samples.")
            if count < 4:  # Not enough samples to stratify
                print("🚫 Removing stratification due to small dataset")
                stratify_param = None
                break
                """

    # Preprocess dataset
    X_processed = preprocess_dataset(X)

    # Normalize per window
    X_normalized = (X_processed - X_processed.mean(axis=1, keepdims=True)) / (X_processed.std(axis=1, keepdims=True) + 1e-8)

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y_relabeled, test_size=test_size, stratify=stratify_param, random_state=0
    )
    
    print(f"📊 Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                           torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⚡ Using device:", device)
    model = ECGModel(num_classes=NUM_CLASSES).to(device)

    # Weighted loss
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"📊 Epoch {epoch+1} | Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

    # Save model
    os.makedirs("model_out", exist_ok=True)
    torch.save(model.state_dict(), "model_out/ecg_classifier.pth")
    
    # Save label mapping for later use
    np.save("model_out/label_mapping.npy", label_map)
    print("✅ Model saved at model_out/ecg_classifier.pth")
    print("✅ Label mapping saved at model_out/label_mapping.npy")

    # Example fuzzy logic
    fuzzy = build_fuzzy_system()
    example_hr = 150; example_conf = 0.9
    fuzzy.input['hr'] = example_hr
    fuzzy.input['confidence'] = example_conf
    fuzzy.compute()
    print(f"🤖 Fuzzy decision for HR={example_hr}, Conf={example_conf} → {fuzzy.output['decision']:.2f}")

    # 0 -> Normal
    # 1 -> Bradycardia 
    # 2 -> Tachycardia
    # 3 -> Arrhythmia

if __name__ == "__main__":
    main()
    '''

#above code with only accuracy

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, iirnotch
import pywt
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# ---------------- Config ----------------
FS = 250
WINDOW_LEN = 5 * FS
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 4   # Normal, Bradycardia, Tachycardia, Arrhythmia

# ---------------- Preprocessing ----------------
def bandpass_filter(sig, low=0.5, high=40, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def notch_filter(sig, freq=50.0, fs=FS, Q=30.0):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, sig)

def wavelet_denoise(sig, wavelet="db4", level=1, thr=0.02):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def preprocess_dataset(X):
    print("🩺 Applying preprocessing filters (bandpass + notch + wavelet)...")
    X_filt = []
    for win in tqdm(X):
        sig = bandpass_filter(win)
        sig = notch_filter(sig)
        sig = wavelet_denoise(sig)
        X_filt.append(sig[:WINDOW_LEN])
    return np.array(X_filt)

# ---------------- CNN + GRU Model ----------------
class ECGModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)                
        x = x.permute(0, 2, 1)        
        _, h = self.gru(x)            
        return self.fc(h[-1])         

# ---------------- Fuzzy Logic ----------------
def build_fuzzy_system():
    hr = ctrl.Antecedent(np.arange(30, 201, 1), 'hr')
    confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
    decision = ctrl.Consequent(np.arange(0, 4, 1), 'decision')

    hr['brady'] = fuzz.trimf(hr.universe, [30, 30, 60])
    hr['normal'] = fuzz.trimf(hr.universe, [60, 75, 100])
    hr['tachy'] = fuzz.trimf(hr.universe, [100, 150, 200])

    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

    decision['brady'] = fuzz.trimf(decision.universe, [0, 0, 1])
    decision['normal'] = fuzz.trimf(decision.universe, [1, 1, 2])
    decision['tachy'] = fuzz.trimf(decision.universe, [2, 2, 3])
    decision['arrhythmia'] = fuzz.trimf(decision.universe, [3, 3, 3])

    rules = [
        ctrl.Rule(hr['brady'] & confidence['high'], decision['brady']),
        ctrl.Rule(hr['normal'] & confidence['high'], decision['normal']),
        ctrl.Rule(hr['tachy'] & confidence['high'], decision['tachy']),
        ctrl.Rule(confidence['low'], decision['arrhythmia'])
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# ---------------- Training ----------------
def main():
    # Load dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(BASE_DIR, "X.npy")
    y_path = os.path.join(BASE_DIR, "y.npy")

    print("🔎 Looking for dataset files in:", BASE_DIR)

    X = np.load(X_path)
    y = np.load(y_path)

    print("📂 Loaded dataset:", X.shape, y.shape)
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("🔎 Original class distribution:", dict(zip(unique, counts)))

    # Remove under-represented classes
    min_samples = 2
    valid_classes = unique[counts >= min_samples]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    
    unique_after, counts_after = np.unique(y, return_counts=True)
    print("✅ After filtering:", dict(zip(unique_after, counts_after)))

    # Relabel classes
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_after)}
    y_relabeled = np.array([label_map[label] for label in y])
    global NUM_CLASSES
    NUM_CLASSES = len(unique_after)
    print("🏷️ Relabeled classes:", label_map)

    # Preprocess
    X_processed = preprocess_dataset(X)

    # Normalize
    X_normalized = (X_processed - X_processed.mean(axis=1, keepdims=True)) / (
        X_processed.std(axis=1, keepdims=True) + 1e-8
    )

    # Train-val split
    test_size = 0.15
    stratify_param = y_relabeled if len(unique_after) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y_relabeled, test_size=test_size, stratify=stratify_param, random_state=0
    )
    
    print(f"📊 Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Torch datasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                           torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⚡ Using device:", device)
    model = ECGModel(num_classes=NUM_CLASSES).to(device)

    # Weighted loss
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    all_preds, all_labels = [], []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"📊 Epoch {epoch+1} | Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

    # Metrics
    print("\n---------------- Classification Report ----------------")
    print(classification_report(all_labels, all_preds, target_names=[str(k) for k in unique_after]))
    print("-------------------------------------------------------")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

    # Save model
    os.makedirs("model_out", exist_ok=True)
    torch.save(model.state_dict(), "model_out/ecg_classifier.pth")
    np.save("model_out/label_mapping.npy", label_map)
    print("✅ Model + label mapping saved")

    # Example fuzzy logic
    fuzzy = build_fuzzy_system()
    example_hr = 150; example_conf = 0.9
    fuzzy.input['hr'] = example_hr
    fuzzy.input['confidence'] = example_conf
    fuzzy.compute()
    print(f"🤖 Fuzzy decision for HR={example_hr}, Conf={example_conf} → {fuzzy.output['decision']:.2f}")

if __name__ == "__main__":
    main()
