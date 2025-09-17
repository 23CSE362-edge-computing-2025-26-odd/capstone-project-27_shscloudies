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

#!/usr/bin/env python3
"""
stage_2.py - End-to-end ECG classification pipeline for ECGdata.csv

Place ECGdata.csv in the same folder and run:
    python stage_2.py --csv ECGdata.csv --outdir outputs

The script will:
 - load CSV
 - run basic EDA (shape + class counts)
 - preprocess (drop high-missing columns, impute medians, scale)
 - train RandomForest baseline and a PyTorch MLP (optional)
 - evaluate and save models + artifacts (scaler, label encoder, feature list)
"""
'''
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Optional PyTorch model
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

RANDOM_STATE = 42

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def basic_eda(df):
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample head:\n", df.head().T.iloc[:10])
    if 'ECG_signal' in df.columns:
        print("Label distribution:\n", df['ECG_signal'].value_counts())

def preprocess(df, label_col='ECG_signal', missing_threshold=0.5):
    # Drop identifier columns if present
    df = df.copy()
    drop_candidates = ['RECORD', 'record', 'id', 'Id']
    for c in drop_candidates:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Drop columns with too many missing values
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > missing_threshold].index.tolist()
    if to_drop:
        print(f"Dropping {len(to_drop)} columns with >{int(missing_threshold*100)}% missing: {to_drop}")
        df = df.drop(columns=to_drop)
    # Separate X/y
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)
    # Impute numeric missing values with median
    num_cols = X.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(med)
    # If any non-numeric feature remains, drop or encode
    nonnum = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if nonnum:
        print(f"Warning: Dropping non-numeric columns: {nonnum}")
        X = X.drop(columns=nonnum)
    return X, y

def train_rf(X_train, y_train, n_estimators=200, random_state=RANDOM_STATE):
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test, label_encoder=None, outdir=None, prefix='rf'):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    if outdir:
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion matrix: {prefix}")
        plt.colorbar()
        ticks = range(len(set(y_test)))
        plt.xticks(ticks, sorted(set(y_test)), rotation=45)
        plt.yticks(ticks, sorted(set(y_test)))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_confusion.png"))
        plt.close()
    return y_pred

# Simple feed-forward classifier in PyTorch
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_torch(X_train, y_train, X_val, y_val, num_classes, device=None, epochs=30, batch_size=32, lr=1e-3):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install torch to run the neural network.")
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('Using device:', device)
    Xtr = torch.tensor(X_train.values, dtype=torch.float32)
    ytr = torch.tensor(y_train.values, dtype=torch.long)
    Xv = torch.tensor(X_val.values, dtype=torch.float32)
    yv = torch.tensor(y_val.values, dtype=torch.long)
    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xv, yv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = MLP(input_dim=X_train.shape[1], hidden_dims=[128,64], num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    # load best
    model.load_state_dict(best_state)
    return model

def save_artifacts(outdir, scaler, label_encoder, rf_model=None, torch_model=None, feature_columns=None):
    os.makedirs(outdir, exist_ok=True)
    if scaler is not None:
        joblib.dump(scaler, os.path.join(outdir, 'scaler.joblib'))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(outdir, 'label_encoder.joblib'))
    if rf_model is not None:
        joblib.dump(rf_model, os.path.join(outdir, 'rf_model.joblib'))
    if feature_columns is not None:
        with open(os.path.join(outdir, 'feature_columns.json'), 'w') as fh:
            json.dump(list(feature_columns), fh)
    if torch_model is not None:
        import torch as _torch
        _torch.save(torch_model.state_dict(), os.path.join(outdir, 'torch_model.pt'))

def main(args):
    df = load_data(args.csv)
    basic_eda(df)
    X, y = preprocess(df, label_col='ECG_signal', missing_threshold=0.5)
    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print('Classes:', le.classes_)
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y_enc)
    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train RandomForest
    rf = train_rf(pd.DataFrame(X_train_scaled, columns=X.columns), y_train)
    # Evaluate RF
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    evaluate_model(rf, pd.DataFrame(X_test_scaled, columns=X.columns), y_test, outdir=args.outdir, prefix='rf')
    # save artifacts (note: save scaler and label encoder with original X columns)
    save_artifacts(args.outdir or '.', scaler, le, rf_model=rf, feature_columns=X.columns)
    # Optional: train PyTorch net
    if args.torch and TORCH_AVAILABLE:
        # create a small val split from training set
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)
        X_tr_df = pd.DataFrame(X_tr, columns=X.columns)
        X_val_df = pd.DataFrame(X_val, columns=X.columns)
        torch_model = train_torch(X_tr_df, pd.Series(y_tr), X_val_df, pd.Series(y_val), num_classes=len(le.classes_), epochs=args.epochs)
        save_artifacts(args.outdir or '.', None, None, torch_model=torch_model, feature_columns=X.columns)
        # evaluate torch on test set
        import torch as _torch
        device = _torch.device('cuda') if _torch.cuda.is_available() else _torch.device('cpu')
        X_test_t = _torch.tensor(X_test_scaled, dtype=_torch.float32).to(device)
        model = MLP(input_dim=X_test_t.shape[1], hidden_dims=[128,64], num_classes=len(le.classes_)).to(device)
        model.load_state_dict(torch_model.state_dict())
        model.eval()
        with _torch.no_grad():
            logits = model(X_test_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        print('PyTorch test accuracy:', accuracy_score(y_test, preds))
    elif args.torch and not TORCH_AVAILABLE:
        print('Skipping PyTorch training: torch not available.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECG classification pipeline')
    parser.add_argument('--csv', type=str, default='ECGdata.csv', help='Path to ECG CSV (default: ECGdata.csv)')
    parser.add_argument('--outdir', type=str, default='outputs', help='Where to save models/artifacts')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--torch', action='store_true', help='Also train a PyTorch MLP (if torch installed)')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs for PyTorch model')
    args = parser.parse_args()
    main(args)
'''
