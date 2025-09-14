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
