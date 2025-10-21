# train_model.py
import numpy as np
import wfdb
from wfdb import processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout, Flatten
import joblib
import skfuzzy as fuzz

# -----------------------
# Load MIT-BIH dataset
# -----------------------
# Example: use record '100'
record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/100')
annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/100', 'atr')

# Extract ECG signal and labels
signal = record.p_signal[:,0]  # use first lead
labels = annotation.symbol

# -----------------------
# Preprocessing
# -----------------------
# Simple bandpass: 0.5-40Hz
fs = record.fs
b, a = processing.butter_bandpass(0.5, 40, fs)
signal_filtered = processing.filter_signal(signal, 'bandpass', f_low=0.5, f_high=40, fs=fs)

# Sliding window segmentation (e.g., 5s windows)
window_size = int(fs*5)
X = []
y = []

for i in range(0, len(signal_filtered)-window_size, int(window_size/2)):
    segment = signal_filtered[i:i+window_size]
    # Label: if any abnormal beat in window, mark as abnormal
    beat_labels = labels[np.searchsorted(annotation.sample, np.arange(i,i+window_size))]
    label = 0 if any(l not in ['N','~'] for l in beat_labels) else 1  # 1=normal, 0=abnormal
    X.append(segment)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for 1D-CNN/GRU [samples, timesteps, features]
X = X[..., np.newaxis]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Build 1D-CNN + GRU
# -----------------------
model = Sequential()
model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(GRU(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Save model and scaler
model.save('router_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully.")

# -----------------------
# Fuzzy Logic (simple example)
# -----------------------
# Define fuzzy membership functions for HR
hr_range = np.arange(0, 201, 1)
normal = fuzz.trimf(hr_range, [50, 75, 100])
brady = fuzz.trimf(hr_range, [0, 30, 60])
tachy = fuzz.trimf(hr_range, [100, 150, 200])

# Save fuzzy sets
np.savez('fuzzy_hr_sets.npz', normal=normal, brady=brady, tachy=tachy)
print("✅ Fuzzy sets saved.")