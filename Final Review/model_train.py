import numpy as np
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wfdb
import glob

# -----------------------
# ECG Preprocessing
# -----------------------
def bandpass_filter(signal, fs=250, low=0.5, high=40):
    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def extract_segments(signal, ann_samples, window_size=250):
    """
    Extract ECG windows of length `window_size` with label based on beats in that window.
    """
    X, y = [], []
    num_windows = len(signal)//window_size
    for i in range(num_windows):
        seg = signal[i*window_size:(i+1)*window_size]
        beats = ann_samples[(ann_samples >= i*window_size) & (ann_samples < (i+1)*window_size)]
        # Label assignment
        if len(beats) == 0:
            label = 0  # normal
        else:
            hr_est = 60.0 / (np.mean(np.diff(beats)/250) if len(beats) > 1 else 1)
            if hr_est < 45:
                label = 1  # bradycardia
            elif hr_est > 140:
                label = 2  # tachycardia
            else:
                label = 3  # arrhythmia
        X.append(seg)
        y.append(label)
    return np.array(X), np.array(y)

# -----------------------
# Load MIT-BIH dataset
# -----------------------
record_paths = glob.glob("mit-bih-database/*.dat")  # path to MIT-BIH .dat files
X_all, y_all = [], []

for rec_file in record_paths:
    record_name = rec_file.split('/')[-1].replace('.dat','')
    record = wfdb.rdrecord(record_name, pn_dir='mit-bih-database')
    ann = wfdb.rdann(record_name, 'atr', pn_dir='mit-bih-database')
    signal = record.p_signal[:,0]  # Lead II
    filtered = bandpass_filter(signal)
    X_seg, y_seg = extract_segments(filtered, ann.sample)
    X_all.append(X_seg)
    y_all.append(y_seg)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

# -----------------------
# Scale and reshape
# -----------------------
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)
X_all = X_all[..., np.newaxis]  # for Conv1D

# One-hot encode labels
y_all = to_categorical(y_all, num_classes=4)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# -----------------------
# Model
# -----------------------
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1],1)),
    Dropout(0.2),
    GRU(32),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------
# Train
# -----------------------
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# -----------------------
# Save model and scaler
# -----------------------
model.save("ecg_classifier_4class.h5")
import joblib
joblib.dump(scaler, "scaler.save")