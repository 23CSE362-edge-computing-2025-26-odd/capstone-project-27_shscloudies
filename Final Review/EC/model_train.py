import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wfdb  # MIT-BIH dataset reader

# -----------------------
# Load MIT-BIH dataset
# -----------------------
def load_mitbih(signals_path):
    X = []
    y = []
    records = wfdb.get_record_list('mit-bih-arrhythmia-database-1.0.0')
    for rec_name in records[:10]:  # subset for example, adjust as needed
        record = wfdb.rdrecord(signals_path + '/' + rec_name)
        annotation = wfdb.rdann(signals_path + '/' + rec_name, 'atr')
        sig = record.p_signal[:, 0]  # first channel
        ann_symbols = annotation.symbol
        for idx, sym in zip(annotation.sample, ann_symbols):
            start = max(0, idx - 125)
            end = min(len(sig), idx + 125)
            beat = sig[start:end]
            if len(beat) == 250:
                X.append(beat)
                # Map symbols to classes
                if sym in ['N', 'L', 'R']:
                    y.append('normal')
                elif sym in ['V', 'E']:
                    y.append('tachycardia')
                elif sym in ['A', 'J']:
                    y.append('arrhythmia')
                elif sym in ['F']:
                    y.append('bradycardia')
                else:
                    y.append('normal')
    return np.array(X), np.array(y)

X, y = load_mitbih("mit-bih-records")  # replace with your local path
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc, num_classes=4)

X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, channels)

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# -----------------------
# Model: 1D-CNN + GRU
# -----------------------
inp = Input(shape=(250, 1))
x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inp)
x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
x = GRU(64, return_sequences=False)(x)
x = Dropout(0.3)(x)
out = Dense(4, activation='softmax')(x)

model = Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------
# Train
# -----------------------
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
model.save("ecg_classifier_model.h5")
