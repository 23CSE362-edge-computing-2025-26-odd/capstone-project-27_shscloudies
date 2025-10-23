'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wfdb  # MIT-BIH dataset reader

#-----------------Load MIT-BIH dataset-----------------
def load_mitbih(signals_path):
    X = []
    y = []
    records = wfdb.get_record_list('mit-bih-arrhythmia-database-1.0.0')
    for rec_name in records[:10]:
        record = wfdb.rdrecord(signals_path + '/' + rec_name)
        annotation = wfdb.rdann(signals_path + '/' + rec_name, 'atr')
        sig = record.p_signal[:, 0]
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

X, y = load_mitbih("mit-bih-records")
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc, num_classes=4)

X = X.reshape((X.shape[0], X.shape[1], 1))  #(samples, timesteps, channels)

#-----------------Train-test split-----------------
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

#-----------------Model: 1D-CNN + GRU-----------------
inp = Input(shape=(250, 1))
x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inp)
x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
x = GRU(64, return_sequences=False)(x)
x = Dropout(0.3)(x)
out = Dense(4, activation='softmax')(x)

model = Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#-----------------Train-----------------
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
model.save("ecg_classifier_model_1.h5")
'''


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wfdb
import random

#-----------------Load MIT-BIH dataset (offline version)-----------------
def load_mitbih(signals_path):
    X = []
    y = []

    records = list({f.split('.')[0] for f in os.listdir(signals_path) if f.endswith('.dat')})

    for rec_name in records[:10]:
        record_file = os.path.join(signals_path, rec_name)
        record = wfdb.rdrecord(record_file)
        annotation = wfdb.rdann(record_file, 'atr')
        sig = record.p_signal[:, 0]
        ann_symbols = annotation.symbol

        for idx, sym in zip(annotation.sample, ann_symbols):
            start = max(0, idx - 125)
            end = min(len(sig), idx + 125)
            beat = sig[start:end]
            if len(beat) == 250:
                X.append(beat)
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


#-----------------Load and preprocess data-----------------
signals_path = r"D:\BTECH CSE\sem5\CI\project\CI\capstone-project-27_shscloudies\Review 2\CI\mitdb"
X, y = load_mitbih(signals_path)
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc, num_classes=4)
X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

#-----------------Define model creation function-----------------
def create_model(params):
    filters1 = int(params['filters1'])
    filters2 = int(params['filters2'])
    gru_units = int(params['gru_units'])
    dropout_rate = params['dropout']
    lr = params['lr']

    inp = Input(shape=(250, 1))
    x = Conv1D(filters1, kernel_size=5, activation='relu', padding='same')(inp)
    x = Conv1D(filters2, kernel_size=5, activation='relu', padding='same')(x)
    x = GRU(gru_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(4, activation='softmax')(x)

    model = Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#-----------------Genetic Algorithm Setup-----------------
def random_params():
    return {
        'filters1': random.choice([16, 32, 64]),
        'filters2': random.choice([32, 64, 128]),
        'gru_units': random.choice([32, 64, 128]),
        'dropout': random.uniform(0.2, 0.5),
        'lr': random.choice([0.001, 0.0005, 0.0001])
    }

def mutate(params):
    """Mutate one random hyperparameter"""
    key = random.choice(list(params.keys()))
    new_params = params.copy()
    if key == 'dropout':
        new_params[key] = min(0.6, max(0.1, new_params[key] + random.uniform(-0.1, 0.1)))
    else:
        new_params[key] = random_params()[key]
    return new_params

def crossover(p1, p2):
    """Crossover two sets of params"""
    child = {}
    for k in p1.keys():
        child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return child

#-----------------GA Evaluation Function-----------------
def evaluate(params):
    model = create_model(params)
    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=3, batch_size=64, verbose=0)
    val_acc = history.history['val_accuracy'][-1]
    return val_acc

#-----------------Run GA Optimization-----------------
POP_SIZE = 5
GENERATIONS = 3

population = [random_params() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    print(f"\n=== Generation {gen+1}/{GENERATIONS} ===")
    scores = []
    for i, params in enumerate(population):
        print(f" Evaluating model {i+1} with params: {params}")
        acc = evaluate(params)
        scores.append((acc, params))
        print(f"  → Validation Accuracy: {acc:.4f}")

    # Sort by accuracy
    scores.sort(reverse=True, key=lambda x: x[0])
    best = scores[0][1]
    print(f"\nBest so far: {best}")

    # Select top 2 and create new population
    new_pop = [scores[0][1], scores[1][1]]
    while len(new_pop) < POP_SIZE:
        if random.random() < 0.5:
            new_pop.append(mutate(random.choice(new_pop)))
        else:
            p1, p2 = random.sample(new_pop, 2)
            new_pop.append(crossover(p1, p2))

    population = new_pop

#-----------------Train final model with best params-----------------
print("\nTraining final model with best parameters...")
best_params = best
final_model = create_model(best_params)
final_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
final_model.save("ecg_classifier_model_2.h5")
print("Model saved as ecg_classifier_model_2.h5")