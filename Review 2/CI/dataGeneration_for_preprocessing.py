
# prepare_waveform.py
import os
import wfdb
import numpy as np
import scipy.signal as signal
import pywt
from tqdm import tqdm

DL_DIR = "./mitdb"      # directory with MIT-BIH .dat/.hea/.atr
TARGET_FS = 250
WINDOW_S = 5
WINDOW_LEN = WINDOW_S * TARGET_FS
NOTCH_FREQ = 50.0       # change to 60.0 if mains is 60Hz in your region

def bandpass_filter(sig, fs=TARGET_FS, low=0.5, high=40.0, order=4):
    b, a = signal.butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, sig)

def notch_filter(sig, fs=TARGET_FS, freq=NOTCH_FREQ, Q=30.0):
    # use frequency in Hz and pass fs so iirnotch interprets freq as Hz
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, sig)

def wavelet_denoise(sig, wavelet='db4', level=1):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(sig)))
    coeffs[1:] = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)
    return rec[:len(sig)]

def label_window(resamp_ann_samples, resamp_ann_symbols, start, end, bpm):
    # If any non-'N' annotation in window -> Arrhythmia label (2)
    idxs = np.where((resamp_ann_samples >= start) & (resamp_ann_samples < end))[0]
    if len(idxs) > 0:
        for ii in idxs:
            sym = resamp_ann_symbols[ii]
            if sym != 'N':
                return 2   # Arrhythmia
    # else use BPM thresholds
    if bpm < 60:
        return 1   # Bradycardia
    if bpm > 100:
        return 3   # Tachycardia
    return 0       # Normal

def process_all(dest=DL_DIR, sample_limit=None):
    # Expect MIT-BIH files already present in dest (100.dat, 100.hea, 100.atr, etc.)
    files = os.listdir(dest)
    recs = sorted(list({os.path.splitext(f)[0] for f in files if f.endswith('.dat')}))
    if sample_limit:
        recs = recs[:sample_limit]
    X_list = []
    y_list = []
    for rec in tqdm(recs, desc="Records"):
        rec_path = os.path.join(dest, rec)
        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal[:,0]   # channel 0
            orig_fs = int(record.fs)
        except Exception as e:
            print("Failed to load", rec, e)
            continue

        # resample if needed
        if orig_fs != TARGET_FS:
            sig_rs = signal.resample_poly(sig, TARGET_FS, orig_fs)
        else:
            sig_rs = sig.astype(np.float32)

        # preprocess
        try:
            sig_bp = bandpass_filter(sig_rs, fs=TARGET_FS)
            sig_nt = notch_filter(sig_bp, fs=TARGET_FS, freq=NOTCH_FREQ)
            sig_clean = wavelet_denoise(sig_nt)
        except Exception as e:
            print("Filtering error for", rec, e)
            continue

        # annotations
        try:
            ann = wfdb.rdann(rec_path, 'atr')
            ann_samples = np.array(ann.sample)
            ann_symbols = np.array(ann.symbol)
            resamp_ann_samples = np.round(ann_samples * TARGET_FS / orig_fs).astype(int)
            resamp_ann_symbols = ann_symbols
        except Exception:
            resamp_ann_samples = np.array([], dtype=int)
            resamp_ann_symbols = np.array([], dtype='U1')

        L = len(sig_clean)
        for start in range(0, L - WINDOW_LEN + 1, WINDOW_LEN):   # non-overlapping windows
            end = start + WINDOW_LEN
            seg = sig_clean[start:end]
            peaks, _ = signal.find_peaks(seg, distance=int(0.3*TARGET_FS))
            bpm = (len(peaks) * 60.0) / WINDOW_S
            lbl = label_window(resamp_ann_samples, resamp_ann_symbols, start, end, bpm)
            X_list.append(seg.astype(np.float32))
            y_list.append(lbl)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    print("Total windows:", X.shape, "counts:", np.bincount(y))
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Saved X.npy and y.npy")

if __name__ == "__main__":
    # For quick dev run use process_all(sample_limit=3)
    process_all()
