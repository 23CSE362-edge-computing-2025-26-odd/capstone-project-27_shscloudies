#code used for testing the data

#!/usr/bin/env python3
"""
stage_2_relabel.py
- Auto-detects BPM / RR columns when available.
- Creates new labels: arrhythmia, bradycardia, tachycardia, normal (4-class).
- Optionally collapses to 3-class (arrhythmia, bradycardia, tachycardia)
  by merging or dropping 'normal'.
- Trains RandomForest and saves artifacts in --outdir.

Usage examples:
  python stage_2_relabel.py --csv ECGdata.csv --outdir outputs_relabel --target four
  python stage_2_relabel.py --csv ECGdata.csv --outdir outputs_relabel_three --target three --merge-normal
"""
import os, re, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def detect_hr_rr_columns(df):
    """
    Try to find a column that directly contains BPM or an RR-interval column.
    Return a dict { 'type': 'bpm'|'rr'|'count_duration'|'none', 'col': colname or (count_col, duration_col) }
    """
    cols = df.columns.tolist()
    low = [c.lower() for c in cols]
    # direct BPM candidates
    bpm_patterns = ['bpm','heart_rate','heartrate','hr','pulse','beats_per_min']
    for p in bpm_patterns:
        for c in cols:
            if p in c.lower():
                return {'type':'bpm', 'col':c}
    # RR / RRI patterns
    rr_patterns = ['rr','rri','r_to_r','rto r','r_to_r_interval','rr_interval','rr_ms','rrmsec','r_peak_interval']
    for c in cols:
        if any(p in c.lower().replace('-','_').replace(' ','_') for p in rr_patterns):
            return {'type':'rr', 'col':c}
    # num beats + duration fields (can compute bpm)
    numc = None
    durc = None
    for c in cols:
        if re.search(r'num.*beat|beat_count|n_beats|r_peaks|rpeak', c, re.I):
            numc = c
        if re.search(r'duration|length_sec|seconds|duration_seconds|record_length|rec_len', c, re.I):
            durc = c
    if numc and durc:
        return {'type':'count_duration', 'col': (numc, durc)}
    return {'type':'none', 'col': None}

def compute_bpm_series(df, detector):
    """
    Given df and the detector result, attempt to compute a BPM series (pandas Series).
    Returns series of bpm or None if not possible.
    """
    typ = detector['type']
    col = detector['col']
    if typ == 'bpm':
        s = df[col].astype(float)
        # check plausible range, otherwise return None
        med = s.median(skipna=True)
        if 20 <= med <= 250:
            return s
        # maybe it's a 0-1 normalized? but we won't try extreme heuristics
        return s  # still return it; we'll validate later
    elif typ == 'rr':
        s = df[col].astype(float).replace(0, np.nan)
        med = s.median(skipna=True)
        if pd.isna(med):
            return None
        # detect units:
        if 0.2 <= med <= 3.0:  # likely seconds
            return 60.0 / s
        if 200 <= med <= 2000:  # likely ms
            return 60000.0 / s
        # unknown scale — attempt both heuristics and pick values that are plausible
        # prefer ms->bpm if med > 50
        if med > 50:
            return 60000.0 / s
        else:
            return 60.0 / s
    elif typ == 'count_duration':
        numc, durc = col
        n = pd.to_numeric(df[numc], errors='coerce')
        d = pd.to_numeric(df[durc], errors='coerce')
        # assume duration in seconds; if duration seems like minutes, check median
        med_d = d.median(skipna=True)
        if med_d > 1000:  # unlikely seconds; maybe ms
            # convert ms -> seconds
            d = d / 1000.0
        bpm = n * 60.0 / d.replace(0, np.nan)
        return bpm
    return None

def create_target_labels(df, original_label_col='ECG_signal'):
    """
    Use original label and computed bpm (if available) to produce a target column:
    - 'arrhythmia' if original label suggests ARR/AFF (irregular)
    - else if bpm exists:
        bpm < 60 -> 'bradycardia'
        bpm > 100 -> 'tachycardia'
        else -> 'normal'
    - if bpm not available: fallback: ARR/AFF -> 'arrhythmia', NSR -> 'normal', CHF -> 'arrhythmia'
    """
    detector = detect_hr_rr_columns(df)
    bpm_series = None
    if detector['type'] != 'none':
        bpm_series = compute_bpm_series(df, detector)
        if bpm_series is None or bpm_series.isna().all():
            bpm_series = None
        else:
            # put bpm into df
            df['bpm_detected'] = bpm_series
    else:
        # no HR/RR info
        pass

    def map_one(row):
        orig = str(row.get(original_label_col)).upper() if pd.notna(row.get(original_label_col)) else ''
        # direct arrhythmia mapping for known arrhythmia labels
        if orig in ('ARR','AFF'):
            return 'arrhythmia'
        # if bpm available, use it
        if 'bpm_detected' in row.index and pd.notna(row['bpm_detected']):
            bpm = float(row['bpm_detected'])
            if bpm < 60:
                return 'bradycardia'
            elif bpm > 100:
                return 'tachycardia'
            else:
                return 'normal'
        # fallback mapping when no bpm:
        if orig == 'NSR':
            return 'normal'
        if orig == 'CHF':
            return 'arrhythmia'   # heuristic fallback
        # last resort
        return 'arrhythmia'

    df['new_target'] = df.apply(map_one, axis=1)
    return df, detector

# Reuse simple RF training/evaluation
def train_rf(X_train, y_train, n_estimators=200):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_and_save(model, X_test, y_test, label_encoder, outdir, prefix='rf'):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    cm = confusion_matrix(y_test, y_pred)
    # plot
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion matrix ({prefix})')
    plt.colorbar()
    ticks = range(len(label_encoder.classes_))
    plt.xticks(ticks, label_encoder.classes_, rotation=45)
    plt.yticks(ticks, label_encoder.classes_)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_confusion.png'))
    plt.close()

def save_artifacts(outdir, scaler, label_encoder, rf_model, feature_columns):
    os.makedirs(outdir, exist_ok=True)
    if scaler is not None:
        joblib.dump(scaler, os.path.join(outdir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(outdir, 'label_encoder.joblib'))
    joblib.dump(rf_model, os.path.join(outdir, 'rf_model.joblib'))
    with open(os.path.join(outdir, 'feature_columns.json'), 'w') as fh:
        json.dump(list(feature_columns), fh)
    # also save label map (int -> label)
    label_map = {int(i): str(c) for i,c in enumerate(label_encoder.classes_)}
    with open(os.path.join(outdir, 'label_map.json'), 'w') as fh:
        json.dump(label_map, fh)

def main(args):
    df = load_data(args.csv)
    print("Loaded:", df.shape)
    if 'ECG_signal' not in df.columns:
        raise RuntimeError("Label column 'ECG_signal' not found in CSV.")
    # create new target
    df_with_labels, detector = create_target_labels(df, original_label_col='ECG_signal')
    print("HR/RR detector result:", detector)
    if 'bpm_detected' in df_with_labels.columns:
        print("Sample of detected BPM (first 5):")
        print(df_with_labels['bpm_detected'].head().to_list())
    # show distribution
    print("New label counts (before merging/dropping):")
    print(df_with_labels['new_target'].value_counts())

    # handle --target option
    if args.target == 'three':
        if args.drop_normal:
            df_with_labels = df_with_labels[df_with_labels['new_target'] != 'normal'].copy()
            print("Dropped 'normal' rows. New counts:")
            print(df_with_labels['new_target'].value_counts())
        elif args.merge_normal:
            # merge 'normal' into 'arrhythmia' (user requested 3 classes)
            df_with_labels['new_target'] = df_with_labels['new_target'].replace({'normal':'arrhythmia'})
            print("Merged 'normal' -> 'arrhythmia'. New counts:")
            print(df_with_labels['new_target'].value_counts())
        else:
            # default: merge normal
            df_with_labels['new_target'] = df_with_labels['new_target'].replace({'normal':'arrhythmia'})
            print("Merged 'normal' -> 'arrhythmia' (default). New counts:")
            print(df_with_labels['new_target'].value_counts())

    # Prepare features X and target y
    X = df_with_labels.drop(columns=['ECG_signal','new_target'], errors='ignore').copy()
    y = df_with_labels['new_target'].astype(str)

    # Drop high-missing columns and non-numeric columns (same as earlier)
    missing_frac = X.isna().mean()
    to_drop = missing_frac[missing_frac > 0.5].index.tolist()
    if to_drop:
        print("Dropping high-missing columns:", to_drop)
        X = X.drop(columns=to_drop)
    # numeric only
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    nonnum = set(X.columns) - set(X_numeric.columns)
    if nonnum:
        print("Dropping non-numeric columns:", nonnum)
    # impute medians
    for c in X_numeric.columns:
        if X_numeric[c].isna().any():
            X_numeric[c] = X_numeric[c].fillna(X_numeric[c].median())

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Final classes:", le.classes_)
    # split
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_enc, test_size=args.test_size, stratify=y_enc, random_state=RANDOM_STATE)
    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = train_rf(X_train_scaled, y_train, n_estimators=args.n_estimators)
    # evaluate
    os.makedirs(args.outdir, exist_ok=True)
    evaluate_and_save(rf, X_test_scaled, y_test, le, args.outdir, prefix='rf')
    # save
    save_artifacts(args.outdir, scaler, le, rf, X_numeric.columns)
    print("Saved artifacts to", args.outdir)
    # if the detector type == none, warn user
    if detector['type'] == 'none':
        print("\nWARNING: No HR/RR-like column was detected automatically.")
        print("-> bradycardia/tachycardia labels were made using fallbacks (ARR/AFF->arrhythmia, NSR->normal, CHF->arrhythmia).")
        print("If you want correct brady/tachy detection, provide BPM or RR interval columns (or R-peak timestamps).")
    else:
        print("\nDetector used:", detector)
        print("If labels look odd, inspect 'bpm_detected' column in your CSV and adjust heuristics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='ECGdata.csv')
    parser.add_argument('--outdir', type=str, default='outputs_relabel')
    parser.add_argument('--target', choices=['three','four'], default='four', help='four -> arrhythmia/brady/tachy/normal; three -> collapse normal into arrhythmia (or drop with --drop-normal)')
    parser.add_argument('--merge-normal', action='store_true', help='When --target three, merge normal -> arrhythmia (default behavior)')
    parser.add_argument('--drop-normal', action='store_true', help='When --target three, drop normal rows from dataset')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_estimators', type=int, default=200)
    args = parser.parse_args()
    # default behaviour: if target three and neither merge nor drop specified, merge
    if args.target == 'three' and not (args.merge_normal or args.drop_normal):
        args.merge_normal = True
    main(args)
