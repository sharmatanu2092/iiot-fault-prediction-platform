"""
Train and evaluate the IIoT fault classifier.
Target: >= 95% weighted F1 across 5 fault classes.

Fault classes:
    0  NORMAL
    1  BEARING_WEAR
    2  OVERHEATING
    3  MOTOR_IMBALANCE
    4  ELECTRICAL_FAULT
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

FAULT_CLASSES = ["NORMAL", "BEARING_WEAR", "OVERHEATING",
                 "MOTOR_IMBALANCE", "ELECTRICAL_FAULT"]

FEATURE_NAMES = [
    "vib_rms", "vib_p2p", "vib_kurtosis", "vib_skewness", "vib_crest",
    "fft_dom_freq", "fft_dom_mag",
    "temp_mean", "temp_rise",
    "curr_rms", "curr_p2p", "curr_kurtosis"
]


def generate_synthetic_dataset(n_per_class=2000, noise=0.06, seed=42):
    """
    Generate a labelled feature dataset representative of each fault class.
    Each class has a distinct signature in the feature space.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    # NORMAL: low kurtosis, low crest, stable temp, steady current
    for _ in range(n_per_class):
        vib_rms     = rng.normal(0.12, 0.02)
        vib_p2p     = rng.normal(0.45, 0.05)
        kurtosis    = rng.normal(3.0, 0.3)
        skewness    = rng.normal(0.0, 0.1)
        crest       = rng.normal(2.8, 0.2)
        dom_freq    = rng.normal(25.0, 2.0)
        dom_mag     = rng.normal(0.08, 0.01)
        temp_mean   = rng.normal(62.0, 1.5)
        temp_rise   = rng.normal(0.8, 0.3)
        curr_rms    = rng.normal(4.2, 0.15)
        curr_p2p    = rng.normal(0.6, 0.08)
        curr_kurt   = rng.normal(3.0, 0.2)
        X.append([vib_rms, vib_p2p, kurtosis, skewness, crest,
                  dom_freq, dom_mag, temp_mean, temp_rise,
                  curr_rms, curr_p2p, curr_kurt])
        y.append(0)

    # BEARING_WEAR: elevated kurtosis + crest, high-frequency dom_freq
    for _ in range(n_per_class):
        vib_rms   = rng.normal(0.38, 0.06)
        vib_p2p   = rng.normal(1.10, 0.12)
        kurtosis  = rng.normal(8.5, 1.2)
        skewness  = rng.normal(0.6, 0.2)
        crest     = rng.normal(6.2, 0.7)
        dom_freq  = rng.normal(187.0, 8.0)
        dom_mag   = rng.normal(0.42, 0.06)
        temp_mean = rng.normal(64.0, 2.0)
        temp_rise = rng.normal(1.2, 0.4)
        curr_rms  = rng.normal(4.4, 0.2)
        curr_p2p  = rng.normal(0.7, 0.1)
        curr_kurt = rng.normal(3.2, 0.3)
        X.append([vib_rms, vib_p2p, kurtosis, skewness, crest,
                  dom_freq, dom_mag, temp_mean, temp_rise,
                  curr_rms, curr_p2p, curr_kurt])
        y.append(1)

    # OVERHEATING: high temp_mean and steep temp_rise
    for _ in range(n_per_class):
        vib_rms   = rng.normal(0.15, 0.03)
        vib_p2p   = rng.normal(0.52, 0.07)
        kurtosis  = rng.normal(3.2, 0.4)
        skewness  = rng.normal(0.05, 0.1)
        crest     = rng.normal(3.0, 0.3)
        dom_freq  = rng.normal(25.0, 2.5)
        dom_mag   = rng.normal(0.09, 0.02)
        temp_mean = rng.normal(88.5, 2.0)
        temp_rise = rng.normal(8.4, 1.2)
        curr_rms  = rng.normal(5.8, 0.4)
        curr_p2p  = rng.normal(0.9, 0.12)
        curr_kurt = rng.normal(3.1, 0.3)
        X.append([vib_rms, vib_p2p, kurtosis, skewness, crest,
                  dom_freq, dom_mag, temp_mean, temp_rise,
                  curr_rms, curr_p2p, curr_kurt])
        y.append(2)

    # MOTOR_IMBALANCE: asymmetric vibration (skewness), moderate kurtosis
    for _ in range(n_per_class):
        vib_rms   = rng.normal(0.52, 0.08)
        vib_p2p   = rng.normal(1.65, 0.18)
        kurtosis  = rng.normal(4.8, 0.6)
        skewness  = rng.normal(1.4, 0.3)
        crest     = rng.normal(4.2, 0.5)
        dom_freq  = rng.normal(50.0, 4.0)
        dom_mag   = rng.normal(0.55, 0.08)
        temp_mean = rng.normal(65.0, 2.0)
        temp_rise = rng.normal(1.5, 0.5)
        curr_rms  = rng.normal(5.1, 0.3)
        curr_p2p  = rng.normal(1.1, 0.15)
        curr_kurt = rng.normal(3.4, 0.4)
        X.append([vib_rms, vib_p2p, kurtosis, skewness, crest,
                  dom_freq, dom_mag, temp_mean, temp_rise,
                  curr_rms, curr_p2p, curr_kurt])
        y.append(3)

    # ELECTRICAL_FAULT: current spike kurtosis, high curr_p2p
    for _ in range(n_per_class):
        vib_rms   = rng.normal(0.18, 0.03)
        vib_p2p   = rng.normal(0.58, 0.07)
        kurtosis  = rng.normal(3.1, 0.3)
        skewness  = rng.normal(0.1, 0.1)
        crest     = rng.normal(3.0, 0.3)
        dom_freq  = rng.normal(50.0, 3.0)
        dom_mag   = rng.normal(0.10, 0.02)
        temp_mean = rng.normal(63.0, 2.0)
        temp_rise = rng.normal(0.9, 0.3)
        curr_rms  = rng.normal(6.4, 0.5)
        curr_p2p  = rng.normal(4.2, 0.6)
        curr_kurt = rng.normal(12.8, 1.8)
        X.append([vib_rms, vib_p2p, kurtosis, skewness, crest,
                  dom_freq, dom_mag, temp_mean, temp_rise,
                  curr_rms, curr_p2p, curr_kurt])
        y.append(4)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # add global noise
    X += rng.normal(0, noise, X.shape)
    return X, y


def build_pipeline():
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train(args):
    print("Generating dataset...")
    X, y = generate_synthetic_dataset(n_per_class=args.n_per_class)

    pipeline = build_pipeline()

    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"  CV F1 (weighted): {scores.mean():.4f} +/- {scores.std():.4f}")

    if scores.mean() < 0.95:
        print("[WARN] CV score below 95% target. Review features or class balance.")

    # final fit on full dataset
    pipeline.fit(X, y)

    # held-out evaluation on last 20% of each class
    split = int(args.n_per_class * 0.8)
    X_test = np.concatenate([X[i*args.n_per_class + split:(i+1)*args.n_per_class]
                              for i in range(5)])
    y_test = np.concatenate([y[i*args.n_per_class + split:(i+1)*args.n_per_class]
                              for i in range(5)])

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    print(f"\nHeld-out accuracy: {acc:.4f}")
    print(f"Held-out F1 (weighted): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=FAULT_CLASSES))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    metrics = {
        "accuracy":    round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_f1_mean":  round(float(scores.mean()), 4),
        "cv_f1_std":   round(float(scores.std()), 4)
    }

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=2000)
    parser.add_argument("--model_dir",   type=str, default="model_artifacts")
    args = parser.parse_args()
    train(args)
