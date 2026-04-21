# =============================================================
# run_experiment.py — Main entry point
#
# Usage:
#   python run_experiment.py
#
# Edit config.py first to point at your dataset.
# =============================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import config
from models import (
    get_classical_model, evaluate_classical,
    evaluate_ann, train_ann, predict_ann,
)
from plots import plot_learning_curves, plot_confusion_matrices, print_summary_table

warnings.filterwarnings("ignore")
np.random.seed(config.RANDOM_STATE)


# ── 1. Load & preprocess ──────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(config.CSV_PATH)
    #print(f"Loaded:  {df.shape[0]:,} rows × {df.shape[1]} columns")

    if config.DROP_COLS:
        df = df.drop(columns=config.DROP_COLS, errors="ignore")

    # Missing value report
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        #print(f"\nMissing values found:\n{missing.to_string()}")
        df = df.dropna()
        #print(f"After dropping NaN rows: {df.shape[0]:,} samples")
    #else:
        #print("Missing values: none ✓")

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    # Encode categorical feature columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target
    n_classes = 1
    if config.TASK == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))
        n_classes = len(le.classes_)
        #print(f"Classes ({n_classes}): {le.classes_}")
    else:
        y = y.astype(float).values

    X = X.astype(float).values
    #print(f"Features: {X.shape[1]}  |  Task: {config.TASK.upper()}  |  Classes: {n_classes}")
    return X, y, n_classes


# ── 2. Experiment loop ────────────────────────────────────────────────────

def run_experiments(X_train_full, y_train_full, X_test, y_test, n_classes):
    records = []
    fracs   = config.TRAIN_FRACTIONS
    total   = len(fracs) * config.N_RUNS * 2

    #print(f"\nRunning {total} experiments "
          #f"({len(fracs)} fractions × {config.N_RUNS} runs × 2 models)\n")

    done = 0
    for frac in fracs:
        for run in range(config.N_RUNS):
            n_samples = max(10, int(len(X_train_full) * frac))
            idx = np.random.choice(len(X_train_full), size=n_samples, replace=False)
            X_tr = X_train_full[idx]
            y_tr = y_train_full[idx]

            # Classical
            clf    = get_classical_model(config.TASK, config.RANDOM_STATE)
            c_mets = evaluate_classical(clf, X_tr, y_tr, X_test, y_test,
                                        config.TASK, n_classes)
            records.append({"model": "Classical", "fraction": frac,
                            "run": run + 1, "n_train": n_samples, **c_mets})
            done += 1

            # ANN
            a_mets = evaluate_ann(
                X_tr, y_tr, X_test, y_test,
                config.TASK, n_classes,
                config.ANN_HIDDEN_LAYERS, config.ANN_EPOCHS,
                config.ANN_LR, config.ANN_BATCH_SIZE,
            )
            records.append({"model": "ANN (MLP)", "fraction": frac,
                            "run": run + 1, "n_train": n_samples, **a_mets})
            done += 1

            #print(f"  [{done:>3}/{total}]  frac={int(frac*100):>3}%  "
                  #f"run={run+1}  n_train={n_samples:,}")

    return pd.DataFrame(records)


# ── 3. Main ───────────────────────────────────────────────────────────────

def main():
    #print("=" * 60)
    #print("  ML Experiment: Training Data Size vs. Performance")
    #print("=" * 60)
    #print(f"  Dataset : {config.CSV_PATH}")
    #print(f"  Target  : {config.TARGET_COL}")
    #print(f"  Task    : {config.TASK.upper()}")
    #print("=" * 60 + "\n")

    # Load
    X, y, n_classes = load_data()

    # Fixed train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if config.TASK == "classification" else None,
    )

    # Scale (fit only on training pool)
    scaler       = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test       = scaler.transform(X_test)
    #print(f"\nTrain pool: {X_train_full.shape[0]:,}  |  Test (fixed): {X_test.shape[0]:,}")

    # Run experiments
    df = run_experiments(X_train_full, y_train_full, X_test, y_test, n_classes)

    # Save raw results
    df.to_csv(config.RESULTS_CSV, index=False)
    #print(f"\n  Saved raw results → {config.RESULTS_CSV}")

    # Summary table
    print_summary_table(df)

    # Learning curves
    #print("\nGenerating plots...")
    plot_learning_curves(df, config.TASK, config.PLOT_FILE)

    # Confusion matrix (classification only)
    if config.TASK == "classification":
        clf = get_classical_model(config.TASK, config.RANDOM_STATE)
        clf.fit(X_train_full, y_train_full)
        classical_preds = clf.predict(X_test)

        ann_model = train_ann(
            X_train_full, y_train_full, config.TASK, n_classes,
            config.ANN_HIDDEN_LAYERS, config.ANN_EPOCHS,
            config.ANN_LR, config.ANN_BATCH_SIZE,
        )
        ann_preds = predict_ann(ann_model, X_test, config.TASK, n_classes)

        plot_confusion_matrices(y_test, classical_preds, ann_preds)

    # Final summary
    #print("\n" + "=" * 60)
    #print("DONE")
    #print("=" * 60)
    for model in df["model"].unique():
        sub = df[df["model"] == model].groupby("fraction")
        if config.TASK == "classification":
            best_frac = sub["test_acc"].mean().idxmax()
            best_val  = sub["test_acc"].mean().max()
            print(f"  {model:12s}  best test accuracy = {best_val:.4f} "
                  f"at {int(best_frac*100)}% data")
        else:
            best_frac = sub["test_r2"].mean().idxmax()
            best_val  = sub["test_r2"].mean().max()
            print(f"  {model:12s}  best test R² = {best_val:.4f} "
                  f"at {int(best_frac*100)}% data")


if __name__ == "__main__":
    main()
