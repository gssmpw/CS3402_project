# =============================================================
# run_experiment.py — Main entry point
#
# Usage:
#   python run_experiment.py
#
# Edit config.py first to configure your 4 datasets.
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
from plots import plot_all_datasets, plot_confusion_matrices, print_summary_table

warnings.filterwarnings("ignore")
np.random.seed(config.RANDOM_STATE)


# ── 1. Load & preprocess ──────────────────────────────────────────────────

def load_data(dataset_cfg):
    """Load and preprocess a single dataset dict from config.DATASETS."""
    df = pd.read_csv(dataset_cfg["csv_path"])

    drop_cols  = dataset_cfg.get("drop_cols", [])
    target_col = dataset_cfg["target_col"]
    task       = dataset_cfg["task"]

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical feature columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target
    n_classes = 1
    if task == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))
        n_classes = len(le.classes_)
    else:
        y = y.astype(float).values

    X = X.astype(float).values
    return X, y, n_classes


# ── 2. Experiment loop ────────────────────────────────────────────────────

def run_experiments(X_train_full, y_train_full, X_test, y_test, n_classes, task):
    records = []
    fracs   = config.TRAIN_FRACTIONS

    for frac in fracs:
        for run in range(config.N_RUNS):
            n_samples = max(10, int(len(X_train_full) * frac))
            idx  = np.random.choice(len(X_train_full), size=n_samples, replace=False)
            X_tr = X_train_full[idx]
            y_tr = y_train_full[idx]

            # Classical
            clf    = get_classical_model(task, config.RANDOM_STATE)
            c_mets = evaluate_classical(clf, X_tr, y_tr, X_test, y_test, task, n_classes)
            records.append({"model": "Classical", "fraction": frac,
                            "run": run + 1, "n_train": n_samples, **c_mets})

            # ANN
            a_mets = evaluate_ann(
                X_tr, y_tr, X_test, y_test,
                task, n_classes,
                config.ANN_HIDDEN_LAYERS, config.ANN_EPOCHS,
                config.ANN_LR, config.ANN_BATCH_SIZE,
            )
            records.append({"model": "ANN (MLP)", "fraction": frac,
                            "run": run + 1, "n_train": n_samples, **a_mets})

    return pd.DataFrame(records)


# ── 3. Per-dataset runner ─────────────────────────────────────────────────

def process_dataset(dataset_cfg):
    """Run the full pipeline for one dataset. Returns result dict for plotting."""
    task = dataset_cfg["task"]
    name = dataset_cfg["name"]

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Target  : {dataset_cfg['target_col']}")
    print(f"  Task    : {task.upper()}")
    print(f"{'='*60}")

    X, y, n_classes = load_data(dataset_cfg)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if task == "classification" else None,
    )

    scaler       = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test       = scaler.transform(X_test)

    print(f"  Train pool: {X_train_full.shape[0]:,}  |  Test (fixed): {X_test.shape[0]:,}")

    df = run_experiments(X_train_full, y_train_full, X_test, y_test, n_classes, task)

    # Save raw results per dataset (slug the name for safe filenames)
    safe_name = name.lower().replace(" ", "_")
    csv_path  = f"results_{safe_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved raw results → {csv_path}")

    # Per-dataset summary
    print_summary_table(df)

    # Best performance summary
    for model in df["model"].unique():
        sub = df[df["model"] == model].groupby("fraction")
        if task == "classification":
            best_frac = sub["test_acc"].mean().idxmax()
            best_val  = sub["test_acc"].mean().max()
            print(f"  {model:12s}  best test accuracy = {best_val:.4f} "
                  f"at {int(best_frac*100)}% data")
        else:
            best_frac = sub["test_r2"].mean().idxmax()
            best_val  = sub["test_r2"].mean().max()
            print(f"  {model:12s}  best test R² = {best_val:.4f} "
                  f"at {int(best_frac*100)}% data")

    # Confusion matrices (classification only, full training data)
    if task == "classification":
        clf = get_classical_model(task, config.RANDOM_STATE)
        clf.fit(X_train_full, y_train_full)
        classical_preds = clf.predict(X_test)

        ann_model = train_ann(
            X_train_full, y_train_full, task, n_classes,
            config.ANN_HIDDEN_LAYERS, config.ANN_EPOCHS,
            config.ANN_LR, config.ANN_BATCH_SIZE,
        )
        ann_preds = predict_ann(ann_model, X_test, task, n_classes)

        cm_path = f"confusion_matrices_{safe_name}.png"
        plot_confusion_matrices(y_test, classical_preds, ann_preds, save_path=cm_path)
        print(f"  Saved confusion matrices → {cm_path}")

    return {"name": name, "df": df, "task": task}


# ── 4. Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ML Experiment: Training Data Size vs. Performance")
    print(f"  Running {len(config.DATASETS)} datasets")
    print("=" * 60)

    dataset_results = []
    for dataset_cfg in config.DATASETS:
        result = process_dataset(dataset_cfg)
        dataset_results.append(result)

    # Combined learning-curve plot (4 rows × 4 panels)
    print(f"\nGenerating combined plot → {config.PLOT_FILE}")
    plot_all_datasets(dataset_results, save_path=config.PLOT_FILE)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()