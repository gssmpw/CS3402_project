# =============================================================
# plots.py — Learning curves, confusion matrix, summary table
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves(df, task, save_path="learning_curves.png"):
    fracs    = sorted(df["fraction"].unique())
    x_labels = [f"{int(f*100)}%" for f in fracs]
    models   = df["model"].unique()
    colors   = {"Classical": "#F59E0B", "ANN (MLP)": "#14B8A6"}

    if task == "classification":
        metrics = [
            ("test_acc",  "Test Accuracy"),
            ("test_f1",   "Test F1-Score"),
            ("train_acc", "Train Accuracy"),
            ("train_f1",  "Train F1-Score"),
        ]
    else:
        metrics = [
            ("test_mse",  "Test MSE"),
            ("test_r2",   "Test R²"),
            ("train_mse", "Train MSE"),
            ("train_r2",  "Train R²"),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Learning Curves — Effect of Training Data Size",
                 fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for ax, (col, label) in zip(axes[:3], metrics[:3]):
        for model in models:
            sub   = df[df["model"] == model].groupby("fraction")[col]
            means = sub.mean().reindex(fracs).values
            stds  = sub.std().reindex(fracs).values
            ax.plot(x_labels, means, marker="o", label=model,
                    color=colors.get(model), linewidth=2.5, markersize=7)
            ax.fill_between(x_labels, means - stds, means + stds,
                            alpha=0.15, color=colors.get(model))
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Training Data Fraction")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Fourth panel: generalization gap
    ax = axes[3]
    for model in models:
        grp = df[df["model"] == model].groupby("fraction")
        if task == "classification":
            gap = (grp["train_acc"].mean() - grp["test_acc"].mean()).reindex(fracs).values
        else:
            gap = (grp["test_mse"].mean()  - grp["train_mse"].mean()).reindex(fracs).values
        ax.plot(x_labels, gap, marker="s", linestyle="--", linewidth=2,
                label=f"{model}", color=colors.get(model))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Generalization Gap (Train − Test)", fontweight="bold")
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    #print(f"  Saved: {save_path}")


def plot_confusion_matrices(y_test, classical_preds, ann_preds,
                            save_path="confusion_matrices.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Confusion Matrices (100% training data)", fontweight="bold")

    for ax, preds, title, cmap in zip(
        axes,
        [classical_preds, ann_preds],
        ["Classical Model", "ANN (MLP)"],
        ["Blues", "Greens"],
    ):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    #print(f"  Saved: {save_path}")


def print_summary_table(df):
    """Prints mean ± SD for every metric, grouped by model and fraction."""
    metric_cols = [c for c in df.columns
                   if c not in ("model", "fraction", "run", "n_train")]

    rows = []
    for (model, frac), grp in df.groupby(["model", "fraction"]):
        row = {
            "Model":    model,
            "Fraction": f"{int(frac * 100)}%",
            "N Train":  int(grp["n_train"].mean()),
        }
        for col in metric_cols:
            row[col] = f"{grp[col].mean():.4f} ± {grp[col].std():.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    #print("\n" + "=" * 70)
    #print("SUMMARY TABLE (mean ± std across runs)")
    #print("=" * 70)
    #print(summary.to_string(index=False))
    return summary