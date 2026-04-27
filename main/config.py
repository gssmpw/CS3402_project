# =============================================================
# config.py — Edit this file for each run
# =============================================================

# --- Datasets (define exactly 4) ---
DATASETS = [
    {
        "name":       "StudentsPerformance",
        "csv_path":   r"\Users\gsmit\Documents\CS3402_project\datasets\StudentsPerformance.csv",
        "target_col": "gender",
        "task":       "classification",   # "classification" or "regression"
        "drop_cols":  [],
    },
    {
        "name":       "heart",
        "csv_path":   r"\Users\gsmit\Documents\CS3402_project\datasets\heart.csv",
        "target_col": "target",
        "task":       "classification",
        "drop_cols":  [],
    },
    {
        "name":       "healthcare-dataset-stroke-data",
        "csv_path":   r"\Users\gsmit\Documents\CS3402_project\datasets\healthcare-dataset-stroke-data.csv",
        "target_col": "stroke",
        "task":       "classification",
        "drop_cols":  ["id"],
    },
    {
        "name":       "AIvsHumanTextDataset",
        "csv_path":   r"\Users\gsmit\Documents\CS3402_project\datasets\AIvsHumanTextDataset.csv",
        "target_col": "label",
        "task":       "classification",
        "drop_cols":  ["text_id"],
    },
]

# --- Experiment ---
TRAIN_FRACTIONS = [0.10, 0.30, 0.50, 1.00]
N_RUNS          = 3
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# --- ANN Hyperparameters ---
ANN_HIDDEN_LAYERS = [128, 64]
ANN_EPOCHS        = 50
ANN_LR            = 1e-3
ANN_BATCH_SIZE    = 64

# --- Output ---
RESULTS_CSV = "results.csv"
PLOT_FILE   = "learning_curves.png"