# =============================================================
# config.py — Edit this file for each dataset you run
# =============================================================

# --- Dataset ---
CSV_PATH   = r"\Users\gsmit\Documents\CS3402_project\datasets\StudentsPerformance.csv"  # Path to your CSV file
TARGET_COL = "gender"            # Column to predict
TASK       = "regression"    # "classification" or "regression"
DROP_COLS  = []                  # Columns to ignore, e.g. ["id", "name"]

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
