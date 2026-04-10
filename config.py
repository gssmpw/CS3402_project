# =============================================================
# config.py — Edit this file for each dataset you run
# =============================================================

# --- Dataset ---
CSV_PATH   = "bank_transactions_data_2.csv"  # Path to your CSV file
TARGET_COL = "TransactionAmount"            # Column to predict
TASK       = "regression"    # "classification" or "regression"
DROP_COLS  = ["TransactionID", "AccountID"]                  # Columns to ignore, e.g. ["id", "name"]

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
