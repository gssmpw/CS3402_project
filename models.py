# =============================================================
# models.py — Classical and ANN model definitions + evaluation
# =============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)


# ── Classical Models ──────────────────────────────────────────────────────

def get_classical_model(task, random_state=42):
    """Returns a Logistic Regression (classification) or Ridge (regression)."""
    if task == "classification":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        return Ridge(alpha=1.0)


def evaluate_classical(model, X_tr, y_tr, X_te, y_te, task, n_classes):
    """Fits the classical model and returns a dict of metrics."""
    model.fit(X_tr, y_tr)
    train_pred = model.predict(X_tr)
    test_pred  = model.predict(X_te)
    return _compute_metrics(y_tr, train_pred, y_te, test_pred, task, n_classes)


# ── ANN (MLP) ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, task):
        super().__init__()
        self.task = task
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        if task == "classification" and output_dim == 1:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_ann(X_tr, y_tr, task, n_classes, hidden_layers, epochs, lr, batch_size):
    """Trains an MLP and returns the trained model."""
    input_dim  = X_tr.shape[1]
    output_dim = 1 if (task == "regression" or n_classes == 2) else n_classes

    model     = MLP(input_dim, hidden_layers, output_dim, task)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task == "classification":
        criterion = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    if task == "classification" and n_classes > 2:
        yt = torch.tensor(y_tr, dtype=torch.long)
    elif task == "classification":
        yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    else:
        yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model


def predict_ann(model, X, task, n_classes):
    """Returns predictions from a trained MLP."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
        if task == "classification":
            if n_classes == 2:
                return (out.squeeze().numpy() > 0.5).astype(int)
            else:
                return out.argmax(dim=1).numpy()
        else:
            return out.squeeze().numpy()


def evaluate_ann(X_tr, y_tr, X_te, y_te, task, n_classes,
                 hidden_layers, epochs, lr, batch_size):
    """Trains and evaluates an MLP, returning a dict of metrics."""
    model      = train_ann(X_tr, y_tr, task, n_classes,
                           hidden_layers, epochs, lr, batch_size)
    train_pred = predict_ann(model, X_tr, task, n_classes)
    test_pred  = predict_ann(model, X_te, task, n_classes)
    return _compute_metrics(y_tr, train_pred, y_te, test_pred, task, n_classes)


# ── Shared metric computation ─────────────────────────────────────────────

def _compute_metrics(y_tr, train_pred, y_te, test_pred, task, n_classes):
    if task == "classification":
        avg = "binary" if n_classes == 2 else "macro"
        return {
            "train_acc": accuracy_score(y_tr, train_pred),
            "test_acc":  accuracy_score(y_te, test_pred),
            "train_f1":  f1_score(y_tr, train_pred, average=avg, zero_division=0),
            "test_f1":   f1_score(y_te, test_pred,  average=avg, zero_division=0),
        }
    else:
        return {
            "train_mse": mean_squared_error(y_tr, train_pred),
            "test_mse":  mean_squared_error(y_te, test_pred),
            "train_r2":  r2_score(y_tr, train_pred),
            "test_r2":   r2_score(y_te, test_pred),
        }
