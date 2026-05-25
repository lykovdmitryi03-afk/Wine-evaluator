"""
Wine Quality Classification with Pandas, NumPy, and PyTorch

Task:
Predict whether a red wine is "good" based on chemical measurements.
A wine is labeled good if quality >= 7, otherwise not good.

Libraries used:
- Pandas: load and inspect CSV data
- NumPy: split data, scale features, compute metrics helpers
- PyTorch: build and train a neural network classifier

Run:
    python wine_quality_pytorch.py
"""

import os
import urllib.request
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_PATH = "winequality-red.csv"
MODEL_PATH = "wine_quality_model.pt"
PREDICTIONS_PATH = "wine_quality_predictions.csv"
DECISION_THRESHOLD = 0.4


# -----------------------------
# 1. Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2. Data loading
# -----------------------------
def download_dataset_if_needed():
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"Saved to {DATA_PATH}")


def load_data():
    download_dataset_if_needed()
    df = pd.read_csv(DATA_PATH, sep=";")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nQuality distribution:")
    print(df["quality"].value_counts().sort_index())

    # Convert original quality score into binary target:
    # 1 = good wine, 0 = not good wine
    df["good_wine"] = (df["quality"] >= 7).astype(int)

    feature_cols = [col for col in df.columns if col not in ["quality", "good_wine"]]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["good_wine"].to_numpy(dtype=np.float32).reshape(-1, 1)

    return df, feature_cols, X, y


# -----------------------------
# 3. NumPy train/val/test split
# -----------------------------
def train_val_test_split(X, y, train_ratio=0.70, val_ratio=0.15, seed=42):
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx], test_idx


# -----------------------------
# 4. NumPy feature scaling
# -----------------------------
def standardize_train_val_test(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled, mean, std


# -----------------------------
# 5. PyTorch model
# -----------------------------
class WineClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    def forward(self, x):
        # Raw logits. BCEWithLogitsLoss applies sigmoid internally.
        return self.network(x)


# -----------------------------
# 6. Metrics
# -----------------------------
def binary_metrics(y_true, probabilities, threshold=DECISION_THRESHOLD):
    y_true = y_true.reshape(-1).astype(int)
    y_pred = (probabilities.reshape(-1) >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@torch.no_grad()
def predict_probabilities(model, X, device):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    logits = model(X_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def predict_custom_wine(model, feature_cols, mean, std, device, threshold=DECISION_THRESHOLD):
    """
    Ask the user to enter wine chemical values, then predict whether the wine is good.
    """

    print("\nEnter wine chemical values.")
    print("Use decimal numbers. Example: 7.4, 0.70, 1.9, etc.\n")

    values = []

    for col in feature_cols:
        while True:
            try:
                value = float(input(f"{col}: "))
                values.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Convert user input to NumPy array
    user_wine = np.array(values, dtype=np.float32).reshape(1, -1)

    # Standardize using the same mean and std from training data
    user_wine_scaled = (user_wine - mean) / std

    # Predict probability
    probability = predict_probabilities(model, user_wine_scaled, device)[0][0]

    # Convert probability to class prediction
    predicted_class = int(probability >= threshold)

    print("\nPrediction result:")
    print(f"Probability of being a good wine: {probability:.4f}")
    print(f"Decision threshold: {threshold}")

    if predicted_class == 1:
        print("Prediction: GOOD wine")
    else:
        print("Prediction: NOT good wine")

    return probability, predicted_class

# -----------------------------
# 7. Training loop
# -----------------------------
def train_model(model, train_loader, X_val, y_val, pos_weight, device, epochs=250, lr=0.004):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1
    best_state = None
    train_losses = []
    val_f1_scores = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_probs = predict_probabilities(model, X_val, device)
        val_metrics = binary_metrics(y_val, val_probs)
        train_losses.append(train_loss)
        val_f1_scores.append(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation F1 during training: {best_val_f1:.4f}")
    return model, train_losses, val_f1_scores

def plot_training_curves(train_losses, val_f1_scores):
    """
    Plot training loss and validation F1-score over epochs.
    Saves the plot as training_curves.png.
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Validation F1 curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1_scores, label="Validation F1-score", color="green")
    plt.title("Validation F1-score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()

    print("Saved training curves to training_curves.png")

def plot_confusion_matrix(metrics):
    """
    Plot and save a confusion matrix using the final test metrics.
    Matrix layout:

                Predicted Not Good   Predicted Good
    Actual Not Good        TN              FP
    Actual Good            FN              TP
    """

    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Not Good", "Good"])
    ax.set_yticklabels(["Not Good", "Good"])

    # Add numbers inside boxes
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold"
            )

    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    print("Saved confusion matrix to confusion_matrix.png")

# -----------------------------
# 8. Main project pipeline
# -----------------------------
def main():
    set_seed(42)

    df, feature_cols, X, y = load_data()

    X_train, y_train, X_val, y_val, X_test, y_test, test_idx = train_val_test_split(X, y)
    X_train, X_val, X_test, mean, std = standardize_train_val_test(X_train, X_val, X_test)

    print("\nClass balance:")
    print("Training positives:", int(y_train.sum()), "/", len(y_train))
    print("Validation positives:", int(y_val.sum()), "/", len(y_val))
    print("Test positives:", int(y_test.sum()), "/", len(y_test))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = WineClassifier(input_dim=X_train.shape[1]).to(device)

    # Because good wines are less common, use positive-class weighting.
    num_positive = y_train.sum()
    num_negative = len(y_train) - num_positive
    pos_weight_value = num_negative / (num_positive + 1e-8)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    model, train_losses, val_f1_scores = train_model(
        model=model,
        train_loader=train_loader,
        X_val=X_val,
        y_val=y_val,
        pos_weight=pos_weight,
        device=device,
        epochs=250,
        lr=0.004,
    )
    plot_training_curves(train_losses, val_f1_scores)

    test_probs = predict_probabilities(model, X_test, device)
    test_metrics = binary_metrics(y_test, test_probs)

    print("\nFinal test metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
            
    plot_confusion_matrix(test_metrics)

    # Save model checkpoint and preprocessing values.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
            "decision_threshold": DECISION_THRESHOLD,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH}")

    # Save predictions for inspection in Pandas/Excel.
    test_rows = df.iloc[test_idx].copy()
    test_rows["predicted_probability_good"] = test_probs.reshape(-1)
    test_rows["predicted_good_wine"] = (test_probs.reshape(-1) >= DECISION_THRESHOLD).astype(int)
    test_rows.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved test predictions to {PREDICTIONS_PATH}")

    # Example prediction on one test sample.
    example = test_rows.iloc[0]
    print("\nExample prediction:")
    print(example[feature_cols + ["quality", "good_wine", "predicted_probability_good", "predicted_good_wine"]])
    answer = input("\nDo you want to enter your own wine values? (y/n): ").strip().lower()

    if answer == "y":
        predict_custom_wine(
            model=model,
            feature_cols=feature_cols,
            mean=mean,
            std=std,
            device=device,
            threshold=DECISION_THRESHOLD,
        )    



if __name__ == "__main__":
    main()
