import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def ensure_results_dir(results_dir: Path) -> None:
    """Create results/ folder if it doesn't exist."""
    results_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from CSV.
    Assumes label column contains class labels (0/1 or benign/ransomware).
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in dataset columns: {list(df.columns)}"
        )

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Convert labels if they are strings like "benign"/"ransomware"
    if y.dtype == object:
        y_lower = y.astype(str).str.lower().str.strip()
        # Common mappings
        mapping = {
            "benign": 0,
            "goodware": 0,
            "normal": 0,
            "legit": 0,
            "legitimate": 0,
            "ransomware": 1,
            "malware": 1
        }
        if set(y_lower.unique()).issubset(set(mapping.keys())):
            y = y_lower.map(mapping)
        else:
            raise ValueError(
                "Label column appears to be non-numeric strings but does not match "
                "expected values (e.g., benign/ransomware). Please convert to 0/1."
            )

    # Force numeric features
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are fully NaN after conversion
    X = X.dropna(axis=1, how="all")

    # Fill remaining missing values (simple strategy)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def generate_demo_dataset(n_samples: int = 1500, n_features: int = 25, random_state: int = 42):
    """
    Generates a synthetic dataset so the pipeline can run even if
    you don't have the real dataset wired into the repo yet.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.55, 0.45],
        random_state=random_state
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(y, name="label")
    return X, y


def evaluate_model(model_name: str, model, X_test, y_test, results_dir: Path) -> None:
    """Evaluate model and save confusion matrix plot."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Ransomware"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")

    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()

    output_path = results_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved confusion matrix to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression and Random Forest on static ransomware features."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to CSV dataset file. If not provided, a demo dataset is generated."
    )

    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of label column in CSV (default: label)."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size (default: 0.2)."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)."
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    ensure_results_dir(results_dir)

    # Load dataset
    if args.data.strip():
        csv_path = Path(args.data)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")

        print(f"Loading dataset from: {csv_path}")
        X, y = load_dataset(csv_path, args.label_col)
    else:
        print("No dataset provided. Generating demo dataset...")
        X, y = generate_demo_dataset(random_state=args.random_state)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    print("\nDataset Summary")
    print("-" * 70)
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Label distribution (train):\n{pd.Series(y_train).value_counts(normalize=True)}")

    # Logistic Regression pipeline
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=250,
        random_state=args.random_state,
        n_jobs=-1
    )

    # Train models
    print("\nTraining Logistic Regression...")
    lr_model.fit(X_train, y_train)

    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)

    # Evaluate + save plots
    evaluate_model("Logistic Regression", lr_model, X_test, y_test, results_dir)
    evaluate_model("Random Forest", rf_model, X_test, y_test, results_dir)

    print("\nDone ✅")


if __name__ == "__main__":
    main()
