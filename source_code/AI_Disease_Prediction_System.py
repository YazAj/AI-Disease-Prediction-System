
"""
AI-Based Disease Prediction System
Student: AHMAD YOUSEF NEMER AYAAD
Student ID: 243039

This script trains and compares multiple machine learning models using the
Breast Cancer Wisconsin Diagnostic dataset included in scikit-learn.
Run in Spyder: open this file and press Run. Output files will be saved in
figures/, results/, models/, and dataset/ folders.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

for folder in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR, DATASET_DIR]:
    os.makedirs(folder, exist_ok=True)


def load_data():
    """Load the dataset and create a clean pandas DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    df.to_csv(os.path.join(DATASET_DIR, "breast_cancer_dataset.csv"), index=False)
    return df, data


def perform_eda(df):
    """Create exploratory analysis outputs and visualizations."""
    summary = df.describe().T
    summary.to_csv(os.path.join(RESULTS_DIR, "data_summary.csv"))

    missing_values = df.isnull().sum()
    missing_values.to_csv(os.path.join(RESULTS_DIR, "missing_values.csv"))

    # Class distribution
    plt.figure(figsize=(6, 4))
    df["diagnosis"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=200)
    plt.close()

    # Correlation heatmap for top 10 features with target
    correlations = df.drop(columns=["diagnosis"]).corr(numeric_only=True)["target"].abs().sort_values(ascending=False)
    top_features = correlations.index[1:11].tolist() + ["target"]
    corr_matrix = df[top_features].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index, fontsize=8)
    plt.title("Correlation Heatmap for Top Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=200)
    plt.close()

    # Boxplot for selected important features
    selected = ["mean radius", "mean texture", "mean concavity", "worst radius"]
    for feature in selected:
        plt.figure(figsize=(6, 4))
        df.boxplot(column=feature, by="diagnosis")
        plt.title(f"{feature.title()} by Diagnosis")
        plt.suptitle("")
        plt.xlabel("Diagnosis")
        plt.ylabel(feature.title())
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"boxplot_{feature.replace(' ', '_')}.png"), dpi=200)
        plt.close()


def train_models(df):
    """Train models, evaluate them, and save results."""
    X = df.drop(columns=["target", "diagnosis"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42
        ),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
        ])
    }

    metrics = []
    reports = {}
    best_model_name = None
    best_f1 = -1
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
            "CV F1 Mean": float(np.mean(cv_scores)),
            "CV F1 Std": float(np.std(cv_scores))
        }
        metrics.append(row)
        reports[name] = classification_report(
            y_test, predictions,
            target_names=["malignant", "benign"],
            output_dict=True
        )

        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["malignant", "benign"]
        )
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), dpi=200)
        plt.close()

        if row["F1 Score"] > best_f1:
            best_f1 = row["F1 Score"]
            best_model_name = name
            best_model = model

    metrics_df = pd.DataFrame(metrics).sort_values(by="F1 Score", ascending=False)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "classification_reports.json"), "w") as f:
        json.dump(reports, f, indent=4)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

    # Metrics comparison plot
    plot_df = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
    plot_df.plot(kind="bar", figsize=(9, 5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0.85, 1.01)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=200)
    plt.close()

    # Random forest feature importance
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    importances.to_csv(os.path.join(RESULTS_DIR, "top_feature_importance.csv"))
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh")
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=200)
    plt.close()

    return metrics_df, best_model_name


def main():
    df, data = load_data()
    print("Dataset loaded successfully.")
    print(f"Samples: {df.shape[0]}, Features: {len(data.feature_names)}")
    print(df.head())

    perform_eda(df)
    metrics_df, best_model_name = train_models(df)

    print("\nModel Metrics:")
    print(metrics_df)
    print(f"\nBest model based on F1 Score: {best_model_name}")
    print("\nProject outputs saved successfully.")


if __name__ == "__main__":
    main()
