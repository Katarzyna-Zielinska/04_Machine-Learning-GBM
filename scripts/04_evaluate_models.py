import pandas as pd
import numpy as np
from pathlib import Path
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


# ============================================================
# Project:
# Machine Learning Classification of Glioblastoma
#
# Script:
# 04_evaluate_models.py
#
# Author:
# Katarzyna Zielińska
#
# Description:
# Evaluate Logistic Regression and Random Forest models
# on the independent test set.
#
# Metrics:
# - Accuracy
# - Precision
# - Recall
# - F1-score
# - ROC-AUC
#
# Visualizations:
# - Confusion matrices
# - ROC curve
# - Model performance comparison
# ============================================================


# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 2. Load trained models
# ------------------------------------------------------------

print("Loading trained models...")

logistic_model = joblib.load(
    RESULTS_DIR /
    "logistic_regression_pipeline.joblib"
)

random_forest_model = joblib.load(
    RESULTS_DIR /
    "random_forest_pipeline.joblib"
)


# ------------------------------------------------------------
# 3. Load test predictions
# ------------------------------------------------------------

print("Loading test predictions...")

predictions = pd.read_csv(
    RESULTS_DIR /
    "test_predictions.csv"
)


# ------------------------------------------------------------
# 4. Extract true labels and predictions
# ------------------------------------------------------------

y_true = predictions["true_class"]

logistic_pred = (
    predictions["logistic_prediction"]
)

logistic_prob = (
    predictions["logistic_probability_GBM"]
)

rf_pred = (
    predictions["random_forest_prediction"]
)

rf_prob = (
    predictions["random_forest_probability_GBM"]
)


# ------------------------------------------------------------
# 5. Define evaluation function
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1-score":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_true,
                y_probability
            )
    }


# ------------------------------------------------------------
# 6. Calculate model metrics
# ------------------------------------------------------------

print("\nCalculating model performance...")

logistic_metrics = calculate_metrics(
    y_true,
    logistic_pred,
    logistic_prob
)

rf_metrics = calculate_metrics(
    y_true,
    rf_pred,
    rf_prob
)


# ------------------------------------------------------------
# 7. Create metrics table
# ------------------------------------------------------------

metrics = pd.DataFrame(
    [
        logistic_metrics,
        rf_metrics
    ],
    index=[
        "Logistic Regression",
        "Random Forest"
    ]
)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================\n")

print(
    metrics.round(4).to_string()
)


# ------------------------------------------------------------
# 8. Save metrics
# ------------------------------------------------------------

metrics.to_csv(
    RESULTS_DIR /
    "model_performance.csv"
)


# ------------------------------------------------------------
# 9. Classification reports
# ------------------------------------------------------------

print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================\n")

print(
    classification_report(
        y_true,
        logistic_pred,
        target_names=[
            "NON_TUMOR",
            "GBM"
        ],
        zero_division=0
    )
)


print("\n========================================")
print("RANDOM FOREST")
print("========================================\n")

print(
    classification_report(
        y_true,
        rf_pred,
        target_names=[
            "NON_TUMOR",
            "GBM"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 10. Confusion matrix function
# ------------------------------------------------------------

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "NON_TUMOR",
            "GBM"
        ]
    )

    ax.set_yticklabels(
        [
            "NON_TUMOR",
            "GBM"
        ]
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ------------------------------------------------------------
# 11. Save confusion matrices
# ------------------------------------------------------------

print("\nCreating confusion matrices...")

save_confusion_matrix(
    y_true,
    logistic_pred,
    "Logistic Regression",
    "confusion_matrix_logistic_regression.png"
)

save_confusion_matrix(
    y_true,
    rf_pred,
    "Random Forest",
    "confusion_matrix_random_forest.png"
)


# ------------------------------------------------------------
# 12. ROC curves
# ------------------------------------------------------------

print("Creating ROC curve...")

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_true,
    logistic_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_true,
    rf_prob
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.plot(
    logistic_fpr,
    logistic_tpr,
    label=(
        f"Logistic Regression "
        f"(AUC = {logistic_metrics['ROC-AUC']:.3f})"
    )
)

ax.plot(
    rf_fpr,
    rf_tpr,
    label=(
        f"Random Forest "
        f"(AUC = {rf_metrics['ROC-AUC']:.3f})"
    )
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curve - GBM Classification"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_DIR /
    "ROC_curve_models.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 13. Model performance comparison
# ------------------------------------------------------------

print("Creating model comparison plot...")

plot_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

x = np.arange(
    len(plot_metrics)
)

width = 0.35

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.bar(
    x - width / 2,
    [
        logistic_metrics[m]
        for m in plot_metrics
    ],
    width,
    label="Logistic Regression"
)

ax.bar(
    x + width / 2,
    [
        rf_metrics[m]
        for m in plot_metrics
    ],
    width,
    label="Random Forest"
)

ax.set_xticks(x)

ax.set_xticklabels(
    plot_metrics
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Machine Learning Model Performance"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_DIR /
    "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------

print("\n========================================")
print("MODEL EVALUATION COMPLETED")
print("========================================")

print("\nSaved results:")

print(
    RESULTS_DIR /
    "model_performance.csv"
)

print("\nSaved figures:")

print(
    FIGURES_DIR /
    "confusion_matrix_logistic_regression.png"
)

print(
    FIGURES_DIR /
    "confusion_matrix_random_forest.png"
)

print(
    FIGURES_DIR /
    "ROC_curve_models.png"
)

print(
    FIGURES_DIR /
    "model_performance_comparison.png"
)
