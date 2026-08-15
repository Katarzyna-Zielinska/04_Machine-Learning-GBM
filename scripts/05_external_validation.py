import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Project:
# Machine Learning Classification of Glioblastoma
#
# Script:
# 05_external_validation.py
#
# Author:
# Katarzyna Zielińska
#
# Description:
# Assess the stability and generalization of the machine
# learning models using 5-fold stratified cross-validation.
#
# Feature selection is performed inside each fold to prevent
# data leakage.
#
# Models:
# - Logistic Regression
# - Random Forest
# ============================================================


# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

EXPRESSION_FILE = (
    PROJECT_DIR /
    "data/processed/"
    "GSE4290_GBM_vs_NonTumor_gene_expression.csv"
)

METADATA_FILE = (
    PROJECT_DIR /
    "data/processed/"
    "GSE4290_GBM_vs_NonTumor_metadata.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"

FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(
    exist_ok=True
)

FIGURES_DIR.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 2. Parameters
# ------------------------------------------------------------

N_FEATURES = 100

N_SPLITS = 5

RANDOM_STATE = 42


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------

print("Loading gene-level expression data...")

expression = pd.read_csv(
    EXPRESSION_FILE
)

metadata = pd.read_csv(
    METADATA_FILE
)


# ------------------------------------------------------------
# 4. Prepare feature matrix and target
# ------------------------------------------------------------

gene_names = expression[
    "Gene_symbol"
].tolist()

X = expression[
    metadata["sample"].tolist()
].T

X.columns = gene_names

y = metadata[
    "condition"
].map(
    {
        "NON_TUMOR": 0,
        "GBM": 1
    }
)


# ------------------------------------------------------------
# 5. Check dataset
# ------------------------------------------------------------

print("\nDataset dimensions:")

print(
    f"Samples: {X.shape[0]}"
)

print(
    f"Genes: {X.shape[1]}"
)

print("\nClass distribution:")

print(
    metadata["condition"].value_counts()
)


# ------------------------------------------------------------
# 6. Define cross-validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# 7. Define Logistic Regression pipeline
# ------------------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=N_FEATURES
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ------------------------------------------------------------
# 8. Define Random Forest pipeline
# ------------------------------------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=N_FEATURES
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt"
            )
        )
    ]
)


# ------------------------------------------------------------
# 9. Define evaluation metrics
# ------------------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ------------------------------------------------------------
# 10. Cross-validation function
# ------------------------------------------------------------

def run_cross_validation(
    model,
    model_name
):

    print(
        f"\nRunning 5-fold CV: "
        f"{model_name}"
    )

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    summary = {}

    for metric in scoring.keys():

        values = results[
            f"test_{metric}"
        ]

        summary[
            f"{metric}_mean"
        ] = values.mean()

        summary[
            f"{metric}_std"
        ] = values.std()

    return results, summary


# ------------------------------------------------------------
# 11. Run Logistic Regression CV
# ------------------------------------------------------------

logistic_results, logistic_summary = (
    run_cross_validation(
        logistic_pipeline,
        "Logistic Regression"
    )
)


# ------------------------------------------------------------
# 12. Run Random Forest CV
# ------------------------------------------------------------

rf_results, rf_summary = (
    run_cross_validation(
        random_forest_pipeline,
        "Random Forest"
    )
)


# ------------------------------------------------------------
# 13. Create summary table
# ------------------------------------------------------------

summary_table = pd.DataFrame(
    [
        logistic_summary,
        rf_summary
    ],
    index=[
        "Logistic Regression",
        "Random Forest"
    ]
)


# ------------------------------------------------------------
# 14. Print results
# ------------------------------------------------------------

print("\n========================================")
print("5-FOLD CROSS-VALIDATION RESULTS")
print("========================================\n")

for model_name in summary_table.index:

    print(model_name)

    for metric in scoring.keys():

        mean_value = summary_table.loc[
            model_name,
            f"{metric}_mean"
        ]

        std_value = summary_table.loc[
            model_name,
            f"{metric}_std"
        ]

        print(
            f"  {metric.upper():<10}: "
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
        )

    print()


# ------------------------------------------------------------
# 15. Save summary table
# ------------------------------------------------------------

summary_table.to_csv(
    RESULTS_DIR /
    "cross_validation_summary.csv"
)


# ------------------------------------------------------------
# 16. Save fold-level results
# ------------------------------------------------------------

logistic_fold_results = pd.DataFrame(
    logistic_results
)

logistic_fold_results["model"] = (
    "Logistic Regression"
)

rf_fold_results = pd.DataFrame(
    rf_results
)

rf_fold_results["model"] = (
    "Random Forest"
)

fold_results = pd.concat(
    [
        logistic_fold_results,
        rf_fold_results
    ],
    ignore_index=True
)

fold_results.to_csv(
    RESULTS_DIR /
    "cross_validation_fold_results.csv",
    index=False
)


# ------------------------------------------------------------
# 17. Create CV comparison plot
# ------------------------------------------------------------

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

logistic_means = [
    logistic_summary[
        f"{metric}_mean"
    ]
    for metric in metrics
]

rf_means = [
    rf_summary[
        f"{metric}_mean"
    ]
    for metric in metrics
]

logistic_stds = [
    logistic_summary[
        f"{metric}_std"
    ]
    for metric in metrics
]

rf_stds = [
    rf_summary[
        f"{metric}_std"
    ]
    for metric in metrics
]


x = np.arange(
    len(metrics)
)

width = 0.35

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.bar(
    x - width / 2,
    logistic_means,
    width,
    yerr=logistic_stds,
    capsize=4,
    label="Logistic Regression"
)

ax.bar(
    x + width / 2,
    rf_means,
    width,
    yerr=rf_stds,
    capsize=4,
    label="Random Forest"
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
)

ax.set_ylim(
    0,
    1.1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "5-Fold Cross-Validation Performance"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_DIR /
    "cross_validation_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 18. Create ROC-AUC stability plot
# ------------------------------------------------------------

logistic_auc = (
    logistic_results["test_roc_auc"]
)

rf_auc = (
    rf_results["test_roc_auc"]
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.boxplot(
    [
        logistic_auc,
        rf_auc
    ],
    tick_labels=[
        "Logistic Regression",
        "Random Forest"
    ]
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_title(
    "ROC-AUC Across 5 Cross-Validation Folds"
)

ax.set_ylim(
    0,
    1.05
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR /
    "cross_validation_ROC_AUC.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 19. Final summary
# ------------------------------------------------------------

print("\n========================================")
print("CROSS-VALIDATION COMPLETED")
print("========================================")

print("\nSaved results:")

print(
    RESULTS_DIR /
    "cross_validation_summary.csv"
)

print(
    RESULTS_DIR /
    "cross_validation_fold_results.csv"
)

print("\nSaved figures:")

print(
    FIGURES_DIR /
    "cross_validation_model_comparison.png"
)

print(
    FIGURES_DIR /
    "cross_validation_ROC_AUC.png"
)
