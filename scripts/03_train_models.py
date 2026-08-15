import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
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
# 03_train_models.py
#
# Author:
# Katarzyna Zielińska
#
# Description:
# Split the data into training and test sets, perform feature
# selection using only the training data, and train two
# machine-learning models:
#
# - Logistic Regression
# - Random Forest
#
# The feature selection step is performed inside a pipeline
# to prevent data leakage.
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

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 2. Parameters
# ------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FEATURES = 100


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
# 4. Prepare feature matrix X and target y
# ------------------------------------------------------------

gene_names = expression["Gene_symbol"].tolist()

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
# 5. Check input dimensions
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
# 6. Train/test split
# ------------------------------------------------------------

print("\nCreating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# 7. Save sample split information
# ------------------------------------------------------------

train_samples = X_train.index.tolist()
test_samples = X_test.index.tolist()

split_metadata = pd.DataFrame(
    {
        "sample": train_samples + test_samples,
        "set": (
            ["TRAIN"] * len(train_samples)
            +
            ["TEST"] * len(test_samples)
        )
    }
)

split_metadata.to_csv(
    RESULTS_DIR /
    "train_test_split.csv",
    index=False
)


# ------------------------------------------------------------
# 8. Report split
# ------------------------------------------------------------

print("\nTraining set:")
print(
    y_train.value_counts()
)

print("\nTest set:")
print(
    y_test.value_counts()
)


# ------------------------------------------------------------
# 9. Define Logistic Regression pipeline
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
# 10. Define Random Forest pipeline
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
# 11. Train Logistic Regression
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression training completed."
)


# ------------------------------------------------------------
# 12. Train Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

random_forest_pipeline.fit(
    X_train,
    y_train
)

print(
    "Random Forest training completed."
)


# ------------------------------------------------------------
# 13. Extract selected features
# ------------------------------------------------------------

selector = (
    logistic_pipeline
    .named_steps["feature_selection"]
)

selected_mask = selector.get_support()

selected_features = np.array(
    X_train.columns
)[selected_mask]

feature_scores = selector.scores_[selected_mask]

feature_results = pd.DataFrame(
    {
        "gene": selected_features,
        "f_score": feature_scores
    }
).sort_values(
    "f_score",
    ascending=False
)


# ------------------------------------------------------------
# 14. Save selected features
# ------------------------------------------------------------

feature_results.to_csv(
    RESULTS_DIR /
    "selected_features.csv",
    index=False
)


# ------------------------------------------------------------
# 15. Generate test predictions
# ------------------------------------------------------------

logistic_predictions = (
    logistic_pipeline.predict(X_test)
)

logistic_probabilities = (
    logistic_pipeline.predict_proba(X_test)[:, 1]
)

rf_predictions = (
    random_forest_pipeline.predict(X_test)
)

rf_probabilities = (
    random_forest_pipeline.predict_proba(X_test)[:, 1]
)


# ------------------------------------------------------------
# 16. Save predictions
# ------------------------------------------------------------

predictions = pd.DataFrame(
    {
        "sample": X_test.index,
        "true_class": y_test.values,
        "logistic_prediction":
            logistic_predictions,
        "logistic_probability_GBM":
            logistic_probabilities,
        "random_forest_prediction":
            rf_predictions,
        "random_forest_probability_GBM":
            rf_probabilities
    }
)

predictions.to_csv(
    RESULTS_DIR /
    "test_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 17. Save trained pipelines
# ------------------------------------------------------------

joblib.dump(
    logistic_pipeline,
    RESULTS_DIR /
    "logistic_regression_pipeline.joblib"
)

joblib.dump(
    random_forest_pipeline,
    RESULTS_DIR /
    "random_forest_pipeline.joblib"
)


# ------------------------------------------------------------
# 18. Final summary
# ------------------------------------------------------------

print("\n========================================")
print("Model training completed.")
print("========================================")

print(
    f"\nTraining samples: "
    f"{len(X_train)}"
)

print(
    f"Test samples: "
    f"{len(X_test)}"
)

print(
    f"Selected features: "
    f"{len(selected_features)}"
)

print("\nTop selected genes:")

print(
    feature_results.head(20).to_string(
        index=False
    )
)

print("\nSaved results:")

print(
    RESULTS_DIR /
    "train_test_split.csv"
)

print(
    RESULTS_DIR /
    "selected_features.csv"
)

print(
    RESULTS_DIR /
    "test_predictions.csv"
)

print(
    RESULTS_DIR /
    "logistic_regression_pipeline.joblib"
)

print(
    RESULTS_DIR /
    "random_forest_pipeline.joblib"
)
