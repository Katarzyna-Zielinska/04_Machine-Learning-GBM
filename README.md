Machine Learning Classification of Glioblastoma

A machine learning project for classification of glioblastoma (GBM) versus non-tumor brain samples using gene expression data.

The project demonstrates a complete machine learning workflow for transcriptomic data, including data preparation, probe-to-gene annotation, feature selection, model training, evaluation, and cross-validation.

Biological Question

Can gene expression profiles be used to distinguish glioblastoma (GBM) samples from non-tumor brain samples using machine learning?

Dataset

The analysis uses the publicly available GSE4290 dataset from the NCBI Gene Expression Omnibus (GEO).

The dataset contains gene expression measurements generated using the Affymetrix Human Genome U133 Plus 2.0 Array (GPL570).

For this project, the analysis was restricted to two diagnostic groups:

Glioblastoma, grade 4 (GBM): 77 samples
Non-tumor: 23 samples

Total:

100 samples
54,613 probes in the original expression matrix
22,189 genes after probe annotation and aggregation

The original GEO dataset is not included in this repository.

Workflow

GEO GSE4290 → Probe-level expression matrix → Sample selection (GBM vs Non-Tumor) → GPL570 probe annotation → Probe-to-gene mapping → Multiple probes aggregated per gene → Gene-level expression matrix → Feature selection → Top 100 genes → Logistic Regression and Random Forest → Model evaluation → 5-fold stratified cross-validation.

Methods
1. Data Preparation

The GEO series matrix was downloaded from NCBI GEO.

The analysis was restricted to samples annotated as glioblastoma, grade 4, and non-tumor.

The resulting dataset contained 100 samples.

2. Probe Annotation

The original Affymetrix expression matrix contained probe-level measurements.

GPL570 annotation was used to map probes to gene symbols.

Multiple probes corresponding to the same gene were aggregated to obtain a gene-level expression matrix.

After annotation and aggregation, 22,189 genes were available for downstream analysis.

3. Feature Selection

The original gene expression matrix contains thousands of genes, while the number of samples is relatively small.

To reduce dimensionality, univariate statistical feature selection was performed using SelectKBest with the f_classif scoring function.

The top 100 genes were selected for machine learning.

Feature selection was performed inside the machine learning pipeline. This prevents information from the test folds from influencing feature selection during cross-validation.

4. Train/Test Split

For the initial model evaluation, the dataset was divided using a stratified train/test split.

Training set: 80 samples

Test set: 20 samples

Training set:

GBM: 62
NON_TUMOR: 18

Test set:

GBM: 15
NON_TUMOR: 5
Machine Learning Models

Two classification algorithms were evaluated.

Logistic Regression

Logistic Regression was used as a linear classification model.

Before classification, the selected features were standardized using StandardScaler.

The model used balanced class weights.

Random Forest

Random Forest was used as a nonlinear ensemble classification model.

The model consisted of multiple decision trees and used 500 trees, balanced class weights and max_features = sqrt.

Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC

Confusion matrices and ROC curves were also generated.

Test Set Performance
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	1.00	1.00	1.00	1.00	1.00
Random Forest	0.95	1.00	0.93	0.97	1.00

The Logistic Regression model correctly classified all 20 test samples.

Random Forest correctly classified 19 of 20 test samples.

5-Fold Stratified Cross-Validation

To assess model stability beyond a single train/test split, 5-fold stratified cross-validation was performed.

Feature selection was performed independently within each fold to prevent data leakage.

Cross-Validation Results
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.950 ± 0.055	0.988 ± 0.024	0.947 ± 0.078	0.965 ± 0.041	0.997 ± 0.005
Random Forest	0.950 ± 0.055	0.988 ± 0.024	0.947 ± 0.078	0.965 ± 0.041	0.992 ± 0.016

The cross-validation results indicate consistently high classification performance for both models.

Logistic Regression achieved a slightly higher and more stable ROC-AUC than Random Forest.

Results and Figures

The following figures are included in the figures/ directory.

Confusion Matrix – Logistic Regression

Confusion Matrix – Random Forest

ROC Curve

Model Performance Comparison

5-Fold Cross-Validation Performance

ROC-AUC Stability Across Cross-Validation Folds

Software and Tools

The project was developed using Python and Linux/WSL.

Python
Python 3.11
pandas
NumPy
scikit-learn
matplotlib
seaborn
PyYAML
Machine Learning
LogisticRegression
RandomForestClassifier
SelectKBest
f_classif
StandardScaler
StratifiedKFold
cross_validate
Data Source
NCBI Gene Expression Omnibus (GEO)
GSE4290
GPL570 Affymetrix Human Genome U133 Plus 2.0 Array
Repository Structure

04_Machine-Learning-GBM/

├── README.md

├── data/
│ └── README.md

├── figures/
│ ├── confusion_matrix_logistic_regression.png
│ ├── confusion_matrix_random_forest.png
│ ├── ROC_curve_models.png
│ ├── model_performance_comparison.png
│ ├── cross_validation_model_comparison.png
│ └── cross_validation_ROC_AUC.png

├── results/
│ ├── model_performance.csv
│ ├── cross_validation_summary.csv
│ ├── cross_validation_fold_results.csv
│ ├── selected_features.csv
│ ├── train_test_split.csv
│ └── test_predictions.csv

└── scripts/
├── 01_prepare_data.py
├── 02_feature_selection.py
├── 03_train_models.py
├── 04_evaluate_models.py
└── 05_external_validation.py

Reproducibility

Create and activate the project environment:

conda create -n ml-gbm python=3.11

conda activate ml-gbm

Install the required Python packages:

pip install pandas numpy scikit-learn matplotlib seaborn pyyaml

Download the GSE4290 series matrix and GPL570 annotation files from NCBI GEO.

Place them in:

data/raw/

Then execute the scripts in order:

python scripts/01_prepare_data.py

python scripts/02_feature_selection.py

python scripts/03_train_models.py

python scripts/04_evaluate_models.py

python scripts/05_external_validation.py

Limitations

This project is intended as a machine learning and bioinformatics portfolio project.

The dataset contains only 100 samples and is imbalanced toward the GBM class.

The very high classification performance should therefore not be interpreted as evidence that the models are clinically validated.

Independent external validation using an additional dataset would be required before considering clinical or translational applications.

Conclusion

This project demonstrates a complete machine learning workflow for transcriptomic classification of glioblastoma.

Gene expression data were transformed from probe-level measurements to gene-level features, reduced to the most informative 100 genes, and used to train Logistic Regression and Random Forest classifiers.

Both models achieved high classification performance, with Logistic Regression showing the highest and most stable ROC-AUC during 5-fold stratified cross-validation:

ROC-AUC = 0.997 ± 0.005

The project demonstrates the integration of bioinformatics preprocessing, dimensionality reduction, machine learning, model evaluation, and reproducible computational workflows.

Author

Katarzyna Zielińska

Bioinformatics Portfolio

2026
