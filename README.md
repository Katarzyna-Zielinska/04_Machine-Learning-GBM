Machine Learning Classification of Glioblastoma

Machine learning analysis of gene expression profiles for classification of glioblastoma (GBM) versus non-tumor brain samples.

The project demonstrates a reproducible workflow combining transcriptomic data preprocessing, gene annotation, feature selection, machine learning classification and model validation.

Biological Question

Can gene expression profiles be used to distinguish glioblastoma from non-tumor brain samples using machine learning?

Dataset

The project uses the publicly available GSE4290 dataset from the NCBI Gene Expression Omnibus (GEO).

The expression data were generated using the Affymetrix Human Genome U133 Plus 2.0 Array (GPL570).

For this analysis, two diagnostic groups were selected:

Group	Samples
Glioblastoma (GBM)	77
Non-tumor	23
Total	100

The original expression matrix contained 54,613 probes.

After probe annotation and aggregation to gene level, 22,189 genes were available for analysis.

The original GEO data are not included in this repository.

Workflow
flowchart TD
    A[GSE4290<br/>100 samples] --> B[Select GBM and Non-Tumor samples]
    B --> C[Probe-level expression<br/>54,613 probes]
    C --> D[GPL570 annotation]
    D --> E[Probe → Gene mapping]
    E --> F[Gene-level expression<br/>22,189 genes]
    F --> G[Feature selection<br/>Top 100 genes]
    G --> H[Train/Test Split]

    H --> I[Logistic Regression]
    H --> J[Random Forest]

    I --> K[Model Evaluation]
    J --> K

    K --> L[5-fold Stratified<br/>Cross-Validation]

Methods
Data Preparation

Samples were selected according to their histopathological diagnosis, retaining only GBM and non-tumor samples.

Probe Annotation

Affymetrix GPL570 annotation was used to map probe IDs to gene symbols.

When multiple probes corresponded to the same gene, their expression values were aggregated to obtain a gene-level expression matrix.

Feature Selection

Because the dataset contains thousands of genes but only 100 samples, dimensionality was reduced using:

SelectKBest + ANOVA F-test (f_classif)

The 100 highest-ranking genes were selected as machine learning features.

Feature selection was performed inside the machine learning pipeline to prevent information leakage during cross-validation.

Classification

Two models were compared:

Logistic Regression

A linear classification model with standardized features.

Random Forest

An ensemble model based on multiple decision trees.

Validation

Model performance was assessed using:

Accuracy
Precision
Recall
F1-score
ROC-AUC

Two validation strategies were used:

Stratified 80/20 train-test split
5-fold stratified cross-validation
Results
Test Set
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	1.00	1.00	1.00	1.00	1.00
Random Forest	0.95	1.00	0.93	0.97	1.00

On the 20-sample test set, Logistic Regression correctly classified all samples, while Random Forest correctly classified 19 of 20 samples.

5-Fold Cross-Validation
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.950 ± 0.055	0.988 ± 0.024	0.947 ± 0.078	0.965 ± 0.041	0.997 ± 0.005
Random Forest	0.950 ± 0.055	0.988 ± 0.024	0.947 ± 0.078	0.965 ± 0.041	0.992 ± 0.016

Logistic Regression showed a slightly higher and more stable ROC-AUC across the five folds.

Visual Results
ROC Curve

Model Performance

Logistic Regression – Confusion Matrix

Random Forest – Confusion Matrix

Cross-Validation Performance

ROC-AUC Stability

Software

Programming language

Python 3.11

Main libraries

pandas
NumPy
scikit-learn
matplotlib
seaborn
PyYAML

Machine learning methods

Logistic Regression
Random Forest
SelectKBest
ANOVA F-test
StandardScaler
StratifiedKFold
Cross-validation

Data source

NCBI GEO
GSE4290
GPL570

Repository Structure
04_Machine-Learning-GBM/
│
├── README.md
│
├── data/
│   └── README.md
│
├── figures/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── ROC_curve_models.png
│   ├── model_performance_comparison.png
│   ├── cross_validation_model_comparison.png
│   └── cross_validation_ROC_AUC.png
│
├── results/
│   ├── model_performance.csv
│   ├── cross_validation_summary.csv
│   ├── cross_validation_fold_results.csv
│   ├── selected_features.csv
│   ├── train_test_split.csv
│   └── test_predictions.csv
│
└── scripts/
    ├── 01_prepare_data.py
    ├── 02_feature_selection.py
    ├── 03_train_models.py
    ├── 04_evaluate_models.py
    └── 05_external_validation.py

Reproducibility
Create the project environment:
conda create -n ml-gbm python=3.11
conda activate ml-gbm
Install the required packages:
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml
Download the GSE4290 expression matrix and GPL570 annotation from NCBI GEO and place them in:
data/raw/
Run the analysis sequentially:
python scripts/01_prepare_data.py
python scripts/02_feature_selection.py
python scripts/03_train_models.py
python scripts/04_evaluate_models.py
python scripts/05_external_validation.py

Limitations

The dataset contains only 100 samples, with an unequal number of GBM and non-tumor samples.

The high classification performance should therefore be interpreted as a result of this dataset and workflow rather than as evidence of clinical validity.

External validation using an independent dataset would be required to assess whether the observed performance generalizes to other patient cohorts.

Conclusion

This project demonstrates how transcriptomic data can be combined with machine learning to distinguish glioblastoma from non-tumor brain samples.

Both Logistic Regression and Random Forest achieved high classification performance. Logistic Regression showed the strongest and most stable cross-validation performance, with:

ROC-AUC = 0.997 ± 0.005

The project integrates bioinformatics preprocessing, gene-level feature construction, statistical feature selection, machine learning and model validation in a reproducible Python workflow.

Author

Katarzyna Zielińska
Bioinformatics Portfolio
2026
