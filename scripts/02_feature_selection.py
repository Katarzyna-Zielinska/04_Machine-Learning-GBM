import pandas as pd
from pathlib import Path
import gzip
from io import StringIO


# ============================================================
# Project:
# Machine Learning Classification of Glioblastoma
#
# Script:
# 02_feature_selection.py
#
# Author:
# Katarzyna Zielińska
#
# Description:
# Convert Affymetrix probe-level expression data into
# gene-level expression data using GPL570 annotation.
#
# This script performs preprocessing only.
#
# Feature selection for machine learning will be performed
# after the train/test split to avoid data leakage.
# ============================================================


# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

EXPRESSION_FILE = (
    PROJECT_DIR /
    "data/processed/"
    "GSE4290_GBM_vs_NonTumor_expression.csv"
)

METADATA_FILE = (
    PROJECT_DIR /
    "data/processed/"
    "GSE4290_GBM_vs_NonTumor_metadata.csv"
)

ANNOTATION_FILE = (
    PROJECT_DIR /
    "data/raw/"
    "GPL570_annotation.txt.gz"
)

OUTPUT_FILE = (
    PROJECT_DIR /
    "data/processed/"
    "GSE4290_GBM_vs_NonTumor_gene_expression.csv"
)


# ------------------------------------------------------------
# 2. Load probe-level expression matrix
# ------------------------------------------------------------

print("Loading probe-level expression matrix...")

expression = pd.read_csv(
    EXPRESSION_FILE
)

metadata = pd.read_csv(
    METADATA_FILE
)

sample_columns = metadata["sample"].tolist()

print(
    f"Input matrix: "
    f"{expression.shape[0]} probes × "
    f"{len(sample_columns)} samples"
)


# ------------------------------------------------------------
# 3. Load GPL570 annotation
# ------------------------------------------------------------

print("\nLoading GPL570 annotation...")

with gzip.open(
    ANNOTATION_FILE,
    "rt",
    encoding="utf-8"
) as f:

    lines = f.readlines()


# ------------------------------------------------------------
# 4. Locate annotation table
# ------------------------------------------------------------

annotation_start = next(
    i
    for i, line in enumerate(lines)
    if line.startswith(
        "!platform_table_begin"
    )
)

annotation_end = next(
    i
    for i, line in enumerate(lines)
    if line.startswith(
        "!platform_table_end"
    )
)


# ------------------------------------------------------------
# 5. Read annotation table
# ------------------------------------------------------------

annotation_lines = lines[
    annotation_start + 1:annotation_end
]

annotation = pd.read_csv(
    StringIO(
        "".join(annotation_lines)
    ),
    sep="\t",
    dtype=str
)


# ------------------------------------------------------------
# 6. Clean annotation column names
# ------------------------------------------------------------

annotation.columns = [
    column.strip()
    for column in annotation.columns
]


# ------------------------------------------------------------
# 7. Keep probe ID and gene symbol
# ------------------------------------------------------------

annotation = annotation[
    ["ID", "Gene symbol"]
].copy()

annotation.columns = [
    "ID_REF",
    "Gene_symbol"
]


# ------------------------------------------------------------
# 8. Clean annotation values
# ------------------------------------------------------------

annotation["ID_REF"] = (
    annotation["ID_REF"]
    .astype(str)
    .str.strip()
)

annotation["Gene_symbol"] = (
    annotation["Gene_symbol"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# 9. Remove probes without gene annotation
# ------------------------------------------------------------

annotation = annotation[
    annotation["Gene_symbol"] != ""
].copy()

print(
    f"Annotated probes: "
    f"{len(annotation)}"
)


# ------------------------------------------------------------
# 10. Merge expression with annotation
# ------------------------------------------------------------

print("\nMapping probes to gene symbols...")

expression = expression.merge(
    annotation,
    on="ID_REF",
    how="inner"
)

print(
    f"Mapped probes: "
    f"{expression.shape[0]}"
)


# ------------------------------------------------------------
# 11. Select expression columns
# ------------------------------------------------------------

gene_expression = expression[
    ["Gene_symbol"] + sample_columns
].copy()


# ------------------------------------------------------------
# 12. Convert expression values to numeric
# ------------------------------------------------------------

for column in sample_columns:

    gene_expression[column] = pd.to_numeric(
        gene_expression[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# 13. Aggregate multiple probes per gene
# ------------------------------------------------------------

print("\nAggregating multiple probes per gene...")

gene_expression = (
    gene_expression
    .groupby("Gene_symbol")[sample_columns]
    .mean()
    .reset_index()
)


# ------------------------------------------------------------
# 14. Remove genes with excessive missing values
# ------------------------------------------------------------

before_filtering = gene_expression.shape[0]

gene_expression = gene_expression[
    gene_expression[sample_columns]
    .notna()
    .mean(axis=1) >= 0.90
].copy()

after_filtering = gene_expression.shape[0]

print(
    f"Genes before missing-value filtering: "
    f"{before_filtering}"
)

print(
    f"Genes after missing-value filtering: "
    f"{after_filtering}"
)


# ------------------------------------------------------------
# 15. Save gene-level expression matrix
# ------------------------------------------------------------

gene_expression.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 16. Final summary
# ------------------------------------------------------------

print("\nGene-level preprocessing completed.")

print(
    f"Final number of genes: "
    f"{gene_expression.shape[0]}"
)

print(
    f"Number of samples: "
    f"{len(sample_columns)}"
)

print("\nSaved file:")

print(
    OUTPUT_FILE
)
