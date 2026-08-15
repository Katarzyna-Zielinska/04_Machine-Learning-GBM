import pandas as pd
from pathlib import Path
import gzip
from io import StringIO


# ============================================================
# Project:
# Machine Learning Classification of Glioblastoma
#
# Script:
# 01_prepare_data.py
#
# Author:
# Katarzyna Zielińska
#
# Description:
# Prepare GSE4290 expression data for machine learning.
#
# The analysis retains only:
# - glioblastoma, grade 4
# - non-tumor
#
# All other diagnostic categories are excluded.
# ============================================================


# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR /
    "data/raw/GSE4290_series_matrix.txt.gz"
)

OUTPUT_EXPRESSION = (
    PROJECT_DIR /
    "data/processed/GSE4290_GBM_vs_NonTumor_expression.csv"
)

OUTPUT_METADATA = (
    PROJECT_DIR /
    "data/processed/GSE4290_GBM_vs_NonTumor_metadata.csv"
)


# ------------------------------------------------------------
# 2. Read GEO series matrix
# ------------------------------------------------------------

print("Reading GSE4290 series matrix...")

with gzip.open(
    INPUT_FILE,
    "rt",
    encoding="utf-8"
) as f:
    lines = f.readlines()


# ------------------------------------------------------------
# 3. Extract sample IDs
# ------------------------------------------------------------

sample_line = next(
    line
    for line in lines
    if line.startswith("!Sample_geo_accession")
)

sample_ids = sample_line.strip().split("\t")[1:]

sample_ids = [
    sample.strip().strip('"')
    for sample in sample_ids
]


# ------------------------------------------------------------
# 4. Extract sample characteristics
# ------------------------------------------------------------

characteristics_line = next(
    line
    for line in lines
    if line.startswith("!Sample_characteristics_ch1")
)

characteristics = (
    characteristics_line
    .strip()
    .split("\t")[1:]
)

characteristics = [
    value.strip().strip('"')
    for value in characteristics
]


# ------------------------------------------------------------
# 5. Create metadata table
# ------------------------------------------------------------

metadata = pd.DataFrame({
    "sample": sample_ids,
    "characteristics": characteristics
})


# ------------------------------------------------------------
# 6. Assign biological classes
# ------------------------------------------------------------

metadata["condition"] = pd.NA

metadata.loc[
    metadata["characteristics"].str.contains(
        "glioblastoma, grade 4",
        case=False,
        na=False
    ),
    "condition"
] = "GBM"

metadata.loc[
    metadata["characteristics"].str.contains(
        "non-tumor",
        case=False,
        na=False
    ),
    "condition"
] = "NON_TUMOR"


# ------------------------------------------------------------
# 7. Keep only relevant samples
# ------------------------------------------------------------

metadata = metadata[
    metadata["condition"].isin(
        ["GBM", "NON_TUMOR"]
    )
].copy()


# ------------------------------------------------------------
# 8. Check class distribution
# ------------------------------------------------------------

print("\nClass distribution:")

print(
    metadata["condition"].value_counts()
)


# ------------------------------------------------------------
# 9. Locate expression matrix
# ------------------------------------------------------------

matrix_start = next(
    i
    for i, line in enumerate(lines)
    if line.startswith(
        "!series_matrix_table_begin"
    )
)

matrix_end = next(
    i
    for i, line in enumerate(lines)
    if line.startswith(
        "!series_matrix_table_end"
    )
)


# ------------------------------------------------------------
# 10. Read expression matrix
# ------------------------------------------------------------

expression_lines = lines[
    matrix_start + 1:matrix_end
]

expression = pd.read_csv(
    StringIO(
        "".join(expression_lines)
    ),
    sep="\t"
)


# ------------------------------------------------------------
# 11. Clean column names
# ------------------------------------------------------------

expression.columns = [
    column.strip().strip('"')
    for column in expression.columns
]

expression["ID_REF"] = (
    expression["ID_REF"]
    .astype(str)
    .str.strip()
    .str.strip('"')
)


# ------------------------------------------------------------
# 12. Select relevant samples
# ------------------------------------------------------------

selected_samples = metadata["sample"].tolist()

expression = expression[
    ["ID_REF"] + selected_samples
]


# ------------------------------------------------------------
# 13. Convert expression values to numeric
# ------------------------------------------------------------

for column in selected_samples:

    expression[column] = pd.to_numeric(
        expression[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# 14. Check for missing values
# ------------------------------------------------------------

missing_values = expression[
    selected_samples
].isna().sum().sum()

print(
    f"\nTotal missing expression values: "
    f"{missing_values}"
)


# ------------------------------------------------------------
# 15. Save processed data
# ------------------------------------------------------------

metadata.to_csv(
    OUTPUT_METADATA,
    index=False
)

expression.to_csv(
    OUTPUT_EXPRESSION,
    index=False
)


# ------------------------------------------------------------
# 16. Final summary
# ------------------------------------------------------------

print("\nData preparation completed.")

print(
    f"Number of samples: "
    f"{len(selected_samples)}"
)

print(
    f"Number of probes: "
    f"{expression.shape[0]}"
)

print("\nSamples by condition:")

print(
    metadata["condition"].value_counts()
)

print("\nSaved files:")

print(
    OUTPUT_EXPRESSION
)

print(
    OUTPUT_METADATA
)
