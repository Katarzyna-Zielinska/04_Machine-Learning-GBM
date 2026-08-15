# Data

The original expression data are not included in this repository because the files are publicly available from the NCBI Gene Expression Omnibus (GEO).

The dataset used in this project is:

- **GEO accession:** GSE4290
- **Organism:** *Homo sapiens*
- **Platform:** Affymetrix Human Genome U133 Plus 2.0 Array (GPL570)
- **Samples analyzed:** 100
- **Glioblastoma (GBM):** 77
- **Non-tumor:** 23

The original GSE4290 dataset can be downloaded from NCBI GEO:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4290

The GPL570 platform annotation was used for probe-to-gene mapping and gene-level data preparation.

The downloaded files used during the analysis were:

- `GSE4290_series_matrix.txt.gz`
- `GPL570_annotation.txt.gz`

The files should be placed in:

```text
data/raw/
```

The raw data are excluded from the GitHub repository.
