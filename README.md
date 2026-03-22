## Quick Summary

Benchmarks nine graph topology algorithms on cancer-specific PPI subnetworks to predict CRISPR-confirmed gene essentiality, validated against DepMap 23Q4 Chronos scores across three cancer types: BRCA, LUAD, and LIHC.

---

## Setup

**Data files required**:

```
data/ppi/9606.protein.links.full.v12.0.txt   # STRING v12.0
data/ppi/9606.protein.aliases.v12.0.txt      # STRING v12.0
data/depmap/CRISPRGeneEffect.csv             # DepMap Public 23Q4
data/dgidb/interactions.tsv                  # DGIdb 4.0
processed/BRCA_clean.csv                     # pre-processed TCGA DE genes
processed/LUAD_clean.csv                     # columns: gene, logfc, qval
processed/LIHC_clean.csv
```

**Install dependencies:**

```bash
pip install requirements.txt
```

---

## Usage

Run in order:

```bash
python3 fix.py                   # if there's new data: data cleaning + pre-processing (TCGA DE genes)
python3 analysis_final.py        # main pipeline
python3 report.py                # full validation tables, DeLong tests, commentary
```

---

## Output

```
output/
├── all_scores.csv          gene × algorithm × cancer scores
├── top20_genes.csv         top-20 candidates per cancer
├── validation_summary.csv  AUROC, rho, P@k for all 9 algorithms
├── delong_tests.csv        pairwise DeLong AUROC significance tests
├── figures/                publication figures (from supplementary_figures.py)
└── tables/                 LaTeX + CSV result tables
```

---

## Algorithms benchmarked

| Algorithm | Paradigm | Complexity |
|---|---|---|
| Strength Centrality | Local heuristic | O(m) |
| Weighted K-Core | Core-periphery | O(m) |
| DependANT Composite | Feature composite | O(m·Δ) |
| HITS Hub Score | Spectral | O(m·k) |
| Eigenvector Centrality | Spectral | O(m·k) |
| LeaderRank | Random walk | O(m·k) |
| PageRank | Random walk | O(m·k) |
| ECC (Link Cluster) | Local modularity | O(m·Δ) |
| VoteRank | Spreader (neg. ctrl) | O(m·n) |

---

## Key parameters

| Parameter | Value | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 700 | STRING combined_score cutoff |
| `ESSENTIALITY_CUTOFF` | -0.6 | DepMap Chronos threshold |
| `FDR_THRESHOLD` | 0.05 | DE gene significance cutoff |
| `LOG2FC_THRESHOLD` | 1.0 | Upregulation filter (>2x) |
| `TOP_N` | 20 | Candidate genes per cancer |