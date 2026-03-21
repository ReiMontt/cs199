"""
Script to clean and standardize differential expression gene files. 
Used to fix inconsistencies in column names and formats across files.
"""

import pandas as pd
import os
import glob

INPUT_DIR = "data/de_genes"
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df

files = glob.glob(f"{INPUT_DIR}/TCGA-*_upregulated.txt")

for path in files:
    cancer = os.path.basename(path).split("_")[0].replace("TCGA-", "")
    print(f"\nProcessing {cancer}...")

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df = normalize_columns(df)

        print("  Columns found:", list(df.columns))

        # ---- REQUIRED COLUMN RESOLUTION ----
        if "genesymbol" in df.columns:
            gene_col = "genesymbol"
        elif "symbol" in df.columns:
            gene_col = "symbol"
        elif "gene" in df.columns:
            gene_col = "gene"
        else:
            raise ValueError("No gene symbol column found")

        if "log2foldchange" in df.columns:
            df["logfc"] = df["log2foldchange"]
        else:
            raise ValueError("log2foldchange column missing")

        if "adjp" not in df.columns:
            raise ValueError("adjp column missing")

        df = df[[gene_col, "logfc", "adjp"]].dropna()
        df.columns = ["gene", "logfc", "qval"]

        out = f"{OUTPUT_DIR}/{cancer}_clean.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out}")

    except Exception as e:
        print(f"ERROR processing {cancer}: {e}")

print("\nData preparation finished.")
