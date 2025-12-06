# FINAL_NUCLEAR_FIX_DE_FILES.py
import pandas as pd
import os

cancers = ["BRCA", "LUAD", "COAD", "PRAD", "LIHC"]

for cancer in cancers:
    path = f"data/de_genes/TCGA-{cancer}_upregulated.txt"
    if not os.path.exists(path):
        print(f"Missing {path}")
        continue
    
    # Read raw lines
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Skip header, split by whitespace
    data = []
    for line in lines[1:]:  # skip header
        parts = line.strip().split()
        if len(parts) < 6: 
            continue
        gene = parts[0]
        # Find Log2FC (it's split across columns)
        try:
            logfc = float(parts[-3])  # usually the third from the end
        except:
            continue
        if logfc > 1:
            data.append(gene)
    
    # Save top 3000
    data = data[:3000]
    pd.DataFrame(data, columns=["gene"]).to_csv(
        f"data/de_genes/TCGA-{cancer}_upregulated.csv", index=False
    )
    print(f"SAVED {len(data)} genes for {cancer}")