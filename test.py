import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import mannwhitneyu
import re
import sys
import os
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
PPI_FILE = "data/ppi/9606.protein.links.full.v12.0.txt"
ALIAS_FILE = "data/ppi/9606.protein.aliases.v12.0.txt"
DEPMAP_FILE = "data/depmap/CRISPRGeneEffect.csv"
DGIDB_FILE = "data/dgidb/interactions.tsv"
DE_GENES_DIR = "data/de_genes/"
CONFIDENCE_THRESHOLD = 700  # High confidence for robust stats

def extract_ensp(series):
    """Robustly extracts 'ENSP12345' from '9606.ENSP12345.1'"""
    return series.astype(str).str.extract(r'(ENSP\d+)', expand=False)

def main():
    print("--- LOADING DATA FOR BEST THESIS 2026 ---")

    # 1. BUILD GENE WHITELIST
    # We load DepMap & DGIdb first to know which Gene Symbols are "Valid"
    # This prevents mapping ENSP IDs to obscure aliases like "HGNC:123"
    print("1. Building Gene Symbol Whitelist...")
    whitelist = set()
    
    # Load DepMap
    try:
        # Check orientation by peeking
        peek = pd.read_csv(DEPMAP_FILE, index_col=0, nrows=5)
        if peek.columns.str.contains(r" \(").any():
            # Standard DepMap: Columns are "BRAF (673)" -> Extract "BRAF"
            depmap_genes = set(pd.read_csv(DEPMAP_FILE, index_col=0, nrows=1).columns.str.split(" ").str[0].str.upper())
            depmap_orient = 'col'
        else:
            # User Format: Index is Gene
            depmap_genes = set(pd.read_csv(DEPMAP_FILE, usecols=[0]).iloc[:,0].astype(str).str.upper())
            depmap_orient = 'idx'
        whitelist.update(depmap_genes)
        print(f"   DepMap loaded: {len(depmap_genes):,} genes")
    except Exception as e:
        print(f"   ! Error loading DepMap: {e}")
        return

    # Load DGIdb
    try:
        dgidb = pd.read_csv(DGIDB_FILE, sep="\t")
        drug_col = 'gene_name' if 'gene_name' in dgidb.columns else dgidb.columns[0]
        druggable = set(dgidb[drug_col].dropna().astype(str).str.upper())
        whitelist.update(druggable)
        print(f"   DGIdb loaded: {len(druggable):,} druggable genes")
    except Exception as e:
        print(f"   ! Error loading DGIdb: {e}")
        druggable = set()

    # Load DE Genes (to add to whitelist)
    cancers = ["BRCA", "LUAD", "COAD", "PRAD", "LIHC"]
    cancer_genes_dict = {}
    print("   Loading DE Gene Lists...")
    for c in cancers:
        fpath = os.path.join(DE_GENES_DIR, f"TCGA-{c}_upregulated.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # Robust column finding: first column or 'gene'
            col = 'gene' if 'gene' in df.columns else df.columns[0]
            # CRITICAL: Strip whitespace
            genes = set(df[col].astype(str).str.upper().str.strip())
            cancer_genes_dict[c] = genes
            whitelist.update(genes)
            print(f"     {c}: Found {len(genes)} genes")
        else:
            print(f"     {c}: File not found")
            cancer_genes_dict[c] = set()

    # 2. MAP ALIASES (Targeted Mapping)
    print("\n2. Mapping ENSP IDs to Symbols...")
    ens_to_gene = {}
    try:
        # Read aliases
        # Filter 1: Source contains 'HGNC', 'Symbol', 'BioMart' to avoid obscure IDs
        # Filter 2: The alias MUST be in our whitelist (DepMap/DE genes)
        
        # Using chunks to handle large file safely, though v12 is okay in memory usually
        chunksize = 500000
        for chunk in pd.read_csv(ALIAS_FILE, sep="\t", header=None, names=["ensp", "alias", "source"], chunksize=chunksize):
            # Clean Alias
            chunk['alias_clean'] = chunk['alias'].astype(str).str.upper().str.strip()
            
            # Filter by whitelist immediately
            mask_valid = chunk['alias_clean'].isin(whitelist)
            valid = chunk[mask_valid].copy()
            
            if valid.empty: continue
            
            # Extract ENSP
            valid['ensp_clean'] = extract_ensp(valid['ensp'])
            
            # Update Dict (Last seen overwrites, which is fine for Symbols)
            ens_to_gene.update(dict(zip(valid.ensp_clean, valid.alias_clean)))
            
        print(f"   Mapping Dictionary Ready: {len(ens_to_gene):,} valid protein-gene pairs")
        
        if len(ens_to_gene) == 0:
            print("   CRITICAL ERROR: No aliases matched the whitelist. Check file formats.")
            return

    except Exception as e:
        print(f"   ! Error reading aliases: {e}")
        return

    # 3. BUILD GRAPH
    print("\n3. Building NetworkX Graph...")
    try:
        df_links = pd.read_csv(PPI_FILE, sep=" ")
        df_links = df_links[df_links.combined_score >= CONFIDENCE_THRESHOLD]
        
        # Extract ENSP
        df_links['p1'] = extract_ensp(df_links['protein1'])
        df_links['p2'] = extract_ensp(df_links['protein2'])
        
        # Map
        df_links['g1'] = df_links['p1'].map(ens_to_gene)
        df_links['g2'] = df_links['p2'].map(ens_to_gene)
        
        # Drop unmapped
        edges = df_links.dropna(subset=['g1', 'g2'])
        
        # Build
        G = nx.from_pandas_edgelist(edges, 'g1', 'g2')
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Largest Component
        if len(G) > 0:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
        print(f"   PPI LOADED: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
    except Exception as e:
        print(f"   ! Error building graph: {e}")
        return

    # 4. LOAD SCORES
    print("\n4. Loading Essentiality Scores...")
    try:
        if depmap_orient == 'col':
            dm = pd.read_csv(DEPMAP_FILE, index_col=0)
            dm.columns = dm.columns.str.split(" ").str[0].str.upper()
            gene_ess = dm.mean(axis=0).to_dict()
        else:
            dm = pd.read_csv(DEPMAP_FILE, index_col=0)
            dm.index = dm.index.str.split(" ").str[0].str.upper()
            gene_ess = dm.mean(axis=1).to_dict()
    except:
        print("   ! Error calculating essentiality")
        gene_ess = {}

    # 5. RUN ANALYSIS
    print("\n5. Running Analysis...")
    results = []
    
    for cancer in cancers:
        genes = cancer_genes_dict[cancer]
        overlap = genes.intersection(set(G.nodes()))
        
        print(f"   {cancer}: Overlap {len(overlap)} genes")
        
        if len(overlap) < 500:
            continue
            
        # Subgraph & Metrics
        subG = G.subgraph(overlap)
        core = nx.core_number(subG)
        deg = dict(subG.degree())
        
        # DataFrame for stats
        data = []
        for n in subG.nodes():
            if n in gene_ess:
                # Metric: Coreness * log(Degree)
                c_score = core[n] * np.log1p(deg[n])
                data.append({
                    'gene': n, 
                    'score': c_score, 
                    'ess': gene_ess[n], 
                    'drug': n in druggable
                })
        
        df = pd.DataFrame(data)
        if df.empty: continue
        
        # Compare Top 20% Core vs Bottom 20% Periphery
        high = df[df.score >= df.score.quantile(0.8)]
        low = df[df.score <= df.score.quantile(0.2)]
        
        if len(high) > 10 and len(low) > 10:
            # Mann-Whitney (High Coreness should be MORE essential = Lower DepMap score)
            p = mannwhitneyu(high.ess, low.ess, alternative='less').pvalue
            
            # Druggability Enrichment
            drug_fold = high.drug.mean() / (low.drug.mean() + 1e-9)
            
            results.append([cancer, len(overlap), p, drug_fold])

    # 6. FINAL TABLE
    print("\nBEST THESIS 2026 — FINAL TABLE")
    results_df = pd.DataFrame(results, columns=["Cancer", "Genes", "p-value", "Druggability"])
    
    if not results_df.empty:
        results_df["p-value"] = results_df["p-value"].apply(lambda x: f"{x:.2e}")
        results_df["Druggability"] = results_df["Druggability"].apply(lambda x: f"{x:.1f}x")
        print(results_df.to_string(index=False))
    else:
        print("No significant results found (Check input data paths).")

if __name__ == "__main__":
    main()

