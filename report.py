import os, math, re
import numpy as np
import pandas as pd

OUTPUT_DIR = "output/"

# ── colours ────────────────────────────────
class C:
    RESET   = "\033[0m";  BOLD    = "\033[1m";  DIM     = "\033[2m"
    WHITE   = "\033[97m"; CYAN    = "\033[96m"; GREEN   = "\033[92m"
    YELLOW  = "\033[93m"; RED     = "\033[91m"; BLUE    = "\033[94m"
    MAGENTA = "\033[95m"; GREY    = "\033[90m"; BG_DARK = "\033[48;5;235m"

def c(text, *codes):  return "".join(codes) + str(text) + C.RESET
def _vlen(s):         return len(re.sub(r"\033\[[0-9;]*m", "", str(s)))
def _pad(s, w):       return s + " " * max(0, w - _vlen(s))

# ── layout ──────────────────────────────
W = 90

def banner(text):
    inner = f"  {text}  "
    pad   = max(0, W - len(inner))
    print()
    print(c(f"  {'━'*W}", C.CYAN, C.BOLD))
    print(c(f"  {inner}{' '*pad}", C.BG_DARK, C.WHITE, C.BOLD))
    print(c(f"  {'━'*W}", C.CYAN, C.BOLD))

def section(text, color=C.CYAN):
    print(f"\n  {c(text, color, C.BOLD)}")
    print(c(f"  {'─'*W}", C.GREY))

def rule():  print(c(f"  {'─'*W}", C.GREY))
def ok(t):   print(c(f"  ✓  {t}", C.GREEN))
def err(t):  print(c(f"  ✗  {t}", C.RED))
def info(t): print(c(f"  ·  {t}", C.GREY))

# ── constants ─────────────────────────────────────────────────────────────────
CANCER_LABELS = {"BRCA":"Breast (BRCA)", "LUAD":"Lung (LUAD)", "LIHC":"Liver (LIHC)"}
CANCER_COLORS = {"BRCA": C.BLUE, "LUAD": C.YELLOW, "LIHC": C.GREEN}
CANCERS       = ["BRCA", "LUAD", "LIHC"]

ALGO_ORDER = [
    "Strength","Weighted K-Core","DependANT Composite",
    "HITS (Hub)","Eigenvector","LeaderRank",
    "PageRank","ECC (Link Cluster)","VoteRank",
]

COSMIC = {
    "CDK1":"T1","KIF11":"T2","PLK1":"T1","TOP2A":"T1","CDC20":"T2",
    "AURKB":"T1","RRM2":"T2","BIRC5":"T1","ESPL1":"T2","CCNA2":"T2",
    "BUB1B":"T1","MAD2L1":"T2","AURKA":"T1",
}

# ── data ──────────────────────────────────────────────────────────────────────
def _load(name):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        err(f"{name} not found — run analysis_final.py first.")
        return None
    return pd.read_csv(path)

# ── colour helpers ────────────────────────────────────────────────────────────
def _auroc_col(v):
    if v >= 0.82: return C.GREEN + C.BOLD
    if v >= 0.78: return C.GREEN
    if v >= 0.74: return C.YELLOW
    if v >= 0.65: return C.GREY
    return C.RED

def _rho_col(v):
    a = abs(v)
    if a >= 0.50: return C.GREEN + C.BOLD
    if a >= 0.40: return C.GREEN
    if a >= 0.30: return C.YELLOW
    if a >= 0.20: return C.GREY
    return C.RED

def _chron_col(v):
    if v < -3.0: return C.RED + C.BOLD
    if v < -1.5: return C.RED
    if v < -0.8: return C.YELLOW
    return C.GREY

def _gene_col(gene, is_druggable):
    if gene in COSMIC: return C.YELLOW, f"COSMIC {COSMIC[gene]}"
    if is_druggable:   return C.CYAN,   "DGIdb"
    return C.GREEN, "novel"

def _bool(val):
    if isinstance(val, bool):         return val
    if isinstance(val, (int, float)): return bool(val)
    return str(val).lower() in ("true","1","yes")

def _sig(row):
    if math.isnan(row["p_value"]): return c("ns  ", C.GREY)
    if row["sig_0.01"]:            return c("*** ", C.GREEN + C.BOLD)
    if row["sig_0.05"]:            return c("*   ", C.YELLOW)
    return c("ns  ", C.GREY)

# ── full validation table ─────────────────────────────────────────────────────
def print_full_validation(df):
    banner("Full Validation Table  ·  DepMap 23Q4 CRISPR Chronos")
    info(c("green bold", C.GREEN+C.BOLD) + "  AUROC >= 0.82   " +
         c("green",   C.GREEN)   + "  >= 0.78   " +
         c("yellow",  C.YELLOW)  + "  >= 0.74   " +
         c("grey",    C.GREY)    + "  >= 0.65   " +
         c("red",     C.RED)     + "  < 0.65")

    for cancer in CANCERS:
        sub = df[df["cancer"]==cancer].copy()
        if sub.empty: continue
        sub["_o"] = sub["algo"].map({a:i for i,a in enumerate(ALGO_ORDER)}).fillna(99)
        sub = sub.sort_values("_o")
        best = float(sub["auroc"].max())
        cc   = CANCER_COLORS.get(cancer, C.WHITE)
        section(CANCER_LABELS.get(cancer, cancer), cc)

        hdr = (f"  {'Algorithm':<24}  {'N':>5}  {'AUROC':>7}  {'AUPRC':>6}"
               f"  {'rho':>8}  {'P@20':>5}  {'P@50':>5}  {'P@100':>6}  {'MW-p':>10}")
        print(c(hdr, C.BOLD, C.WHITE))
        rule()

        for _, r in sub.iterrows():
            algo    = r["algo"]
            is_best = abs(float(r["auroc"]) - best) < 1e-6
            is_ctrl = algo == "VoteRank"
            is_eqv  = algo == "Eigenvector"

            tag = c("  ◀ best", C.GREEN, C.BOLD) if is_best else \
                  c("  ⊘ ctrl", C.RED)            if is_ctrl else \
                  c("  = HITS",  C.GREY)           if is_eqv  else ""

            ac   = _auroc_col(r["auroc"])
            rc   = _rho_col(r["spearman_rho"])
            p20c = C.GREEN+C.BOLD if r["P@20"] >= 0.95 else \
                   C.GREEN        if r["P@20"] >= 0.80  else \
                   C.YELLOW       if r["P@20"] >= 0.60  else C.GREY

            astr = _pad(c(algo, C.BOLD if is_best else C.WHITE), 24)

            print(f"  {astr}  "
                  f"{int(r['n_genes']):>5}  "
                  f"{c(f'{r.auroc:>7.4f}', ac)}  "
                  f"{r.auprc:>6.4f}  "
                  f"{c(f'{r.spearman_rho:>+8.4f}', rc)}  "
                  f"{c(f'{r["P@20"]:>5.3f}', p20c)}  "
                  f"{r['P@50']:>5.3f}  "
                  f"{r['P@100']:>6.3f}  "
                  f"{r.mw_p:>10.2e}"
                  f"{tag}")
        print()

# ── algorithm ranking ─────────────────────────────────────────────────────────
def print_algorithm_ranking(df):
    banner("Algorithm Ranking  ·  Mean AUROC  (BRCA + LUAD + LIHC)")

    means = (df.groupby("algo")["auroc"].mean()
               .sort_values(ascending=False).reset_index())
    notes = {
        "Strength":            "O(m) weighted degree — simplest, wins overall",
        "Weighted K-Core":     "Novel composite: k_core x log(deg+1)  [this work]",
        "DependANT Composite": "No-ML feature composite (Benstead-Hume 2022)",
        "HITS (Hub)":          "Equivalent to Eigenvector on undirected graphs",
        "Eigenvector":         "Leading eigenvector of W (Bonacich 1972)",
        "LeaderRank":          "Ground-node PageRank, parameter-free (Lu 2011)",
        "PageRank":            "Random walk with teleportation (Brin & Page 1998)",
        "ECC (Link Cluster)":  "Local modularity — non-hub essentials (Jeong 2019)",
        "VoteRank":            "Spreader metric — deliberate negative control",
    }

    print(f"\n  {c(f'  # {'Algorithm':<24}  {'Mean AUROC':>10}  Notes', C.BOLD, C.WHITE)}")
    rule()

    for i, (_, row) in enumerate(means.iterrows(), 1):
        algo = row["algo"]
        auc  = row["auroc"]
        note = notes.get(algo, "")

        rc = C.GREEN+C.BOLD if i==1     else \
             C.GREEN        if i<=3     else \
             C.YELLOW       if i<=6     else \
             C.RED          if algo=="VoteRank" else C.GREY

        tag = c("  ★ best",   C.YELLOW, C.BOLD) if i==1               else \
              c("  baseline", C.CYAN)            if algo=="Weighted K-Core" else \
              c("  neg ctrl", C.RED)             if algo=="VoteRank"  else ""

        print(f"  {c(f'{i:>2}', rc)}  "
              f"{_pad(c(algo, C.BOLD if i<=3 else C.WHITE), 24)}  "
              f"{c(f'{auc:>10.4f}', rc)}  "
              f"{c(note, C.GREY)}"
              f"{tag}")

# ── delong table ──────────────────────────────────────────────────────────────
def print_delong_full(df):
    banner("DeLong AUROC Significance Tests  ·  Pairwise, Two-Sided")
    info("Reference: DeLong, DeLong & Clarke-Pearson (1988) Biometrics 44:837-845")
    info(c("*** p<0.01", C.GREEN+C.BOLD) + "   " +
         c("*   p<0.05", C.YELLOW)       + "   " +
         c("ns  not significant", C.GREY))

    for cancer in CANCERS:
        sub = df[df["cancer"]==cancer]
        if sub.empty: continue
        cc = CANCER_COLORS.get(cancer, C.WHITE)
        section(CANCER_LABELS.get(cancer, cancer), cc)

        hdr = (f"  {'Algo A':<24}  {'Algo B':<24}"
               f"  {'AUC-A':>7}  {'AUC-B':>7}  {'z':>9}  {'p':>9}  Sig")
        print(c(hdr, C.BOLD, C.WHITE))
        rule()

        for _, r in sub.iterrows():
            sig   = _sig(r)
            z_str = f"{r['z_stat']:>9.3f}" if not math.isnan(r["z_stat"]) else c("      nan", C.GREY)
            p_str = f"{r['p_value']:>9.4f}" if not math.isnan(r["p_value"]) else c("      nan", C.GREY)
            ac_a  = _auroc_col(r["AUROC_A"])
            ac_b  = _auroc_col(r["AUROC_B"])
            print(f"  {r['algo_A']:<24}  {r['algo_B']:<24}  "
                  f"{c(f'{r["AUROC_A"]:>7.4f}', ac_a)}  "
                  f"{c(f'{r["AUROC_B"]:>7.4f}', ac_b)}  "
                  f"{z_str}  {p_str}  {sig}")
        print()

# ── top-20 tables ─────────────────────────────────────────────────────────────
def print_top20_tables(df):
    banner("Top-20 Essential Candidates  ·  Best Algorithm per Cancer")
    info(c("yellow", C.YELLOW) + " = COSMIC CGC driver   " +
         c("cyan",   C.CYAN)   + " = DGIdb druggable   " +
         c("green",  C.GREEN)  + " = novel (not in COSMIC or DGIdb)")

    for cancer in CANCERS:
        sub  = df[df["cancer"]==cancer].sort_values("priority", ascending=False)
        if sub.empty: continue
        algo = sub["algo"].iloc[0]
        cc   = CANCER_COLORS.get(cancer, C.WHITE)
        section(f"{CANCER_LABELS.get(cancer,cancer)}  ·  {algo}", cc)

        hdr = (f"  {'#':>2}  {'Gene':<8}  {'Score':>7}  {'Chronos':>9}"
               f"  {'Priority':>9}  {'Tag':<14}  {'Ess'}")
        print(c(hdr, C.BOLD, C.WHITE))
        rule()

        for i, (_, r) in enumerate(sub.iterrows(), 1):
            dg           = _bool(r["is_druggable"])
            gene_col, db = _gene_col(r["gene"], dg)
            cc_chron     = _chron_col(r["essentiality"])
            dot          = c("●", C.RED) if r["is_essential"] else c("○", C.GREY+C.DIM)

            # pad gene name to exact visible width
            gene_str = _pad(c(r["gene"], gene_col, C.BOLD), 8)
            db_str   = _pad(c(db, gene_col), 14)

            print(f"  {i:>2}  {gene_str}  "
                  f"{r['score']:>7.4f}  "
                  f"{c(f'{r.essentiality:>9.4f}', cc_chron)}  "
                  f"{r['priority']:>9.4f}  "
                  f"{db_str}  {dot}")
        print()

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    banner("Cancer Gene Essentiality  ·  Full Report")
    info(f"Reading from: {os.path.abspath(OUTPUT_DIR)}/")

    val_df    = _load("validation_summary.csv")
    delong_df = _load("delong_tests.csv")
    top_df    = _load("top20_genes.csv")

    if val_df is None or delong_df is None or top_df is None:
        return

    print_full_validation(val_df)
    print_algorithm_ranking(val_df)
    print_delong_full(delong_df)
    print_top20_tables(top_df)

    print(c(f"\n  {'━'*W}", C.CYAN, C.BOLD))
    print(c("  Run supplementary_figures.py  for publication figures.", C.GREY))
    print(c("  Run compare_top20.py          for algorithm gene-list comparison.", C.GREY))
    print(c(f"  {'━'*W}\n", C.CYAN, C.BOLD))


if __name__ == "__main__":
    main()