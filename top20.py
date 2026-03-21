"""
compare_top20.py
================
Compares top-20 gene lists across the most informative algorithm pairs.
Explains what makes each algorithm's list different and why.

Covers the 5 most important comparisons:
  1. Strength vs HITS (Hub)          — best overall vs best spectral
  2. Strength vs Weighted K-Core     — best vs thesis baseline
  3. HITS vs DependANT Composite     — spectral vs composite
  4. Any of the above vs ECC         — hub cluster vs locally-embedded essentials
  5. VoteRank vs Strength            — spreader vs essential (why neg ctrl fails)

All per-cancer.  Reads from output/all_scores.csv + output/top20_genes.csv.
Outputs:
  Terminal: ranked comparison tables with colour annotations
  output/figures/fig_compare_*.png  — visual comparison grids

Usage:  python3 compare_top20.py
"""

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output/"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures/")
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# ANSI COLOURS
# =============================================================================
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    GREY    = "\033[90m"
    BG_DARK = "\033[48;5;235m"

def c(text, *codes):
    return "".join(codes) + str(text) + C.RESET

W = 90

def banner(text):
    inner = f"  {text}  "
    pad   = max(0, W - len(inner))
    print()
    print(c(f"  {'━'*W}", C.CYAN, C.BOLD))
    print(c(f"  {inner}{' '*pad}", C.BG_DARK, C.WHITE, C.BOLD))
    print(c(f"  {'━'*W}", C.CYAN, C.BOLD))

def section(title, sub=""):
    print(f"\n  {c(title, C.CYAN, C.BOLD)}")
    if sub: print(f"  {c(sub, C.GREY)}")
    print(c(f"  {'─'*W}", C.GREY))

def rule():
    print(c(f"  {'─'*W}", C.GREY))

# =============================================================================
# DESIGN
# =============================================================================
CANCER_LABEL = {"BRCA":"Breast (BRCA)", "LUAD":"Lung (LUAD)", "LIHC":"Liver (LIHC)"}
CANCER_COLOR = {"BRCA":"#4e79a7", "LUAD":"#f28e2b", "LIHC":"#59a14f"}
CANCERS      = ["BRCA","LUAD","LIHC"]

ALGO_SHORT = {
    "Strength":            "Strength",
    "HITS (Hub)":          "HITS",
    "Weighted K-Core":     "K-Core",
    "DependANT Composite": "DependANT",
    "ECC (Link Cluster)":  "ECC",
    "VoteRank":            "VoteRank",
    "Eigenvector":         "Eigvec",
    "LeaderRank":          "LeaderRk",
    "PageRank":            "PageRank",
}

ALGO_HEX = {
    "Strength":            "#4e79a7",
    "HITS (Hub)":          "#e15759",
    "Weighted K-Core":     "#76b7b2",
    "DependANT Composite": "#9c755f",
    "ECC (Link Cluster)":  "#b07aa1",
    "VoteRank":            "#bab0ac",
    "Eigenvector":         "#ff9da7",
}

COSMIC_TIERS = {
    "CDK1":"T1","KIF11":"T2","PLK1":"T1","TOP2A":"T1","CDC20":"T2",
    "AURKB":"T1","RRM2":"T2","BIRC5":"T1","ESPL1":"T2","CCNA2":"T2",
    "BUB1B":"T1","MAD2L1":"T2","AURKA":"T1",
}

# The 5 comparisons to run
COMPARISONS = [
    ("Strength",        "HITS (Hub)",
     "Best overall (local) vs best spectral (global diffusion)",
     "Strength uses raw interaction weight. HITS propagates importance through "
     "the eigenvector structure. After upregulation filtering, local weight "
     "contains the same essentiality signal without global computation."),

    ("Strength",        "Weighted K-Core",
     "Best overall vs thesis baseline",
     "Both are local/hierarchical. K-Core adds shell depth (how embedded in "
     "the network core) while Strength adds edge weight sum. "
     "Agreement = structurally central AND strongly connected."),

    ("HITS (Hub)",      "DependANT Composite",
     "Best spectral vs composite (4-feature average)",
     "DependANT averages k-core + degree + eigenvector + betweenness. "
     "HITS is the leading eigenvector of A². In LUAD (DeLong p=0.14) they "
     "are statistically indistinguishable; in LIHC HITS is significantly better."),

    ("Strength",        "ECC (Link Cluster)",
     "Hub cluster (Strength) vs locally-embedded essentials (ECC)",
     "ECC targets genes in dense local modules that are NOT network hubs. "
     "Low Strength score + high ECC = essential gene that sits in a tight "
     "clique rather than a global hub. ECC-unique genes are the discovery set "
     "that hub metrics systematically miss."),

    ("Strength",        "VoteRank",
     "Best predictor vs negative control",
     "VoteRank ranks network spreaders — nodes that maximise information "
     "diffusion. These are peripheral bridging nodes, structurally the "
     "opposite of dense-core essential genes. The disagreement between "
     "Strength and VoteRank directly shows what the evaluation is measuring."),
]

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.titlesize":11,"axes.titleweight":"bold","axes.labelsize":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.facecolor":"white","axes.facecolor":"white",
    "savefig.dpi":200,"savefig.bbox":"tight","savefig.facecolor":"white",
})

# =============================================================================
# DATA LOADING
# =============================================================================
def _load(name):
    p = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(p):
        print(c(f"  ✗  {name} not found — run analysis_final.py first.", C.RED))
        return None
    return pd.read_csv(p)

def _gene_tag(gene, is_druggable):
    if gene in COSMIC_TIERS:
        return c(f"COSMIC {COSMIC_TIERS[gene]}", C.YELLOW)
    if is_druggable:
        return c("DGIdb", C.CYAN)
    return c("novel", C.GREEN)

# =============================================================================
# CORE COMPARISON FUNCTIONS
# =============================================================================

def get_ranked_genes(scores_df, cancer, algo, n=30):
    """Return top-n genes by score for a given algo/cancer."""
    sub = scores_df[(scores_df["cancer"]==cancer) & (scores_df["algo"]==algo)]
    return sub.sort_values("score", ascending=False).head(n)[["gene","score","essentiality","is_druggable","is_essential"]].reset_index(drop=True)


def compare_gene_lists(scores_df, cancer, algo_a, algo_b):
    """
    Compare ranked gene lists from two algorithms on one cancer.
    Returns three sets: only_a, shared (top-20 of each), only_b,
    plus full ranked DataFrames for each.
    """
    df_a = get_ranked_genes(scores_df, cancer, algo_a, n=20)
    df_b = get_ranked_genes(scores_df, cancer, algo_b, n=20)
    genes_a = set(df_a["gene"])
    genes_b = set(df_b["gene"])
    shared  = genes_a & genes_b
    only_a  = genes_a - genes_b
    only_b  = genes_b - genes_a
    return df_a, df_b, shared, only_a, only_b


def score_corr(scores_df, cancer, algo_a, algo_b):
    """Spearman correlation between two algorithms' full score vectors."""
    sa = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_a)][["gene","score"]].set_index("gene")
    sb = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_b)][["gene","score"]].set_index("gene")
    common = sa.index.intersection(sb.index)
    if len(common) < 10: return 0.0
    rho, _ = spearmanr(sa.loc[common,"score"].values,
                       sb.loc[common,"score"].values)
    return round(float(rho), 4)


# =============================================================================
# TERMINAL PRINT
# =============================================================================

def print_comparison(scores_df, cancer, algo_a, algo_b, explanation):
    df_a, df_b, shared, only_a, only_b = compare_gene_lists(
        scores_df, cancer, algo_a, algo_b)
    rho = score_corr(scores_df, cancer, algo_a, algo_b)

    label = CANCER_LABEL.get(cancer, cancer)
    short_a = ALGO_SHORT.get(algo_a, algo_a)
    short_b = ALGO_SHORT.get(algo_b, algo_b)

    print(f"\n  {c(label, C.BOLD)}  ·  "
          f"Rank corr (all genes): {c(f'ρ = {rho:.4f}', C.CYAN, C.BOLD)}")
    print(f"  Overlap: {c(str(len(shared)), C.GREEN, C.BOLD)}/20  |  "
          f"{c(short_a, C.BOLD)} only: {c(str(len(only_a)), C.YELLOW)}  |  "
          f"{c(short_b, C.BOLD)} only: {c(str(len(only_b)), C.YELLOW)}")

    # Side-by-side top-10 from each
    header = (f"  {'#':>2}  {'━'*7}  {c(short_a, C.BOLD):<28}"
              f"  {'#':>2}  {'━'*7}  {c(short_b, C.BOLD):<28}")
    print(f"\n{header}")
    rule()

    col_hdr = (f"  {'#':>2}  {'Gene':<8}  {'Score':>6}  {'Chron':>6}  {'Tag':<16}"
               f"  {'#':>2}  {'Gene':<8}  {'Score':>6}  {'Chron':>6}  {'Tag'}")
    print(c(col_hdr, C.BOLD))
    rule()

    for i in range(20):
        # Row for algo_a
        if i < len(df_a):
            ra   = df_a.iloc[i]
            gene_a = ra["gene"]
            in_b = gene_a in set(df_b["gene"])
            gc_a = C.GREEN if in_b else C.YELLOW
            dg_a = bool(ra["is_druggable"]) if ra["is_druggable"] in (True,False,0,1) \
                   else str(ra["is_druggable"]).lower() in ("true","1")
            tag_a = _gene_tag(gene_a, dg_a)
            ess_a = c("●", C.RED) if ra["is_essential"] else c("○", C.GREY)
            a_str = (f"  {i+1:>2}  "
                     f"{c(f'{gene_a:<8}', gc_a, C.BOLD)}"
                     f"  {ra['score']:>6.4f}"
                     f"  {ra['essentiality']:>6.2f}"
                     f"  {ess_a} {tag_a:<22}")
        else:
            a_str = " " * 52

        # Row for algo_b
        if i < len(df_b):
            rb   = df_b.iloc[i]
            gene_b = rb["gene"]
            in_a = gene_b in set(df_a["gene"])
            gc_b = C.GREEN if in_a else C.MAGENTA
            dg_b = bool(rb["is_druggable"]) if rb["is_druggable"] in (True,False,0,1) \
                   else str(rb["is_druggable"]).lower() in ("true","1")
            tag_b = _gene_tag(gene_b, dg_b)
            ess_b = c("●", C.RED) if rb["is_essential"] else c("○", C.GREY)
            b_str = (f"  {i+1:>2}  "
                     f"{c(f'{gene_b:<8}', gc_b, C.BOLD)}"
                     f"  {rb['score']:>6.4f}"
                     f"  {rb['essentiality']:>6.2f}"
                     f"  {ess_b} {tag_b}")
        else:
            b_str = ""

        print(f"{a_str}{b_str}")

    # Exclusive genes
    if only_a or only_b:
        print()
        if only_a:
            excl = df_a[df_a["gene"].isin(only_a)].sort_values("score", ascending=False)
            genes_str = ", ".join(c(g, C.YELLOW, C.BOLD) for g in excl["gene"])
            print(f"  {c('Only in ' + short_a + ':', C.BOLD)} {genes_str}")
        if only_b:
            excl = df_b[df_b["gene"].isin(only_b)].sort_values("score", ascending=False)
            genes_str = ", ".join(c(g, C.MAGENTA, C.BOLD) for g in excl["gene"])
            print(f"  {c('Only in ' + short_b + ':', C.BOLD)} {genes_str}")

    # Explanation
    print()
    words = explanation.split()
    line  = "  "
    for w in words:
        if len(line) + len(w) + 1 > W - 2:
            print(c(line, C.GREY)); line = "  " + w + " "
        else:
            line += w + " "
    if line.strip(): print(c(line, C.GREY))


# =============================================================================
# FIGURE: SCORE SCATTER GRID
# =============================================================================

def fig_score_scatter(scores_df, algo_a, algo_b, pair_title):
    cancers = [c for c in CANCERS if c in scores_df["cancer"].unique()]
    fig, axes = plt.subplots(1, len(cancers), figsize=(5*len(cancers), 5))
    if len(cancers)==1: axes=[axes]

    for ax, cancer in zip(axes, cancers):
        sa = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_a)][["gene","score","is_essential"]].set_index("gene")
        sb = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_b)][["gene","score","is_essential"]].set_index("gene")
        common = sa.index.intersection(sb.index)
        if len(common) < 10: continue

        x = sa.loc[common,"score"].values
        y = sb.loc[common,"score"].values
        ess = sa.loc[common,"is_essential"].values.astype(bool)

        ax.scatter(x[~ess], y[~ess], alpha=0.25, s=12,
                   color="#cccccc", edgecolors="none", label="Non-essential")
        ax.scatter(x[ess],  y[ess],  alpha=0.65, s=20,
                   color="#e15759", edgecolors="none", label="Essential (Chron<−0.6)", zorder=3)

        # highlight top-20 of algo_a
        top_a = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_a)].nlargest(20,"score")["gene"]
        top_b = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_b)].nlargest(20,"score")["gene"]
        shared_top = set(top_a) & set(top_b) & set(common)
        only_a_top = set(top_a) - set(top_b)
        only_b_top = set(top_b) - set(top_a)

        for gene_set, col, marker, zorder, lbl in [
            (shared_top, "#2ca02c", "D", 5, "Top-20 both"),
            (only_a_top, ALGO_HEX.get(algo_a,"#4e79a7"), "^", 5,
             f"Top-20 {ALGO_SHORT.get(algo_a,algo_a)} only"),
            (only_b_top, ALGO_HEX.get(algo_b,"#e15759"), "v", 5,
             f"Top-20 {ALGO_SHORT.get(algo_b,algo_b)} only"),
        ]:
            gl = [g for g in gene_set if g in common]
            if not gl: continue
            xi = sa.loc[gl,"score"].values
            yi = sb.loc[gl,"score"].values
            ax.scatter(xi, yi, s=60, color=col, edgecolors="black",
                       linewidths=0.5, marker=marker, zorder=zorder, label=lbl)
            for g, gx, gy in zip(gl, xi, yi):
                ax.annotate(g, (gx, gy), fontsize=7, ha="left",
                            xytext=(3,3), textcoords="offset points")

        rho, _ = spearmanr(x, y)
        ax.set_xlabel(f"{ALGO_SHORT.get(algo_a,algo_a)} score")
        ax.set_ylabel(f"{ALGO_SHORT.get(algo_b,algo_b)} score")
        ax.set_title(f"{CANCER_LABEL.get(cancer,cancer)}\nρ = {rho:.4f}", fontweight="bold")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.tick_params(length=0)

    fig.suptitle(f"Score Comparison: {ALGO_SHORT.get(algo_a,algo_a)} vs {ALGO_SHORT.get(algo_b,algo_b)}\n"
                 f"{pair_title}",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    safe = f"{ALGO_SHORT.get(algo_a,algo_a)}_vs_{ALGO_SHORT.get(algo_b,algo_b)}".replace(" ","_")
    p = os.path.join(FIG_DIR, f"fig_compare_{safe}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓  figures/{os.path.basename(p)}")


# =============================================================================
# FIGURE: UPSET-STYLE OVERLAP GRID  (all comparisons per cancer)
# =============================================================================

def fig_overlap_summary(scores_df):
    """
    A grid figure showing pairwise overlap counts between key algorithms.
    Rows/cols = the 6 key algorithms, cells = |intersection of top-20|.
    One subplot per cancer.
    """
    key_algos = ["Strength","HITS (Hub)","Weighted K-Core",
                 "DependANT Composite","ECC (Link Cluster)","VoteRank"]
    key_algos = [a for a in key_algos if a in scores_df["algo"].unique()]
    n   = len(key_algos)

    cancers = [c for c in CANCERS if c in scores_df["cancer"].unique()]
    fig, axes = plt.subplots(1, len(cancers), figsize=(6.5*len(cancers), 5.5))
    if len(cancers)==1: axes=[axes]

    cmap = LinearSegmentedColormap.from_list("ov", ["#f5f5f5","#1a5276"], N=21)

    for ax, cancer in zip(axes, cancers):
        mat = np.zeros((n, n), dtype=int)
        top_sets = {}
        for algo in key_algos:
            sub = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo)]
            top_sets[algo] = set(sub.nlargest(20,"score")["gene"])

        for i, a1 in enumerate(key_algos):
            for j, a2 in enumerate(key_algos):
                mat[i,j] = len(top_sets.get(a1,set()) & top_sets.get(a2,set()))

        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=20, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([ALGO_SHORT.get(a,a) for a in key_algos],
                           rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels([ALGO_SHORT.get(a,a) for a in key_algos], fontsize=9)
        ax.tick_params(length=0)

        for i in range(n):
            for j in range(n):
                v   = mat[i,j]
                col = "white" if v > 12 else "#222222"
                fw  = "bold" if i==j else "normal"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=10, color=col, fontweight=fw)

        plt.colorbar(im, ax=ax, shrink=0.8).set_label("Shared genes in top-20")
        ax.set_title(CANCER_LABEL.get(cancer,cancer), fontweight="bold")

    fig.suptitle("Pairwise Top-20 Gene List Overlap — Key Algorithms",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig_compare_overlap_grid.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓  figures/{os.path.basename(p)}")


# =============================================================================
# FIGURE: RANK POSITION COMPARISON  (where do genes land per algo?)
# =============================================================================

def fig_rank_position(scores_df, cancer, algo_a, algo_b):
    """
    For each gene that appears in either top-30, show its rank in algo_a vs algo_b.
    Genes in both = green line, only_a = blue arrow, only_b = orange arrow.
    """
    top_a = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_a)].nlargest(30,"score").reset_index(drop=True)
    top_b = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_b)].nlargest(30,"score").reset_index(drop=True)

    rank_a = {row["gene"]: i+1 for i,row in top_a.iterrows()}
    rank_b = {row["gene"]: i+1 for i,row in top_b.iterrows()}

    all_genes = sorted(set(rank_a.keys()) | set(rank_b.keys()))
    fig, ax   = plt.subplots(figsize=(7, 8))

    short_a = ALGO_SHORT.get(algo_a, algo_a)
    short_b = ALGO_SHORT.get(algo_b, algo_b)

    for gene in all_genes:
        ra = rank_a.get(gene)
        rb = rank_b.get(gene)
        if ra is not None and rb is not None:
            ax.plot([0, 1], [ra, rb], color="#2ca02c", alpha=0.6, lw=1.5)
            ax.text(-0.04, ra, gene, ha="right", va="center", fontsize=8.5,
                    color=ALGO_HEX.get(algo_a,"#4e79a7"),
                    fontweight="bold" if ra<=20 else "normal")
            ax.text(1.04, rb, gene, ha="left", va="center", fontsize=8.5,
                    color=ALGO_HEX.get(algo_b,"#e15759"),
                    fontweight="bold" if rb<=20 else "normal")
        elif ra is not None:
            ax.scatter(0, ra, color=ALGO_HEX.get(algo_a,"#4e79a7"),
                       s=40, zorder=4)
            ax.text(-0.04, ra, gene, ha="right", va="center", fontsize=8.5,
                    color=ALGO_HEX.get(algo_a,"#4e79a7"))
        elif rb is not None:
            ax.scatter(1, rb, color=ALGO_HEX.get(algo_b,"#e15759"),
                       s=40, zorder=4)
            ax.text(1.04, rb, gene, ha="left", va="center", fontsize=8.5,
                    color=ALGO_HEX.get(algo_b,"#e15759"))

    ax.axhline(20.5, color="#aaaaaa", ls="--", lw=1, label="Top-20 cutoff")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(31, 0)   # rank 1 at top
    ax.set_xticks([0, 1])
    ax.set_xticklabels([short_a, short_b], fontsize=11, fontweight="bold")
    ax.set_ylabel("Rank (1 = highest score)")
    ax.set_title(f"{CANCER_LABEL.get(cancer,cancer)}: Rank Positions\n"
                 f"{short_a} vs {short_b}",
                 fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    safe    = f"{short_a}_vs_{short_b}_{cancer}".replace(" ","_")
    p       = os.path.join(FIG_DIR, f"fig_compare_ranks_{safe}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓  figures/{os.path.basename(p)}")


# =============================================================================
# FIGURE: EXCLUSIVE GENES DEEP DIVE  (what are the algo-unique genes?)
# =============================================================================

def fig_exclusive_genes(scores_df, cancer, algo_a, algo_b):
    """
    Show scores AND essentiality of the genes exclusive to each algorithm's top-20.
    Helps explain the biological meaning of divergence.
    """
    top_a = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_a)].nlargest(20,"score")
    top_b = scores_df[(scores_df["cancer"]==cancer)&(scores_df["algo"]==algo_b)].nlargest(20,"score")

    genes_a = set(top_a["gene"]); genes_b = set(top_b["gene"])
    only_a  = genes_a - genes_b;  only_b  = genes_b - genes_a
    if not only_a and not only_b: return

    # Get full scores for exclusive genes
    excl_a = scores_df[(scores_df["cancer"]==cancer) &
                       (scores_df["algo"]==algo_a) &
                       (scores_df["gene"].isin(only_a))].copy()
    excl_b = scores_df[(scores_df["cancer"]==cancer) &
                       (scores_df["algo"]==algo_b) &
                       (scores_df["gene"].isin(only_b))].copy()

    # Also get the OTHER algo's score for these exclusive genes
    other_a = scores_df[(scores_df["cancer"]==cancer) &
                        (scores_df["algo"]==algo_b) &
                        (scores_df["gene"].isin(only_a))].set_index("gene")["score"].to_dict()
    other_b = scores_df[(scores_df["cancer"]==cancer) &
                        (scores_df["algo"]==algo_a) &
                        (scores_df["gene"].isin(only_b))].set_index("gene")["score"].to_dict()

    n_a = len(excl_a); n_b = len(excl_b)
    if n_a == 0 and n_b == 0: return

    short_a  = ALGO_SHORT.get(algo_a, algo_a)
    short_b  = ALGO_SHORT.get(algo_b, algo_b)
    col_a    = ALGO_HEX.get(algo_a, "#4e79a7")
    col_b    = ALGO_HEX.get(algo_b, "#e15759")
    n_panels = (1 if n_a>0 else 0) + (1 if n_b>0 else 0)
    if n_panels == 0: return

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6*n_panels, max(4, 0.5*max(n_a,n_b)+2)))
    if n_panels == 1: axes = [axes]
    ax_idx = 0

    for excl, other_scores, algo, col, label in [
        (excl_a, other_a, algo_a, col_a,
         f"Unique to {short_a}\n(rank in {short_b} > 20)"),
        (excl_b, other_b, algo_b, col_b,
         f"Unique to {short_b}\n(rank in {short_a} > 20)"),
    ]:
        if excl.empty: continue
        ax  = axes[ax_idx]; ax_idx += 1
        excl = excl.sort_values("score", ascending=False)
        genes = list(excl["gene"])
        y     = np.arange(len(genes))

        # Primary: this algo's score
        ax.barh(y, excl["score"].values, height=0.35,
                color=col, edgecolor="white", alpha=0.85,
                label=f"{ALGO_SHORT.get(algo,algo)} score")
        # Secondary: other algo's score (ghosted)
        other_vals = [other_scores.get(g, 0) for g in genes]
        ax.barh(y-0.38, other_vals, height=0.35,
                color="#cccccc", edgecolor="white", alpha=0.75,
                label=f"{ALGO_SHORT.get(algo_b if algo==algo_a else algo_a, '')} score")

        ax.set_yticks(y)
        ax.set_yticklabels(genes, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised topology score")
        ax.set_title(label, fontweight="bold")

        # Essentiality markers
        for i, (gene, row) in enumerate(excl.iterrows()):
            ess = row.get("is_essential", 0)
            chron = row.get("essentiality", 0)
            marker = "●" if ess else "○"
            color  = "#c0392b" if ess else "#aaaaaa"
            ax.text(0.02, i, f"{marker} {chron:.2f}",
                    va="center", fontsize=8, color=color)

        ax.legend(fontsize=8.5, frameon=False)
        ax.tick_params(length=0)

    fig.suptitle(f"{CANCER_LABEL.get(cancer,cancer)}: Exclusive Genes — "
                 f"{short_a} vs {short_b}\n"
                 f"● = CRISPR-essential (Chronos < −0.6)",
                 fontweight="bold", fontsize=11, y=1.02)
    plt.tight_layout()
    safe = f"{short_a}_vs_{short_b}_{cancer}".replace(" ","_")
    p    = os.path.join(FIG_DIR, f"fig_compare_exclusive_{safe}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓  figures/{os.path.basename(p)}")


# =============================================================================
# LEGEND SUMMARY
# =============================================================================

def print_legend():
    print(f"\n  {c('Legend', C.BOLD)}")
    rule()
    items = [
        (C.GREEN,   "Gene",  "appears in BOTH top-20 lists"),
        (C.YELLOW,  "Gene",  "unique to left algorithm"),
        (C.MAGENTA, "Gene",  "unique to right algorithm"),
        (C.YELLOW,  "COSMIC T1/T2", "established cancer driver (COSMIC CGC)"),
        (C.CYAN,    "DGIdb", "druggable gene (DGIdb 4.0)"),
        (C.GREEN,   "novel", "not in COSMIC or DGIdb — discovery candidate"),
        (C.RED,     "●",     "CRISPR-essential (Chronos < −0.6)"),
        (C.GREY,    "○",     "not essential at threshold"),
    ]
    for col, sym, desc in items:
        print(f"  {c(f'{sym:<12}', col, C.BOLD)}  {c(desc, C.GREY)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("Top-20 Gene List Comparison  ·  Key Algorithm Pairs")

    scores_df = _load("all_scores.csv")
    if scores_df is None: return

    # Ensure boolean columns
    scores_df["is_essential"] = scores_df["is_essential"].astype(bool)
    scores_df["is_druggable"] = scores_df["is_druggable"].astype(bool)

    print_legend()

    print(f"\n  {c('Figures will be saved to output/figures/', C.GREY)}")
    print(f"\n  {c('Generating figures...', C.CYAN)}")
    fig_overlap_summary(scores_df)

    for cancer in CANCERS:
        if cancer not in scores_df["cancer"].unique(): continue
        # Rank-position figures for the two most important pairs per cancer
        fig_rank_position(scores_df, cancer, "Strength", "HITS (Hub)")
        fig_rank_position(scores_df, cancer, "Strength", "ECC (Link Cluster)")
        # Exclusive-gene deep dives
        fig_exclusive_genes(scores_df, cancer, "Strength", "HITS (Hub)")
        fig_exclusive_genes(scores_df, cancer, "Strength", "ECC (Link Cluster)")
        fig_exclusive_genes(scores_df, cancer, "Strength", "VoteRank")

    for algo_a, algo_b, _, _ in COMPARISONS:
        if algo_a in scores_df["algo"].unique() and algo_b in scores_df["algo"].unique():
            fig_score_scatter(scores_df, algo_a, algo_b,
                              f"{ALGO_SHORT.get(algo_a,algo_a)} vs "
                              f"{ALGO_SHORT.get(algo_b,algo_b)}")

    # ─── terminal tables ──────────────────────────────────────────────────────
    for algo_a, algo_b, pair_title, explanation in COMPARISONS:
        if algo_a not in scores_df["algo"].unique(): continue
        if algo_b not in scores_df["algo"].unique(): continue

        short_a = ALGO_SHORT.get(algo_a, algo_a)
        short_b = ALGO_SHORT.get(algo_b, algo_b)
        banner(f"{short_a}  vs  {short_b}  ·  {pair_title}")

        for cancer in CANCERS:
            if cancer not in scores_df["cancer"].unique(): continue
            print_comparison(scores_df, cancer, algo_a, algo_b, explanation)

    # ─── final summary ────────────────────────────────────────────────────────
    print(f"\n  {c('━'*W, C.CYAN, C.BOLD)}")
    print(f"  {c('Summary: what the comparisons tell you', C.WHITE, C.BOLD)}")
    print(c(f"  {'─'*W}", C.GREY))

    insights = [
        ("Strength vs HITS",
         "Near-identical gene lists, especially in LUAD and LIHC. High score "
         "correlation (ρ ≈ 0.85–0.95). Strength wins by AUROC because weighted "
         "degree directly captures the sum of interaction confidence — no noise "
         "from global diffusion. The key insight is that after the upregulation "
         "filter, the network is concentrated enough that local weight suffices."),

        ("Strength vs K-Core",
         "High agreement on the top genes (CDK1, PLK1, KIF11). Divergence happens "
         "in the 10–20 range, where K-Core penalises high-degree peripheral nodes "
         "that Strength rewards. K-Core's shell index adds a depth requirement: "
         "a gene must be embedded deep in the network core, not just well-connected."),

        ("HITS vs DependANT",
         "Very high overlap — DependANT includes eigenvector as one of its four "
         "features, so HITS (which equals eigenvector) dominates the composite. "
         "DeLong shows p>0.05 in LUAD: statistically indistinguishable. The "
         "DependANT bonus is k-core + betweenness, which occasionally promotes "
         "bridge genes not captured by spectral methods."),

        ("Strength vs ECC",
         "Low overlap (often 0–3 shared genes). This is the most important "
         "comparison: ECC finds a completely different gene population — locally-"
         "embedded essential genes in tight cliques that are peripheral by "
         "spectral/degree metrics. These are the 'dark matter' essentials. "
         "Low AUROC (ECC ≈ 0.67–0.72) means they are rare, not absent."),

        ("Strength vs VoteRank",
         "Near-zero overlap. VoteRank maximises geographic spread across the "
         "network; Strength maximises local interaction weight. The gene lists "
         "are structurally disjoint. This is why VoteRank AUROC ≈ 0.66 — it "
         "is finding the opposite of what CRISPR essentiality selects for."),
    ]

    for title, body in insights:
        print(f"\n  {c('▸ ' + title, C.CYAN, C.BOLD)}")
        words = body.split()
        line  = "    "
        for w in words:
            if len(line)+len(w)+1 > W-2:
                print(c(line, C.GREY)); line = "    " + w + " "
            else:
                line += w + " "
        if line.strip(): print(c(line, C.GREY))

    print(f"\n  {c('━'*W, C.CYAN, C.BOLD)}\n")


if __name__ == "__main__":
    main()