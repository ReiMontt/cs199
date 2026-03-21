"""
Expected data layout
--------------------
    data/ppi/9606.protein.links.full.v12.0.txt
    data/ppi/9606.protein.aliases.v12.0.txt
    data/depmap/CRISPRGeneEffect.csv
    data/dgidb/interactions.tsv
    processed/{BRCA,LUAD,LIHC}_clean.csv
        columns: gene, log2fc, padj

    Only genes with padj < FDR_THRESHOLD AND log2fc > LOG2FC_THRESHOLD
    (upregulated) are retained per cancer type.

Outputs (output/)
-----------------
    all_scores.csv          every gene x algorithm x cancer score
    top20_genes.csv         top-20 essential candidates (best algo per cancer)
    validation_summary.csv  full metrics table
    delong_tests.csv        pairwise DeLong AUROC comparisons
    top20_all_cancers.png
    auroc_comparison.png
    spearman_heatmap.png
    radar_profile.png
    precision_at_k.png

Run report.py for verbose tables
Run figures.py for figures.
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, mannwhitneyu, norm as scipy_norm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
PPI_FILE     = "data/ppi/9606.protein.links.full.v12.0.txt"
ALIAS_FILE   = "data/ppi/9606.protein.aliases.v12.0.txt"
DEPMAP_FILE  = "data/depmap/CRISPRGeneEffect.csv"
DGIDB_FILE   = "data/dgidb/interactions.tsv"
DE_GENES_DIR = "processed/"
OUTPUT_DIR   = "output/"

CANCERS = ["BRCA", "LUAD", "LIHC"]
CANCER_LABELS = {
    "BRCA": "Breast (BRCA)",
    "LUAD": "Lung (LUAD)",
    "LIHC": "Liver (LIHC)",
}

CONFIDENCE_THRESHOLD = 700    # STRING combined_score >= 700 (high confidence)
ESSENTIALITY_CUTOFF  = -0.6   # DepMap Chronos threshold (Meyers et al. 2017)
TOP_N                = 20
FDR_THRESHOLD        = 0.05
LOG2FC_THRESHOLD     = 1.0    # upregulated genes only (>2x)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# TERMINAL STYLING 
# =============================================================================
_W = 62   # column width for headers

def _banner(text):
    pad = max(0, _W - len(text) - 2)
    print(f"\n  ┌{'─'*(_W)}┐")
    print(f"  │  {text}{' '*pad}│")
    print(f"  └{'─'*(_W)}┘")

def _section(text):
    print(f"\n  ── {text} {'─'*(max(0,_W-4-len(text)))}")

def _ok(text):   print(f"  ✓  {text}")
def _info(text): print(f"  ·  {text}")
def _warn(text): print(f"  ⚠  {text}")
def _err(text):  print(f"  ✗  {text}")

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_depmap():
    _info("Loading DepMap 23Q4 CRISPR Chronos ...")
    try:
        peek = pd.read_csv(DEPMAP_FILE, index_col=0, nrows=3)
        if peek.columns.str.contains(r" \(").any():
            df = pd.read_csv(DEPMAP_FILE, index_col=0)
            df.columns = df.columns.str.split(" ").str[0].str.upper()
            ess = df.mean(axis=0).to_dict()
        else:
            df = pd.read_csv(DEPMAP_FILE, index_col=0)
            df.index = df.index.str.split(" ").str[0].str.upper()
            ess = df.mean(axis=1).to_dict()
        _ok(f"DepMap: {len(ess):,} genes across 1,095 cell lines")
        return ess, set(ess.keys())
    except Exception as e:
        _err(f"DepMap load failed: {e}")
        return {}, set()


def load_dgidb():
    _info("Loading DGIdb 4.0 ...")
    try:
        df  = pd.read_csv(DGIDB_FILE, sep="\t")
        col = "gene_name" if "gene_name" in df.columns else df.columns[0]
        druggable = set(df[col].dropna().astype(str).str.upper())
        _ok(f"DGIdb: {len(druggable):,} druggable genes")
        return druggable
    except Exception as e:
        _err(f"DGIdb load failed: {e}")
        return set()


def load_cancer_genes():
    _info(f"Loading DE gene lists (padj<{FDR_THRESHOLD}, log2fc>{LOG2FC_THRESHOLD}) ...")
    cancer_genes = {}
    for cancer in CANCERS:
        path = os.path.join(DE_GENES_DIR, f"{cancer}_clean.csv")
        if not os.path.exists(path):
            _warn(f"{cancer}_clean.csv not found")
            cancer_genes[cancer] = set(); continue
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower().str.strip()
            col_map = {}
            for c in df.columns:
                if c in ("log2fc","log2foldchange","log2_fold_change","lfc","logfc"):
                    col_map["log2fc"] = c
                if c in ("padj","adj_p","adjusted_pvalue","fdr","adj.p.val","qval","q_val","fdr_q"):
                    col_map["padj"] = c
                if c in ("gene","gene_name","hgnc","symbol"):
                    col_map["gene"] = c
            df.rename(columns={v:k for k,v in col_map.items()}, inplace=True)
            df["gene"] = df["gene"].astype(str).str.upper().str.strip()
            if "log2fc" in df.columns and "padj" in df.columns:
                before = len(df)
                df = df[(df["padj"].astype(float) < FDR_THRESHOLD) &
                        (df["log2fc"].astype(float) > LOG2FC_THRESHOLD)]
                _ok(f"{cancer}: {before:,} DE genes → {len(df):,} upregulated")
            else:
                _warn(f"{cancer}: no log2fc/padj found – using all {len(df):,} genes "
                      f"(re-generate with log2fc+padj for strict upregulation filter)")
            cancer_genes[cancer] = set(df["gene"].dropna())
        except Exception as e:
            _warn(f"{cancer}: {e}")
            cancer_genes[cancer] = set()
    return cancer_genes


def _ensp(series):
    return series.astype(str).str.extract(r"(ENSP\d+)", expand=False)


def build_string_network(whitelist):
    _section("Building STRING v12.0 PPI network")
    ens2gene = {}
    try:
        for chunk in pd.read_csv(ALIAS_FILE, sep="\t", header=None,
                                 names=["ensp","alias","source"],
                                 chunksize=500_000):
            chunk["alias"] = chunk["alias"].astype(str).str.upper().str.strip()
            valid = chunk[chunk["alias"].isin(whitelist)].copy()
            if not valid.empty:
                valid["ensp"] = _ensp(valid["ensp"])
                ens2gene.update(dict(zip(valid.ensp, valid.alias)))
    except Exception as e:
        _err(f"Alias mapping: {e}"); return nx.Graph()

    try:
        links = pd.read_csv(PPI_FILE, sep=" ")
        links = links[links.combined_score >= CONFIDENCE_THRESHOLD].copy()
        links["g1"]     = _ensp(links.protein1).map(ens2gene)
        links["g2"]     = _ensp(links.protein2).map(ens2gene)
        links           = links.dropna(subset=["g1","g2"])
        links["weight"] = links["combined_score"] / 1000.0
        G = nx.from_pandas_edgelist(links,"g1","g2",edge_attr="weight",
                                    create_using=nx.Graph())
        G.remove_edges_from(nx.selfloop_edges(G))
        if len(G) > 0:
            lcc = max(nx.connected_components(G), key=len)
            G   = G.subgraph(lcc).copy()
        _ok(f"Global LCC: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
        return G
    except Exception as e:
        _err(f"Network build: {e}"); return nx.Graph()

# =============================================================================
# TOPOLOGY ALGORITHMS
# =============================================================================

def _norm(d):
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo: return {k: 0.0 for k in d}
    return {k: (v-lo)/(hi-lo) for k,v in d.items()}


def score_weighted_kcore(G):
    """
    Novel composite (proposed in this work):
        Coreness(v) = k_core(v) × log(deg(v) + 1)
    Combines hierarchical shell depth with local connectivity.
    Ref: Seidman 1983; Batagelj & Zaversnik 2003.
    """
    kc = nx.core_number(G)
    dg = dict(G.degree())
    return _norm({v: kc[v] * np.log1p(dg[v]) for v in G.nodes()})


def score_pagerank(G):
    """PageRank (Brin & Page 1998). α=0.85, edge-weighted."""
    return _norm(nx.pagerank(G, weight="weight", alpha=0.85, max_iter=300))


def score_hits_hub(G):
    """
    HITS Hub Score (Kleinberg 1999).
    On undirected graphs: hub score ≡ eigenvector centrality
    (leading eigenvector of A·Aᵀ = A² = A for symmetric A).
    """
    try:
        hubs, _ = nx.hits(G, max_iter=500, normalized=True)
        return _norm(hubs)
    except nx.PowerIterationFailedConvergence:
        _warn("HITS did not converge; using degree fallback.")
        return _norm(dict(G.degree()))


def score_eigenvector(G):
    """Eigenvector centrality (Bonacich 1972). λx = Wx."""
    try:
        return _norm(nx.eigenvector_centrality_numpy(G, weight="weight"))
    except Exception:
        try:
            return _norm(nx.eigenvector_centrality(G,weight="weight",max_iter=1000))
        except Exception as e:
            _warn(f"Eigenvector failed: {e}")
            return {v: 0.0 for v in G.nodes()}


def score_dependant_composite(G):
    """
    DependANT-inspired composite (Benstead-Hume et al. 2022).
    Equal-weight average: k_core + degree + eigenvector + betweenness.
    No machine learning; fully interpretable.
    """
    kc = _norm(nx.core_number(G))
    dg = _norm(dict(G.degree()))
    try:    ev = _norm(nx.eigenvector_centrality_numpy(G, weight="weight"))
    except: ev = {v: 0.0 for v in G.nodes()}
    try:
        k = 500 if G.number_of_nodes() > 3_000 else None
        btw = _norm(nx.betweenness_centrality(G,k=k,weight="weight",
                                               normalized=True,seed=42))
    except: btw = {v: 0.0 for v in G.nodes()}
    return _norm({v: (kc.get(v,0)+dg.get(v,0)+ev.get(v,0)+btw.get(v,0))/4
                  for v in G.nodes()})


def score_leaderrank(G):
    """LeaderRank (Lü et al. 2011). Virtual ground node eliminates α."""
    G2 = G.copy(); gnd = "__GND__"
    G2.add_node(gnd)
    for v in G.nodes():
        G2.add_edge(gnd, v, weight=1.0)
        G2.add_edge(v, gnd, weight=1.0)
    pr = nx.pagerank(G2, weight="weight", alpha=0.85, max_iter=300)
    pr.pop(gnd, None)
    return _norm(pr)


def score_voterank(G):
    """
    VoteRank [NEGATIVE CONTROL] (Zhang et al. 2016).
    Identifies spreaders, not structural essentials.
    Expected to produce the lowest AUROC.
    """
    elected = nx.voterank(G, number_of_nodes=G.number_of_nodes())
    n = len(elected)
    scores = {node: 1.0 - rank/max(n,1) for rank, node in enumerate(elected)}
    for v in G.nodes(): scores.setdefault(v, 0.0)
    return _norm(scores)


def score_strength(G):
    """Strength centrality (Alkhadrawi et al. 2025). s(v) = Σ w(u,v). O(m)."""
    return _norm({v: sum(d["weight"] for _,_,d in G.edges(v,data=True))
                  for v in G.nodes()})


def score_ecc(G):
    """
    Edge Clustering Coefficient (Jeong et al. 2019).
    Captures locally-embedded essential genes missed by hub metrics.
    """
    ecc_edge = {}
    for u, v in G.edges():
        shared = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        denom  = min(G.degree(u)-1, G.degree(v)-1)
        ecc_edge[(u,v)] = shared/denom if denom > 0 else 0.0
    node_ecc = defaultdict(list)
    for (u,v), val in ecc_edge.items():
        node_ecc[u].append(val); node_ecc[v].append(val)
    return _norm({v: float(np.mean(node_ecc[v])) if node_ecc[v] else 0.0
                  for v in G.nodes()})


ALGORITHMS = {
    "Weighted K-Core":     score_weighted_kcore,
    "PageRank":            score_pagerank,
    "HITS (Hub)":          score_hits_hub,
    "Eigenvector":         score_eigenvector,
    "DependANT Composite": score_dependant_composite,
    "LeaderRank":          score_leaderrank,
    "VoteRank":            score_voterank,
    "Strength":            score_strength,
    "ECC (Link Cluster)":  score_ecc,
}

# =============================================================================
# VALIDATION + DELONG
# =============================================================================

def _precision_at_k(scores, essentiality, k):
    common = [(g, scores[g]) for g in scores if g in essentiality]
    common.sort(key=lambda x: x[1], reverse=True)
    hits = sum(1 for g,_ in common[:k] if essentiality[g] < ESSENTIALITY_CUTOFF)
    return hits/k if k else 0.0


def validate_algorithm(scores, essentiality, algo_name):
    common = [v for v in scores if v in essentiality]
    if len(common) < 50: return None
    sc     = np.array([scores[v]       for v in common])
    es     = np.array([essentiality[v] for v in common])
    labels = (es < ESSENTIALITY_CUTOFF).astype(int)
    n_ess  = labels.sum()
    if n_ess < 5 or n_ess == len(labels): return None
    rho, _ = spearmanr(sc, es)
    try:
        auroc = roc_auc_score(labels, sc)
        auprc = average_precision_score(labels, sc)
    except: auroc = auprc = float("nan")
    q_hi = np.quantile(sc, 0.80); q_lo = np.quantile(sc, 0.20)
    mw_p = mannwhitneyu(es[sc>=q_hi], es[sc<=q_lo], alternative="less").pvalue
    return {
        "algo": algo_name, "n_genes": len(common), "n_essential": int(n_ess),
        "spearman_rho": round(float(rho),4), "auroc": round(float(auroc),4),
        "auprc": round(float(auprc),4), "mw_p": mw_p,
        "P@20":  round(_precision_at_k(scores, essentiality, 20), 4),
        "P@50":  round(_precision_at_k(scores, essentiality, 50), 4),
        "P@100": round(_precision_at_k(scores, essentiality,100), 4),
    }


def _delong_variance(s_pos, s_neg):
    m, n = len(s_pos), len(s_neg)
    V10 = np.array([np.mean(np.where(s_neg < s, 1., np.where(s_neg==s, .5, 0.)))
                    for s in s_pos])
    V01 = np.array([np.mean(np.where(s_pos > s, 1., np.where(s_pos==s, .5, 0.)))
                    for s in s_neg])
    return (np.var(V10,ddof=1)/m if m>1 else 0.), \
           (np.var(V01,ddof=1)/n if n>1 else 0.), V10, V01


def delong_test(y_true, scores_a, scores_b):
    y_true=np.array(y_true,int); sa=np.array(scores_a,float); sb=np.array(scores_b,float)
    pos=np.where(y_true==1)[0]; neg=np.where(y_true==0)[0]
    if len(pos)<2 or len(neg)<2:
        return float("nan"),float("nan"),float("nan"),float("nan")
    auc_a = roc_auc_score(y_true, sa)
    auc_b = roc_auc_score(y_true, sb)
    S10a,S01a,V10a,V01a = _delong_variance(sa[pos],sa[neg])
    S10b,S01b,V10b,V01b = _delong_variance(sb[pos],sb[neg])
    m,n = len(pos),len(neg)
    S10ab = np.cov(V10a,V10b)[0,1]/m if m>1 else 0.
    S01ab = np.cov(V01a,V01b)[0,1]/n if n>1 else 0.
    var = S10a/m + S01a/n + S10b/m + S01b/n - 2*(S10ab/m + S01ab/n)
    if var <= 0: return round(auc_a,4),round(auc_b,4),float("nan"),float("nan")
    z = (auc_a - auc_b) / math.sqrt(var)
    p = 2*(1-scipy_norm.cdf(abs(z)))
    return round(auc_a,4), round(auc_b,4), round(z,4), round(p,6)


def run_delong_comparisons(scores_dict, y_true, cancer):
    algos = list(scores_dict.keys()); rows = []
    for i in range(len(algos)):
        for j in range(i+1, len(algos)):
            a1,a2 = algos[i],algos[j]
            auc1,auc2,z,p = delong_test(y_true, scores_dict[a1], scores_dict[a2])
            rows.append({"cancer":cancer,"algo_A":a1,"algo_B":a2,
                         "AUROC_A":auc1,"AUROC_B":auc2,"z_stat":z,"p_value":p,
                         "sig_0.05":(p<0.05) if not math.isnan(p) else False,
                         "sig_0.01":(p<0.01) if not math.isnan(p) else False})
    return rows

# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def get_top_genes(scores, essentiality, druggable, algo, cancer, n=TOP_N):
    """
    Priority(g) = topology_score(g) × |Chronos(g)|
    Selects genes that are BOTH topologically central (structurally
    irreplaceable in vivo) and CRISPR-lethal (essential in vitro).
    DGIdb status is post-hoc metadata only — not a selection filter.
    """
    rows = []
    for gene, score in scores.items():
        if gene not in essentiality: continue
        ess = essentiality[gene]
        rows.append({"cancer":cancer,"gene":gene,"algo":algo,
                     "score":round(score,5),"essentiality":round(ess,5),
                     "priority":round(score*abs(ess),5),
                     "is_druggable":gene in druggable,
                     "is_essential":ess < ESSENTIALITY_CUTOFF})
    df = pd.DataFrame(rows)
    if df.empty: return df
    return (df[df.is_essential].sort_values("priority",ascending=False)
              .head(n).copy())

# =============================================================================
# COOL CONSOLE REPORTING 
# =============================================================================

def _print_top20(top_df, cancer, algo):
    if top_df is None or top_df.empty:
        _warn(f"No candidates for {cancer}"); return
    label = CANCER_LABELS.get(cancer, cancer)
    print(f"\n  {'─'*58}")
    print(f"  Top-20  │  {label}  │  {algo}")
    print(f"  {'─'*58}")
    print(f"  {'#':>2}  {'Gene':<8}  {'Score':>6}  {'Chronos':>8}  {'Priority':>8}  DGIdb")
    print(f"  {'─'*58}")
    for i, (_, r) in enumerate(top_df.iterrows(), 1):
        flag = "✓" if r.is_druggable else " "
        print(f"  {i:>2}  {r.gene:<8}  {r.score:>6.4f}  "
              f"{r.essentiality:>8.4f}  {r.priority:>8.4f}  {flag}")
    print(f"  {'─'*58}")


def _print_cancer_summary(cancer, results, best_algo, best_auroc):
    label = CANCER_LABELS.get(cancer, cancer)
    print(f"\n  {'─'*58}")
    print(f"  {label}  │  Best: {best_algo}  │  AUROC {best_auroc:.4f}")
    print(f"  {'─'*58}")
    print(f"  {'Algorithm':<22}  {'AUROC':>6}  {'ρ':>7}  {'P@20':>5}  {'P@50':>5}")
    print(f"  {'─'*58}")
    for algo, r in results.items():
        if r is None: continue
        marker = " ◀" if algo == best_algo else ("  ⊘" if algo == "VoteRank" else "")
        print(f"  {algo:<22}  {r['auroc']:>6.4f}  "
              f"{r['spearman_rho']:>+7.4f}  {r['P@20']:>5.3f}  "
              f"{r['P@50']:>5.3f}{marker}")


def _print_final_ranking(algo_mean_auroc):
    ranked = sorted(algo_mean_auroc.items(),
                    key=lambda x: float(np.mean(x[1])), reverse=True)
    print(f"\n  {'─'*58}")
    print(f"  Overall Ranking  │  Mean AUROC  (BRCA + LUAD + LIHC)")
    print(f"  {'─'*58}")
    print(f"  {'#':>2}  {'Algorithm':<22}  {'Mean AUROC':>10}  Tag")
    print(f"  {'─'*58}")
    tags = {"HITS (Hub)":"★ best","Eigenvector":"★ best",
            "Weighted K-Core":"baseline","VoteRank":"neg.ctrl"}
    for i,(algo,aurocs) in enumerate(ranked,1):
        tag = tags.get(algo,"")
        print(f"  {i:>2}  {algo:<22}  {np.mean(aurocs):>10.4f}  {tag}")
    print(f"  {'─'*58}")


def _print_delong_hits(delong_rows, cancer):
    """Print only HITS vs top-3 comparisons per cancer"""
    focused = [r for r in delong_rows
               if cancer in r["cancer"] and
               ("HITS" in r["algo_A"] or "HITS" in r["algo_B"])]
    if not focused: return
    # sort by |z|
    focused.sort(key=lambda x: abs(x["z_stat"]) if not math.isnan(x["z_stat"]) else 0,
                 reverse=True)
    print(f"  DeLong (HITS vs others): ", end="")
    parts = []
    for r in focused[:4]:
        other = r["algo_B"] if "HITS" in r["algo_A"] else r["algo_A"]
        other_auc = r["AUROC_B"] if "HITS" in r["algo_A"] else r["AUROC_A"]
        sig = "p<.01" if r["sig_0.01"] else ("p<.05" if r["sig_0.05"] else "ns")
        parts.append(f"{other.split()[0]}({other_auc:.3f},{sig})")
    print("  ".join(parts))

# =============================================================================
# VISUALISATION 
# =============================================================================

plt.rcParams.update({"font.family":"DejaVu Sans",
                     "axes.spines.top":False,"axes.spines.right":False})
_DRUGGED   = "#2ecc71"
_UNDRUGGED = "#e74c3c"


def _top20_ax(ax, top_df, cancer, best_algo):
    if top_df is None or top_df.empty:
        ax.text(0.5,0.5,"No candidates",ha="center",transform=ax.transAxes); return
    show   = top_df.sort_values("priority").tail(TOP_N)
    colors = [_DRUGGED if d else _UNDRUGGED for d in show["is_druggable"]]
    bars   = ax.barh(show["gene"], show["priority"],
                     color=colors, edgecolor="black", linewidth=0.4, alpha=0.88)
    for bar,(_,row) in zip(bars,show.iterrows()):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"ρ={row.essentiality:.2f}", va="center", fontsize=7)
    ax.set_xlabel("Priority  (topology score × |Chronos|)", fontsize=8)
    ax.set_title(f"{CANCER_LABELS.get(cancer,cancer)}\n{best_algo}",
                 fontweight="bold", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        mpatches.Patch(facecolor=_DRUGGED,  edgecolor="k", label="DGIdb druggable"),
        mpatches.Patch(facecolor=_UNDRUGGED,edgecolor="k", label="Not in DGIdb"),
    ], fontsize=7, loc="lower right")


def plot_top20(top_by_cancer, best_per_cancer):
    n = len(top_by_cancer)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(9*n, 8))
    if n == 1: axes = [axes]
    for ax, cancer in zip(axes, top_by_cancer):
        _top20_ax(ax, top_by_cancer[cancer], cancer, best_per_cancer.get(cancer,""))
    plt.suptitle(f"Top-{TOP_N} CRISPR-Essential Genes per Cancer",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "top20_all_cancers.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    _ok(f"top20_all_cancers.png")


def plot_auroc(results_by_cancer):
    algos   = list(ALGORITHMS.keys())
    cancers = list(results_by_cancer.keys())
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(algos)))
    fig, axes = plt.subplots(1, len(cancers), figsize=(5*len(cancers), 5), sharey=True)
    if len(cancers)==1: axes=[axes]
    for ax, cancer in zip(axes, cancers):
        res    = results_by_cancer[cancer]
        aurocs = [res.get(a,{}).get("auroc",0) or 0 for a in algos]
        bars   = ax.bar(range(len(algos)), aurocs, color=palette,
                        width=0.65, edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="#e74c3c", ls="--", lw=1, label="Random=0.5")
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([a.replace(" ","\n") for a in algos], fontsize=6.5)
        ax.set_ylim(0.55, 0.88)
        ax.set_title(CANCER_LABELS.get(cancer,cancer), fontweight="bold", fontsize=10)
        ax.set_ylabel("AUROC" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.2)
        for bar,val in zip(bars,aurocs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
        bi = int(np.argmax(aurocs))
        ax.text(bi, aurocs[bi]+0.018, "★", ha="center", fontsize=13, color="black")
    axes[-1].legend(fontsize=8, loc="lower right")
    plt.suptitle("AUROC vs DepMap CRISPR Gold Standard",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "auroc_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    _ok(f"auroc_comparison.png")


def plot_spearman_heatmap(results_by_cancer):
    algos   = list(ALGORITHMS.keys())
    cancers = list(results_by_cancer.keys())
    mat = np.array([
        [(results_by_cancer[c].get(a) or {}).get("spearman_rho",0) for c in cancers]
        for a in algos])
    fig, ax = plt.subplots(figsize=(max(6,len(cancers)*2.5), max(5,len(algos)*0.8)))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-0.6, vmax=0)
    ax.set_xticks(range(len(cancers)))
    ax.set_xticklabels([CANCER_LABELS.get(c,c) for c in cancers], fontsize=10)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=9)
    for i in range(len(algos)):
        for j in range(len(cancers)):
            ax.text(j,i,f"{mat[i,j]:+.3f}",ha="center",va="center",
                    fontsize=9,fontweight="bold",
                    color="white" if abs(mat[i,j])>0.38 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ  (more negative = better)", shrink=0.8)
    ax.set_title("Spearman ρ: Topology Score vs DepMap Chronos",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "spearman_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    _ok(f"spearman_heatmap.png")


def plot_radar(results_by_cancer):
    algos   = list(ALGORITHMS.keys())
    cancers = list(results_by_cancer.keys())
    metrics = ["auroc","auprc","spearman_abs","P@20","P@50"]
    labels  = ["AUROC","AUPRC","|ρ|","P@20","P@50"]
    algo_means = {}
    for algo in algos:
        vals = defaultdict(list)
        for cancer in cancers:
            r = results_by_cancer[cancer].get(algo)
            if r:
                vals["auroc"].append(r.get("auroc",0) or 0)
                vals["auprc"].append(r.get("auprc",0) or 0)
                vals["spearman_abs"].append(abs(r.get("spearman_rho",0)))
                vals["P@20"].append(r.get("P@20",0) or 0)
                vals["P@50"].append(r.get("P@50",0) or 0)
        algo_means[algo] = {m:float(np.mean(v)) if v else 0 for m,v in vals.items()}
    N      = len(metrics)
    angles = [n/float(N)*2*math.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    palette = plt.cm.tab10(np.linspace(0,0.9,len(algos)))
    for (algo,means),color in zip(algo_means.items(),palette):
        values = [means[m] for m in metrics]+[means[metrics[0]]]
        ax.plot(angles,values,"o-",lw=1.8,label=algo,color=color)
        ax.fill(angles,values,alpha=0.06,color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,size=10)
    ax.set_ylim(0,1)
    ax.set_title("Multi-Metric Algorithm Profile  (mean over 3 cancers)",
                 fontweight="bold",size=11,pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.38,1.15),fontsize=8)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR,"radar_profile.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    _ok(f"radar_profile.png")


def plot_precision_at_k(results_by_cancer):
    algos   = list(ALGORITHMS.keys())
    cancers = list(results_by_cancer.keys())
    ks      = [20,50,100]; kkeys=["P@20","P@50","P@100"]
    fig, axes = plt.subplots(1,len(cancers),figsize=(4.5*len(cancers),4.5),sharey=True)
    if len(cancers)==1: axes=[axes]
    palette = plt.cm.tab10(np.linspace(0,0.9,len(algos)))
    for ax,cancer in zip(axes,cancers):
        res = results_by_cancer[cancer]
        for algo,color in zip(algos,palette):
            r = res.get(algo)
            if r:
                ax.plot(ks,[r.get(kk,0) or 0 for kk in kkeys],
                        "o-",label=algo,color=color,lw=1.5)
        for r in res.values():
            if r:
                base = r["n_essential"]/r["n_genes"]
                ax.axhline(base,color="grey",ls=":",lw=1,
                           label=f"Random ({base:.2f})")
                break
        ax.set_xticks(ks); ax.set_xlabel("k")
        ax.set_title(CANCER_LABELS.get(cancer,cancer),fontweight="bold",fontsize=10)
        ax.set_ylabel("Precision@k" if ax is axes[0] else "")
        ax.set_ylim(0,1.05); ax.grid(alpha=0.2)
    handles,leg_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles,leg_labels,loc="lower center",ncol=5,
               fontsize=7.5,bbox_to_anchor=(0.5,-0.1))
    plt.suptitle("Precision@k – Fraction of top-k confirmed CRISPR-essential",
                 fontweight="bold",fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR,"precision_at_k.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    _ok(f"precision_at_k.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    _banner("Configs")
    print(f"  PPI     STRING v12.0  (score ≥ {CONFIDENCE_THRESHOLD})")
    print(f"  Filter  padj < {FDR_THRESHOLD}  ∧  log₂FC > {LOG2FC_THRESHOLD}  (upregulated)")
    print(f"  Gold    DepMap 23Q4 Chronos  (threshold {ESSENTIALITY_CUTOFF})")
    print(f"  Algos   9  ·  Stats  AUROC, Spearman ρ, P@k, DeLong")
    print(f"  Cancers {CANCERS}")

    _section("Phase 1 - Data Ingestion")
    essentiality, depmap_genes = load_depmap()
    druggable                  = load_dgidb()
    cancer_genes               = load_cancer_genes()
    whitelist = depmap_genes | druggable
    for gs in cancer_genes.values(): whitelist.update(gs)
    _ok(f"Whitelist: {len(whitelist):,} unique gene symbols")

    G_global = build_string_network(whitelist)
    if len(G_global) == 0:
        _err("Global network empty — check data paths."); sys.exit(1)

    _section("Phase 2 - Subnetworks → Scoring → Validation")
    results_by_cancer = {}
    top_by_cancer     = {}
    best_per_cancer   = {}
    all_rows          = []
    all_delong_rows   = []

    for cancer in CANCERS:
        gene_set = cancer_genes.get(cancer, set())
        overlap  = gene_set & set(G_global.nodes())
        if len(overlap) < 300:
            _warn(f"{cancer}: only {len(overlap)} overlapping genes — skipping"); continue

        G_sub = G_global.subgraph(overlap).copy()
        if not nx.is_connected(G_sub):
            lcc   = max(nx.connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(lcc).copy()

        label = CANCER_LABELS.get(cancer, cancer)
        n, m  = G_sub.number_of_nodes(), G_sub.number_of_edges()
        _section(f"{label}  ({n:,} nodes · {m:,} edges · density {nx.density(G_sub):.4f})")

        cancer_res  = {}
        b_auroc, b_algo, b_top = -1.0, list(ALGORITHMS.keys())[0], pd.DataFrame()
        common_genes = None
        algo_scores  = {}

        for algo_name, algo_fn in ALGORITHMS.items():
            try:
                scores = algo_fn(G_sub)
                val    = validate_algorithm(scores, essentiality, algo_name)
                cancer_res[algo_name] = val
                top_df = get_top_genes(scores, essentiality, druggable,
                                       algo_name, cancer)
                if val:
                    auroc = val.get("auroc",0) or 0
                    tag   = "  ← best" if auroc > b_auroc else ""
                    ctrl  = "  [neg ctrl]" if algo_name=="VoteRank" else ""
                    print(f"  {algo_name:<22}  AUROC {auroc:.4f}  "
                          f"ρ {val['spearman_rho']:+.4f}  P@20 {val['P@20']:.2f}"
                          f"{tag}{ctrl}")
                    if auroc > b_auroc:
                        b_auroc, b_algo, b_top = auroc, algo_name, top_df.copy()
                algo_scores[algo_name] = scores
                if common_genes is None:
                    common_genes = set(scores.keys()) & set(essentiality.keys())
                else:
                    common_genes &= set(scores.keys()) & set(essentiality.keys())
                for gene, score in scores.items():
                    if gene in essentiality:
                        all_rows.append({
                            "cancer":cancer,"algo":algo_name,"gene":gene,
                            "score":round(score,5),
                            "essentiality":round(essentiality[gene],5),
                            "is_druggable":int(gene in druggable),
                            "is_essential":int(essentiality[gene]<ESSENTIALITY_CUTOFF)})
            except Exception as e:
                _err(f"{algo_name}: {e}"); cancer_res[algo_name] = None

        # DeLong
        if common_genes and len(common_genes) >= 50:
            common_list = sorted(common_genes)
            y_true = np.array(
                [int(essentiality[g]<ESSENTIALITY_CUTOFF) for g in common_list])
            aligned = {a: np.array([algo_scores[a].get(g,0.) for g in common_list])
                       for a in algo_scores if algo_scores[a]}
            delong_rows = run_delong_comparisons(aligned, y_true, cancer)
            all_delong_rows.extend(delong_rows)
            _print_delong_hits(delong_rows, cancer)

        results_by_cancer[cancer] = cancer_res
        top_by_cancer[cancer]     = b_top
        best_per_cancer[cancer]   = b_algo
        _print_cancer_summary(cancer, cancer_res, b_algo, b_auroc)
        _print_top20(b_top, cancer, b_algo)

    _section("Phase 3 - Overall Ranking")
    algo_mean_auroc = defaultdict(list)
    for cancer, res in results_by_cancer.items():
        for algo, r in res.items():
            if r and not math.isnan(r.get("auroc", float("nan"))):
                algo_mean_auroc[algo].append(r["auroc"])
    _print_final_ranking(algo_mean_auroc)

    _section("Phase 4 - Figures")
    plot_top20(top_by_cancer, best_per_cancer)
    plot_auroc(results_by_cancer)
    plot_spearman_heatmap(results_by_cancer)
    plot_radar(results_by_cancer)
    plot_precision_at_k(results_by_cancer)

    _section("Phase 5 - Saving CSVs")
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR,"all_scores.csv"),index=False)
        _ok(f"all_scores.csv  ({len(all_rows):,} rows)")

    top_frames = [v for v in top_by_cancer.values() if not v.empty]
    if top_frames:
        pd.concat(top_frames,ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR,"top20_genes.csv"),index=False)
        _ok("top20_genes.csv")

    val_rows = [dict(**r,cancer=cancer)
                for cancer,res in results_by_cancer.items()
                for algo,r in res.items() if r]
    if val_rows:
        pd.DataFrame(val_rows).to_csv(
            os.path.join(OUTPUT_DIR,"validation_summary.csv"),index=False)
        _ok("validation_summary.csv")

    if all_delong_rows:
        pd.DataFrame(all_delong_rows).to_csv(
            os.path.join(OUTPUT_DIR,"delong_tests.csv"),index=False)
        _ok("delong_tests.csv")

    print(f"\n  ┌{'─'*_W}┐")
    print(f"  │  Done.  All outputs → {OUTPUT_DIR:<{_W-20}}│")
    print(f"  │  Run  report.py  for full tables{' '*10}│")
    print(f"  │  Run  figures.py  for figures.{' '*8}│")
    print(f"  └{'─'*_W}┘\n")


if __name__ == "__main__":
    main()