#!/usr/bin/env python3
"""
dfi_group_comparison.py

Compare per-residue Dynamic Flexibility Index (DFI) profiles between two groups of
enzymes (e.g. enzymes that can degrade a ligand vs. enzymes that cannot).

Reads one <name>_dfi.csv per enzyme from a directory and reports:
  * per-position group means, difference (A - B), Cliff's delta and exact label-permutation
    p-values (per position, and family-wise via the max-|difference| statistic)
  * "candidate" residues (large, consistent group difference)
  * contiguous regions with a sustained group difference (smoothed profile)
  * residues that are rigid / flexible in all of one group but none of the other
  * whole-profile similarity (Spearman correlation matrix and PCA) to see whether the
    dynamics separate the groups at all

Residue numbering:
  * default: residue IDs are assumed to be shared by all enzymes
  * --mapping: CSV with one column per enzyme (header = enzyme name) giving the residue ID
               at each aligned position (blank = gap)
  * --fasta:   alignment (aligned FASTA or Clustal .aln) whose sequence names equal the enzyme names; the i-th non-gap
               residue of each sequence is matched to the i-th row of that enzyme's DFI file

Example (files in per-enzyme folders, named to match the groups by folder or file name):
  python dfi_group_comparison.py BlaC/BlaC_TI_dfi.csv CTXM16/CTXM_TI_dfi.csv ... \
      --group-a BlaC CTXM16 SHV1 TEM1 --label-a carbapenem-inhibited \
      --group-b KPC2 NMCA SFC1 SME1   --label-b carbapenemases \
      --fasta enzymes.fasta --reference BlaC --highlight 70 73 118 211
  (a directory of <enzyme>_dfi.csv files can be given instead of individual files)
"""
import argparse
import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESID_CANDIDATES = ["resid", "res_id", "resnum", "resno", "residue", "res",
                    "residue_number", "position", "pos"]
METRIC_CANDIDATES = ["pctdfi", "pct_dfi", "%dfi", "dfi_pct", "dfi_percentile",
                     "percentile", "dfi_rank", "dfi"]
CHAIN_CANDIDATES = ("chainid", "chain_id", "chain")
COL_A, COL_B = "#D55E00", "#0072B2"


# ----------------------------------------------------------------------------- helpers
def die(msg):
    sys.exit(f"ERROR: {msg}")


def norm_id(x):
    """Normalise a residue ID to a string ('12', '12.0' -> '12'); None for blanks/NaN."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "-"):
        return None
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def sort_key(s):
    try:
        return (0, float(s), "")
    except ValueError:
        return (1, 0.0, s)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "group"


def find_col(columns, candidates, override, what):
    lower = {str(c).lower().strip(): c for c in columns}
    if override:
        if override in columns:
            return override
        if override.lower() in lower:
            return lower[override.lower()]
        die(f"column '{override}' not found. Columns available: {list(columns)}")
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    die(f"could not auto-detect the {what}. Columns found: {list(columns)}. "
        f"Specify it explicitly (--resid-col / --metric).")


# ----------------------------------------------------------------------------- loading
def load_dfi(path, args):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    rcol = find_col(df.columns, RESID_CANDIDATES, args.resid_col, "residue-ID column")
    mcol = find_col(df.columns, METRIC_CANDIDATES, args.metric, "DFI metric column")
    ccol = next((c for c in df.columns if c.lower() in CHAIN_CANDIDATES), None)
    if ccol is not None:
        chains = list(dict.fromkeys(df[ccol].astype(str)))
        if args.chain:
            df = df[df[ccol].astype(str) == args.chain]
            if df.empty:
                die(f"{path.name}: chain '{args.chain}' not found (chains: {chains})")
        elif len(chains) > 1:
            print(f"  WARNING: {path.name} has several chains {chains}; using '{chains[0]}'. "
                  f"Use --chain to choose.")
            df = df[df[ccol].astype(str) == chains[0]]
    out = pd.DataFrame({
        "resid": [norm_id(v) for v in df[rcol]],
        "value": pd.to_numeric(df[mcol], errors="coerce").to_numpy(),
    }).dropna(subset=["resid"]).reset_index(drop=True)
    already_pct = any(k in mcol.lower() for k in ("pct", "%", "percent", "rank"))
    ranked = args.rank == "yes" or (args.rank == "auto" and not already_pct)
    if ranked:
        out["value"] = out["value"].rank(pct=True)
    return out, rcol, mcol, ranked


def read_alignment(path):
    """Read an aligned FASTA (>name) or a Clustal-format (.aln) alignment -> {name: aligned seq}."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    first = next((ln for ln in lines if ln.strip()), "")
    seqs = {}
    if first.startswith(">"):
        name = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line)
    else:  # Clustal: header line, then blocks of 'name  sequence  [count]'; consensus lines are indented
        started = False
        for line in lines:
            if not started:
                started = bool(line.strip())  # skip the header line
                continue
            if not line.strip() or line[0].isspace():
                continue
            parts = line.split()
            if len(parts) >= 2:
                seqs.setdefault(parts[0], []).append(parts[1])
    if not seqs:
        die(f"no sequences could be read from {path} (expected aligned FASTA or Clustal format)")
    return {k: "".join(v) for k, v in seqs.items()}


# ----------------------------------------------------------------------------- alignment
def check_unique(data):
    for e, d in data.items():
        if d["resid"].duplicated().any():
            die(f"{e}: duplicate residue IDs in the DFI file (multiple chains or insertion "
                f"codes?). Use --chain, or use --fasta (which matches residues by order).")


def align_by_id(data):
    check_unique(data)
    id_sets = {e: set(d["resid"]) for e, d in data.items()}
    all_ids = sorted(set().union(*id_sets.values()), key=sort_key)
    values = pd.DataFrame({e: d.set_index("resid")["value"].reindex(all_ids).to_numpy()
                           for e, d in data.items()})
    ids = pd.DataFrame({e: [i if i in id_sets[e] else None for i in all_ids] for e in data},
                       dtype=object)
    return values, ids


def align_by_mapping(data, path):
    check_unique(data)
    m = pd.read_csv(path, dtype=str)
    m.columns = [c.strip() for c in m.columns]
    missing = [e for e in data if e not in m.columns]
    if missing:
        die(f"mapping file lacks columns for: {missing}. Columns present: {list(m.columns)}")
    ids = pd.DataFrame({e: [norm_id(v) for v in m[e]] for e in data}, dtype=object)
    vals = {}
    for e, d in data.items():
        lookup = dict(zip(d["resid"], d["value"]))
        unknown = sorted({i for i in ids[e] if i is not None and i not in lookup}, key=sort_key)
        if unknown:
            print(f"  WARNING: {e}: {len(unknown)} mapped residue IDs are absent from its DFI "
                  f"file (e.g. {unknown[:5]}); treated as gaps.")
        vals[e] = [lookup.get(i, np.nan) if i is not None else np.nan for i in ids[e]]
    return pd.DataFrame(vals), ids


def align_by_fasta(data, path):
    seqs = read_alignment(path)
    missing = [e for e in data if e not in seqs]
    if missing:
        die(f"alignment sequences not found for: {missing}. Names present: {list(seqs)}. "
            f"Sequence names must equal the enzyme names.")
    lengths = {len(seqs[e]) for e in data}
    if len(lengths) != 1:
        die(f"aligned sequences have different lengths: {sorted(lengths)}")
    ids, vals = {}, {}
    for e, d in data.items():
        aln = seqs[e]
        n_res = sum(ch not in "-." for ch in aln)
        if n_res != len(d):
            die(f"{e}: alignment contains {n_res} residues but the DFI file has {len(d)} rows. "
                f"Use --mapping instead, or make the sequences match the structures.")
        resid_iter, val_iter = iter(d["resid"]), iter(d["value"])
        col_ids, col_vals = [], []
        for ch in aln:
            if ch in "-.":
                col_ids.append(None)
                col_vals.append(np.nan)
            else:
                col_ids.append(next(resid_iter))
                col_vals.append(next(val_iter))
        ids[e], vals[e] = col_ids, col_vals
    return pd.DataFrame(vals), pd.DataFrame(ids, dtype=object)


# ----------------------------------------------------------------------------- statistics
def group_stats(X, n_a, max_perms=20000, seed=0):
    """X: positions x enzymes, columns ordered [group A..., group B...], no NaN."""
    n_pos, n = X.shape
    n_b = n - n_a
    A, B = X[:, :n_a], X[:, n_a:]
    delta = A.mean(axis=1) - B.mean(axis=1)
    cliff = np.sign(A[:, :, None] - B[:, None, :]).sum(axis=(1, 2)) / (n_a * n_b)

    if math.comb(n, n_a) <= max_perms:
        combos = list(itertools.combinations(range(n), n_a))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        combos = [tuple(rng.choice(n, n_a, replace=False)) for _ in range(max_perms)]
        combos.append(tuple(range(n_a)))  # include the observed labelling
        exact = False
    W = np.full((len(combos), n), -1.0 / n_b)
    for k, c in enumerate(combos):
        W[k, list(c)] = 1.0 / n_a
    perm_delta = X @ W.T                      # positions x permutations
    tol = 1e-12
    abs_obs = np.abs(delta)
    p_perm = (np.abs(perm_delta) >= abs_obs[:, None] - tol).mean(axis=1)
    null_max = np.abs(perm_delta).max(axis=0)  # max over positions, per permutation
    p_fwer = (null_max[None, :] >= abs_obs[:, None] - tol).mean(axis=1)
    return dict(delta=delta, cliff=cliff, p_perm=p_perm, p_fwer=p_fwer,
                exact=exact, n_perm=len(combos))


def find_regions(smooth, delta, labels, window_cut, min_len, label_a, label_b):
    regions, start, sign = [], None, 0

    def close(s, e):
        if e - s + 1 < min_len:
            return
        seg = delta[s:e + 1]
        peak = s + int(np.argmax(np.abs(seg)))
        regions.append({
            "start": labels[s], "end": labels[e], "n_positions": e - s + 1,
            "mean_delta": float(seg.mean()), "peak_delta": float(delta[peak]),
            "peak_residue": labels[peak],
            "higher_dfi_in": label_a if seg.mean() > 0 else label_b,
        })

    for i, s in enumerate(smooth):
        sg = 1 if s >= window_cut else (-1 if s <= -window_cut else 0)
        if sg != sign:
            if sign != 0:
                close(start, i - 1)
            start = i if sg != 0 else None
            sign = sg
    if sign != 0:
        close(start, len(smooth) - 1)
    return pd.DataFrame(regions)


# ----------------------------------------------------------------------------- plotting
def plot_profiles(path, x, X, names, n_a, delta, smooth, cand, regions, args, metric_name):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax, ax2 = axes
    for j in range(X.shape[1]):
        ax.plot(x, X[:, j], color=COL_A if j < n_a else COL_B, alpha=0.3, lw=0.8)
    ax.plot(x, X[:, :n_a].mean(axis=1), color=COL_A, lw=2.2,
            label=f"{args.label_a} mean (n={n_a})")
    ax.plot(x, X[:, n_a:].mean(axis=1), color=COL_B, lw=2.2,
            label=f"{args.label_b} mean (n={X.shape[1] - n_a})")
    ax.set_ylabel(metric_name)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ax2.axhline(0, color="k", lw=0.6)
    ax2.axhline(args.min_delta, color="0.5", ls=":", lw=0.8)
    ax2.axhline(-args.min_delta, color="0.5", ls=":", lw=0.8)
    ax2.plot(x, delta, color="0.55", lw=1, label="per-residue difference")
    ax2.plot(x, smooth, color="k", lw=1.8, label=f"{args.window}-residue moving average")
    if cand.any():
        ax2.scatter(x[cand], delta[cand], color="red", s=18, zorder=3, label="candidate")
    ax2.set_ylabel(f"Δ{metric_name}\n({args.label_a} − {args.label_b})")
    ax2.set_xlabel(f"Residue number ({args.reference} numbering)")
    ax2.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)

    if len(regions):
        for _, r in regions.iterrows():
            for a in axes:
                a.axvspan(float(r["start"]), float(r["end"]), color="gold", alpha=0.25, lw=0)
    for r in args.highlight or []:
        for a in axes:
            a.axvline(float(r), color="grey", ls="--", lw=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_similarity(path, corr, scores, var, names, n_a, args):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    n = len(names)
    off = corr.to_numpy()[~np.eye(n, dtype=bool)]
    im = ax.imshow(corr.to_numpy(), cmap="viridis", vmin=max(-1, np.floor(off.min() * 10) / 10),
                   vmax=1)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.axhline(n_a - 0.5, color="w", lw=2)
    ax.axvline(n_a - 0.5, color="w", lw=2)
    ax.set_title("Spearman correlation of DFI profiles")
    fig.colorbar(im, ax=ax, fraction=0.046)

    for j, nm in enumerate(names):
        ax2.scatter(scores[j, 0], scores[j, 1], color=COL_A if j < n_a else COL_B, s=60)
        ax2.annotate(nm, (scores[j, 0], scores[j, 1]), textcoords="offset points",
                     xytext=(5, 4), fontsize=8)
    ax2.set_xlabel(f"PC1 ({var[0] * 100:.0f}%)")
    ax2.set_ylabel(f"PC2 ({var[1] * 100:.0f}%)")
    ax2.set_title("PCA of DFI profiles")
    ax2.scatter([], [], color=COL_A, label=args.label_a)
    ax2.scatter([], [], color=COL_B, label=args.label_b)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="DFI csv files and/or directories containing *_dfi.csv files. A file's "
                        "enzyme name is its name minus '_dfi.csv', or the name of its parent "
                        "directory if that matches a name given in --group-a/--group-b")
    p.add_argument("--group-a", nargs="+", required=True, help="enzyme names in group A")
    p.add_argument("--group-b", nargs="+", required=True, help="enzyme names in group B")
    p.add_argument("--label-a", default="A", help="label for group A (default: A)")
    p.add_argument("--label-b", default="B", help="label for group B (default: B)")
    p.add_argument("--outdir", default="dfi_group_comparison")
    p.add_argument("--metric", help="DFI column to use (default: auto-detect)")
    p.add_argument("--resid-col", help="residue-ID column (default: auto-detect)")
    p.add_argument("--chain", help="chain to use if files contain several")
    p.add_argument("--rank", choices=["auto", "yes", "no"], default="auto",
                   help="convert the DFI column to a within-enzyme percentile rank (%%DFI, 0-1). "
                        "auto (default): convert unless the column name already looks like a "
                        "percentile/rank (contains pct, %%, percent or rank)")
    al = p.add_mutually_exclusive_group()
    al.add_argument("--mapping", help="CSV mapping aligned positions to residue IDs")
    al.add_argument("--fasta", "--alignment", dest="fasta",
                    help="multiple sequence alignment, aligned FASTA or Clustal (.aln); "
                         "sequence names must equal the enzyme names")
    p.add_argument("--reference", help="enzyme whose numbering is used for labels "
                                       "(default: first enzyme of group A)")
    p.add_argument("--highlight", nargs="+", help="residues (reference numbering) to mark")
    p.add_argument("--min-delta", type=float, default=0.20,
                   help="min |mean difference| for a candidate (default 0.20)")
    p.add_argument("--min-cliff", type=float, default=0.75,
                   help="min |Cliff's delta| for a candidate (default 0.75; 1 = no overlap)")
    p.add_argument("--window", type=int, default=5, help="moving-average window (default 5)")
    p.add_argument("--region-delta", type=float, default=0.15,
                   help="min |smoothed difference| to define a region (default 0.15)")
    p.add_argument("--min-region", type=int, default=3, help="min region length (default 3)")
    p.add_argument("--rigid-cutoff", type=float, default=0.20,
                   help="value <= this is 'rigid' (default 0.20)")
    p.add_argument("--flex-cutoff", type=float, default=0.80,
                   help="value >= this is 'flexible' (default 0.80)")
    p.add_argument("--top", type=int, default=20, help="rows of candidates to print")
    return p.parse_args()


def main():
    args = parse_args()
    order = list(args.group_a) + list(args.group_b)
    if len(set(order)) != len(order):
        die("an enzyme appears more than once / in both groups")
    wanted = set(order)
    files = {}

    def register(name, f):
        if name in files and files[name] != f:
            die(f"two DFI files resolve to the name '{name}': {files[name]} and {f}")
        files[name] = f

    for item in args.inputs:
        pth = Path(item)
        if pth.is_dir():
            for f in sorted(pth.glob("*_dfi.csv")):
                register(re.sub(r"_dfi\.csv$", "", f.name, flags=re.I), f)
        elif pth.is_file():
            stem = re.sub(r"_dfi\.csv$", "", pth.name, flags=re.I)
            parent = pth.parent.name
            register(stem if (stem in wanted or parent not in wanted) else parent, pth)
        else:
            die(f"input not found: {item}")
    if not files:
        die("no *_dfi.csv files found in the inputs")
    missing = [e for e in order if e not in files]
    if missing:
        found = "\n".join(f"    {k}  <-  {v}" for k, v in sorted(files.items()))
        die(f"no DFI file matched {missing}. Names resolved from the inputs:\n{found}")
    n_a = len(args.group_a)
    if n_a < 2 or len(args.group_b) < 2:
        die("each group needs at least 2 enzymes")
    args.reference = args.reference or args.group_a[0]
    if args.reference not in order:
        die("--reference must be one of the enzymes being compared")

    print(f"Loading {len(order)} enzymes ...")
    data, cols_used = {}, set()
    for e in order:
        print(f"  {e}  <-  {files[e]}")
        data[e], rcol, mcol, ranked = load_dfi(files[e], args)
        cols_used.add((rcol, mcol, ranked))
    for rcol, mcol, ranked in sorted(cols_used):
        note = (" -> converted to within-enzyme percentile rank (%DFI, 0-1)" if ranked
                else " (used as is)")
        print(f"  using residue column '{rcol}' and DFI column '{mcol}'{note}")
    metric_name = "%DFI" if any(r for _, _, r in cols_used) else sorted(cols_used)[0][1]

    if args.fasta:
        values, ids = align_by_fasta(data, args.fasta)
    elif args.mapping:
        values, ids = align_by_mapping(data, args.mapping)
    else:
        values, ids = align_by_id(data)

    complete = values[order].notna().all(axis=1).to_numpy()
    print(f"Aligned positions: {len(values)}; present in all {len(order)} enzymes: "
          f"{int(complete.sum())} (only these are analysed)")
    if complete.sum() < 10:
        die("fewer than 10 positions shared by all enzymes - check numbering/alignment")

    X = values.loc[complete, order].to_numpy(float)
    if np.nanmax(X) > 1.0 + 1e-6:
        print("  WARNING: values exceed 1. The default cut-offs assume %DFI on a 0-1 scale; "
              "adjust --min-delta, --region-delta and the rigid/flex cut-offs or pick another "
              "column with --metric.")
    n_pos, n = X.shape
    n_b = n - n_a
    align_pos = np.flatnonzero(complete) + 1
    ref_ids = ids.loc[complete, args.reference].tolist()
    ref_num = pd.to_numeric(pd.Series(ref_ids), errors="coerce").to_numpy()
    if np.isnan(ref_num).any():
        print("  NOTE: reference residue IDs are not all numeric; plotting against aligned "
              "position instead.")
        x = align_pos.astype(float)
    else:
        x = ref_num
    labels = [str(i) for i in ref_ids]

    # ---- statistics
    st = group_stats(X, n_a)
    delta, cliff = st["delta"], st["cliff"]
    smooth = pd.Series(delta).rolling(args.window, center=True, min_periods=1).mean().to_numpy()
    cand = (np.abs(delta) >= args.min_delta) & (np.abs(cliff) >= args.min_cliff)
    perfect = np.abs(cliff) >= 1 - 1e-9
    rigid_a = (X[:, :n_a] <= args.rigid_cutoff).sum(axis=1)
    rigid_b = (X[:, n_a:] <= args.rigid_cutoff).sum(axis=1)
    flex_a = (X[:, :n_a] >= args.flex_cutoff).sum(axis=1)
    flex_b = (X[:, n_a:] >= args.flex_cutoff).sum(axis=1)
    la, lb = args.label_a, args.label_b
    pattern = np.select(
        [(rigid_a == n_a) & (rigid_b == n_b),
         (rigid_a == n_a) & (rigid_b == 0),
         (rigid_b == n_b) & (rigid_a == 0),
         (flex_a == n_a) & (flex_b == n_b),
         (flex_a == n_a) & (flex_b == 0),
         (flex_b == n_b) & (flex_a == 0)],
        ["rigid in all enzymes",
         f"rigid in all {la}, none of {lb}",
         f"rigid in all {lb}, none of {la}",
         "flexible in all enzymes",
         f"flexible in all {la}, none of {lb}",
         f"flexible in all {lb}, none of {la}"],
        default="")

    tab = pd.DataFrame({"align_pos": align_pos, "ref_resid": labels})
    if any(ids.loc[complete, e].tolist() != ref_ids for e in order):
        for e in order:
            tab[f"resid_{e}"] = ids.loc[complete, e].to_numpy()
    for j, e in enumerate(order):
        tab[e] = X[:, j]
    tab[f"mean_{slug(la)}"] = X[:, :n_a].mean(axis=1)
    tab[f"mean_{slug(lb)}"] = X[:, n_a:].mean(axis=1)
    tab["delta"] = delta
    tab["abs_delta"] = np.abs(delta)
    tab["delta_smoothed"] = smooth
    tab["cliff_delta"] = cliff
    tab["perfect_separation"] = perfect
    tab["p_perm"] = st["p_perm"]
    tab["p_fwer"] = st["p_fwer"]
    tab["candidate"] = cand
    tab[f"n_rigid_{slug(la)}"] = rigid_a
    tab[f"n_rigid_{slug(lb)}"] = rigid_b
    tab[f"n_flex_{slug(la)}"] = flex_a
    tab[f"n_flex_{slug(lb)}"] = flex_b
    tab["pattern"] = pattern

    regions = find_regions(smooth, delta, labels, args.region_delta, args.min_region, la, lb)

    # ---- profile similarity
    corr = pd.DataFrame(X, columns=order).corr(method="spearman")
    C = corr.to_numpy()
    grp = np.array([0] * n_a + [1] * n_b)
    off = ~np.eye(n, dtype=bool)
    same_a = C[np.ix_(grp == 0, grp == 0)][off[np.ix_(grp == 0, grp == 0)]].mean()
    same_b = C[np.ix_(grp == 1, grp == 1)][off[np.ix_(grp == 1, grp == 1)]].mean()
    between = C[np.ix_(grp == 0, grp == 1)].mean()
    Xc = X.T - X.T.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores, var = U * S, S ** 2 / (S ** 2).sum()

    # ---- write outputs
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "positions.csv", index=False, float_format="%.5g")
    cand_tab = tab[tab.candidate].sort_values(["p_perm", "abs_delta"], ascending=[True, False])
    cand_tab.to_csv(out / "candidate_residues.csv", index=False, float_format="%.5g")
    regions.to_csv(out / "regions.csv", index=False, float_format="%.5g")
    corr.to_csv(out / "profile_correlations.csv", float_format="%.4f")
    plot_profiles(out / "profiles.png", x, X, order, n_a, delta, smooth, cand, regions, args,
                  metric_name)
    plot_similarity(out / "similarity.png", corr, scores, var, order, n_a, args)

    # ---- console summary
    perm_kind = "exact" if st["exact"] else "sampled"
    print(f"\n=== Group comparison: {la} (n={n_a}) vs {lb} (n={n_b}), {n_pos} positions ===")
    print(f"Label-permutation test: {st['n_perm']} {perm_kind} permutations "
          f"(smallest possible per-position p = {2 / math.comb(n, n_a):.3f})")
    exp_perfect = n_pos * 2 / math.comb(n, n_a)
    print(f"Perfectly separated positions (no overlap between groups): observed "
          f"{int(perfect.sum())}, expected by chance ~{exp_perfect:.1f} if positions were independent")
    print(f"Candidate residues (|Δ| >= {args.min_delta}, |Cliff's δ| >= {args.min_cliff}): "
          f"{int(cand.sum())}")
    print(f"Positions with family-wise p <= 0.05: {int((st['p_fwer'] <= 0.05).sum())}")

    if len(cand_tab):
        print(f"\nTop candidates (reference = {args.reference}):")
        show = cand_tab[["ref_resid", f"mean_{slug(la)}", f"mean_{slug(lb)}", "delta",
                         "cliff_delta", "p_perm", "p_fwer"]].head(args.top)
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(regions):
        print("\nRegions with a sustained difference:")
        print(regions.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("\nNo regions passed the smoothed-difference cut-off.")
    pat = tab[tab.pattern != ""]
    group_specific = pat[~pat.pattern.str.contains("in all enzymes")]
    print(f"\nGroup-specific rigid/flexible residues: {len(group_specific)} "
          f"(see 'pattern' column in positions.csv)")
    print(f"\nProfile similarity (mean Spearman r): within {la} {same_a:.3f}, "
          f"within {lb} {same_b:.3f}, between groups {between:.3f}")
    print(f"PCA: PC1 {var[0] * 100:.0f}% / PC2 {var[1] * 100:.0f}% of variance")
    print(f"\nResults written to {out.resolve()}")
    print("Reminder: with groups this small, a few positions will separate the groups by "
          "chance. Weight regions of adjacent residues, the family-wise p-values and "
          "structural/mechanistic context over single-residue hits.")


if __name__ == "__main__":
    main()
