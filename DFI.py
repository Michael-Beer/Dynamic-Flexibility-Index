#!/usr/bin/env python3
"""
dfi_analysis.py  --  Dynamic Flexibility Index (DFI) / Dynamic Coupling Index (DCI)
from MD trajectories, following the Methods of

    Modi, Risso, ... Sanchez-Ruiz, Ozkan, Nat. Commun. 12, 1852 (2021)
    "Hinge-shift mechanism as a protein design principle ..."
    (reference implementation from the authors: https://github.com/SBOZKAN/DFI-DCI)

Subcommands
-----------
  dfi      trajectory -> per-residue DFI / %DFI (+ DCI to active site, DARC flags)
  compare  two DFI tables -> common / non-common hinges, sequence conservation
  cluster  several DFI tables -> SVD/PCA of %DFI profiles (paper Figs 3-4)

Method (paper, Methods; equation numbers are the paper's)
----------------------------------------------------------
 1. Discard the first --skip-ns (100 ns) of the trajectory.
 2. Slide windows of --window-ns (50 ns) every --lag-ns (25 ns).
 3. Per window: fit every frame to the first frame of the window using backbone
    heavy atoms, then take the 3N x 3N covariance matrix G of the C-alpha
    coordinates (G ~ Hessian inverse; Eq. 2).
 4. Perturbation response scanning, linear response dR = G F (Eq. 2). With random
    unit "Brownian kicks" isotropic in direction at residue j, the RMS response at
    residue i is |dR^j|_i = sqrt(<|G_ij F|^2>) = ||G_ij||_F / sqrt(3), where G_ij is
    the 3x3 block of G (the sqrt(3) cancels in every normalised quantity). This is
    the analytic expectation of the random-kick average, so no sampling noise.
    Response matrix A[i, j] = |dR^j|_i (Eq. 3).
 5. DFI_i = sum_j A[i,j] / sum_ij A[i,j]  (Eq. 5); averaged over windows.
 6. %DFI = percentile rank of DFI across residues (0 = most rigid, 1 = most flexible).
    Hinges: %DFI < 0.2.
 7. DCI to a functional set F (Eq. 6):
        DCI_i = [sum_{j in F} A[i,j] / |F|] / [sum_j A[i,j] / N]
    Pairwise DCI of i upon perturbation at j (Eq. 7):
        DCI^j_i = A[i,j] / [sum_j A[i,j] / N]
    %DCI = percentile rank across residues; "highly coupled" is %DCI > 0.8.
 8. DARC spots: 0.3 < %DFI < 0.5, %DCI (active site) > 0.8, > 8 A from the active
    site (optionally also %DCI > 0.8 to a user-supplied target set).

Requires: numpy, scipy, pandas, MDAnalysis (matplotlib optional, for plots).

Examples
--------
  python dfi_analysis.py dfi system.prmtop -t prod.nc --active-site 70,73,130,132,166,234 -o TEM1
  python dfi_analysis.py dfi system.prmtop -t rep1.nc rep2.nc rep3.nc -o TEM1   # independent repeats, pooled
  python dfi_analysis.py dfi acyl.prmtop -t rep1.nc rep2.nc --protein-sel "protein or resid 42" --aa-override 42=S --active-site 42,...  # covalent adduct
  python dfi_analysis.py dfi system.prmtop -t seg1.nc seg2.nc --concat -o TEM1  # restart segments of one run
  python dfi_analysis.py dfi system.prmtop -t prod.nc --converge -o TEM1_conv
  python dfi_analysis.py compare GNCA_dfi.csv TEM1_dfi.csv --labels GNCA TEM-1 -o GNCA_vs_TEM1
  python dfi_analysis.py cluster GNCA_dfi.csv GNCA-XY_dfi.csv GNCA-XYZ_dfi.csv TEM1_dfi.csv -o pca
  # different numbering: realign on equivalent (e.g. active-site) residues, same order in each list
  python dfi_analysis.py compare TEM1_dfi.csv KPC2_dfi.csv --anchors-a 45,48,105,107,141,209 \\
         --anchors-b 44,47,104,106,140,207 --labels TEM1 KPC2 -o TEM1_vs_KPC2
  python dfi_analysis.py cluster TEM1_dfi.csv KPC2_dfi.csv SHV1_dfi.csv \\
         --anchors 45,48,105,107 44,47,104,106 45,48,105,107
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Amber protonation / disulfide variants
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}


# --------------------------------------------------------------------------- core maths
def pct_rank(x, axis=0):
    """Percentile rank scaled to [0, 1] (0 = lowest value, 1 = highest)."""
    n = x.shape[axis]
    return (rankdata(x, axis=axis, method="average") - 1.0) / (n - 1.0)


def response_matrix(G):
    """3N x 3N covariance -> N x N response matrix A[i, j] = |dR^j|_i (Eq. 3)."""
    n = G.shape[0] // 3
    blocks = G.reshape(n, 3, n, 3)
    return np.sqrt((blocks ** 2).sum(axis=(1, 3)))


def dfi_from_response(A):
    """Eq. 5."""
    return A.sum(axis=1) / A.sum()


def dci_to_sites(A, site_idx):
    """Eq. 6: coupling of every residue to a functional set (indices into A)."""
    return A[:, site_idx].mean(axis=1) / A.mean(axis=1)


def dci_pairwise(A):
    """Eq. 7: M[i, j] = coupling of residue i to a perturbation at residue j."""
    return A / A.mean(axis=1, keepdims=True)


# --------------------------------------------------------------------------- trajectory
_RANK_NOTE_SHOWN = False

def window_covariance(u, ca, fit, frames):
    """Covariance of C-alpha coords after fitting each frame to the window's first frame."""
    from MDAnalysis.analysis import align

    X = np.empty((len(frames), ca.n_atoms, 3))
    ref = None
    for k, f in enumerate(frames):
        u.trajectory[f]
        c = fit.center_of_geometry()
        p = fit.positions.astype(np.float64) - c
        if ref is None:
            ref = p.copy()
        R, _ = align.rotation_matrix(p, ref)
        X[k] = (ca.positions.astype(np.float64) - c) @ R.T
    return np.cov(X.reshape(len(frames), -1), rowvar=False)


def trajectory_profile(u, ca, fit, dt_ns, skip_ns, window_ns, lag_ns, stride):
    """Per-window DFI (n_windows x N) and the window-averaged response matrix A."""
    n_frames = len(u.trajectory)
    first = int(np.ceil(skip_ns / dt_ns))
    wlen = int(round(window_ns / dt_ns))
    lag = max(1, int(round(lag_ns / dt_ns)))
    starts = list(range(first, n_frames - wlen + 1, lag))
    if not starts:
        sys.exit(f"Trajectory too short: {n_frames * dt_ns:.1f} ns total, need > "
                 f"{skip_ns + window_ns:.1f} ns for skip + one window.")
    n_used = len(range(0, wlen, stride))
    global _RANK_NOTE_SHOWN
    if n_used < 3 * ca.n_atoms and not _RANK_NOTE_SHOWN:
        _RANK_NOTE_SHOWN = True
        print(f"NOTE: {n_used} frames per window < 3N = {3 * ca.n_atoms}, so each window covariance is "
              f"rank-deficient. DFI uses the covariance directly (no inversion), so this is not an "
              f"error, only a noisier estimate; judge it with --converge and DFI_sd_repeats.",
              file=sys.stderr)
    dfis, A_sum = [], 0.0
    for s in starts:
        A = response_matrix(window_covariance(u, ca, fit, range(s, s + wlen, stride)))
        dfis.append(dfi_from_response(A))
        A_sum = A_sum + A
    return np.array(dfis), A_sum / len(starts), len(starts)


def pooled_profile(unis, sel, fit_sel, dt_arg, skip_ns, window_ns, lag_ns, stride, quiet=False):
    """Run trajectory_profile on each independent repeat and pool the windows.

    Returns per-window DFI (all repeats stacked), the response matrix averaged over all
    windows, per-repeat mean DFI profiles, and the number of repeats actually used.
    """
    all_dfi, A_sum, n_tot, rep_means = [], 0.0, 0, []
    for k, u in enumerate(unis):
        dt = dt_arg if dt_arg else u.trajectory.dt / 1000.0
        if len(u.trajectory) * dt < skip_ns + window_ns:
            if not quiet:
                print(f"  repeat {k + 1}: {len(u.trajectory) * dt:.1f} ns is too short, skipped")
            continue
        d, A, nw = trajectory_profile(u, u.select_atoms(sel), u.select_atoms(fit_sel),
                                      dt, skip_ns, window_ns, lag_ns, stride)
        if not quiet:
            print(f"  repeat {k + 1}: {len(u.trajectory) * dt:.1f} ns, {nw} windows")
        all_dfi.append(d)
        rep_means.append(d.mean(axis=0))
        A_sum = A_sum + A * nw
        n_tot += nw
    if not all_dfi:
        sys.exit(f"No trajectory is long enough for skip ({skip_ns} ns) + window ({window_ns} ns).")
    return np.vstack(all_dfi), A_sum / n_tot, np.array(rep_means), len(all_dfi)


def min_distance_to_site(u, ca, site_resids, frame, psel="protein"):
    """Minimum heavy-atom distance (A) from each residue to the site residues."""
    from MDAnalysis.lib.distances import distance_array

    u.trajectory[frame]
    heavy = u.select_atoms(f"({psel}) and not name H*")
    site = u.select_atoms(f"({psel}) and not name H* and resid " + " ".join(map(str, site_resids)))
    d = distance_array(heavy.positions, site.positions).min(axis=1)
    per_res = pd.Series(d).groupby(heavy.resindices).min()
    return per_res.reindex(ca.resindices).to_numpy()


def parse_resids(text):
    """'70,73,130-132' -> [70, 73, 130, 131, 132]"""
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def resid_indices(ca, resids, what):
    idx = np.where(np.isin(ca.resids, resids))[0]
    missing = set(resids) - set(ca.resids[idx])
    if missing:
        have = np.asarray(ca.resids)
        gaps = sorted(set(range(have.min(), have.max() + 1)) - set(have.tolist()))
        sys.exit(f"{what}: residue ids {sorted(missing)} not found among the {len(have)} selected "
                 f"residues (resids {have.min()}-{have.max()}; gaps in numbering: {gaps[:10] or 'none'}). "
                 f"--active-site/--targets use the TOPOLOGY's resids, which may differ from "
                 f"literature (e.g. Ambler) numbering.")
    return idx


# --------------------------------------------------------------------------- plotting
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return None


def plot_profile(df, path, hinge_cut):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df["resid"], df["pDFI"], lw=1.2, color="k")
    ax.axhspan(0, hinge_cut, color="tab:blue", alpha=0.15, label=f"hinge (%DFI < {hinge_cut})")
    if "DARC" in df:
        d = df[df["DARC"]]
        ax.scatter(d["resid"], d["pDFI"], color="tab:red", zorder=3, s=18, label="DARC candidate")
    ax.set_xlabel("residue")
    ax.set_ylabel("%DFI")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- subcommand: dfi
def cmd_dfi(a):
    import MDAnalysis as mda

    # Each -t file is an independent repeat unless --concat (e.g. restart segments of ONE run)
    groups = [a.trajectory] if a.concat else [[t] for t in a.trajectory]
    unis = [mda.Universe(a.topology, *g) for g in groups]
    u = unis[0]
    psel = a.protein_sel
    sel = a.selection or f"({psel}) and name CA"
    fit_sel = a.fit_selection or f"({psel}) and name N CA C O"
    ca = u.select_atoms(sel)
    n_res = u.select_atoms(psel).n_residues
    if n_res != ca.n_atoms:
        no_ca = [f"{r.resname}{r.resid}" for r in u.select_atoms(psel).residues
                 if "CA" not in r.atoms.names]
        print(f"NOTE: {n_res} residues in '{psel}' but {ca.n_atoms} C-alpha atoms selected; "
              f"residues without a CA: {no_ca[:10]}", file=sys.stderr)
    dt_ns = a.dt_ns if a.dt_ns else u.trajectory.dt / 1000.0
    print(f"{ca.n_atoms} residues; {len(unis)} {'concatenated run' if a.concat else 'independent repeat(s)'}")
    site_resids = parse_resids(a.active_site) if a.active_site else []
    site_idx = resid_indices(ca, site_resids, "--active-site") if site_resids else None
    target_idx = resid_indices(ca, parse_resids(a.targets), "--targets") if a.targets else None

    if a.converge:  # paper's convergence check: profile should not depend on window size
        profs = {}
        for w in [25, 50, 75, 100]:
            print(f"window {w} ns:")
            try:
                d, _, _, _ = pooled_profile(unis, sel, fit_sel, a.dt_ns,
                                            a.skip_ns, w, a.lag_ns, a.stride)
            except SystemExit:
                print("  no trajectory long enough, skipped")
                continue
            profs[w] = d.mean(axis=0)
        tab = pd.DataFrame(profs).corr(method="pearson")
        print("\nPearson r between mean DFI profiles from different window sizes:")
        print(tab.round(4).to_string())
        pd.DataFrame(profs, index=ca.resids).to_csv(f"{a.out}_convergence.csv", index_label="resid")
        return

    dfis, A, rep_means, n_rep = pooled_profile(unis, sel, fit_sel, a.dt_ns,
                                               a.skip_ns, a.window_ns, a.lag_ns, a.stride)
    nwin = len(dfis)
    print(f"{nwin} windows of {a.window_ns} ns pooled (lag {a.lag_ns} ns, first {a.skip_ns} ns of each repeat discarded)")

    df = pd.DataFrame({
        "resid": ca.resids,
        "resname": ca.resnames,
        "aa": [AA3TO1.get(r, "X") for r in ca.resnames],
        "DFI": dfis.mean(axis=0),
        "DFI_sd": dfis.std(axis=0),
    })
    if n_rep > 1:  # spread of the per-repeat mean profiles
        df["DFI_sd_repeats"] = rep_means.std(axis=0, ddof=1)
    if a.aa_override:  # e.g. 42=S for a covalently modified serine with a non-standard resname
        ov = {int(k): v for k, v in (x.split("=") for x in a.aa_override.split(","))}
        df["aa"] = [ov.get(r, aa) for r, aa in zip(df["resid"], df["aa"])]
    df["pDFI"] = pct_rank(df["DFI"].to_numpy())
    df["hinge"] = df["pDFI"] < a.hinge_cut

    if site_resids:
        df["DCI_site"] = dci_to_sites(A, site_idx)
        df["pDCI_site"] = pct_rank(df["DCI_site"].to_numpy())
        first = int(np.ceil(a.skip_ns / dt_ns))
        df["dist_site"] = min_distance_to_site(u, ca, site_resids, first, psel)
        lo, hi = a.darc_flex
        darc = (df["pDFI"] > lo) & (df["pDFI"] < hi) & (df["pDCI_site"] > a.dci_cut) \
            & (df["dist_site"] > a.darc_dist)
        if a.targets:
            df["pDCI_targets"] = pct_rank(dci_to_sites(A, target_idx))
            darc &= df["pDCI_targets"] > a.dci_cut
        df["DARC"] = darc
        print(f"{int(darc.sum())} DARC candidates: {df.loc[darc, 'resid'].tolist()}")

    df.to_csv(f"{a.out}_dfi.csv", index=False, float_format="%.5f")
    np.save(f"{a.out}_response_matrix.npy", A)
    plot_profile(df, f"{a.out}_dfi.png", a.hinge_cut)
    print(f"{int(df['hinge'].sum())} hinges (%DFI < {a.hinge_cut}): {df.loc[df['hinge'], 'resid'].tolist()}")
    print(f"Wrote {a.out}_dfi.csv, {a.out}_response_matrix.npy, {a.out}_dfi.png")


# --------------------------------------------------------------------------- subcommand: compare
def anchor_map(ref, oth, anch_ref, anch_oth, name="second"):
    """Residue map ref -> other protein from equivalent anchor residues (e.g. active-site residues).

    Between anchors the offset (other - ref) is constant. Where two neighbouring anchors have
    different offsets there is an indel between them; its position is placed where the number of
    identical amino acids is maximised (a deletion leaves the extra residues unmapped).
    Residues before the first / after the last anchor use that anchor's offset.
    Returns a DataFrame with columns resid_a (reference numbering) and resid_b.
    """
    if len(anch_ref) != len(anch_oth) or not anch_ref:
        sys.exit("Anchors: give the same (non-zero) number of residues for both proteins.")
    for lst, df, nm in [(anch_ref, ref, "reference"), (anch_oth, oth, name)]:
        miss = sorted(set(lst) - set(df["resid"]))
        if miss:
            sys.exit(f"Anchor residue(s) {miss} are not in the {nm} protein's table.")
    pairs = sorted(zip(anch_ref, anch_oth))
    ar, ao = [p[0] for p in pairs], [p[1] for p in pairs]
    if any(np.diff(ar) <= 0) or any(np.diff(ao) <= 0):
        sys.exit("Anchors must be listed in the same order in both proteins, without repeats.")
    offs = [o - r for r, o in pairs]
    aa_r = dict(zip(ref["resid"], ref["aa"]))
    aa_o = dict(zip(oth["resid"], oth["aa"]))

    def same(r, off):
        x = aa_r.get(r)
        return x is not None and x != "X" and x == aa_o.get(r + off)

    bad = [(r, aa_r.get(r), o, aa_o.get(o)) for r, o in pairs if aa_r.get(r) != aa_o.get(o)]
    if bad:
        print(f"WARNING: anchors with different amino acids (ref resid, aa, {name} resid, aa): {bad}. "
              f"Check they are really equivalent positions.", file=sys.stderr)

    offset_of = {r: offs[0] for r in aa_r if r <= ar[0]}
    offset_of.update({r: offs[-1] for r in aa_r if r >= ar[-1]})
    print(f"Anchor alignment, reference -> {name}:")
    print("  " + ", ".join(f"{r}->{o} ({o - r:+d})" for r, o in pairs))
    for i in range(len(ar) - 1):
        o1, o2 = offs[i], offs[i + 1]
        if o1 == o2:
            k, d = ar[i + 1] - 1, 0
        else:
            d = max(0, o1 - o2)  # reference residues with no partner (deletion in the other protein)
            best = None
            for k_ in range(ar[i], ar[i + 1] - d):
                score = sum(same(r, o1) for r in range(ar[i], k_ + 1)) + \
                        sum(same(r, o2) for r in range(k_ + d + 1, ar[i + 1] + 1))
                key = (-score, abs(k_ - (ar[i] + ar[i + 1]) / 2))
                if best is None or key < best[0]:
                    best = (key, k_)
            k = best[1]
            kind = "insertion" if o2 > o1 else "deletion"
            print(f"  offset {o1:+d} -> {o2:+d} between reference "
                  f"residues {ar[i]} and {ar[i + 1]}: {kind} of {abs(o2 - o1)} in {name}, placed after "
                  f"reference residue {k} (maximises sequence identity)")
        for r in range(ar[i], ar[i + 1] + 1):
            if r not in aa_r:
                continue
            offset_of[r] = o1 if r <= k else (None if r <= k + d else o2)
    rows = [(r, r + off) for r, off in offset_of.items() if off is not None and (r + off) in aa_o]
    m = pd.DataFrame(sorted(rows), columns=["resid_a", "resid_b"])
    ident = np.mean([same(r, b - r) for r, b in rows if aa_r.get(r) != "X"]) if rows else 0.0
    print(f"  {len(m)} of {len(ref)} reference residues mapped; sequence identity over mapped residues "
          f"= {100 * ident:.0f}%")
    if ident < 0.2:
        print("WARNING: very low identity over the mapped residues; check the anchors.", file=sys.stderr)
    return m


def load_pair(A, B, m=None):
    if m is None:  # assume identical numbering (e.g. Ambler numbering for class A beta-lactamases)
        common = sorted(set(A["resid"]) & set(B["resid"]))
        m = pd.DataFrame({"resid_a": common, "resid_b": common})
    A, B = A.add_suffix("_a"), B.add_suffix("_b")
    return m.merge(A, on="resid_a").merge(B, on="resid_b")


def cmd_compare(a):
    la, lb = a.labels
    A, B = pd.read_csv(a.csv_a), pd.read_csv(a.csv_b)
    m = None
    if a.anchors_a or a.anchors_b:
        if not (a.anchors_a and a.anchors_b) or a.map:
            sys.exit("Use --anchors-a together with --anchors-b (and not together with --map).")
        m = anchor_map(A, B, parse_resids(a.anchors_a), parse_resids(a.anchors_b), name=lb)
        m.to_csv(f"{a.out}_map.txt", sep=" ", header=False, index=False)
        print(f"Wrote {a.out}_map.txt (reusable with --map)")
    elif a.map:
        m = pd.read_csv(a.map, sep=r"\s+", header=None, names=["resid_a", "resid_b"], comment="#")
    d = load_pair(A, B, m)
    d["delta_pDFI"] = d["pDFI_b"] - d["pDFI_a"]
    d["seq_conserved"] = d["aa_a"] == d["aa_b"]
    ha, hb = d["pDFI_a"] < a.hinge_cut, d["pDFI_b"] < a.hinge_cut
    shifted = d["delta_pDFI"].abs() > a.shift
    d["category"] = "other"
    d.loc[ha & hb, "category"] = "common hinge"
    d.loc[(ha ^ hb) & shifted, "category"] = "non-common hinge"
    d["hinge_in"] = np.select([ha & hb, ha & ~hb, ~ha & hb], ["both", la, lb], default="")
    d["class"] = np.where(d["category"] == "other", "",
                          d["category"] + np.where(d["seq_conserved"], ", seq-conserved",
                                                   ", seq-non-conserved"))
    out = d[["resid_a", "resid_b", "aa_a", "aa_b", "pDFI_a", "pDFI_b", "delta_pDFI",
             "category", "hinge_in", "seq_conserved", "class"]].rename(columns={
        "resid_a": f"resid_{la}", "resid_b": f"resid_{lb}", "aa_a": f"aa_{la}", "aa_b": f"aa_{lb}",
        "pDFI_a": f"pDFI_{la}", "pDFI_b": f"pDFI_{lb}"})
    out.to_csv(f"{a.out}_hinges.csv", index=False, float_format="%.4f")
    print(f"{len(d)} matched residues")
    print(out[out["category"] != "other"].groupby(["category", "hinge_in", "seq_conserved"]).size()
          .rename("n").to_string())
    print("\nCandidate pools (paper's sets; the coupling filters are NOT applied here):")
    for name, cond in [("X-like: non-common hinge, seq-non-conserved",
                        (d["category"] == "non-common hinge") & ~d["seq_conserved"]),
                       ("Y-like: common hinge, seq-non-conserved",
                        (d["category"] == "common hinge") & ~d["seq_conserved"]),
                       ("non-common hinge, seq-conserved (targets for DARC/Z)",
                        (d["category"] == "non-common hinge") & d["seq_conserved"])]:
        print(f"  {name}: {d.loc[cond, 'resid_a'].tolist()}")

    plt = _plt()
    if plt:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(d["resid_a"], d["pDFI_a"], color="tab:blue", label=la)
        ax.plot(d["resid_a"], d["pDFI_b"], color="tab:red", label=lb)
        ax.axhline(a.hinge_cut, color="grey", ls=":", lw=0.8)
        ax.set_xlabel(f"residue ({la} numbering)")
        ax.set_ylabel("%DFI")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{a.out}_profiles.png", dpi=200)


# --------------------------------------------------------------------------- subcommand: cluster
def cmd_cluster(a):
    raw = [pd.read_csv(f) for f in a.csvs]
    if a.anchors:
        if len(a.anchors) != len(raw):
            sys.exit("--anchors needs one comma-separated residue list per table, in the same order.")
        lists = [parse_resids(x) for x in a.anchors]
        dfs = [raw[0].set_index("resid")]
        for k in range(1, len(raw)):
            m = anchor_map(raw[0], raw[k], lists[0], lists[k], name=Path(a.csvs[k]).stem)
            dk = raw[k].set_index("resid").loc[m["resid_b"]]
            dk.index = m["resid_a"].to_numpy()  # renumber into the first table's numbering
            dfs.append(dk)
    else:
        dfs = [d.set_index("resid") for d in raw]
    labels = a.labels or [Path(f).stem.replace("_dfi", "") for f in a.csvs]
    common = sorted(set.intersection(*[set(d.index) for d in dfs]))
    X = np.column_stack([d.loc[common, "pDFI"].to_numpy() for d in dfs])  # n x m
    if not a.no_center:
        X = X - X.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(a.components, len(S))
    coords = Vt[:r].T * S[:r]  # m x r   (X* = V* Sigma*, Eq. 8)
    pcs = pd.DataFrame(coords, index=labels, columns=[f"PC{i + 1}" for i in range(r)])
    pcs.to_csv(f"{a.out}_pca.csv", float_format="%.5f")
    diff = coords[:, None, :] - coords[None, :, :]
    dist = pd.DataFrame(np.sqrt((diff ** 2).sum(-1) / r), index=labels, columns=labels)  # Eq. 9
    print(f"{len(common)} common residues; variance fractions: {np.round(S ** 2 / (S ** 2).sum(), 3)}")
    print(pcs.round(3).to_string())
    print("\nEuclidean distances in reduced space (Eq. 9):")
    print(dist.round(3).to_string())
    dist.to_csv(f"{a.out}_distances.csv", float_format="%.5f")
    plt = _plt()
    if plt and r >= 2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.scatter(coords[:, 0], coords[:, 1])
        for lab, (x, y) in zip(labels, coords[:, :2]):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(f"{a.out}_pca.png", dpi=200)


# --------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dfi", help="DFI/DCI from an MD trajectory")
    d.add_argument("topology")
    d.add_argument("-t", "--trajectory", nargs="+", required=True)
    d.add_argument("-o", "--out", default="dfi")
    d.add_argument("--protein-sel", default="protein",
                   help="what counts as protein. Add non-standard residues here, e.g. "
                        "\"protein or resid 42\" for a covalently modified serine")
    d.add_argument("--selection", default=None, help="override: atoms for the covariance "
                   "(default: '(<protein-sel>) and name CA')")
    d.add_argument("--fit-selection", default=None, help="override: atoms for per-window fitting "
                   "(default: '(<protein-sel>) and name N CA C O')")
    d.add_argument("--aa-override", default="", help="one-letter codes for non-standard residues, e.g. 42=S")
    d.add_argument("--skip-ns", type=float, default=100.0, help="equilibration to discard (paper: 100)")
    d.add_argument("--window-ns", type=float, default=50.0, help="covariance window (paper: 50)")
    d.add_argument("--lag-ns", type=float, default=25.0, help="offset between window starts (paper: 25)")
    d.add_argument("--stride", type=int, default=1, help="use every n-th frame inside a window")
    d.add_argument("--dt-ns", type=float, default=None, help="frame spacing if not stored in the trajectory")
    d.add_argument("--active-site", default="", help="resids, e.g. 70,73,130-132,166,234 (topology numbering)")
    d.add_argument("--targets", default="", help="optional resid set; adds %%DCI to it as a DARC filter")
    d.add_argument("--hinge-cut", type=float, default=0.2)
    d.add_argument("--darc-flex", type=float, nargs=2, default=(0.3, 0.5), metavar=("LO", "HI"))
    d.add_argument("--dci-cut", type=float, default=0.8)
    d.add_argument("--darc-dist", type=float, default=8.0)
    d.add_argument("--concat", action="store_true",
                   help="treat all -t files as consecutive segments of ONE continuous run "
                        "(default: each file is an independent repeat)")
    d.add_argument("--converge", action="store_true", help="compare profiles across 25/50/75/100 ns windows")
    d.set_defaults(func=cmd_dfi)

    c = sub.add_parser("compare", help="classify hinges between two proteins")
    c.add_argument("csv_a")
    c.add_argument("csv_b")
    c.add_argument("--labels", nargs=2, default=("A", "B"))
    c.add_argument("--map", help="two-column file resid_A resid_B (default: identical numbering)")
    c.add_argument("--anchors-a", default="", help="equivalent residues (e.g. active site) in protein A, "
                   "e.g. 70,73,130,132,166,234; with --anchors-b realigns B onto A's numbering")
    c.add_argument("--anchors-b", default="", help="the same residues in protein B's numbering, same order")
    c.add_argument("--hinge-cut", type=float, default=0.2)
    c.add_argument("--shift", type=float, default=0.2, help="min |delta %%DFI| for a non-common hinge")
    c.add_argument("-o", "--out", default="compare")
    c.set_defaults(func=cmd_compare)

    k = sub.add_parser("cluster", help="SVD/PCA of %%DFI profiles")
    k.add_argument("csvs", nargs="+")
    k.add_argument("--labels", nargs="+")
    k.add_argument("--anchors", nargs="+", help="one comma-separated list of equivalent residues per table "
                   "(same order as the tables); all are renumbered into the FIRST table's numbering")
    k.add_argument("--components", type=int, default=2)
    k.add_argument("--no-center", action="store_true", help="literal SVD of X (no mean-centering)")
    k.add_argument("-o", "--out", default="cluster")
    k.set_defaults(func=cmd_cluster)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
