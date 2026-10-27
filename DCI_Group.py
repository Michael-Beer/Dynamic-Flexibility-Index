#!/usr/bin/env python3
"""
dci_groups.py  --  which residues couple strongly to the active site in one enzyme group,
the other group, or both?  (default groups: carbapenemases vs carbapenem inhibited)

Input : one DCI table per enzyme (the *_dci.csv written by `dfi_analysis.py dci`) and a
        multiple sequence alignment (FASTA or Clustal format) containing all the enzymes.
Output: a per-residue table, a category for every residue, a text summary and a heat map,
        with residues shown in Ambler numbering.

How it works
------------
 1. Residue numbering differs between enzymes, so every table is mapped onto the alignment:
    the table's amino-acid column (`aa`) is located in that enzyme's (ungapped) alignment
    sequence, which handles different numbering offsets, indels and trimmed termini ('X' in
    either sequence matches anything). The rows of the output are the residues of the
    reference enzyme (--ref); alignment columns where it has a gap are dropped.
 2. Ambler numbering is taken from the alignment, with TEM-1 as the numbering standard
    (--ambler-ref): Ser70 is the column where every enzyme has S (or X, the acylated serine) with
    Lys73 three columns later; TEM-1 residues are then numbered consecutively from Ser70 = 70,
    omitting 239 and 253 (--ambler-skip). Alignment columns where TEM-1 has a gap inside the
    sequence get the preceding number plus a letter (e.g. 121A); columns beyond its ends continue
    the numbering. The residues at Ambler 70, 73, 130, 132, 166 and 234 are printed for every
    enzyme so you can check the numbering. This is only as good as the alignment: near indels,
    and in loops where the alignment is uncertain, it can differ from the published numbering
    of an individual enzyme. --numbering ref shows the reference enzyme's own residue numbers
    instead (they are always in the table's ref_resid column).
 3. A residue "couples strongly" in an enzyme if its %DCI (default column pDCI_active) is
    above --cut (default 0.8, the paper's threshold for strongly coupled).
 4. Per residue, count the enzymes in each group that couple strongly (group A = --groupA,
    default the carbapenemases; group B = --groupB, default the carbapenem-inhibited enzymes)
    and classify:
        carb_only    >= --min-in of group A strong, and <= --max-out of group B
        inhib_only   the reverse
        both         >= --min-in strong in each group
        other        anything else (mixed or weak)
        low_coverage fewer than --min-data enzymes of a group have a residue at that position
    (carb / inhib are the --tagA / --tagB names.) Defaults (--min-in 3, --max-out 1) mean
    "strong in at least 3 of 4, and in at most 1 of the 4 of the other group". Use --min-in 4
    --max-out 0 for the strictest version. The table keeps every enzyme's %DCI and the counts,
    so you can re-threshold without rerunning.

Note: residues in the active-site set itself have high DCI to that set by construction, so
expect them to appear as "both". --mark (default Ambler 70,73 = Ser70 and Lys73) flags them in
the summary; --label (default Ser70, Lys73, Ser130, Glu166, Asn170, Lys234) marks residues on
the heat map and checks that the named amino acid is really there in every enzyme.

Example (acyl-enzyme state; tables in KPC2/KPC_dci.csv, NMCA/NMCA_dci.csv, ...)
-------
  python dci_groups.py --state AEC --fasta enzymes.fasta --ref KPC2 \\
      --template "{enz}/{short}_dci.csv" -o AEC_groups

Requires numpy, pandas (matplotlib optional, for the heat map).
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_A = ["KPC2", "NMCA", "SFC1", "SME1"]    # carbapenemases
GROUP_B = ["BlaC", "CTXM16", "SHV1", "TEM1"]  # non-carbapenemases


# --------------------------------------------------------------------------- alignment helpers
def read_alignment(path):
    """Read an alignment as {name: aligned sequence}. The format is detected from the content
    (not the file extension): FASTA ('>' records) or Clustal / Clustal Omega style blocks."""
    lines = open(path).read().splitlines()
    first = next((l for l in lines if l.strip()), "")
    recs = {}
    if first.startswith(">"):
        name, seq = None, []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs[name] = "".join(seq)
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line)
        if name is not None:
            recs[name] = "".join(seq)
    else:  # blocks of 'name  sequence  [residue count]'; conservation lines start with whitespace
        for line in lines:
            if not line.strip() or line[0].isspace() or line.upper().startswith(("CLUSTAL", "MUSCLE", "MAFFT")):
                continue
            tok = line.split()
            if len(tok) >= 2 and re.fullmatch(r"[A-Za-z.\-*]+", tok[1]):
                recs[tok[0]] = recs.get(tok[0], "") + tok[1]
    if not recs:
        sys.exit(f"No sequences could be read from {path} (expected FASTA or Clustal alignment format).")
    return recs


def norm(x):
    return re.sub(r"[^a-z0-9]", "", x.lower())


def find_record(recs, enz, alias):
    key = norm(alias.get(enz, enz))
    exact = [k for k in recs if norm(k) == key]
    if len(exact) == 1:
        return exact[0]
    part = [k for k in recs if key in norm(k)]
    if len(part) == 1:
        return part[0]
    sys.exit(f"Could not match '{enz}' to exactly one alignment record (candidates: {part or 'none'}). "
             f"Records are: {list(recs)}. Use --alias {enz}=<record id>.")


def locate(t, s):
    """Offset such that table residue i sits at ungapped alignment index i + offset.
    Slides t along s and keeps the best match; 'X' (non-standard residue) in either sequence
    matches anything."""
    ta = np.frombuffer(t.encode(), dtype=np.uint8)
    sa = np.frombuffer(s.encode(), dtype=np.uint8)
    wild = ta == ord("X")
    best = (-1, 0, 1)
    for off in range(-(len(t) - 1), len(s)):
        i0, i1 = max(0, -off), min(len(t), len(s) - off)
        if i1 <= i0:
            continue
        seg = sa[i0 + off:i1 + off]
        m = int(((ta[i0:i1] == seg) | wild[i0:i1] | (seg == ord("X"))).sum())
        if m > best[0]:
            best = (m, off, i1 - i0)
    return best[1], best[0], best[2]


def column_map(df, aligned, enz):
    """Alignment column (0-based) for every row of the table (-1 if outside the alignment sequence)."""
    s = re.sub(r"[-.]", "", aligned)
    cols = [c for c, ch in enumerate(aligned) if ch not in "-."]
    t = "".join(df["aa"].astype(str)).upper()
    off, m, n = locate(t, s)
    ident = m / n
    if ident < 0.9 or n < 0.5 * len(t):
        sys.exit(f"{enz}: the table's sequence does not match its alignment sequence "
                 f"(best ungapped match: {100 * ident:.0f}% identity over {n} of {len(t)} residues). "
                 f"Check the alignment record, the --alias, and that the table's `aa` column is right "
                 f"(use --aa-override in dfi_analysis.py for non-standard residues).")
    idx = np.full(len(df), -1)
    for i in range(len(df)):
        u = i + off
        if 0 <= u < len(s):
            idx[i] = cols[u]
    unmapped = int((idx < 0).sum())
    print(f"  {enz:8s} offset {off:+d}, identity {100 * ident:.0f}% over {n} residues"
          + (f", {unmapped} table residue(s) outside the alignment sequence" if unmapped else ""))
    return idx


# --------------------------------------------------------------------------- main
def parse_resids(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def kv(items, what):
    d = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"{what} expects NAME=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        d[k] = v
    return d


AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
       "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
       "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}


def parse_labels(text):
    """'Ser70,Lys73,130' -> [('Ser70', 70, 'S'), ('Lys73', 73, 'K'), ('130', 130, None)]"""
    out = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        m = re.fullmatch(r"([A-Za-z]{3})?(\d+)", item)
        if not m or (m.group(1) and m.group(1).capitalize() not in AA3):
            sys.exit(f"--label: cannot read '{item}'; use e.g. Ser130 or 130 (Ambler numbers).")
        out.append((item, int(m.group(2)), AA3[m.group(1).capitalize()] if m.group(1) else None))
    return out


def pick_enzyme(name, enzymes, what):
    exact = [e for e in enzymes if norm(e) == norm(name)]
    starts = [e for e in enzymes if norm(e).startswith(norm(name))]
    if len(exact) == 1:
        return exact[0]
    if len(starts) == 1:
        print(f"{what} {name} interpreted as {starts[0]}")
        return starts[0]
    sys.exit(f"{what} {name} does not identify one enzyme. Enzymes are: {enzymes}")


def find_ser70_column(aln, forced=None):
    """0-based alignment column of Ser70: the column where every enzyme has S (or X, the acylated
    serine) and Lys73 sits three columns later, i.e. the conserved S-x-x-K motif."""
    if forced:
        return forced - 1
    names = list(aln)
    n_col = len(aln[names[0]])
    hits = [c for c in range(n_col - 3)
            if all(aln[n][c] in "SX" and aln[n][c + 3] == "K" for n in names)]
    if len(hits) == 1:
        return hits[0]
    sys.exit(("No alignment column has S/X followed three columns later by K in every enzyme"
              if not hits else
              f"{len(hits)} columns look like Ser70-x-x-Lys73 (alignment columns {[h + 1 for h in hits]})")
             + "; give the 1-based alignment column of Ser70 with --ser70-col.")


def ambler_labels(aligned_ref, ser70_col, skip):
    """Ambler label for every alignment column, using one enzyme (TEM-1) as the numbering standard.

    Residues of that enzyme are numbered consecutively from Ser70 = 70, omitting the numbers in
    `skip` (Ambler numbering has no 239 and 253). Columns where it has a gap inside the sequence
    get the preceding number plus a letter (52A, 52B ...); columns beyond its ends continue the
    numbering outwards."""
    allowed = [n for n in range(1, 3000) if n not in skip]
    n_col = len(aligned_ref)
    cols = [c for c, ch in enumerate(aligned_ref) if ch not in "-."]
    if ser70_col not in cols:
        sys.exit("The Ambler reference enzyme has a gap at the Ser70 column; use a different --ambler-ref.")
    j0, p70 = cols.index(ser70_col), allowed.index(70)
    if p70 - j0 < 0:
        sys.exit("The Ambler reference has more residues before Ser70 than Ambler numbers available.")
    num = {c: allowed[p70 + j - j0] for j, c in enumerate(cols)}
    labels = []
    for c in range(n_col):
        if c in num:
            labels.append(str(num[c]))
        elif c < cols[0]:
            k = cols[0] - c
            i = allowed.index(num[cols[0]]) - k
            labels.append(str(allowed[i]) if i >= 0 else f"-{k}")
        elif c > cols[-1]:
            k = c - cols[-1]
            labels.append(str(allowed[allowed.index(num[cols[-1]]) + k]))
        else:
            prev = max(x for x in cols if x < c)
            labels.append(f"{num[prev]}{chr(65 + min(c - prev - 1, 25))}")
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="state name used in the file template, e.g. AEC or TI")
    p.add_argument("--fasta", "--alignment", dest="fasta", required=True,
                   help="multiple sequence alignment containing every enzyme (FASTA or Clustal format, "
                        "detected from the content). Record names must match the enzyme names.")
    p.add_argument("--ref", default="BlaC", help="reference enzyme: its residues are the rows of the table "
                                                 "(default BlaC)")
    p.add_argument("--groupA", nargs="+", default=GROUP_A, help="default: KPC2 NMCA SFC1 SME1")
    p.add_argument("--groupB", nargs="+", default=GROUP_B, help="default: BlaC CTXM16 SHV1 TEM1")
    p.add_argument("--labelA", default="carbapenemases", help="group A name used in the heat map and summary")
    p.add_argument("--labelB", default="carbapenem inhibited", help="group B name used in the heat map and summary")
    p.add_argument("--tagA", default="carb", help="short group A tag used in column and category names")
    p.add_argument("--tagB", default="inhib", help="short group B tag used in column and category names")
    p.add_argument("--numbering", choices=["ambler", "ref"], default="ambler",
                   help="residue numbers shown in the summary and heat map: Ambler (default) or the "
                        "reference enzyme's own numbering")
    p.add_argument("--ambler-ref", default="TEM1",
                   help="enzyme whose sequence defines Ambler numbering (default TEM1)")
    p.add_argument("--ambler-skip", default="239,253", help="numbers absent from Ambler numbering")
    p.add_argument("--ser70-col", type=int, default=None,
                   help="1-based alignment column of Ser70, if the S-x-x-K motif is not found automatically")
    p.add_argument("--template", default="{enz}/{enz}_{state}_dci.csv",
                   help="path of each enzyme's DCI table; {enz} = enzyme name (e.g. KPC2), "
                        "{short} = name without trailing digits (KPC), {state} = --state")
    p.add_argument("--file", action="append", default=[], metavar="ENZ=PATH",
                   help="explicit table for one enzyme (overrides the template)")
    p.add_argument("--alias", action="append", default=[], metavar="ENZ=RECORD",
                   help="alignment record id for an enzyme whose name does not match")
    p.add_argument("--column", default="pDCI_active", help="%%DCI column to use")
    p.add_argument("--cut", type=float, default=0.8, help="%%DCI above this = couples strongly")
    p.add_argument("--min-in", type=int, default=3, help="enzymes of a group that must couple strongly")
    p.add_argument("--max-out", type=int, default=1, help="max enzymes of the other group allowed to couple")
    p.add_argument("--min-data", type=int, default=3, help="min enzymes per group with a residue at the position")
    p.add_argument("--mark", default="70,73",
                   help="residues to flag, in the numbering chosen by --numbering (default Ambler 70,73 = "
                        "Ser70 and Lys73)")
    p.add_argument("--label", default="Ser70,Lys73,Ser130,Glu166,Asn170,Lys234",
                   help="residues labelled (with a dashed line) on the heat map, as name+Ambler number, "
                        "e.g. Ser130, or a bare number; '' for none")
    p.add_argument("-o", "--out", default=None, help="output prefix (default dci_groups_<state>)")
    a = p.parse_args()

    out = a.out or f"dci_groups_{a.state}"
    A_set, B_set = list(a.groupA), [e for e in a.groupB if e not in a.groupA]
    enzymes = A_set + B_set
    a.ref = pick_enzyme(a.ref, enzymes, "--ref")
    a.ambler_ref = pick_enzyme(a.ambler_ref, enzymes, "--ambler-ref")
    tA, tB = a.tagA, a.tagB
    cats = [f"{tA}_only", f"{tB}_only", "both", "other", "low_coverage"]
    files, alias = kv(a.file, "--file"), kv(a.alias, "--alias")
    recs = read_alignment(a.fasta)

    paths = {e: files.get(e) or a.template.format(enz=e, short=re.sub(r"\d+$", "", e), state=a.state)
             for e in enzymes}
    missing = [f"{e}: {pth}" for e, pth in paths.items() if not Path(pth).exists()]
    if missing:
        sys.exit("DCI table(s) not found:\n  " + "\n  ".join(missing) +
                 "\nChange --template ({enz}, {short}, {state} are filled in) or use --file ENZ=PATH.")

    print(f"Mapping {len(enzymes)} enzymes onto the alignment ({a.state}):")
    tabs, colidx, aln = {}, {}, {}
    for e in enzymes:
        path = paths[e]
        df = pd.read_csv(path)
        for c in ("resid", "aa", a.column):
            if c not in df.columns:
                sys.exit(f"{path}: no column '{c}'. Columns are: {list(df.columns)}")
        aln[e] = recs[find_record(recs, e, alias)].upper()
        tabs[e] = df
        colidx[e] = column_map(df, aln[e], e)
    if len({len(v) for v in aln.values()}) != 1:
        sys.exit(f"Alignment records have different lengths {sorted({len(v) for v in aln.values()})}; "
                 f"this must be an alignment.")

    # ---- Ambler numbering from the alignment
    skip = set(parse_resids(a.ambler_skip)) if a.ambler_skip else set()
    ser70 = find_ser70_column(aln, a.ser70_col)
    labels = ambler_labels(aln[a.ambler_ref], ser70, skip)
    col_of = {lab: c for c, lab in enumerate(labels)}
    ar_nums = [int(labels[c]) for c in range(len(labels))
               if aln[a.ambler_ref][c] not in "-." and labels[c].isdigit()]
    print(f"\nAmbler numbering from {a.ambler_ref} (Ser70 at alignment column {ser70 + 1}); "
          f"{a.ambler_ref} spans Ambler {min(ar_nums)}-{max(ar_nums)} ({len(ar_nums)} residues)")
    labelled = parse_labels(a.label) if a.label else []
    land = sorted({70, 73, 130, 132, 166, 170, 234} | {n for _, n, _ in labelled})
    print("Residue at each Ambler landmark in each enzyme (X = non-standard residue, - = gap):")
    print("  " + " " * 8 + "".join(f"{n:>5d}" for n in land))
    for e in enzymes:
        letters = [aln[e][col_of[str(n)]] if str(n) in col_of else "?" for n in land]
        print(f"  {e:8s}" + "".join(f"{x:>5s}" for x in letters))
        if aln[e][ser70] not in "SX" or aln[e][ser70 + 3] != "K":
            print(f"  WARNING: {e} does not have Ser70/Lys73 at the chosen column", file=sys.stderr)
    print("  (class A enzymes normally have S70, K73, S130, N132, E166 and K234; check the "
          "rest of your landmarks look right)")
    for text, n, letter in labelled:  # does each labelled residue really have that amino acid?
        if letter is None or str(n) not in col_of:
            continue
        diff = [f"{e}={aln[e][col_of[str(n)]]}" for e in enzymes
                if aln[e][col_of[str(n)]] not in (letter, "X")]
        if diff:
            print(f"  NOTE: {text} is not {letter} in: {', '.join(diff)}")
    print()

    # ---- table in the reference enzyme's numbering
    ref_cols = colidx[a.ref]
    keep = ref_cols >= 0
    tab = pd.DataFrame({
        "ref_resid": tabs[a.ref]["resid"].to_numpy()[keep],
        "ref_aa": tabs[a.ref]["aa"].to_numpy()[keep],
        "ambler": [labels[c] for c in ref_cols[keep]],
        "align_col": ref_cols[keep] + 1,
    })
    dcol = a.column.replace("pDCI", "DCI", 1)
    for e in enzymes:
        look = {c: i for i, c in enumerate(colidx[e]) if c >= 0}
        idx = np.array([look.get(c - 1, -1) for c in tab["align_col"]])
        has = idx >= 0
        d = tabs[e]
        tab[f"pDCI_{e}"] = np.where(has, d[a.column].to_numpy(float)[np.maximum(idx, 0)], np.nan)
        if dcol in d.columns and dcol != a.column:
            tab[f"DCI_{e}"] = np.where(has, d[dcol].to_numpy(float)[np.maximum(idx, 0)], np.nan)
        r = pd.Series(d["resid"].to_numpy()[np.maximum(idx, 0)]).astype("Int64")
        r[~has] = pd.NA
        tab[f"resid_{e}"] = r

    PA = tab[[f"pDCI_{e}" for e in A_set]].to_numpy(float)
    PB = tab[[f"pDCI_{e}" for e in B_set]].to_numpy(float)
    nA, nB = (PA > a.cut).sum(axis=1), (PB > a.cut).sum(axis=1)
    dA, dB = (~np.isnan(PA)).sum(axis=1), (~np.isnan(PB)).sum(axis=1)
    tab[f"n_strong_{tA}"], tab[f"n_strong_{tB}"] = nA, nB
    tab[f"n_data_{tA}"], tab[f"n_data_{tB}"] = dA, dB
    with warnings.catch_warnings():  # positions where a whole group has a gap give an empty mean (NaN)
        warnings.simplefilter("ignore", RuntimeWarning)
        tab[f"mean_{tA}"], tab[f"mean_{tB}"] = np.nanmean(PA, axis=1), np.nanmean(PB, axis=1)
    tab[f"delta_{tA}_minus_{tB}"] = tab[f"mean_{tA}"] - tab[f"mean_{tB}"]
    cond = [
        (dA < a.min_data) | (dB < a.min_data),
        (nA >= a.min_in) & (nB >= a.min_in),
        (nA >= a.min_in) & (nB <= a.max_out),
        (nB >= a.min_in) & (nA <= a.max_out),
    ]
    tab["category"] = np.select(cond, ["low_coverage", "both", cats[0], cats[1]], default="other")

    shown = tab["ambler"] if a.numbering == "ambler" else tab["ref_resid"].astype(str)
    marks = parse_resids(a.mark) if a.mark else []
    tab["marked"] = shown.isin([str(m) for m in marks])
    tab.to_csv(f"{out}_table.csv", index=False, float_format="%.4f")

    # ---- report
    names = {cats[0]: f"strong in {a.labelA} only", cats[1]: f"strong in {a.labelB} only",
             "both": "strong in both groups", "other": "other (mixed or weak)",
             "low_coverage": "low alignment coverage"}
    num_note = (f"Ambler numbering (from {a.ambler_ref})" if a.numbering == "ambler"
                else f"{a.ref} numbering")
    lines = [f"state {a.state}; residues shown in {num_note}; table rows are {a.ref} residues "
             f"(its own numbers are in the ref_resid column); strong = {a.column} > {a.cut}; "
             f"a group needs >= {a.min_in} strong, the other group <= {a.max_out}",
             f"{a.labelA}: {A_set}", f"{a.labelB}: {B_set}",
             f"{len(tab)} residues in the alignment-mapped table", ""]
    for cat in cats:
        sub = tab[tab["category"] == cat]
        res = [f"{r}{'*' if m else ''}" for r, m in zip(shown[sub.index], sub["marked"])]
        lines.append(f"{names[cat]}: {len(sub)}" + (f"\n  {', '.join(res)}" if cat != "other" and res else ""))
    if marks:
        lines.append("\n(* = in --mark; the active-site residues themselves, which couple to the site by construction)")
    text = "\n".join(lines)
    print("\n" + text)
    Path(f"{out}_summary.txt").write_text(text + "\n")
    print(f"\nWrote {out}_table.csv, {out}_summary.txt")

    # ---- heat map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not installed; skipping heat map")
        return
    M = np.vstack([PA.T, PB.T])
    lab = shown.reset_index(drop=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 4.6),
                                  gridspec_kw={"height_ratios": [len(enzymes), 1.2], "hspace": 0.1})
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(enzymes)))
    ax.set_yticklabels(enzymes, fontsize=8)
    ax.axhline(len(A_set) - 0.5, color="w", lw=2)
    fig.colorbar(im, ax=[ax, ax2], label=a.column, pad=0.01)
    full = ["tab:red", "tab:blue", "tab:purple", "white", "lightgrey"]
    order = {cats[0]: 0, cats[1]: 1, "both": 2, "other": 3, "low_coverage": 4}
    strip = np.array([order[c] for c in tab["category"]])[None, :]
    ax2.imshow(strip, aspect="auto", interpolation="nearest", vmin=0, vmax=4, cmap=ListedColormap(full))
    ax2.set_yticks([0])
    ax2.set_yticklabels(["category"], fontsize=8)
    if a.numbering == "ambler":  # a tick at every Ambler multiple of 10
        tk = [i for i, t in enumerate(lab) if t.isdigit() and int(t) % 10 == 0]
    else:
        tk = []
    if len(tk) < 3:
        tk = np.linspace(0, len(tab) - 1, min(15, len(tab))).astype(int).tolist()
    ax2.set_xticks(tk)
    ax2.set_xticklabels([lab[i] for i in tk], rotation=90, fontsize=7)
    ax.set_xticks([])
    amb = tab["ambler"].reset_index(drop=True)
    pts = []
    for text, n, _ in labelled:
        hit = np.where(amb == str(n))[0]
        if len(hit):
            pts.append((int(hit[0]), text))
        else:
            print(f"NOTE: no {a.ref} residue at Ambler {n}; {text} not labelled on the heat map")
    pts.sort()
    if pts:
        for i, _ in pts:
            for axx in (ax, ax2):
                axx.axvline(i, color="white" if axx is ax else "black", ls="--", lw=0.7)
        ax.set_xticks([i for i, _ in pts])
        ax.set_xticklabels([t for _, t in pts], fontsize=7)
        ax.xaxis.tick_top()
        close = [(pts[k + 1][0] - pts[k][0]) < 12 for k in range(len(pts) - 1)]
        crowded = any(close[k] and close[k + 1] for k in range(len(close) - 1))  # three or more together
        ha = ["center"] * len(pts)
        for k, c in enumerate(close):  # neighbouring labels (70/73, 166/170) must not overlap
            if c and ha[k] == "center":
                ha[k], ha[k + 1] = "right", "left"
        for tl, h in zip(ax.get_xticklabels(), ha):
            tl.set_ha(h)
            if crowded:
                tl.set_rotation(90)
    nm = "Ambler numbering" if a.numbering == "ambler" else f"{a.ref} numbering"
    ax2.set_xlabel(f"residue ({nm})\nred = {a.labelA} only, blue = {a.labelB} only, purple = both",
                   fontsize=9)
    fig.savefig(f"{out}_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {out}_heatmap.png")


if __name__ == "__main__":
    main()
