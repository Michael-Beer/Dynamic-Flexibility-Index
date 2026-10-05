# Dynamic-Flexibility-Index

Dynamic Flexibility Index (DFI) and Dynamic Coupling Index (DCI) analysis of molecular dynamics trajectories, in a single Python script built on [MDAnalysis](https://www.mdanalysis.org/).

The method follows the Methods of:

> Modi T., *et al.* Hinge-shift mechanism as a protein design principle for the evolution of β-lactamases from substrate promiscuity to specificity. *Nat. Commun.* **12**, 1852 (2021). https://doi.org/10.1038/s41467-021-22089-0

From a trajectory it tells you which residues behave as rigid **hinges**, which are flexible, and how strongly each residue is dynamically coupled to a functional site (e.g. the active site). Tables from several proteins or states can then be compared (hinge shifts) or clustered (PCA of flexibility profiles).

> **Status.** This is an independent re-implementation written from the paper's Methods.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The three subcommands](#the-three-subcommands)
  - [`dfi`: trajectory to DFI/DCI](#dfi-trajectory-to-dfidci)
  - [`compare`: hinge shifts between two proteins](#compare-hinge-shifts-between-two-proteins)
  - [`cluster`: PCA of flexibility profiles](#cluster-pca-of-flexibility-profiles)
- [Method summary](#method-summary)
- [Output files](#output-files)
- [Differences from the paper and limitations](#differences-from-the-paper-and-limitations)
- [Validation](#validation)
- [Citation](#citation)
- [License](#license)

---

## Installation

Requires Python 3.8+ and:

```
numpy  scipy  pandas  MDAnalysis  matplotlib (optional, for plots)
```

```bash
conda create -n dfi python=3.11 numpy scipy pandas matplotlib
conda activate dfi
pip install MDAnalysis
```

Then just run the script:

```bash
python dfi_analysis.py --help
python dfi_analysis.py dfi --help
```

It has been run under Python 3.8 and 3.12. Any topology/trajectory format MDAnalysis can read works (Amber `.prmtop`/`.nc`, GROMACS `.tpr`/`.xtc`, etc.).

---

## Quick start

```bash
# 1. DFI/DCI from three independent repeats of one system
python dfi_analysis.py dfi system.prmtop -t rep1.nc rep2.nc rep3.nc \
    --active-site 70,73,130,132,166,234 -o TEM1

# 2. Is the profile stable with respect to window length?
python dfi_analysis.py dfi system.prmtop -t rep1.nc rep2.nc rep3.nc --converge -o TEM1

# 3. Compare hinges between two proteins (same numbering)
python dfi_analysis.py compare TEM1_dfi.csv KPC2_dfi.csv --labels TEM1 KPC2 -o TEM1_vs_KPC2

# 4. Cluster several proteins/states by their %DFI profiles
python dfi_analysis.py cluster TEM1_dfi.csv KPC2_dfi.csv SHV1_dfi.csv -o clusters
```

The residue numbers in `--active-site` are the **topology's** residue ids, which may differ from literature numbering (e.g. Ambler). The numbers above are placeholders.

`compare` and `cluster` work on the `*_dfi.csv` tables written by `dfi`; only `dfi` reads trajectories.

---

## The three subcommands

### `dfi`: trajectory to DFI/DCI

```
python dfi_analysis.py dfi TOPOLOGY -t TRAJ [TRAJ ...] [options] -o PREFIX
```

For each repeat the script discards an equilibration period, slides overlapping windows along the rest, computes a Cα covariance matrix per window, derives the DFI by perturbation response scanning, and averages over windows and repeats. If `--active-site` is given it also computes DCI and flags DARC candidates.

| Option | Default | Meaning |
|---|---|---|
| `-t, --trajectory` | required | One or more trajectory files. **Each file is an independent repeat** unless `--concat` is given. |
| `--concat` | off | Treat all `-t` files as consecutive segments of *one* run (e.g. restarts). |
| `-o, --out` | `dfi` | Output prefix. |
| `--skip-ns` | 100 | Equilibration discarded at the start of each repeat. |
| `--window-ns` | 50 | Length of each covariance window. |
| `--lag-ns` | 25 | Offset between successive window starts. |
| `--stride` | 1 | Use every n-th frame within a window. |
| `--dt-ns` | from file | Frame spacing, if not stored in the trajectory. |
| `--protein-sel` | `protein` | What counts as protein. Add non-standard residues here (see below). |
| `--selection` | `(<protein-sel>) and name CA` | Override: atoms used for the covariance. |
| `--fit-selection` | `(<protein-sel>) and name N CA C O` | Override: atoms used to superimpose each frame onto the window's first frame. |
| `--aa-override` | none | One-letter codes for non-standard residues, e.g. `42=S`. |
| `--active-site` | none | Functional-site residue ids, e.g. `70,73,130-132,166,234`. Enables DCI and DARC output. |
| `--targets` | none | Optional second residue set; adds %DCI to it as an extra DARC filter. |
| `--hinge-cut` | 0.2 | %DFI below this is a hinge. |
| `--darc-flex LO HI` | 0.3 0.5 | %DFI range for DARC candidates. |
| `--dci-cut` | 0.8 | %DCI above this counts as strongly coupled. |
| `--darc-dist` | 8.0 | Minimum distance (Å) from the active site for DARC candidates. |
| `--converge` | off | Convergence check across 25/50/75/100 ns windows (see below). |

**Multiple repeats.** Pass several files and each is processed on its own: the first `--skip-ns` of each is discarded and no window straddles two files. Windows from all repeats are pooled. The output gains a `DFI_sd_repeats` column (spread of the per-repeat mean profiles). All repeats must share one topology and atom ordering. Repeats shorter than skip + window are skipped with a message.

**Non-standard residues** (e.g. a covalently bound ligand, a modified serine). MDAnalysis's `protein` keyword only matches standard residue names, so such a residue would be silently left out of the covariance. Include it explicitly:

```bash
python dfi_analysis.py dfi acyl.prmtop -t rep1.nc rep2.nc \
    --protein-sel "protein or resid 42" --aa-override 42=S \
    --active-site 42,... -o acyl
```

The backbone atoms of that residue must be named `N`, `CA`, `C`, `O`. A note is printed if some selected residue has no `CA`. `--aa-override` supplies the one-letter code for the `aa` column (otherwise `X`), which matters for sequence conservation in `compare`. Only Cα atoms enter the covariance, so ligand atoms influence the result through the dynamics but are not themselves part of the matrix.

**`--converge`.** Recomputes the mean DFI profile with 25, 50, 75 and 100 ns windows (using your skip and lag), prints the Pearson correlation between the profiles and writes `<out>_convergence.csv`. High agreement means the profile does not depend on window length. It runs *instead of* the normal analysis, so run both. It is slower (four passes over the data), and agreement between windows does not prove that all relevant conformations were sampled.

### `compare`: hinge shifts between two proteins

```
python dfi_analysis.py compare A_dfi.csv B_dfi.csv --labels A B [options] -o PREFIX
```

Matches residues between two tables and classifies them:

- **common hinge**: hinge (%DFI < `--hinge-cut`) in both proteins;
- **non-common hinge**: hinge in only one protein, with |Δ%DFI| > `--shift` (default 0.2);
- each of these is also labelled sequence-conserved or not (same amino acid at the matched position).

It prints counts and candidate pools (non-common/non-conserved, common/non-conserved, non-common/conserved).

**Residue matching** (choose one):

1. *Same numbering* (default): residues are matched by residue number.
2. `--map FILE`: a two-column, whitespace-separated file `resid_A resid_B` (`#` comments allowed).
3. **Anchors**: list equivalent residues (e.g. active-site residues) in each protein, in the same order, and the script derives the mapping:

   ```bash
   python dfi_analysis.py compare A_dfi.csv B_dfi.csv \
       --anchors-a 45,48,105,107,141,209 \
       --anchors-b 44,47,104,106,140,207 \
       --labels A B -o A_vs_B
   ```

   - Each anchor pair gives a numbering offset; residues before the first and after the last anchor use that anchor's offset.
   - Where neighbouring anchors have different offsets there is an indel between them. Its position is chosen to maximise the number of identical amino acids, and is printed. For a deletion in B, the extra residues in A are left unmapped.
   - One anchor gives a constant shift; more anchors handle indels between them.
   - Warnings are printed if anchors carry different amino acids in the two proteins, or if identity over mapped residues is below 20%.
   - The mapping is written to `<out>_map.txt` and can be reused with `--map`.
   - Indel placement is a sequence-identity best guess. In poorly conserved regions it may be off by a few residues, so check the printed breakpoints against a structural alignment if that region matters. Indels outside the anchored region are not detected.

### `cluster`: PCA of flexibility profiles

```
python dfi_analysis.py cluster A_dfi.csv B_dfi.csv C_dfi.csv [...] [options] -o PREFIX
```

Builds a matrix X of %DFI profiles (residues × proteins), applies SVD, and reports each protein's coordinates in the first `--components` (default 2) components, plus the pairwise Euclidean distances in that reduced space (smaller = more similar dynamics).

- Only residues present in every table are used.
- `--labels` sets display names (default: file names without `_dfi`).
- `--no-center` uses the literal uncentred SVD. By default the mean profile across proteins is subtracted (see [limitations](#differences-from-the-paper-and-limitations)).
- Different numbering: `--anchors` takes one comma-separated residue list per table, in table order. All tables are renumbered into the first table's numbering using the same anchor logic as `compare`:

  ```bash
  python dfi_analysis.py cluster A_dfi.csv B_dfi.csv C_dfi.csv \
      --anchors 45,48,105,107 44,47,104,106 45,48,105,107 -o clusters
  ```

---

## Method summary

Per window (following the paper's Methods; equation numbers are the paper's):

1. Superimpose every frame onto the window's first frame using backbone heavy atoms.
2. Compute the 3N × 3N covariance matrix **G** of the Cα coordinates; it stands in for the inverse Hessian (Eq. 2).
3. Perturbation response scanning: a random, isotropic unit force applied at residue *j* gives a displacement of residue *i* of **ΔR = G F**. The script uses the analytic expectation of the random-force average: the response `A[i,j]` is the Frobenius norm of the 3×3 block `G_ij` (up to a constant that cancels in all normalised quantities). This is the N × N response matrix (Eq. 3).
4. **DFI** (Eq. 5):

   ```
   DFI_i = sum_j A[i,j] / sum_ij A[i,j]
   ```

   averaged over windows and repeats. **%DFI** is the percentile rank across residues, from 0 (most rigid) to 1 (most flexible). Hinges are %DFI < 0.2.
5. **DCI** to a functional set *F* (Eq. 6):

   ```
   DCI_i = [ sum_{j in F} A[i,j] / |F| ] / [ sum_j A[i,j] / N ]
   ```

   with **%DCI** its percentile rank. Pairwise coupling (Eq. 7) is `A[i,j] / mean_j(A[i,:])`; the full matrix is saved so you can compute it yourself (see below).
6. **DARC candidates**: %DFI between 0.3 and 0.5, %DCI to the active site > 0.8, more than 8 Å (minimum heavy-atom distance, first analysed frame) from the active site, and, if `--targets` is given, %DCI to the targets > 0.8.

For `compare`, `cluster` and the PCA distance (Eq. 9) see the subcommand sections above.

Working with the saved response matrix:

```python
import numpy as np
A = np.load("TEM1_response_matrix.npy")        # A[i, j]: response of residue i to a kick at j
coupling = A / A.mean(axis=1, keepdims=True)   # pairwise DCI, Eq. 7
```

Row/column order follows the `resid` column of `TEM1_dfi.csv`.

---

## Output files

`dfi` (prefix `P`):

| File | Contents |
|---|---|
| `P_dfi.csv` | One row per residue: `resid, resname, aa, DFI, DFI_sd` (spread over windows), `DFI_sd_repeats` (if more than one repeat), `pDFI` (%DFI), `hinge`; with `--active-site` also `DCI_site, pDCI_site, dist_site, DARC` (and `pDCI_targets` with `--targets`). |
| `P_response_matrix.npy` | Window-averaged N × N response matrix. |
| `P_dfi.png` | %DFI along the sequence, hinge region shaded, DARC candidates marked. |
| `P_convergence.csv` | With `--converge` only: mean DFI per window size. |

`compare` (prefix `P`): `P_hinges.csv` (per matched residue: %DFI in each protein, change, category, sequence conservation), `P_profiles.png` (overlaid profiles), and `P_map.txt` (with anchors).

`cluster` (prefix `P`): `P_pca.csv` (coordinates), `P_distances.csv` (reduced-space distances), `P_pca.png` (PC1 vs PC2).

---

## Differences from the paper and limitations

- **Analytic random kicks.** The paper applies random unit forces; this script uses their exact expectation instead of sampling. Results should match up to sampling noise, but this has not been compared directly with the reference code.
- **PCA centring.** The paper describes an SVD of the %DFI matrix and does not mention centring. By default the mean profile is subtracted (otherwise PC1 is just the mean profile); use `--no-center` for the literal version. This is an interpretation, not something confirmed from the paper.
- **X/Y/Z design sets are only partly implemented.** `compare` lists the candidate residue pools and `dfi --targets` provides a coupling filter, but the paper's full selection (the coupling filters in its Supplementary material) is not reproduced.
- **Window sizes.** The paper mentions "three" window sizes but lists four; `--converge` uses all four (25, 50, 75, 100 ns).
- **Sampling.** With few frames per window (fewer than 3N), each window covariance is rank-deficient. DFI uses the covariance directly (no inversion), so this is a noisier estimate rather than an error, and a note is printed. Judge it with `--converge` and `DFI_sd_repeats`.
- **Single chain.** `--active-site` and `--targets` select by residue id, which is ambiguous if resids repeat across chains.
- **Numbering.** `--active-site` uses the topology's residue ids. `compare` and `cluster` match residues by number unless `--map` or anchors are used.
- **DARC distance** is measured in the first analysed frame, not averaged over the trajectory.

---

## Validation

Tested on synthetic data only:

- A trajectory sampled from an elastic-network model's covariance, with a random rigid-body rotation and translation applied to every frame, reproduced the model's analytic DFI (Pearson r = 0.9995), which checks the fitting, covariance and response-matrix pipeline end to end.
- Anchor-based realignment recovered all true residue pairs (57/57) on a synthetic pair of sequences containing a 2-residue insertion, a 3-residue deletion and point mutations, and placed both indels at the correct positions.

Not yet done: comparison with the authors' DFI/DCI code, or reproduction of the paper's reported numbers (e.g. the hinge counts for GNCA vs TEM-1).

---

## Citation

If you use this script, please cite the original method paper (above). Add a citation for this repository here once it has a DOI or release.

## License

Add a license (e.g. MIT) before publishing.
