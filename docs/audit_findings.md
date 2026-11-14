# Code audit, changes, and validation record

## Scope inspected

The candidate is based on the four executable Python scripts attached in this
conversation and on the thesis archive's source map, active JFM manuscript,
Whittaker supplementary derivation, small-radial-gravity/Bessel--Whittaker
supplement, and the archive's `SOURCE_AUDIT.md` and supplementary README.
The thesis, manuscript, figures, template files, and correspondence are not
included in the candidate repository.

The ZIP archive contains no `.py` or `.ipynb` files. Its supplementary
`Wittaker.tex` derivation appears twice in the archive as identical copies; it
is not a second executable Python code. Only one Whittaker Python script was
attached in this conversation.

## Original-to-working mapping

| Uploaded source | Working copy | Role |
|---|---|---|
| `Pasted text(20261114-063743).txt` | `src/annular_swmhd/solvers/bessel_constant_height.py` | Constant-height Bessel determinant |
| `Pasted text (2)(20261114-063751).txt` | `src/annular_swmhd/solvers/bessel_weakly_varying_height.py` | Weak-height Bessel approximation |
| `Pasted text(20261114-064207).txt` | `src/annular_swmhd/solvers/whittaker_radial_gravity.py` | Whittaker radial-gravity solver |
| `Pasted text(20261114-063418).txt` | `src/annular_swmhd/solvers/chebyshev_collocation.py` | Global Chebyshev solver |

Verbatim source snapshots are under `original_sources/`. The copies there differ
from the upload only by filename/extension.

## Changes made to working copies

Only non-mathematical runtime/reporting changes have been made:

1. **Chebyshev convergence helper:** changed the call to the undefined
   `find_eigenmodes(...)` function to call the existing
   `find_eigenmodes_determinant_dict(...)` implementation with the same inputs.
   This fixes a name-resolution failure in `run_convergence_test`; it does not
   change the operator or root algorithm.
2. **Chebyshev printed units:** changed the report label for `C` from
   `m^3/s` to `m^3/s^2`, consistent with the equilibrium-height expression
   `C/g * (1/r - 1/r0)` and the source audit's stated units.
3. **Whittaker export:** added the missing `csv` import, changed zoom-panel
   annotation references from undefined `ax` to the created `ax2`, and changed
   the misleading filename `annular_swmhd_roots_B_0=0.csv` to
   `annular_swmhd_roots.csv` (the source default has nonzero `B0`).

All original uploads are preserved unchanged in `original_sources/`.

## Unresolved scientific/numerical issues

### A. Frequency-sign conventions

- Primary manuscript and thesis source: `omega_star = omega-omega_A^2/omega`.
- Constant-height Bessel script: minus sign.
- Weak-height Bessel script: minus sign.
- Chebyshev global operator: minus sign.
- Whittaker radial-gravity script and supporting supplementary derivation:
  plus sign, `omega_star = omega+omega_A^2/omega`.
- Chebyshev WKB magnetic predictions: use plus-sign polynomials, while the
  global differential operator uses the minus sign.

These are not unified in this candidate.

### B. Weak-height Bessel `Gfac`

The script's `Gfac` denominator is `w - (ws**2 - f**2)`. The supplied
Whittaker derivation instead has `w * (ws**2 - f**2)` in the denominator of
its correction factor. The script expression also subtracts quantities with
incompatible frequency dimensions. No correction was applied because the
intended derivation and sign convention for this particular Bessel script
must first be confirmed.

### C. Chebyshev boundary/diagnostic sign

`build_operator` enforces a boundary row proportional to
`omega_star * eta_prime - (m*f/r) * eta`. `reconstruct_vr` and the reported
boundary residuals use a plus sign. The operator and diagnostics therefore do
not report the same boundary functional. The sign was not changed without
checking the original first-order momentum system.

### D. Unverified determinant bracketing

The Chebyshev determinant scan rejects invalid sample frequencies but does not
explicitly partition every candidate bracket at all singular frequencies
before a `brentq` call. A determinant sign change near a pole/resonance should
be checked against the smallest-singular-value residual and the original
first-order equations. The determinant and SVD routines should be compared as
independent checks in a follow-up validation.

### E. Whittaker branch identity and scan coverage

Whittaker branch continuation uses frequency proximity, not radial node count
or eigenfunction overlap. The source header explicitly accepts that close
branches can exchange labels. The plotting code also skips branch index 4 by
configuration. The seed `n=0` labels are based on smallest positive and
negative roots, not a node-count proof. Scan resolution and resonance exclusion
should be checked for each parameter regime.

### F. Whittaker singular and turning-point handling

The frequency safety logic excludes `omega=0` and `omega_*^2=f^2`. The
transformation also uses `r_d=[2*sqrt(-K^2)]^-1`, so the `K^2=0` transition
requires dedicated handling. The current root-bracketing validity logic does
not explicitly partition intervals at every `K^2=0` crossing. No new condition
has been added without deriving the correct continuation through that point.

### G. Unpublished material / licensing

The Overleaf thesis project contains the thesis PDF, unpublished manuscript
source, figures, institute template files, bibliography files, and extensive
supporting derivations. None are included in this code-only candidate.
Their presence in the uploaded archive is not treated as authorization to
redistribute them. The four Python sources do not carry a clear license notice;
public distribution rights and the desired license require author confirmation.

## Validation performed

The following checks were executed in the preparation environment (Python
3.13.5; NumPy 2.3.5; SciPy 1.17.0; Matplotlib 3.10.8; mpmath 1.3.0):

| Check | Result | Interpretation |
|---|---|---|
| `py_compile` on all four uploaded scripts | PASS | All four parse syntactically |
| Import all four original scripts without running `__main__` | PASS | Imports resolve in this environment |
| `run_numerical_validations` for Chebyshev case 4, `N_col=40` | PASS | Checks depth at `r0`, derivative consistency, `Q` identity, magnetic derivative, polynomial differentiation, positive depth, and ascending grid |
| Constant-height Bessel seed scan, `m=1`, `[-5f,5f]` | 8 roots found | Root-search smoke test only; not a branch-validation result |
| Weak-height Bessel seed scan, `m=1`, `[-5f,5f]` | 8 roots found | Root-search smoke test only; its `Gfac` expression remains under scientific review |
| Whittaker seed scan, `m=1`, `[-5f,5f]` | 9 roots found | Root-search smoke test only; not a branch-validation result |
| Constant-height Bessel boundary recurrence identity | PASS | The implemented `J/Y` boundary expressions equal `omega_* eta' - m*f*eta/r` at the checked point |
| Whittaker `M/W` derivative finite-difference checks | PASS | Analytic derivatives agree with centered finite differences at the tested real argument |

A low-resolution code-path smoke test of the patched Chebyshev convergence
helper at `N_col=12` returned 19 candidate modes; this is not a convergence
result. Full plotting workflows, full azimuthal spectra, high-resolution
convergence sweeps, eigenfunction cross-comparisons, and the limiting-case
sweeps have not been run. The tests in this candidate do not establish global spectral
accuracy. A patched candidate smoke-run of the full `__main__` workflows is
also not claimed.

## Prepublication gate

Not yet resolved:

- Choose an explicit license, or choose to publish without a license (which
  leaves reuse permissions restricted by default copyright).
- Confirm the author/citation metadata for a `CITATION.cff` file.
- Confirm that the four uploaded Python scripts may be publicly redistributed.
- Resolve or explicitly publish the documented scientific inconsistencies as
  known limitations.
- Approve the exact manifest in the final audit report.
