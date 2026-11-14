# Proposed public-file manifest — approval pending

**Repository:** `annular-swmhd-dispersion`  
**Description:** Analytical and numerical solvers for wave dispersion relations
in rotating annular shallow-water magnetohydrodynamic systems with radial gravity.  
**Proposed default branch:** `main`  
**Proposed release tag:** `0.1.0`  
**Visibility:** public, only after explicit approval  
**License:** unresolved; no license file included  
**Citation metadata:** unresolved; no `CITATION.cff` included

## Files proposed for publication

| Path | Purpose |
|---|---|
| `.gitignore` | Exclude caches, virtual environments, and generated outputs |
| `README.md` | Project scope, installation, usage, limitations, and license status |
| `PUBLIC_FILE_MANIFEST.md` | This proposed public-file list and exclusions |
| `pyproject.toml` | Installable Python package metadata and dependencies |
| `requirements.txt` | Runtime dependency list |
| `src/annular_swmhd/__init__.py` | Package marker and version |
| `src/annular_swmhd/solvers/__init__.py` | Solver package marker |
| `src/annular_swmhd/solvers/bessel_constant_height.py` | Constant-height Bessel solver |
| `src/annular_swmhd/solvers/bessel_weakly_varying_height.py` | Weakly varying-height Bessel solver |
| `src/annular_swmhd/solvers/whittaker_radial_gravity.py` | Whittaker radial-gravity solver |
| `src/annular_swmhd/solvers/chebyshev_collocation.py` | Global Chebyshev collocation solver |
| `original_sources/README.md` | Provenance and edit policy for snapshots |
| `original_sources/bessel_constant_height_original.py` | Unmodified uploaded source snapshot |
| `original_sources/bessel_weakly_varying_height_original.py` | Unmodified uploaded source snapshot |
| `original_sources/whittaker_radial_gravity_original.py` | Unmodified uploaded source snapshot |
| `original_sources/chebyshev_collocation_solver_original.py` | Unmodified uploaded source snapshot |
| `docs/formulation_map.md` | Differences among models, equations, assumptions, and sign conventions |
| `docs/audit_findings.md` | Source-to-working map, fixes, validation record, open issues |
| `tests/test_imports_and_source_integrity.py` | Import and syntax checks |
| `tests/test_chebyshev_operator.py` | Chebyshev differentiation and boundary-row checks |
| `tests/test_bessel_boundary.py` | Constant-height Bessel boundary identity |
| `tests/test_whittaker_special_functions.py` | Whittaker derivative and finite-determinant checks |

## Explicit exclusions

The following are **not** proposed for publication in this repository:

- The full IIT Kanpur thesis project ZIP, compiled thesis PDF, thesis chapters,
  appendices, institutional thesis template, and thesis bibliography.
- The original JFM manuscript source, manuscript PDF, figures, and bibliography.
- The supplementary derivation archive, including QG appendices and alternate
  derivations, unless separately approved for publication.
- Private correspondence, emails, CVs, recommendation letters, and personal
  application material.
- Generated figures, root CSVs, large numerical arrays, and cached outputs.
- Any credentials or private local configuration (none were found in the four
  selected Python source files during this review).

The existing public repository `lilmoonchilld/Dispersion-relation` is not part
of this proposal and will not be modified or repurposed.
