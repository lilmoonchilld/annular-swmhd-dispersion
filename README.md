# Waves in a Rotating Annular Shallow-Water Magnetohydrodynamic System with Radial Gravity

Research-code candidate for analytical and numerical dispersion-relation solvers
for a rotating annular shallow-water magnetohydrodynamic (SWMHD) system with an
imposed vertical magnetic field. The research includes constant-depth,
weakly varying-depth, and radial-gravity approximations.

> **Scientific scope:** the solver files are deliberately kept separate. They
> do not all use the same modified-frequency sign convention or the same
> approximation. Do not compare or combine their roots until the formulation
> notes in [`docs/formulation_map.md`](docs/formulation_map.md) have been read.

## Included solvers

| Module | Role | Main method |
|---|---|---|
| `bessel_constant_height.py` | Constant-height annular model | Bessel determinant with frequency scanning and continuation |
| `bessel_weakly_varying_height.py` | Weak radial-depth-gradient model with the parameter choices in the source script | Bessel/modified-Bessel radial basis, determinant roots, frequency continuation |
| `whittaker_radial_gravity.py` | Weak-radial-variation model including the central-gravity contribution | Whittaker M/W radial basis, hybrid special-function evaluation, determinant roots |
| `chebyshev_collocation.py` | Global variable-depth annular problem | Chebyshev-Lobatto collocation, determinant root search, SVD eigenfunction extraction, WKB branch matching |

The mathematical assumptions, sign conventions, source relationships, and
known disagreements are documented in [`docs/formulation_map.md`](docs/formulation_map.md).

## Installation

Tested in the preparation environment with Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0, Matplotlib 3.10.8, and mpmath 1.3.0. The lower Python bound in
`pyproject.toml` is a packaging declaration, not a claim that every older
Python/dependency combination has been tested.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

## Running a solver

Each solver retains its own parameter block and `__main__` workflow. From the
repository root, for example:

```bash
python -m annular_swmhd.solvers.bessel_constant_height
python -m annular_swmhd.solvers.bessel_weakly_varying_height
python -m annular_swmhd.solvers.whittaker_radial_gravity
python -m annular_swmhd.solvers.chebyshev_collocation
```

The last command runs the global spectrum workflow with the parameter values
and resolution currently in that source file. It can be computationally
expensive. The scripts write figures and, where implemented, CSV files to the
current working directory. They are not yet exposed through one shared command
line/configuration interface because their model definitions are not
interchangeable.

## Tests

```bash
pytest
```

The tests exercise source import/compilation, differentiation-matrix identities,
the constant-height Bessel boundary identity, and Whittaker special-function
derivatives. They are regression and implementation checks, not a proof that
every dispersion branch is physically correct.

## Source preservation

`original_sources/` contains the four uploaded Python scripts as immutable
baseline snapshots (the file extensions were changed from `.txt` to `.py`,
but the contents were not changed). The working copies are in
`src/annular_swmhd/solvers/`. Changes made to working copies are recorded in
`docs/audit_findings.md`.

The Overleaf thesis archive, manuscript, private correspondence, CVs,
recommendation letters, and thesis figures are not included in this code-only
candidate. Their mathematical source role was inspected, but they are not
proposed for public redistribution.

## Scientific limitations

Several formulation and implementation questions remain unresolved. In
particular, the weak-height Bessel `Gfac` does not match the documented
Whittaker gradient factor; the Chebyshev global operator and its WKB/diagnostic
helpers contain sign-convention inconsistencies; and the Whittaker implementation
uses the plus-sign modified frequency found in a supplementary derivation,
not the minus-sign convention of the primary manuscript. Details and exact
locations are in [`docs/audit_findings.md`](docs/audit_findings.md).

No solver should be described as independently validated against experiment or
as establishing the correctness of all branches. The reported tests cover only
the checks explicitly listed above.

## License and citation

**No license is included in this candidate.** A public GitHub repository without
an explicit license remains subject to default copyright restrictions; public
visibility alone does not grant permission to reuse, modify, or redistribute
the code. A license and author/citation metadata must be selected and confirmed
before publication. No `CITATION.cff` is included because the software citation
metadata has not yet been author-confirmed.
