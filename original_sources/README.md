# Original uploaded solver snapshots

These four Python files are byte-for-byte copies of the uploaded scripts, with
only their filenames changed from `.txt` to `.py`. They are retained so that
all local candidate edits can be diffed against the submitted source.

The cleaned working copies are under `src/annular_swmhd/solvers/`. The only
changes currently made in those copies are:

- `chebyshev_collocation.py`: route the convergence helper to the already
  existing determinant-mode function; correct the printed units for `C`.
- `whittaker_radial_gravity.py`: import `csv`, point the zoom-panel annotations
  at `ax2`, and use a non-misleading generic CSV output filename.

No mathematical expression, physical parameter, root tolerance, or solver
algorithm has been changed in the working copies. Scientific discrepancies
are documented separately and remain unresolved.
