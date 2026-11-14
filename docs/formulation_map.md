# Scientific formulation and source map

This document records what the uploaded implementations actually encode. It
keeps the model branches separate rather than imposing a common formula set.

## Shared research setting

The project studies normal modes in an annular rotating shallow-water MHD layer,
with inner and outer radii `r1`, `r2`, azimuthal mode number `m`, rotation
`Omega`, Coriolis parameter `f=2*Omega`, an imposed vertical field `B0`, and
impermeable radial walls. The relevant scripts use the normal-mode phase
convention `exp[i(m*theta - omega*t)]`.

## 1. Constant-height Bessel implementation

Source: `src/annular_swmhd/solvers/bessel_constant_height.py`.

The depth is fixed at `H0`, and the script defines

\[
\omega_* = \omega-\frac{\omega_A^2}{\omega},\qquad
k^2=\frac{\omega(\omega_*^2-f^2)}{gH_0\omega_*}.
\]

The radial displacement is represented by independent Bessel solutions
`J_m(k*r)` and `Y_m(k*r)`. The boundary operator encoded in the functions
`boundary_J` and `boundary_Y` is equivalent, by the Bessel recurrence relation,
to

\[
\mathcal N[\eta](r)=\omega_*\eta'(r)-\frac{mf}{r}\eta(r).
\]

The two impermeable-wall conditions produce a 2-by-2 boundary determinant.
The frequency scan excludes `omega=0` and intervals crossing
`omega_*^2=f^2`; roots are refined with `brentq`. Branches are continued in
`m` using a secant-predicted Newton step, with fresh scans to recover missed
roots. The Alfven and magneto-Kelvin seed labels are assigned using the ordering
and dispersive-change heuristics in the script; they are not identified by
radial node count.

## 2. Weakly varying-height Bessel implementation

Source: `src/annular_swmhd/solvers/bessel_weakly_varying_height.py`.

The source takes

\[
H'_{\rm eq}(r)=\frac{\Omega^2r}{g},\qquad
\beta_{\rm eff}=\Omega^2,
\]

and defines a modified radial wavenumber through `K2`. Its source parameter
`Ccg` is assigned but is not used in the computed coefficients. The source
header labels the case as `C=0`; thus this script does not implement the
central-gravity `-C/r^2` contribution in the equilibrium gradient.

The radial basis is selected by the sign of `K^2`: ordinary Bessel functions
for positive `K^2`, modified Bessel functions for negative `K^2`, and
power-law/logarithmic limiting forms near zero. The same minus-sign boundary
operator `omega_star*eta' - m*f*eta/r` is used in `N_operator`.

A significant unresolved difference is the implementation of `Gfac`: it uses
`1 + 4*omegaA2*ws / (w - (ws**2 - f**2))`. This does not match the gradient
factor in the supplied Whittaker derivation,

\[
\mathcal G(\omega)=1+
\frac{4\omega_A^2\omega_*}{\omega(\omega_*^2-f^2)}.
\]

The denominator in the script is also dimensionally inconsistent as written:
it subtracts a quantity with units of frequency squared from a frequency.
This has not been silently changed. The intended derivation for this particular
Bessel script must be confirmed before it is used as a limiting-case reference.

## 3. Whittaker formulation with radial gravity

Source: `src/annular_swmhd/solvers/whittaker_radial_gravity.py`, supported by
the supplementary `Wittaker.tex` derivation in the thesis archive.

The reduced radial equation is

\[
\eta''+\frac1r\eta'+\left(K^2+\frac{\Lambda}{r}
-\frac{m^2}{r^2}\right)\eta=0.
\]

The code defines

\[
\omega_* = \omega+\frac{\omega_A^2}{\omega},\qquad
\mathcal G(\omega)=1+
\frac{4\omega_A^2\omega_*}{\omega(\omega_*^2-f^2)},
\]

\[
\beta_{\rm eff}=\Omega^2+\frac{2C}{r_0^3},
\]

\[
K^2=\frac{\omega(\omega_*^2-f^2)}{gD_0\omega_*}
-\frac{mf\beta_{\rm eff}\mathcal G}{gD_0\omega_*},
\qquad
\Lambda=\frac{3mfC\mathcal G}{gD_0\omega_*r_0^2}.
\]

With `r_d=[2*sqrt(-K^2)]^-1`, `z=r/r_d`,
`kappa=Lambda*r_d`, and `mu=abs(m)`, the displacement basis is
`r^-1/2 M_{kappa,mu}(z)` and `r^-1/2 W_{kappa,mu}(z)`. The script provides
real-domain SciPy paths, complex-domain mpmath paths, and a fast scan path
followed by scalar root refinement.

**Convention separation:** this Whittaker script uses the plus-sign modified
frequency. The primary JFM manuscript and active thesis source use
`omega_star = omega - omega_A^2/omega`. The supplied source audit identifies
the plus-sign derivation as an alternate supplementary formulation and says
it must not be substituted into the primary model without re-deriving it from
the chosen momentum and induction equations. The two conventions remain
separate here.

The submitted supplementary TeX file `Wittaker(1).tex` and the earlier
`Wittaker.tex` in the archive are identical copies, not two independent
executable Python solvers. Only one Whittaker Python script was attached in
this conversation.

## 4. Global Chebyshev collocation implementation

Source: `src/annular_swmhd/solvers/chebyshev_collocation.py`.

The solver uses the variable equilibrium profile

\[
H_{\rm eq}(r)=H_0+\frac{\Omega^2}{2g}(r^2-r_0^2)
+\frac{C}{g}\left(\frac1r-\frac1{r_0}\right),
\]

with local magnetic frequency `omega_A^2(r)=B0^2/[mu0*rho0*H_eq^2(r)]`
and the minus-sign definition `omega_*(r)=omega-omega_A^2(r)/omega`.
It constructs Chebyshev-Lobatto points and differentiation matrices, assembles
the second-order radial operator, imposes boundary rows, scans the real
frequency axis with a determinant-based method, and extracts eigenfunctions
from the smallest right-singular vector. WKB local predictions are used to
label/associate branches; they are not the global eigenvalue problem.

Two inconsistencies remain in the uploaded source: the global operator boundary
rows use `omega_* eta' - m*f*eta/r`, while the velocity reconstruction and
boundary diagnostics use `omega_* eta' + m*f*eta/r`; and the magnetic WKB
polynomials construct `omega + omega_A^2/omega`, despite the global operator
using the minus-sign convention. These differences affect diagnostics and
branch matching and must be resolved from the intended first-order equations
before their outputs are interpreted as a single consistent spectrum.

## 5. Relationship between the models

The four solvers provide complementary calculations, but they are not yet a
verified sequence of exact limiting cases:

- The constant-height Bessel script is the analytical uniform-depth baseline
  for the minus-sign convention.
- The weak-height Bessel script introduces a gradient correction but contains
  the unresolved `Gfac` mismatch described above.
- The Whittaker radial-gravity script uses a plus-sign convention and a
  central-gravity-dependent `1/r` radial potential.
- The Chebyshev solver is a global variable-depth numerical problem using the
  minus-sign convention in its differential operator, but its WKB/diagnostic
  helpers have the inconsistencies described above.

Consequently, agreement or disagreement among their roots cannot yet be
attributed solely to the effect of radial gravity or depth variation. First
re-derive the formulas from a single selected linear perturbation system and
then compare like-for-like limits.
