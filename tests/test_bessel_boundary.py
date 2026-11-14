import numpy as np
from scipy.special import jv, yv, jvp, yvp

from annular_swmhd.solvers import bessel_constant_height as solver


def test_constant_height_bessel_boundary_functions_match_velocity_operator():
    m = 2
    omega = 3.0 * solver.f
    r = 0.7 * (solver.r1v + solver.r2v) / 2.0
    k = solver.radial_k(omega, m)
    ws = solver.omega_star(omega)

    expected_j = ws * k * jvp(m, k * r) - (m * solver.f / r) * jv(m, k * r)
    expected_y = ws * k * yvp(m, k * r) - (m * solver.f / r) * yv(m, k * r)

    np.testing.assert_allclose(solver.boundary_J(omega, m, r), expected_j, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(solver.boundary_Y(omega, m, r), expected_y, rtol=1e-10, atol=1e-22)
