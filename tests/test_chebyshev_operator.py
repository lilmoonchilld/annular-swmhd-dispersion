import numpy as np

from annular_swmhd.solvers import chebyshev_collocation as solver


def test_uploaded_chebyshev_numerical_validations():
    params = solver.Params()
    solver.set_case_parameters(4, params)
    solver.run_numerical_validations(params, N_col=40)


def test_boundary_rows_match_operator_minus_sign():
    params = solver.Params()
    solver.set_case_parameters(4, params)
    r, d1, d2 = solver.chebyshev_lobatto(24, params.r1, params.r2)
    omega = 2.0 * params.f
    m = 2

    operator = solver.build_operator(omega, m, r, d1, d2, params)
    ws_left = solver.omega_star(r[0], omega, params)
    ws_right = solver.omega_star(r[-1], omega, params)

    expected_left = ws_left * d1[0, :].copy()
    expected_left[0] -= m * params.f / r[0]
    expected_right = ws_right * d1[-1, :].copy()
    expected_right[-1] -= m * params.f / r[-1]

    np.testing.assert_allclose(operator[0, :], expected_left)
    np.testing.assert_allclose(operator[-1, :], expected_right)
