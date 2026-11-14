import numpy as np
import pytest

from annular_swmhd.solvers import whittaker_radial_gravity as solver


@pytest.mark.parametrize("function", [solver.whitM_and_dz, solver.whitW_and_dz])
def test_whittaker_special_function_derivative(function):
    kappa = 0.3
    mu = 1.0
    z = 2.1
    h = 1.0e-6

    value, derivative = function(kappa, mu, z)
    forward = function(kappa, mu, z + h)[0]
    backward = function(kappa, mu, z - h)[0]
    finite_difference = (forward - backward) / (2.0 * h)

    assert np.isfinite(value)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7, atol=1e-9)


def test_whittaker_dispersion_function_is_finite_at_regular_frequency():
    omega = 3.0 * solver.f
    assert solver.frequency_is_safe(omega)
    assert np.isfinite(solver.disp(omega, 1))
