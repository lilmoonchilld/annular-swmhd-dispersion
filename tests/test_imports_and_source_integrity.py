import importlib
import py_compile
from pathlib import Path


MODULES = [
    "annular_swmhd.solvers.bessel_constant_height",
    "annular_swmhd.solvers.bessel_weakly_varying_height",
    "annular_swmhd.solvers.whittaker_radial_gravity",
    "annular_swmhd.solvers.chebyshev_collocation",
]


def test_solver_modules_import_without_running_main():
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_all_working_solver_files_compile():
    root = Path(__file__).parents[1] / "src" / "annular_swmhd" / "solvers"
    files = sorted(path for path in root.glob("*.py") if path.name != "__init__.py")
    assert len(files) == 4
    for source in files:
        py_compile.compile(str(source), doraise=True)
