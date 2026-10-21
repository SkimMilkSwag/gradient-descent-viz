"""Tests for gdviz.contour — 2D objective contours with trajectory overlays."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from gdviz.contour import _xy_grid, contour_plot
from gdviz.optim import (
    gradient_descent_linesearch,
    gradient_descent_vec,
    l_bfgs,
    grad_rosenbrock,
    rosenbrock,
)


def test_xy_grid_evaluates_objective_on_mesh():
    # Z[i, j] must equal f([x_j, y_i]) — the row-major layout contourf wants.
    X, Y, Z = _xy_grid(rosenbrock, x_range=(-1.0, 1.0), y_range=(0.0, 2.0), n=4)
    xs = np.linspace(-1.0, 1.0, 4)
    ys = np.linspace(0.0, 2.0, 4)
    assert X.shape == (4, 4) and Z.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            assert abs(Z[i, j] - rosenbrock([X[i, j], Y[i, j]])) < 1e-9


def test_xy_grid_masks_nonfinite_values():
    # An objective that blows up outside its domain must not poison the
    # color scale — non-finite cells become nan.
    def f(p):
        x, y = float(p[0]), float(p[1])
        if abs(x) > 1.5:
            return np.inf
        return x * x + y * y

    X, Y, Z = _xy_grid(f, x_range=(-2.0, 2.0), y_range=(-1.0, 1.0), n=9)
    finite_cols = [j for j in range(9) if abs(X[0, j]) <= 1.5]
    for i in range(9):
        for j in range(9):
            if j in finite_cols:
                assert np.isfinite(Z[i, j])
            else:
                assert np.isnan(Z[i, j])


def test_contour_plot_saves_file():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    x0 = [-1.2, 1.0]
    _, fixed = gradient_descent_vec(rosenbrock, grad_rosenbrock, x0, lr=0.001, steps=50)
    out = "/tmp/gh_gdviz_contour_test.png"
    if os.path.exists(out):
        os.remove(out)
    path = contour_plot(rosenbrock, {"gd": fixed}, out_path=out, n=60)
    assert path == out
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_contour_plot_skips_scalar_histories():
    # A 1-D (scalar problem) trajectory can't live on a 2D surface — it must
    # be skipped without raising, while the valid 2D one still draws.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    def f(x):
        return float(np.dot(x, x))

    scalar_hist = [5.0, 4.2, 3.1, 2.0]  # 1-D history
    vec_hist = np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 0.5]])
    out = "/tmp/gh_gdviz_contour_mixed_test.png"
    path = contour_plot(f, {"scalar": scalar_hist, "vec": vec_hist}, out_path=out, n=40)
    assert path == out and os.path.getsize(out) > 0


def test_contour_plot_returns_none_without_matplotlib(monkeypatch):
    # The no-matplotlib import path must degrade to None, not raise — same
    # contract as plot_convergence.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = contour_plot(rosenbrock, {"gd": [[0.0, 0.0], [1.0, 1.0]]}, out_path="/tmp/x.png")
    assert out is None


def test_contour_plot_draws_full_rosenbrock_trajectories():
    # End-to-end: real optimizer runs overlaid on the surface. Line-search and
    # L-BFGS should each reach the minimum in far fewer recorded points than
    # fixed-step GD over the same step budget — that gap is exactly what this
    # plot exists to show.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    x0 = [-1.2, 1.0]
    _, fixed = gradient_descent_vec(rosenbrock, grad_rosenbrock, x0, lr=0.001, steps=300)
    _, ls = gradient_descent_linesearch(rosenbrock, grad_rosenbrock, x0, lr=1.0, steps=300)
    _, lb = l_bfgs(rosenbrock, grad_rosenbrock, x0, steps=300)
    out = "/tmp/gh_gdviz_contour_e2e.png"
    path = contour_plot(
        rosenbrock,
        {"fixed-step gd": fixed, "gd + armijo": ls, "l-bfgs": lb},
        out_path=out,
        n=80,
    )
    assert path == out and os.path.getsize(out) > 0
    assert len(lb) - 1 < len(fixed) - 1
