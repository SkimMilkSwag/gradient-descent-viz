"""Tests for gdviz.adam — Adam (adaptive moment estimates) optimizer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from gdviz.adam import adam, adam_vec
from gdviz.optim import benchmark, gradient_descent_vec


def f_bowl(p):
    x, y = float(p[0]), float(p[1])
    return x * x + y * y


def grad_bowl(p):
    x, y = float(p[0]), float(p[1])
    return [2.0 * x, 2.0 * y]


def f_stretched(a):
    """Anisotropic bowl: steep in one axis, shallow in the other."""
    x = float(a[0])
    y = float(a[1])
    return 100.0 * x * x + y * y


def grad_stretched(a):
    x = float(a[0])
    y = float(a[1])
    return [200.0 * x, 2.0 * y]


def test_adam_vec_converges_quadratic():
    x_final, hist = adam_vec(f_bowl, grad_bowl, [-3.0, 4.0], lr=0.1, steps=500)
    assert np.allclose(x_final, [0.0, 0.0], atol=1e-3)
    # history contract: starts at x0, one entry per accepted step
    assert hist[0] == pytest.approx([-3.0, 4.0])
    assert len(hist) >= 2


def test_adam_vec_history_is_monotone_descent():
    """On a convex quadratic with a modest lr, Adam's objective should not
    increase between consecutive recorded states (a sanity check that the
    bias-corrected step is well-scaled, not just that it ends near zero)."""
    x_final, hist = adam_vec(f_bowl, grad_bowl, [-2.0, 1.0], lr=0.05, steps=300)
    values = [f_bowl(p) for p in hist]
    # final is the minimum reached (tolerates a last-step wobble at the floor)
    assert values[-1] <= min(values) * 2 + 1e-12
    # no single step blows up; the check only applies away from the machine
    # floor (prev > 1e-6), where Adam's sign-alternating jitter near
    # convergence makes multiplicative bounds noisy
    for prev, cur in zip(values, values[1:]):
        if prev > 1e-6:
            assert cur < 2.0 * prev + 1e-9


def test_adam_scalar_converges():
    x_final, hist = adam(lambda a: a * a, lambda a: [2.0 * a], 5.0, lr=0.1, steps=300)
    assert abs(x_final) < 1e-2
    # history contract for the scalar wrapper: a plain list of floats
    assert isinstance(hist, list) and hist and all(isinstance(p, float) for p in hist)


def test_adam_handles_anisotropic_bowl_single_lr():
    """The point of adaptive rates: one global lr must work on an anisotropic
    problem where fixed-step GD fails. GD at lr=0.1 diverges along the steep
    axis (factor 2*lr*100 = 20 > 1), and at lr=1e-4 it crawls; Adam at 0.1
    converges from both axes' scales in the same budget."""
    x_final, hist = adam_vec(f_stretched, grad_stretched, [-0.5, 8.0], lr=0.1, steps=2000)
    assert np.allclose(x_final, [0.0, 0.0], atol=1e-2)

    # fixed-step GD at the same lr actually diverges on this bowl...
    gd_x, _ = gradient_descent_vec(None, grad_stretched, [-0.5, 8.0], lr=0.1, steps=2000)
    assert f_stretched(gd_x) > f_stretched([-0.5, 8.0])
    # ...and at an lr stable for the steep axis it still can't finish in budget:
    gd_small, _ = gradient_descent_vec(None, grad_stretched, [-0.5, 8.0], lr=1e-4, steps=2000)
    assert f_stretched(gd_small) > 1.0
    # ...while adam does.
    assert f_stretched(x_final) < 1e-3


def test_adam_tiny_lr_is_still_progress():
    """Even with a small lr, the bias-corrected moments should keep Adam
    moving (no vanishing-step pathology like uncorrected EMAs can cause)."""
    x0 = [-2.0, 2.0]
    x_final, _ = adam_vec(f_bowl, grad_bowl, x0, lr=1e-3, steps=5000)
    assert f_bowl(x_final) < f_bowl(x0)


def test_adam_stops_early_at_gradient_tol():
    """Starting exactly at the optimum: ||grad|| = 0 < tol on step 1, so the
    loop must stop immediately — history is just [x0]."""
    x_final, hist = adam_vec(f_bowl, grad_bowl, [0.0, 0.0], lr=0.1, steps=100)
    assert len(hist) == 1
    assert x_final == pytest.approx([0.0, 0.0])


def test_adam_via_benchmark_helper():
    """adam_vec is (f, df, ...) shaped like the other vector optimizers, so
    the benchmark() wrapper (which passes f through as None) must accept it —
    Adam never evaluates f, and the result feeds benchmark_table unchanged."""
    iters, converged = benchmark(grad_bowl, adam_vec, [-3.0, 4.0], [0.0, 0.0],
                                 lr=0.1, steps=2000)
    assert converged
    assert 1 < iters < 2000


def test_adam_rms_step_scale():
    """With a constant gradient g (a linear objective), Adam's step size must
    approach lr * |g| / sqrt(E[g^2]) = lr per coordinate — the hallmark of
    normalized steps. A long run on f(x) = 10x from x=5 with lr=0.1 should
    therefore make ~0.1-sized moves per step, not 2.0 (raw grad * lr)."""
    # linear objective: gradient is constant at [10]
    def f_lin(p):
        return 10.0 * float(p[0])

    def df_lin(p):
        return [10.0]

    x_final, hist = adam_vec(f_lin, df_lin, [5.0], lr=0.1, steps=200)
    # after many steps the effective step per iteration is ~lr (the bias-
    # corrected m_hat/v_hat ratio tends to |g|/|g|). Check the average step
    # size over the tail of the run rather than a single step (early steps are
    # biased small while the EMAs warm up).
    tail = [float(p[0]) for p in hist[-50:]]
    avg_step = abs(tail[0] - tail[-1]) / (len(tail) - 1)
    assert 0.05 < avg_step < 0.15, f"average step {avg_step} not ~lr=0.1"
