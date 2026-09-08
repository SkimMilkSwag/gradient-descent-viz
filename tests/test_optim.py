import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from gdviz.optim import (
    adam,
    benchmark,
    benchmark_table,
    converged_at_tol,
    gradient_descent,
    newton,
    grad_quadratic,
    gradient_descent_vec,
    newton_vec,
    rosenbrock,
    grad_rosenbrock,
    hess_rosenbrock,
    loss_history,
)


def test_adam_converges_to_zero():
    x, hist = adam(lambda x: 0, grad_quadratic, x0=5.0, lr=0.1, steps=200)
    assert abs(x) < 1e-3
    assert len(hist) > 1


def test_adam_matches_reference_trajectory():
    # independent reference implementation to pin the math (bias-corrected Adam)
    x, m, v, t = 5.0, 0.0, 0.0, 0
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    ref = [x]
    for _ in range(200):
        g = 2.0 * x
        if abs(g) < 1e-8:
            break
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x -= lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)) ** 0.5 + eps)
        ref.append(x)
    x_end, hist = adam(lambda x: 0, grad_quadratic, x0=5.0, lr=0.1, steps=200)
    assert len(hist) == len(ref)
    for a, b in zip(hist, ref):
        assert abs(a - b) < 1e-9


def test_adam_robust_to_learning_rate():
    # Plain GD on the quadratic diverges when lr > 1/2 (contraction factor
    # |1 - 2*lr| > 1); Adam's adaptive step size stays ~lr-bounded per
    # iteration, so it keeps converging at learning rates where GD blows up.
    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=1.2, steps=200)
    x_ad, _ = adam(lambda x: 0, grad_quadratic, x0=5.0, lr=1.2, steps=200)
    assert abs(gd[-1]) > 1e6   # GD diverged at lr=1.2
    assert abs(x_ad) < 1e-3    # Adam still converged to the minimum


def test_gradient_descent_converges_to_zero():
    x, hist = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.1, steps=100)
    assert abs(x) < 1e-3
    assert len(hist) > 1


def test_momentum_converges_faster_than_plain_gd():
    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=200)
    _, mom = gradient_descent(
        lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, momentum=0.9, steps=200
    )
    # Same lr; momentum damps oscillation and needs far fewer iterations to tol.
    assert len(mom) - 1 < len(gd) - 1


def test_momentum_default_matches_plain_gd():
    x_plain, _ = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=50)
    x_mom, _ = gradient_descent(
        lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, momentum=0.0, steps=50
    )
    assert x_plain == x_mom


def test_newton_faster():
    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.1, steps=200)
    _, nw = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    assert len(nw) - 1 < len(gd) - 1  # Newton should need fewer iterations


def test_newton_converges():
    x, _ = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    assert abs(x) < 1e-6


# --- Rosenbrock 2D -----------------------------------------------------------

def test_rosenbrock_value_and_minimum():
    # f(1, 1) = 0 is the global minimum; values elsewhere are positive
    assert rosenbrock(np.array([1.0, 1.0])) == 0.0
    assert rosenbrock(np.array([-1.0, 2.0])) > 0.0
    # known reference value: f(0, 0) = (1-0)^2 + 100*(0-0)^2 = 1
    assert rosenbrock(np.array([0.0, 0.0])) == 1.0


def test_rosenbrock_gradient_matches_finite_difference():
    # central-difference check of the analytic gradient at a few points
    h = 1e-6
    for pt in [np.array([-1.0, 2.0]), np.array([0.5, -0.3]), np.array([1.2, 0.8])]:
        g = grad_rosenbrock(pt)
        num = np.zeros(2)
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            num[i] = (rosenbrock(pt + e) - rosenbrock(pt - e)) / (2 * h)
        assert np.allclose(g, num, atol=1e-5), (pt, g, num)


def test_rosenbrock_hessian_matches_fd():
    # Hessian entries checked against central-difference gradients
    h = 1e-5
    for pt in [np.array([-0.4, 1.6]), np.array([0.7, 1.9])]:
        H = hess_rosenbrock(pt)
        for i in range(2):
            e_i = np.zeros(2)
            e_i[i] = h
            g_plus = grad_rosenbrock(pt + e_i)
            g_minus = grad_rosenbrock(pt - e_i)
            for j in range(2):
                H_ij = (g_plus[j] - g_minus[j]) / (2 * h)
                assert abs(H[i, j] - H_ij) < 1e-3, (pt, i, j, H[i, j], H_ij)


def test_gd_vec_converges_on_rosenbrock():
    x, hist = gradient_descent_vec(rosenbrock, grad_rosenbrock, [-1.2, 1.0], lr=0.001, steps=20000)
    assert np.allclose(x, [1.0, 1.0], atol=1e-3)
    assert len(hist) > 100  # the valley floor is slow going


def test_momentum_vec_beats_plain_gd_on_rosenbrock():
    # Same lr; momentum should need fewer iterations to reach tol on the
    # Rosenbrock valley (it carries through the shallow, curved sections).
    _, gd = gradient_descent_vec(rosenbrock, grad_rosenbrock, [-1.2, 1.0], lr=0.001, steps=30000)
    _, mom = gradient_descent_vec(rosenbrock, grad_rosenbrock, [-1.2, 1.0], lr=0.001, momentum=0.9, steps=30000)
    assert len(mom) - 1 < len(gd) - 1


def test_gd_vec_stops_on_divergence():
    # A learning rate past the stability bound sends Rosenbrock off to
    # infinity; the runner must break on non-finite gradient, not loop forever.
    x, hist = gradient_descent_vec(rosenbrock, grad_rosenbrock, [-1.0, 2.0], lr=0.01, steps=1000)
    assert len(hist) < 1000  # stopped early rather than burning all steps


def test_newton_vec_converges_fast_on_rosenbrock():
    # Newton with the analytic Hessian should land on (1,1) in a handful of
    # iterations even from a distant start.
    x, hist = newton_vec(rosenbrock, grad_rosenbrock, hess_rosenbrock, [0.5, -0.5], steps=50)
    assert np.allclose(x, [1.0, 1.0], atol=1e-6)
    assert len(hist) - 1 <= 15


# --- benchmark table ---------------------------------------------------------

def test_converged_at_tol():
    assert converged_at_tol([1.0, 1.0], [1.0, 1.0])
    assert converged_at_tol(np.array([0.999999, 1.0000004]), [1.0, 1.0])
    assert not converged_at_tol([1.0, 1.5], [1.0, 1.0])


def test_benchmark_counts_iterations_and_flag():
    # Momentum converges on the Rosenbrock valley within a 30k budget; the
    # benchmark reports its step count and flags it converged.
    iters, ok = benchmark(
        grad_rosenbrock, gradient_descent_vec, [-1.2, 1.0], [1.0, 1.0],
        lr=0.001, momentum=0.9, steps=30000,
    )
    assert ok
    assert 0 < iters < 30000


def test_benchmark_flags_nonconverged_run():
    # A tiny step budget exhausts before reaching the optimum; iters equals
    # the full budget and converged is False.
    iters, ok = benchmark(
        grad_rosenbrock, gradient_descent_vec, [-1.2, 1.0], [1.0, 1.0],
        lr=0.001, steps=50,
    )
    assert not ok
    assert iters == 50


def test_benchmark_table_format():
    results = {
        "gradient_descent": (28943, True),
        "gd_momentum_0.9": (3121, True),
        "newton": (6, True),
        "diverged_method": (100, False),
    }
    table = benchmark_table(results)
    lines = table.splitlines()
    assert lines[2] == "| method | iterations | converged |"
    assert lines[3] == "|---|---|---|"
    assert "| gradient_descent | 28943 | yes |" in lines
    assert "| diverged_method | 100 | no |" in lines
    # dict order is preserved in the output
    assert [l.split("|")[1].strip() for l in lines[4:]] == list(results)


# --- loss history / overlay ---------------------------------------------------

def test_loss_history_matches_manual_eval():
    # The loss at each recorded point must equal f(x) evaluated by hand — this
    # is the curve the plot's second axis draws.
    _, hist = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=10)
    losses = loss_history(lambda x: x ** 2, hist)
    assert len(losses) == len(hist)
    for i, x in enumerate(hist):
        assert abs(losses[i] - x ** 2) < 1e-9


def test_loss_history_monotone_decreasing_on_quadratic():
    # On a convex bowl with a stable lr the objective must never increase as we
    # descend; this pins the "loss-vs-iteration" curve the plot is meant to show.
    _, hist = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=50)
    losses = loss_history(lambda x: x ** 2, hist)
    for a, b in zip(losses, losses[1:]):
        assert b <= a + 1e-9


def test_loss_history_works_on_vector_trajectory():
    # The same helper must accept an array-valued trajectory (Rosenbrock run)
    # and return the objective at each recorded point.
    _, hist = gradient_descent_vec(rosenbrock, grad_rosenbrock, [-1.2, 1.0], lr=0.001, steps=30)
    losses = loss_history(rosenbrock, hist)
    assert len(losses) == len(hist)
    for i, pt in enumerate(hist):
        assert abs(losses[i] - rosenbrock(pt)) < 1e-9


def test_plot_with_loss_overlay():
    # The overlay must actually save a file when matplotlib is available. It
    # degrades to None on systems without it, so guard the real assertions.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("matplotlib not installed")
    import os

    from gdviz.plot import plot_convergence

    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=30)
    _, nw = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    histories = {"gd": gd, "newton": nw}
    losses = {label: loss_history(lambda x: x ** 2, h) for label, h in histories.items()}
    out = "/tmp/gh_gdviz_overlay_test.png"
    if os.path.exists(out):
        os.remove(out)
    path = plot_convergence(histories, out_path=out, losses=losses)
    assert path == out
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_without_loss_keeps_single_panel():
    # No losses -> the original single-panel trajectory plot, no error.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("matplotlib not installed")
    from gdviz.plot import plot_convergence

    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=20)
    out = "/tmp/gh_gdviz_traj_test.png"
    path = plot_convergence({"gd": gd}, out_path=out)
    assert path == out and os.path.exists(out)
