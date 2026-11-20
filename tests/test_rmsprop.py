"""Tests for gdviz.rmsprop — RMSProp (root mean square propagation) optimizer."""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from gdviz.adam import adam_vec
from gdviz.optim import benchmark, gradient_descent_vec
from gdviz.rmsprop import rmsprop, rmsprop_vec


def f_bowl(p):
    x, y = float(p[0]), float(p[1])
    return x * x + y * y


def grad_bowl(p):
    x, y = float(p[0]), float(p[1])
    return [2.0 * x, 2.0 * y]


def f_stretched(a):
    """Anisotropic bowl: steep in one axis (curvature 200), shallow in the other."""
    x = float(a[0])
    y = float(a[1])
    return 100.0 * x * x + y * y


def grad_stretched(a):
    x, y = float(a[0]), float(a[1])
    return [200.0 * x, 2.0 * y]


def test_rmsprop_vec_converges_quadratic():
    x_final, hist = rmsprop_vec(f_bowl, grad_bowl, [-3.0, 4.0], lr=0.1, steps=2000)
    assert np.allclose(x_final, [0.0, 0.0], atol=1e-3)
    # history contract: starts at x0, one entry per accepted step
    assert hist[0] == pytest.approx([-3.0, 4.0])
    assert len(hist) >= 2


def test_rmsprop_scalar_converges():
    x_final, hist = rmsprop(lambda a: a * a, lambda a: [2.0 * a], 5.0, lr=0.1, steps=2000)
    assert abs(x_final) < 1e-2
    # history contract for the scalar wrapper: a plain list of floats
    assert isinstance(hist, list) and hist and all(isinstance(p, float) for p in hist)


def test_rmsprop_handles_anisotropic_bowl_single_lr():
    """Per-coordinate normalization: one global lr must not DIVERGE on an
    anisotropic problem where fixed-step GD blows up. GD at lr=0.1 diverges
    along the steep axis (factor 2*lr*100 = 20 > 1); RMSProp's RMS step keeps
    every coordinate within a bounded neighborhood of the start — no momentum
    to ride through, but no blowup either."""
    x_final, hist = rmsprop_vec(f_stretched, grad_stretched, [-0.5, 8.0], lr=0.1, steps=2000)
    # every recorded state stays in a bounded band around the optimum
    for p in hist:
        assert f_stretched(p) < f_stretched([-0.5, 8.0]) * 1.5
    # and it lands much closer than the start
    assert f_stretched(x_final) < f_stretched([-0.5, 8.0])

    # fixed-step GD at the same lr actually diverges on this bowl
    gd_x, _ = gradient_descent_vec(None, grad_stretched, [-0.5, 8.0], lr=0.1, steps=2000)
    assert f_stretched(gd_x) > f_stretched([-0.5, 8.0])


def test_rmsprop_no_momentum_means_limited_cycle_on_steep_bowl():
    """The no-momentum contrast, pinned concretely: on the steep axis, RMSProp's
    step is ~lr every iteration regardless of distance, so it oscillates
    around the optimum instead of converging to it. The run settles into a
    limit cycle whose objective stays flat for the whole tail — while Adam
    (bias-corrected momentum) grinds down to the floor on the same budget."""
    x0 = [-0.5, 8.0]
    rp_x, rp_hist = rmsprop_vec(f_stretched, grad_stretched, x0, lr=0.1, steps=2000)
    tail = [f_stretched(p) for p in rp_hist[-500:]]
    # the objective is essentially flat over the last 500 steps (limit cycle),
    # and still far from the machine floor
    assert max(tail) - min(tail) < 1e-3
    assert tail[-1] > 1e-2

    ad_x, _ = adam_vec(f_stretched, grad_stretched, x0, lr=0.1, steps=2000)
    # Adam's momentum carries it across the oscillation to the actual minimum
    assert f_stretched(ad_x) < tail[-1]


def test_rmsprop_vs_adam_on_stretched_bowl():
    """RMSProp and Adam both converge on the stretched bowl with one lr; the
    contrast is in how fast they grind along the flat axis. Adam's bias-
    corrected momentum keeps it moving once the gradient shrinks, while
    RMSProp's raw-gradient step is proportional to |g| — so as g -> 0 the
    steps vanish and the run stalls at a worse point on the same budget.
    Pin that gap: both converge-ish (Adam strictly), RMSProp makes real but
    smaller progress."""
    x0 = [-0.5, 8.0]
    ad_x, _ = adam_vec(f_stretched, grad_stretched, x0, lr=0.1, steps=2000)
    rp_x, _ = rmsprop_vec(f_stretched, grad_stretched, x0, lr=0.1, steps=2000)

    # Adam's adaptive rates finish the job on this budget
    assert np.allclose(ad_x, [0.0, 0.0], atol=1e-2)
    # RMSProp makes genuine progress from the start...
    assert f_stretched(rp_x) < f_stretched(x0)
    # ...but Adam lands strictly closer to the optimum (no-momentum stall)
    assert f_stretched(ad_x) < f_stretched(rp_x)


def test_rmsprop_normalized_step_scale():
    """With a constant gradient g, the per-step move is lr * g / sqrt(E[g^2])
    = lr * sign(g) — ~lr-sized steps per iteration regardless of |g|. A long
    run on f(x) = 10x from x=5 with lr=0.1 should therefore make ~0.1-sized
    moves per step, not 2.0 (raw grad * lr)."""
    def f_lin(p):
        return 10.0 * float(p[0])

    def df_lin(p):
        return [10.0]

    x_final, hist = rmsprop_vec(f_lin, df_lin, [5.0], lr=0.1, steps=200)
    tail = [float(p[0]) for p in hist[-50:]]
    avg_step = abs(tail[0] - tail[-1]) / (len(tail) - 1)
    assert 0.05 < avg_step < 0.15, f"average step {avg_step} not ~lr=0.1"


def test_rmsprop_stops_early_at_gradient_tol():
    """Starting exactly at the optimum: ||grad|| = 0 < tol on step 1, so the
    loop must stop immediately — history is just [x0]."""
    x_final, hist = rmsprop_vec(f_bowl, grad_bowl, [0.0, 0.0], lr=0.1, steps=100)
    assert len(hist) == 1
    assert x_final == pytest.approx([0.0, 0.0])


def test_rmsprop_via_benchmark_helper():
    """rmsprop_vec is (f, df, ...) shaped like the other vector optimizers, so
    the benchmark() wrapper (which passes f through as None) must accept it —
    RMSProp never evaluates f."""
    iters, converged = benchmark(grad_bowl, rmsprop_vec, [-3.0, 4.0], [0.0, 0.0],
                                 lr=0.1, steps=20000)
    assert converged
    assert 1 < iters < 20000


# --- demo CLI (--optimizer flag) -----------------------------------------


def _cli_report(argv):
    """Run the demo main() with argv and return its quadratic_bowl report dict."""
    from gdviz.__main__ import main

    captured = {}

    def fake_dumps(obj, *a, **k):
        captured["report"] = obj
        return "captured"

    with mock.patch("gdviz.__main__.json.dumps", side_effect=fake_dumps), \
         mock.patch("gdviz.__main__.print"):
        main(argv)
    return captured["report"]["quadratic_bowl"]


def test_cli_defaults_to_all_optimizers():
    report = _cli_report([])
    assert set(report) == {"gd", "gd_momentum_0.9", "adam", "rmsprop", "newton"}


def test_cli_single_optimizer_selects_only_that_method():
    report = _cli_report(["--optimizer", "rmsprop"])
    assert set(report) == {"rmsprop"}


def test_cli_repeatable_flag_selects_multiple_methods():
    report = _cli_report(["--optimizer", "newton", "--optimizer", "gd"])
    assert set(report) == {"newton", "gd"}


def test_cli_duplicate_optimizer_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        _cli_report(["--optimizer", "adam", "--optimizer", "adam"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "--optimizer given more than once" in err


def test_cli_unknown_optimizer_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        _cli_report(["--optimizer", "sgd"])
    assert exc.value.code != 0
    assert "invalid choice" in capsys.readouterr().err


def test_cli_selected_methods_report_the_same_numbers_as_full_run():
    full = _cli_report([])
    partial = _cli_report(["--optimizer", "newton"])
    for name in partial:
        assert partial[name] == full[name]

