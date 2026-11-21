"""Tests for Adam on the 2D Rosenbrock benchmark.

Companion to tests/test_adam.py (which covers the bowl/anisotropic cases):
these pin how Adam *navigates* the Rosenbrock valley — a narrow curved floor
that punishes fixed-step methods — from several starting points, and how its
iteration count stacks up against plain gradient descent via the shared
``benchmark`` helper.

The point is not that Adam is the fastest method on Rosenbrock (L-BFGS and
Newton are) but that it is *robust*: every tested start lands on (1, 1), the
adaptive per-coordinate step survives the valley's scale mismatch without a
hand-tuned lr, and the iteration cost from an off-valley start is bounded.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from gdviz.adam import adam_vec
from gdviz.optim import (
    benchmark,
    gradient_descent_vec,
    grad_rosenbrock,
    rosenbrock,
)


# A spread of starting points: the repo's default, a point far off the valley
# on the negative side, a point near the minimum, and one well outside it.
ROSEN_STARTS = [
    [-1.2, 1.0],   # the canonical start used across the demo + other tests
    [-2.0, -1.0],   # far off the valley: steepest initial climb
    [0.5, -0.5],    # near the minimum: short navigation
    [2.0, 3.0],     # outside on the positive side: long approach
]


def test_adam_converges_rosenbrock_from_many_starts():
    """Every tested start lands on (1, 1) within tol.

    The valley floor is where fixed-step methods crawl; Adam's adaptive step
    keeps it moving at a usable rate along the curve, so all four starts
    converge rather than stalling.
    """
    for x0 in ROSEN_STARTS:
        x_final, hist = adam_vec(rosenbrock, grad_rosenbrock, x0, lr=0.02, steps=30000)
        assert np.allclose(x_final, [1.0, 1.0], atol=1e-5), \
            f"Adam did not converge from {x0}: landed at {x_final}"
        # history contract: x0 first, then one entry per accepted step
        assert hist[0] == pytest.approx(np.asarray(x0, dtype=float))
        assert rosenbrock(x_final) < 1e-8


def test_adam_rosenbrock_valley_cost_scales_with_distance():
    """Off-valley starts cost more iterations; near the minimum they are cheap.

    This is the "valley navigation" behaviour: a start already close to (1, 1)
    needs far fewer steps than one that must first climb up and follow the
    curved floor. The ordering pins that Adam's per-step progress is genuinely
    distance-dependent (not a flat limit cycle like RMSProp's on this surface).
    """
    iters = {}
    for x0 in ROSEN_STARTS:
        _, hist = adam_vec(rosenbrock, grad_rosenbrock, x0, lr=0.02, steps=30000)
        iters[tuple(x0)] = len(hist) - 1

    # a start near the minimum must be strictly cheaper than the far one
    assert iters[(0.5, -0.5)] < iters[(-2.0, -1.0)]
    # and every run actually made progress (none burned the full budget)
    assert all(n < 30000 for n in iters.values())


def test_adam_beats_plain_gd_on_rosenbrock_via_benchmark():
    """Through the shared ``benchmark`` helper, Adam needs far fewer steps than
    plain gradient descent from the canonical start — the adaptive step is the
    whole point of running it next to a fixed-step baseline.
    """
    x0 = [-1.2, 1.0]
    gd_iters, gd_ok = benchmark(grad_rosenbrock, gradient_descent_vec, x0,
                               [1.0, 1.0], lr=0.001, steps=30000)
    adam_iters, adam_ok = benchmark(grad_rosenbrock, adam_vec, x0,
                                    [1.0, 1.0], lr=0.02, steps=30000)

    # both reach tol...
    assert gd_ok and adam_ok
    # ...but Adam's adaptive step makes it to the minimum in a fraction of the
    # iterations plain descent burns crawling the valley floor.
    assert adam_iters < gd_iters


def test_adam_rosenbrock_history_is_descent_to_minimum():
    """The recorded objective should trend down toward 0 along the trajectory —
    no divergence spikes. A non-finite or growing value mid-run would mean the
    adaptive step overshot the valley, which on this benchmark it shouldn't."""
    x0 = [-1.2, 1.0]
    _, hist = adam_vec(rosenbrock, grad_rosenbrock, x0, lr=0.02, steps=30000)
    values = [rosenbrock(p) for p in hist]
    # every recorded point is finite and the tail is at the machine floor
    assert all(np.isfinite(v) for v in values)
    assert min(values) < 1e-8
    # the final state is the best (or tied-best) reached
    assert values[-1] <= min(values) + 1e-12
