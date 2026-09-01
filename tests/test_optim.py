import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gdviz.optim import adam, gradient_descent, newton, grad_quadratic


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
