import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gdviz.optim import gradient_descent, newton, grad_quadratic


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
