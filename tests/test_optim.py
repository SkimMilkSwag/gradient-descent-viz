import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gdviz.optim import gradient_descent, newton, grad_quadratic


def test_gradient_descent_converges_to_zero():
    x, hist = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=100)
    assert abs(x) < 1e-3
    assert len(hist) > 1


def test_newton_faster():
    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=200)
    _, nw = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    assert len(nw) - 1 < len(gd) - 1  # Newton should need fewer iterations


def test_newton_converges():
    x, _ = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    assert abs(x) < 1e-6
