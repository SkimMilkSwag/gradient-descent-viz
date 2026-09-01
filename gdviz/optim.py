"""From-scratch first-order optimizers on scalar and vector problems."""
import numpy as np


def quadratic(x, b=0.0):
    """f(x) = x^2 + b*x (a simple convex bowl). Returns value."""
    return x ** 2 + b * x


def grad_quadratic(x, b=0.0):
    return 2.0 * x + b


def gradient_descent(f, df, x0, lr=0.1, steps=50, tol=1e-8, momentum=0.0):
    """Run (momentum) gradient descent, recording the full trajectory.

    With momentum > 0 this is Polyak heavy-ball (SGD with velocity):
    v = momentum * v - lr * grad, x += v. Returns (x, history).
    """
    x = float(x0)
    v = 0.0
    history = [x]
    for _ in range(steps):
        g = df(x)
        if abs(g) < tol:
            break
        v = momentum * v - lr * g
        x += v
        history.append(x)
    return x, history


def newton(f, df, d2f, x0, steps=30, tol=1e-8):
    """Newton's method using first and second derivatives."""
    x = float(x0)
    history = [x]
    for _ in range(steps):
        d, dd = df(x), d2f(x)
        if abs(dd) < 1e-12:
            break
        step = d / dd
        if abs(step) < tol:
            break
        x -= step
        history.append(x)
    return x, history


def adam(f, df, x0, lr=0.1, steps=50, tol=1e-8,
         beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam (Kingma & Ba 2014) implemented from scratch for the scalar bowl.

    m = beta1*m + (1-beta1)*g;  v = beta2*v + (1-beta2)*g^2
    mhat = m/(1-beta1^t);       vhat = v/(1-beta2^t)
    x -= lr * mhat / (sqrt(vhat) + eps)

    Records the full trajectory like gradient_descent. Returns (x, history).
    """
    x = float(x0)
    m = 0.0
    v = 0.0
    t = 0
    history = [x]
    for _ in range(steps):
        g = df(x)
        if abs(g) < tol:
            break
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x)
    return x, history


def poly_bowl(x, coeffs):
    """Evaluate a polynomial sum(coeffs[i] * x**i)."""
    return sum(c * x ** i for i, c in enumerate(coeffs))


def grad_poly(x, coeffs):
    return sum(i * c * x ** (i - 1) for i, c in enumerate(coeffs) if i >= 1)


def hess_poly(x, coeffs):
    return sum(i * (i - 1) * c * x ** (i - 2) for i, c in enumerate(coeffs) if i >= 2)
