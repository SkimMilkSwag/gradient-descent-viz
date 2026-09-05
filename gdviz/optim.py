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


def gradient_descent_vec(f, df, x0, lr=0.1, steps=50, tol=1e-8, momentum=0.0):
    """Vector (multi-dimensional) gradient descent with the same contract as
    the scalar variant: ``v = momentum*v - lr*grad``; converges when the
    gradient norm drops below ``tol``.

    Returns ``(x_final, history)`` where history is a list of arrays —
    the starting point followed by one entry per accepted step.
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(steps):
        g = np.asarray(df(x), dtype=float)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < tol:
            break  # diverged to inf/nan, or converged
        v = momentum * v - lr * g
        x = x + v
        history.append(x.copy())
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


def newton_vec(f, df, hess, x0, steps=30, tol=1e-8):
    """Vector Newton's method: x -= H(x)^{-1} g(x), with a 2x2 solve.

    Uses the same stopping rules as the scalar variant (step norm or
    gradient norm below ``tol``). Returns ``(x_final, history)`` of arrays.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    for _ in range(steps):
        g = np.asarray(df(x), dtype=float)
        H = np.asarray(hess(x), dtype=float)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            break  # diverged off the manifold; solve would be garbage
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if min(np.linalg.norm(step), np.linalg.norm(g)) < tol:
            break
        x = x - step
        history.append(x.copy())
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


def rosenbrock(x, a=1.0, b=100.0):
    """Rosenbrock function f(x, y) = (a-x)^2 + b*(y - x^2)^2.

    The classic non-convex benchmark: a narrow, curved valley running from
    the origin toward (a, a). Minimum at (a, a) with value 0; the flat valley
    floor makes it unfriendly to plain gradient descent.
    """
    x, y = float(x[0]), float(x[1])
    return (a - x) ** 2 + b * (y - x * x) ** 2


def grad_rosenbrock(x, a=1.0, b=100.0):
    """Gradient of :func:`rosenbrock`: [-2(a-x) - 4b*x*(y-x^2), 2b*(y-x^2)]."""
    x, y = float(x[0]), float(x[1])
    return np.array([-2.0 * (a - x) - 4.0 * b * x * (y - x * x),
                     2.0 * b * (y - x * x)])


def hess_rosenbrock(x, a=1.0, b=100.0):
    """Hessian of :func:`rosenbrock` (2x2).

    [[12b*x^2 - 4b*y + 2, -4b*x], [-4b*x, 2b]] — note it is only positive
    definite near the minimum; far away one eigenvalue goes negative, which
    is why raw Newton overshoots on this benchmark.
    """
    x, y = float(x[0]), float(x[1])
    return np.array([[12.0 * b * x * x - 4.0 * b * y + 2.0, -4.0 * b * x],
                     [-4.0 * b * x, 2.0 * b]])
