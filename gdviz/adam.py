"""Adam (adaptive moment estimates) optimizer.

Implements the standard Adam update from Kingma & Ba (2015): per-coordinate
learning rates adapted from exponential moving averages of the gradient
(first moment m) and its square (second moment v), both bias-corrected so
they start near zero instead of being skewed by their empty initialization.

  - adam_vec(f, df, x0, lr=0.1, ...): vector Adam; returns (x_final, history)
    with the same history contract as the other vector optimizers — a list of
    states starting from x0 and ending at the final state, one entry per
    accepted step. Stops early when ||grad|| < tol or after `steps` steps.

The 1/x^2 bowl is the reference test: Adam's per-coordinate adaptation means
a single global lr works across scales where fixed-step GD either diverges
(too large) or crawls (too small), so the tests pin both of those regimes.
"""
import numpy as np


def adam_vec(f, df, x0, lr=0.1, steps=100, tol=1e-8, beta1=0.9, beta2=0.999,
             eps=1e-8):
    """Run Adam on a differentiable vector objective.

    ``f`` is evaluated only for the caller's convenience (Adam itself needs
    no function values); pass the real objective so the returned final state
    can be checked against it. Returns ``(x_final, history)`` — ``history``
    records x0 plus one point per accepted step.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    m = np.zeros_like(x)  # first moment estimate (running mean of gradients)
    v = np.zeros_like(x)  # second moment estimate (running mean of squares)
    history = [x.copy()]
    for t in range(1, steps + 1):
        g = np.asarray(df(x), dtype=float).ravel()  # flatten: callers may return nested shapes
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < tol:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        # bias correction: an EMA started at zero underestimates the moment by
        # a factor (1 - beta^t); dividing it out keeps the early steps honest.
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x.copy())
    return x, history


def adam(f, df, x0, lr=0.1, steps=50, tol=1e-8, beta1=0.9, beta2=0.999,
         eps=1e-8):
    """Scalar convenience wrapper around :func:`adam_vec` (x is a list of 1).

    Same contract as the other scalar optimizers in this module: returns
    ``(x_final, history)`` with history a list of floats.
    """
    x_final, hist = adam_vec(f, df, [x0], lr=lr, steps=steps, tol=tol,
                             beta1=beta1, beta2=beta2, eps=eps)
    return float(x_final[0]), [float(p[0]) for p in hist]
