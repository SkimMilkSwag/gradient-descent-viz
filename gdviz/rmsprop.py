"""RMSProp (root mean square propagation) optimizer.

Implements the classic RMSProp update: a per-coordinate step normalized by
the root of an exponential moving average of *squared* gradients, with no
first-moment momentum term and no bias correction:

    v = decay * v + (1 - decay) * g * g
    x -= lr * g / (sqrt(v) + eps)

The contrast this exists for is Adam: where Adam divides a bias-corrected
momentum estimate (m_hat) by a bias-corrected RMS estimate, RMSProp divides
the raw gradient itself by an uncorrected RMS. On the stretched bowl that
means the same single global lr handles both axis scales as well as Adam
does (the normalization is what rescales the step, not the momentum), but
without the first moment it takes more iterations to grind along a flat
valley floor — the tests pin that gap against adam_vec on the same budget.

  - rmsprop_vec(f, df, x0, lr=0.1, ...): vector RMSProp; returns (x_final,
    history) with the same history contract as the other vector optimizers —
    a list of states starting from x0 and ending at the final state, one
    entry per accepted step. Stops early when ||grad|| < tol or after `steps`.

The 100*x^2 + y^2 bowl is the reference test: fixed-step GD diverges on it
at lr=0.1 (factor 2*lr*100 = 20 > 1) and crawls at stable lrs, while RMSProp's
per-coordinate normalization converges from both axes' scales in one budget.
"""
import numpy as np


def rmsprop_vec(f, df, x0, lr=0.1, steps=100, tol=1e-8, decay=0.9, eps=1e-8):
    """Run RMSProp on a differentiable vector objective.

    ``f`` is evaluated only for the caller's convenience (RMSProp itself
    needs no function values); pass the real objective so the returned final
    state can be checked against it. Returns ``(x_final, history)`` —
    ``history`` records x0 plus one point per accepted step.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    v = np.zeros_like(x)  # running mean of squared gradients (RMS estimate)
    history = [x.copy()]
    for _ in range(steps):
        g = np.asarray(df(x), dtype=float).ravel()  # flatten: callers may return nested shapes
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < tol:
            break
        v = decay * v + (1.0 - decay) * g * g
        # raw gradient divided by the (uncorrected) RMS: no momentum, no bias
        # correction — the normalization alone rescales the step per coordinate
        x = x - lr * g / (np.sqrt(v) + eps)
        history.append(x.copy())
    return x, history


def rmsprop(f, df, x0, lr=0.1, steps=50, tol=1e-8, decay=0.9, eps=1e-8):
    """Scalar convenience wrapper around :func:`rmsprop_vec` (x is a list of 1).

    Same contract as the other scalar optimizers in this package: returns
    ``(x_final, history)`` with history a list of floats.
    """
    x_final, hist = rmsprop_vec(f, df, [x0], lr=lr, steps=steps, tol=tol,
                                decay=decay, eps=eps)
    return float(x_final[0]), [float(p[0]) for p in hist]
