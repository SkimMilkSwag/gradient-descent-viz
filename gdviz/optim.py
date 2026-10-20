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


def armijo_alpha(f, x, direction, grad, c1=1e-4, rho=0.5, alpha=1.0, max_iter=60):
    """Backtracking (Armijo) line search for a fixed search direction.

    Shrinks ``alpha`` by the factor ``rho`` until the Armijo sufficient
    decrease condition holds:

        f(x + alpha*d) <= f(x) + c1 * grad . d * alpha

    where ``grad . d`` is the directional derivative at ``x``. Returns the
    accepted step size (never larger than the initial ``alpha``). A line
    search only makes sense when moving downhill, so ``grad . d < 0`` is a
    precondition — otherwise every step is uphill and the caller should pick
    a different direction.
    """
    cur = f(x)
    slope = float(np.dot(np.asarray(grad, dtype=float), np.asarray(direction, dtype=float)))
    if slope >= 0:
        raise ValueError("search direction is not descending (grad . d = %r)" % slope)
    a = float(alpha)
    for _ in range(max_iter):
        cand = f(x + a * direction)
        # A non-finite candidate means the trial step overshot into blowup:
        # shrink instead of accepting garbage.
        if np.isfinite(cand) and cand <= cur + c1 * slope * a:
            return a
        a *= rho
    return a


def gradient_descent_linesearch(f, df, x0, lr=0.5, steps=50, tol=1e-8,
                                armijo_c1=1e-4, armijo_rho=0.5):
    """Gradient descent with an Armijo backtracking line search per step.

    The direction is always ``-grad(x)``; the initial trial step length is
    ``lr`` and is shrunk by ``armijo_rho`` until the Armijo condition is met
    (see :func:`armijo_alpha`). The stopping rules match the fixed-step
    optimizers in this module: stop when the gradient norm drops below
    ``tol``, on non-finite values, or after ``steps`` iterations.

    Returns ``(x_final, history)`` — the starting point followed by one
    entry per accepted step — with the same contract as
    :func:`gradient_descent_vec`.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    for _ in range(steps):
        g = np.asarray(df(x), dtype=float)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < tol:
            break  # diverged to inf/nan, or converged
        a = armijo_alpha(f, x, -g, g, c1=armijo_c1, rho=armijo_rho, alpha=lr)
        x = x + a * (-g)
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


def _l_bfgs_two_loop(pairs: list, g: np.ndarray) -> np.ndarray:
    """L-BFGS two-loop recursion (Nocedal & Wright, eq. 8.34-8.35).

    Given the most recent ``pairs`` of (s_i, y_i) — steps and gradient
    differences — and the current gradient ``g``, returns the L-BFGS search
    direction ``d = -B_k^{-1} g`` where B_k is the implicit limited-memory
    Hessian approximation. With an empty history it degenerates to steepest
    descent (``-g``), which is exactly what a fresh run should do.

    Memory cost is O(m*d): only the ``m`` most recent pairs are kept, so the
    method approximates full BFGS without ever forming the d x d matrix.
    """
    n = len(pairs)
    if n == 0:
        return -g
    ys = [p[1] for p in pairs]
    ss = [p[0] for p in pairs]
    rhos = [p[2] for p in pairs]

    # backward loop: r_i = y_{i-1} + (rho_{i-1} * s_{i-1} . r_i) * ...
    alphas = [0.0] * n
    r = g.copy()
    for i in range(n - 1, -1, -1):
        alphas[i] = rhos[i] * float(np.dot(ss[i], r))
        r = r - alphas[i] * ys[i]

    # initial Hessian inverse scale gamma: (s_{last} . y_{last}) / (y_{last} . y_{last})
    s_last, y_last = ss[-1], ys[-1]
    yy = float(np.dot(y_last, y_last))
    gamma = float(np.dot(s_last, y_last)) / yy if yy > 0 else 1.0
    q = gamma * r

    # forward loop: s_i + (alpha_i - rho_i * y_i . q) ...
    for i in range(n):
        beta = rhos[i] * float(np.dot(ys[i], q))
        q = q + (alphas[i] - beta) * ss[i]

    return -q


def l_bfgs(f, df, x0, steps=100, tol=1e-6, m=10):
    """L-BFGS with Armijo backtracking line search.

    Signature matches :func:`gradient_descent_vec`: ``f`` is the objective,
    ``df`` its gradient, ``x0`` the start point (a vector). Each iteration:

    1. build the search direction with :func:`_l_bfgs_two_loop`,
    2. accept a step along it via :func:`armijo_alpha` (with a steepest-
       descent fallback if the quasi-Newton direction is not descending —
       the inverse-Hessian approximation is only locally valid),
    3. record the (s, y) pair in a circular buffer capped at ``m`` entries;
       pairs with y . s <= 0 are skipped (no curvature information).

    Stopping rules match the other optimizers: gradient norm below ``tol``,
    non-finite values, or the step budget. Returns ``(x_final, history)`` —
    the start point followed by one entry per accepted step.
    """
    x = np.asarray(x0, dtype=float).copy()
    g = np.asarray(df(x), dtype=float)
    history = [x.copy()]

    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []  # (s, y, rho)

    for _ in range(steps):
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < tol:
            break

        d = _l_bfgs_two_loop(pairs, g)
        slope = float(np.dot(g, d))
        # the two-loop direction should be descending; if the implicit
        # approximation went stale, fall back to steepest descent
        if slope >= 0 or not np.all(np.isfinite(d)):
            d = -g

        a = armijo_alpha(f, x, d, g, alpha=0.5)
        x_new = x + a * d
        g_new = np.asarray(df(x_new), dtype=float)
        if not np.all(np.isfinite(g_new)):
            break  # the step overshot into blowup; stop rather than record garbage

        s = x_new - x
        y = g_new - g
        sy = float(np.dot(y, s))
        if sy > 1e-12:
            pairs.append((s, y, 1.0 / sy))
            if len(pairs) > m:
                pairs.pop(0)

        x, g = x_new, g_new
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


def loss_history(f, trajectory):
    """Evaluate objective ``f`` at every recorded point in a trajectory.

    ``trajectory`` is the history list returned by any optimizer in this
    module (a sequence of scalars or arrays, starting from x0). Returns one
    objective value per point, so the i-th entry is the loss at the i-th
    iteration — the curve you want on a convergence plot's second axis.
    """
    return [float(f(pt)) for pt in trajectory]


def converged_at_tol(x_final, x_opt):
    """True when the final state sits within 1e-6 of the optimum coordinate-wise."""
    return bool(np.allclose(np.asarray(x_final, dtype=float),
                            np.asarray(x_opt, dtype=float), atol=1e-6))


def benchmark(df, optim, x0, x_opt, tol=1e-6, steps=100000, **kw):
    """Run ``optim`` from ``x0`` and report the convergence benchmark.

    ``df`` is the gradient function passed as the optimizer's second
    positional argument. Returns ``(iters, converged)``: ``iters`` is the
    number of accepted steps (the optimizer's recorded history minus the
    start point), and ``converged`` is True only when the run ended within
    ``tol`` of ``x_opt`` — i.e. it actually reached the optimum rather than
    exhausting its step budget or diverging. Extra kwargs pass straight
    through to the optimizer (e.g. ``lr=0.001``, ``momentum=0.9``).

    For optimizers with extra positional args (like :func:`newton_vec` which
    also needs a Hessian), call them directly and format the result with
    :func:`benchmark_table` instead.
    """
    x_final, hist = optim(None, df, x0=x0, tol=tol, steps=steps, **kw)
    return len(hist) - 1, converged_at_tol(x_final, x_opt)


def benchmark_table(results: dict, header="Benchmark (tol=1e-6)") -> str:
    """Format benchmark rows as a markdown table.

    ``results`` maps label -> (iters, converged). Order of the dict is
    preserved. Non-converged runs show their iteration count with a ``no`` in
    the last column so the table stays machine-parseable.
    """
    lines = [header, "", "| method | iterations | converged |", "|---|---|---|"]
    for label, (iters, ok) in results.items():
        lines.append(f"| {label} | {iters} | {'yes' if ok else 'no'} |")
    return "\n".join(lines)
