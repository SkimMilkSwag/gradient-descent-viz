"""2D contour plot of a scalar surface with optimizer trajectories overlaid.

The complement to :mod:`gdviz.plot`: instead of convergence *curves*
(loss vs iteration), this renders the objective itself as a filled contour
and draws where each optimizer actually walked across it. Filled contours
read better on the Rosenbrock valley than plain wireframe lines — the
narrow curved floor is visible at a glance, and the trajectory markers make
the difference between crawling methods obvious (a line-search or
quasi-Newton run reaches the minimum in a few dozen points, while fixed-step
descent leaves thousands of evenly spaced dots tracing the valley floor).

``contour_plot`` degrades gracefully like ``plot_convergence``: if
matplotlib is missing it returns ``None`` instead of raising, so callers can
keep it optional. The objective itself is evaluated on a grid here (no
gradient needed for the surface — only the recorded history is required).
"""
import os


def _xy_grid(f, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), n=200):
    """Evaluate ``f`` on an (n x n) grid over the given ranges.

    Returns ``(x, y, z)`` where ``z[i, j] = f([x_j, y_i])`` — the row-major
    layout matplotlib's contour functions expect. Points where ``f`` is not
    finite (blowup outside the interesting region) are masked so they don't
    stretch the color scale.
    """
    import numpy as np

    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.empty_like(X)
    for i in range(n):
        for j in range(n):
            Z[i, j] = f([X[i, j], Y[i, j]])
    bad = ~np.isfinite(Z)
    Z[bad] = np.nan
    return X, Y, Z


def contour_plot(f, histories: dict, out_path="rosenbrock_contour.png",
                 x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), n=200):
    """Draw the objective ``f`` as filled contours with trajectories overlaid.

    ``histories`` maps label -> trajectory (the history list returned by any
    vector optimizer in :mod:`gdviz.optim` — a sequence of 2D points starting
    from x0). Each trajectory is drawn as a thin line with small markers,
    color-matched to its label. Returns the saved path, or ``None`` when
    matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None  # no matplotlib -> skip gracefully
    import numpy as np

    X, Y, Z = _xy_grid(f, x_range=x_range, y_range=y_range, n=n)
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.contour(X, Y, Z, levels=12, colors="white", linewidths=0.5, alpha=0.4)
    fig.colorbar(cs, ax=ax, label="objective value")

    for label, hist in histories.items():
        pts = np.asarray(hist, dtype=float)
        # a 1-D history (scalar problem) can't be drawn on the 2D surface
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.0, alpha=0.9, label=label)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Rosenbrock surface with optimizer trajectories")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import os
    from gdviz.optim import (
        gradient_descent_vec,
        gradient_descent_linesearch,
        l_bfgs,
        rosenbrock,
        grad_rosenbrock,
    )

    x0 = [-1.2, 1.0]
    _, fixed = gradient_descent_vec(rosenbrock, grad_rosenbrock, x0, lr=0.001, steps=300)
    _, ls = gradient_descent_linesearch(rosenbrock, grad_rosenbrock, x0, lr=1.0, steps=300)
    _, lb = l_bfgs(rosenbrock, grad_rosenbrock, x0, steps=300)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rosenbrock_contour.png")
    print(contour_plot(rosenbrock, {
        "fixed-step gd": fixed,
        "gd + armijo linesearch": ls,
        "l-bfgs": lb,
    }, out_path=os.path.abspath(out)))
