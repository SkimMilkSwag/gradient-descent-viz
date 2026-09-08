"""Plot optimizer convergence without requiring matplotlib at import time.

Renders two panels in one figure: the x-trajectory (top) and the objective /
loss vs iteration (bottom, log scale). Loss curves are optional — when ``losses``
is omitted only the trajectory panel is drawn, which keeps the no-matplotlib
import path cheap for callers that only want the trajectory.
"""
import os


def plot_convergence(histories: dict, out_path="convergence.png", losses=None):
    """histories: {label: [x0, x1, ...]}. Saves a PNG if matplotlib is available.

    ``losses`` (optional) maps label -> list of objective values evaluated at
    each recorded trajectory point (see :func:`gdviz.optim.loss_history`). When
    supplied it draws a second, log-scaled axis with one loss curve per method so
    the objective's descent is visible alongside the x-trajectory. Returns the
    saved path, or ``None`` when matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None  # no matplotlib -> skip gracefully

    has_loss = bool(losses)
    if has_loss:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 7), gridspec_kw={"height_ratios": [1, 1]}
        )
    else:
        fig, ax_top = plt.subplots(figsize=(8, 5))

    # --- trajectory panel -----------------------------------------------------
    for label, hist in histories.items():
        ax_top.plot(range(len(hist)), hist, marker="o", ms=3, label=label)
    ax_top.set_ylabel("x")
    ax_top.set_title(
        "Optimizer convergence (trajectory + objective)" if has_loss
        else "Optimizer convergence"
    )
    ax_top.legend()

    # --- objective / loss panel (log scale, optional) -------------------------
    if has_loss:
        for label, curve in losses.items():
            ax_bot.plot(range(len(curve)), curve, marker=".", ms=3, label=label)
        ax_bot.set_yscale("log")
        ax_bot.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from gdviz.optim import gradient_descent, newton, grad_quadratic, loss_history

    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=30)
    _, nw = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    histories = {"gradient descent": gd, "newton": nw}
    losses = {label: loss_history(lambda x: x ** 2, hist) for label, hist in histories.items()}
    print(plot_convergence(histories, losses=losses))
