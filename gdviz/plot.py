"""Plot optimizer convergence without requiring matplotlib at import time."""
import os


def plot_convergence(histories: dict, out_path="convergence.png"):
    """histories: {label: [x0, x1, ...]}. Saves a PNG if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None  # no matplotlib -> skip gracefully

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        ax.plot(range(len(hist)), hist, marker="o", ms=3, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("x")
    ax.set_title("Optimizer convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from gdviz.optim import gradient_descent, newton, grad_quadratic

    _, gd = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=30)
    _, nw = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    print(plot_convergence({"gradient descent": gd, "newton": nw}))
