"""Demo: minimize x^2 from x0=5 with GD, momentum-GD and Newton; then run all
three optimizers on the 2D Rosenbrock benchmark and print a markdown
convergence table (method, iterations to tol, converged?)."""
import json

from .optim import (
    gradient_descent,
    gradient_descent_linesearch,
    newton,
    grad_quadratic,
    benchmark,
    benchmark_table,
    converged_at_tol,
    gradient_descent_vec,
    newton_vec,
    rosenbrock,
    grad_rosenbrock,
    hess_rosenbrock,
    l_bfgs,
    loss_history,
)


def main():
    gd_x, gd_hist = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=40)
    mom_x, mom_hist = gradient_descent(
        lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, momentum=0.9, steps=40
    )
    nw_x, nw_hist = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    report = {
        "quadratic_bowl": {
            "gradient_descent": {"final_x": round(gd_x, 6), "iterations": len(gd_hist) - 1},
            "gd_momentum_0.9": {"final_x": round(mom_x, 6), "iterations": len(mom_hist) - 1},
            "newton": {"final_x": round(nw_x, 6), "iterations": len(nw_hist) - 1},
        },
    }

    # Rosenbrock 2D benchmark: minimum at (1, 1). Plain descent crawls the
    # curved valley floor and needs tens of thousands of steps; momentum cuts
    # it to thousands. Newton (Hessian solve) lands in five.
    x0 = [-1.2, 1.0]
    # benchmark() passes the objective through as None (the vector optimizers
    # only need the gradient), but the line-search variant evaluates f for its
    # Armijo test — so it gets rosenbrock explicitly.
    bench_results = {
        "gradient_descent": benchmark(grad_rosenbrock, gradient_descent_vec, x0, [1.0, 1.0], lr=0.001, steps=30000),
        "gd_momentum_0.9": benchmark(
            grad_rosenbrock, gradient_descent_vec, x0, [1.0, 1.0], lr=0.001, momentum=0.9, steps=30000
        ),
    }
    ls_x, ls_hist = gradient_descent_linesearch(rosenbrock, grad_rosenbrock, x0, lr=1.0, steps=5000)
    bench_results["gd_armijo_linesearch"] = (
        len(ls_hist) - 1,
        converged_at_tol(ls_x, [1.0, 1.0]),
    )
    # L-BFGS takes (f, df) like the other vector optimizers — but benchmark()
    # passes the gradient as its first arg, so call it directly here, the same
    # way Newton is handled below. Two-loop recursion + Armijo line search;
    # should land near Newton's iteration count using only gradients.
    lb_x, lb_hist = l_bfgs(rosenbrock, grad_rosenbrock, x0, steps=300)
    bench_results["l_bfgs"] = (len(lb_hist) - 1, converged_at_tol(lb_x, [1.0, 1.0]))
    # Newton needs a Hessian as a separate arg, so call it directly
    nw_v, nw_hist_v = newton_vec(rosenbrock, grad_rosenbrock, hess_rosenbrock, [0.5, -0.5], steps=50)
    bench_results["newton"] = (len(nw_hist_v) - 1, converged_at_tol(nw_v, [1.0, 1.0]))
    report["rosenbrock_2d"] = {
        label: {"iterations": iters, "converged": ok}
        for label, (iters, ok) in bench_results.items()
    }
    print(json.dumps(report, indent=2))
    print()
    print(benchmark_table(bench_results))

    # Convergence plot: trajectory on top, objective (log scale) below. The
    # loss panel is the real signal — x-trajectories look similar for methods
    # that actually differ in how fast the objective drops.
    from .plot import plot_convergence
    trajectories = {
        "gradient_descent": gd_hist,
        "gd_momentum_0.9": mom_hist,
        "newton": nw_hist,
    }
    losses = {label: loss_history(lambda x: x ** 2, hist) for label, hist in trajectories.items()}
    print(plot_convergence(trajectories, out_path="convergence.png", losses=losses))


if __name__ == "__main__":
    main()
