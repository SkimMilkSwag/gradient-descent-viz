"""Demo: minimize x^2 from x0=5 with GD, momentum-GD and Newton; then run all
three optimizers on the 2D Rosenbrock benchmark and report convergence."""
import json

from .optim import (
    gradient_descent,
    newton,
    grad_quadratic,
    gradient_descent_vec,
    newton_vec,
    rosenbrock,
    grad_rosenbrock,
    hess_rosenbrock,
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
    gd_v, gd_hist_v = gradient_descent_vec(rosenbrock, grad_rosenbrock, x0, lr=0.001, steps=30000)
    mom_v, mom_hist_v = gradient_descent_vec(
        rosenbrock, grad_rosenbrock, x0, lr=0.001, momentum=0.9, steps=30000
    )
    nw_v, nw_hist_v = newton_vec(rosenbrock, grad_rosenbrock, hess_rosenbrock, [0.5, -0.5], steps=50)
    report["rosenbrock_2d"] = {
        "start": x0,
        "gradient_descent": {"final_xy": [round(c, 4) for c in gd_v], "iterations": len(gd_hist_v) - 1},
        "gd_momentum_0.9": {"final_xy": [round(c, 4) for c in mom_v], "iterations": len(mom_hist_v) - 1},
        "newton": {"final_xy": [round(c, 4) for c in nw_v], "iterations": len(nw_hist_v) - 1},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
