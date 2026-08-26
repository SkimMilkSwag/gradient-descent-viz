"""Demo: minimize x^2 from x0=5 with both optimizers and report convergence."""
import json

from .optim import gradient_descent, newton, grad_quadratic


def main():
    gd_x, gd_hist = gradient_descent(lambda x: 0, grad_quadratic, x0=5.0, lr=0.2, steps=40)
    nw_x, nw_hist = newton(lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0, x0=5.0)
    report = {
        "gradient_descent": {"final_x": round(gd_x, 6), "iterations": len(gd_hist) - 1},
        "newton": {"final_x": round(nw_x, 6), "iterations": len(nw_hist) - 1},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
