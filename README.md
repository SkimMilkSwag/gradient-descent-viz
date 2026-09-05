# gradient-descent-viz

A from-scratch implementation of **gradient descent** and **Newton's method**
on simple convex problems, with iteration-by-iteration convergence tracking and
optional matplotlib plots. Written to make the *mechanics* of optimization
concrete: watch how many iterations each method needs, see the trajectories, and
understand why second-order methods converge faster on smooth bowls.

## Install

```bash
pip install -e .            # core (numpy only)
pip install -e .[plot]      # + matplotlib for convergence plots
```

## Run

```bash
python -m gdviz
# {
#   "quadratic_bowl": {
#     "gradient_descent": {"final_x": 0.0, "iterations": 40},
#     "gd_momentum_0.9":  {"final_x": 0.0, "iterations": 2},
#     "newton":           {"final_x": 0.0, "iterations": 1}
#   },
#   "rosenbrock_2d": {
#     "start": [-1.2, 1.0],
#     "gradient_descent": {"final_xy": [0.9999, 0.9997], "iterations": 30000},
#     "gd_momentum_0.9":  {"final_xy": [1.0, 1.0], "iterations": 4128},
#     "newton":           {"final_xy": [1.0, 1.0], "iterations": 5}
#   }
# }
```

Generate a convergence plot:

```bash
python -m gdviz.plot
# -> convergence.png
```

## What's here

- `gdviz/optim.py` — scalar `gradient_descent` (with optional momentum/velocity), `newton`, plus polynomial bowl helpers; vector `gradient_descent_vec` / `newton_vec`; and the 2D **Rosenbrock** benchmark (`rosenbrock`, `grad_rosenbrock`, `hess_rosenbrock`)
- `gdviz/plot.py` — optional convergence plotting (gracefully skipped without matplotlib)

## Notes

Momentum (velocity SGD) damps the oscillation plain descent shows on steep bowls and
cuts iteration count — but second-order info still wins. Newton's method uses the
second derivative and typically converges in 2–4 steps on these problems, versus
dozens for plain gradient descent — a good concrete demo of why second-order
information is expensive but powerful.

The Rosenbrock function is the non-convex stress test: its narrow curved valley
makes plain gradient descent crawl (30k+ iterations to the default tolerance from
`[-1.2, 1.0]`), momentum cuts that to ~4k, and Newton with the analytic Hessian
lands on `(1, 1)` in five. The vector runners stop early when a diverging
trajectory produces a non-finite gradient, so a bad learning rate can't burn
through all `steps`.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
