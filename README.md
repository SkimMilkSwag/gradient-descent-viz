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
#     "adam":             {"final_x": 0.0, "iterations": 358},
#     "rmsprop":          {"final_x": 0.0, "iterations": 94},
#     "newton":           {"final_x": 0.0, "iterations": 1}
#   },
#   "rosenbrock_2d": {
#     "gradient_descent":   {"iterations": 30000, "converged": true},
#     "gd_momentum_0.9":    {"iterations": 3020, "converged": true},
#     "adam":               {"iterations": 3930, "converged": true},
#     "rmsprop":            {"iterations": 30000, "converged": false},
#     "gd_armijo_linesearch": {"iterations": 5000, "converged": false},
#     "l_bfgs":             {"iterations": 63, "converged": true},
#     "newton":             {"iterations": 5, "converged": true}
#   }
# }
#
# Benchmark (tol=1e-6)
#
# | method               | iterations | converged |
# |---|---|---|
# | gradient_descent     | 30000      | yes       |
# | gd_momentum_0.9      | 3020       | yes       |
# | adam                 | 3930       | yes       |
# | rmsprop              | 30000      | no        |
# | gd_armijo_linesearch | 5000       | no        |
# | l_bfgs               | 63         | yes       |
# | newton_damped        | 21         | yes       |
# | newton               | 5          | yes       |
```

Generate a convergence plot:

```bash
python -m gdviz.plot
# -> convergence.png
```

## What's here

- `gdviz/optim.py` — scalar `gradient_descent` (with optional momentum/velocity), `newton`, plus polynomial bowl helpers; vector `gradient_descent_vec` / `newton_vec`; an **Armijo backtracking line search** (`armijo_alpha`) and a line-search descent runner (`gradient_descent_linesearch`); **L-BFGS** from scratch (two-loop recursion + Armijo line search, `l_bfgs`); **damped Newton** with an Armijo step size (`newton_vec_damped`, steepest-descent fallback when the Hessian is indefinite at the current point); the 2D **Rosenbrock** benchmark (`rosenbrock`, `grad_rosenbrock`, `hess_rosenbrock`); and a `benchmark` runner + `benchmark_table` for printing method-vs-iteration-count markdown tables
- `gdviz/adam.py` — **Adam** (Kingma & Ba 2015): bias-corrected first/second moment estimates with per-coordinate adaptive steps (`adam_vec`, scalar wrapper `adam`). No line search — the adaptive step size is the point; it's the workhorse of deep learning and a useful contrast to both fixed-step descent and quasi-Newton methods
- `gdviz/rmsprop.py` — **RMSProp**: per-coordinate steps normalized by the root of an EMA of squared gradients, with *no* momentum term and no bias correction (`rmsprop_vec`, scalar wrapper `rmsprop`). The deliberate contrast to Adam: where Adam divides a bias-corrected momentum estimate by a bias-corrected RMS, RMSProp divides the raw gradient by an uncorrected RMS. On the stretched 100*x²+y² bowl that means the same global lr survives the scale mismatch without diverging (where fixed-step GD blows up), but without momentum to damp it the run settles into a flat limit cycle on the steep axis instead of converging — the tests pin that gap against `adam_vec` on identical budgets
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

The Armijo line search is the other antidote to hand-tuned learning rates:
instead of a fixed step it takes the largest trial step (shrunk geometrically)
that satisfies the sufficient-decrease condition. On Rosenbrock it still needs
tens of thousands of steps — the valley floor is genuinely flat — but every step
is admissible, so it never diverges regardless of the initial trial length.

L-BFGS sits between first- and second-order: it approximates the inverse
Hessian from recent gradient differences (two-loop recursion, O(m*d) memory,
no Hessian evaluations) and pairs each quasi-Newton direction with an Armijo
line search. On Rosenbrock from `[-1.2, 1.0]` it lands on `(1, 1)` in ~60
iterations — thousands of times fewer than plain descent, using only gradient
information, at the cost of a line search per step.

Adam (`gdviz/adam.py`) is the adaptive-first-order contrast: bias-corrected
exponential moving averages of the gradient and its square rescale each
coordinate's step individually, so one global learning rate survives scale
mismatches that would make fixed-step GD diverge or crawl. No line search —
the adaptation *is* the stability mechanism. On Rosenbrock it beats momentum
descent but stays an order of magnitude behind L-BFGS: adaptive steps handle
the valley's curvature better than a constant step, but without a quasi-Newton
direction they still march along the floor.

RMSProp (`gdviz/rmsprop.py`) is Adam with the first moment removed: the same
per-coordinate RMS normalization, applied to the raw gradient instead of a
bias-corrected momentum estimate, and without bias correction on the second
moment. Two consequences show up in the demo numbers. On the scalar x² bowl it
finishes *faster* than Adam (94 vs 358 iterations to tol) — with a constant-
scale gradient the normalized step is ~lr per iteration regardless of distance,
so it grinds straight through. On the stretched anisotropic bowl, though, the
missing momentum means nothing damps the oscillation on the steep axis: the run
settles into a flat limit cycle whose objective stays put for thousands of
iterations, while Adam's momentum carries it across to the actual minimum. On
Rosenbrock the same no-momentum stall burns the full 30k-step budget without
reaching tol — the honest `no` row in the benchmark table is the whole point
of running it next to Adam.

Damped (line-search) Newton is the other end of the same tradeoff: it uses
the exact Hessian like Newton, but shrinks each Newton step with the Armijo
test until the sufficient-decrease condition holds (`newton_vec_damped`). The
result is a run that never diverges on non-convex regions where raw Newton's
full step overshoots — on Rosenbrock it converges from `[-1.2, 1.0]` in ~20
iterations, roughly 3x faster than L-BFGS while paying for one Hessian solve
per step instead of a line search.

## Tests

```bash
python -m pytest tests/ -v
```

## Plotting

Two plot modules, both optional (skipped gracefully without matplotlib):

- `gdviz/plot.py` — convergence *curves*: the x-trajectory plus an objective-vs-iteration panel (`plot_convergence`).
- `gdviz/contour.py` — the objective itself as a **2D filled contour** with each optimizer's recorded trajectory overlaid (`contour_plot`). On Rosenbrock this makes the iteration gap between crawling fixed-step descent and line-search / L-BFGS runs visible at a glance: one method leaves thousands of evenly spaced dots tracing the valley floor, the other reaches the minimum in a handful. Run it directly for a ready-made comparison plot:

```bash
python -m gdviz.contour
# -> rosenbrock_contour.png (fixed-step GD vs GD+Armijo line search vs L-BFGS on the Rosenbrock surface)
```

Non-finite surface values are masked before plotting so blowup outside the interesting region doesn't stretch the color scale, and 1-D (scalar-problem) trajectories are skipped when overlaying a 2D surface.

## License

MIT — see [LICENSE](LICENSE).
