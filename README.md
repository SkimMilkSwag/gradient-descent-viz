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
#   "gradient_descent": {"final_x": 0.0, "iterations": 40},
#   "newton":           {"final_x": 0.0, "iterations": 3}
# }
```

Generate a convergence plot:

```bash
python -m gdviz.plot
# -> convergence.png
```

## What's here

- `gdviz/optim.py` — `gradient_descent`, `newton`, plus polynomial bowl helpers
- `gdviz/plot.py` — optional convergence plotting (gracefully skipped without matplotlib)

## Notes

Newton's method uses the second derivative and typically converges in 2–4 steps on
these problems, versus dozens for plain gradient descent — a good concrete demo of
why second-order information is expensive but powerful.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
