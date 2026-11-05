# Minipick

Sampling with certified minimax risk bounds.

Code is available under the [MIT license](LICENSE).

Minipick chooses a sampling design and unbiased linear estimator for a specified target and uncertainty model. It returns a sampler that meets an exact sample size or disjoint stratum quotas, together with independently checkable bounds on its risk relative to the best admissible strategy.

The [research note](paper/note.pdf) gives the method, proofs, analytic comparisons, and limitations. This is a reference implementation for modest conditioning sets and feature dimensions; practical superiority over existing sampling methods has not been established.

## Install and try it

Requires Python 3.11 or newer. From a checkout:

```sh
uv sync --group dev
```

This installs Minipick in editable mode using the versions in `uv.lock`. Run Python with `uv run python`.

```python
from random import Random

from minipick import solve

solution = solve(
    a=[0.25, 1.0, 1.0],
    features=[[2.0], [0.0], [0.0]],
    noise=[1.0, 1.0, 1.0],
    sample_size=2,
    head=[0],
)

assert solution.verify_certificate()
assert solution.certificate.guarantees(0.001)
print(float(solution.certificate.lower))
print(float(solution.certificate.risk_upper))

draw = solution.draw(Random(42))
print(draw.indices)
print(draw.weights)
```

The two risk bounds are approximately 0.961784. The guarantee check certifies that this sampler's risk is within approximately 0.1% of the minimax optimum under the supplied model. Here all structured uncertainty lies on the conditioned unit, so the relaxation is exact. The example also has a classical closed-form solution; it is a correctness check.

## Specify the target and uncertainty

The target is $a^T y$. Before collection, fix a prediction $m_0$ and a residual model:

$$
y-m_0=F\beta+\varepsilon,\qquad
\|\beta\|_2\le\rho,\quad E\varepsilon=0,\quad
\operatorname{Cov}(\varepsilon)=\operatorname{diag}(d_i).
$$

Risk averages over sampling and noise and takes the worst case over $\beta$. Sampling must be independent of the noise. The comparison class contains supported, sample-dependent linear weights satisfying $Ew=a$; weights can be negative, and zero-target units can serve as control variates.

| Argument | Meaning |
|---|---|
| `a` | Target coefficients, one per population unit. Use ones for a total or `1/N` for a mean. |
| `features` | Matrix $F$ with one row per unit; use `None` for no structured term. |
| `noise` | Strictly positive variances $d_i$, one per unit. |
| `sample_size` | Integer number of units to select. Can be omitted when supplying quotas. |
| `rho` | Nonnegative uncertainty radius; default `1.0`. Setting it to zero removes the structured term. |
| `head` | Distinct, zero-based unit indices whose membership defines the conditioning cells; default empty. |
| `strata`, `quotas` | Nonnegative integer labels for the units and a mapping from each label to its sample size. |

Conditioning on a unit lets the design depend on whether it is selected; it does not force inclusion. With `head=()`, the relaxation reduces to noise allocation, and the sampler's actual risk can be arbitrarily worse than optimal as structured uncertainty grows. Within each cell, rounding ignores the features and depends on row order. The optimizer minimizes the relaxed objective; it does not minimize the final risk upper bound, which also includes a rounding allowance. Use `certificate.guarantees(eta)` to check the returned strategy's guarantee.

Conditioning on all units gives the exact finite benchmark but can enumerate the complete sample space. If all nonzero target coordinates fit within their quotas, Minipick includes them and attains zero risk without optimization.

For example, select one unit from stratum 2 and two from stratum 7:

```python
stratified = solve(
    a=[1, 1, 1, 1, 1],
    features=[[1], [-1], [0.1], [-0.1], [0]],
    noise=[1, 1, 1, 1, 1],
    strata=[2, 2, 7, 7, 7],
    quotas={2: 1, 7: 2},
    head=[0],
)
```

If supplied together, `sample_size` must equal the sum of the quotas.

## Draw and estimate

`solution.draw(rng)` returns a `SamplingDraw` with selected `indices` and exact rational `weights`. Pass a `random.Random` instance for reproducibility. The sampler uses rational probabilities and pairwise rounding to meet every quota exactly, under the usual ideal-randomness interpretation of the generator.

After collection, supply residuals $y_i-m_{0i}$ in `draw.indices` order and the known baseline $a^T m_0$:

```python
observed_residuals = [0.5 for _ in draw.indices]
estimate = draw.estimate(observed_residuals, baseline=0)
exact_estimate = draw.estimate_exact(observed_residuals, baseline=0)
```

The values above are illustrative observations. `estimate_exact` returns a `Fraction`; `estimate` converts it to a float. The risk guarantee applies to the exact estimator, with ordinary rounding error from any final float conversion. Returned weights belong to this estimator and need not be inverse unconditional inclusion probabilities.

## Read the certificate

`solution.certificate` stores exact `Fraction` values:

| Field | Meaning |
|---|---|
| `lower` | Lower bound on the conditioned relaxation optimum and the full minimax risk. |
| `relaxation_upper` | Upper bound on the repaired strategy's relaxed objective. |
| `rounding_allowance` | Bound on the additional structured variance from sampling within cells. |
| `risk_upper` | `relaxation_upper + rounding_allowance`, an upper bound on the sampler's risk. |
| `noise_risk` | Exact noise contribution to the sampler's risk. |
| `relaxation_gap` | `relaxation_upper - lower`. |
| `ratio_upper` | `risk_upper / lower`; `None` if the lower bound is zero and the upper bound is positive, or one if both are zero. |

**Use `certificate.guarantees(eta)` to check near optimality.** It checks `risk_upper <= (1 + eta) * lower` using rational arithmetic. `solution.converged` instead reports whether the relaxation gap meets the requested optimization accuracy. A converged relaxation can still give a poor sampling guarantee.

CVXPY/Clarabel proposes a solution. Minipick repairs it in rational arithmetic and verifies its risk bounds before returning it. `solution.verify_certificate()` rechecks feasibility and the stored witnesses without calling an optimizer, including whether the conditioned dual covers every feasible head pattern. It shares arithmetic routines with certificate construction; the stored witnesses also permit third-party checking. It does not recheck the historical `converged` flag. Convert bounds to floats for display; use `guarantees` for comparisons.

Model inputs are converted to binary64 and then interpreted exactly, with $\rho F$ formed rationally. Certificates cover those represented inputs. Accuracy tolerances and the `eta` argument to `guarantees` also use their binary64 values. Certificates do not account for uncertainty in choosing the model itself.

## Accuracy, limits, and failures

The default relaxation tolerance is `accuracy_atol + accuracy_rtol * lower`, with `accuracy_atol=1e-9` and `accuracy_rtol=1e-6`, in the original risk units. Use `accuracy_atol=0` for purely relative accuracy. Numerical solver controls are separate: `solver_tolerance=1e-9` and `max_iter=300`.

Conditioning can require exponentially many cells. The defaults `max_cells=256` and `max_cone_entries=1_000_000` cap the representation before optimization; they do not bound runtime. Rational verification can become expensive with large feature dimensions or coefficient bit lengths.

| Failure | Behavior |
|---|---|
| `CertificationError` | Requested relaxation accuracy was not reached. The exception's `.result` retains a feasible sampler and valid bounds. `require_accuracy=False` returns that result with `converged=False`. |
| `SolverFailure` | No usable numerical candidate, including solver errors, iteration limits, nonfinite primal values, excessive constraint violations, or unrepresentable scaling. |
| `ValueError` | Invalid inputs or a representation exceeding the resource caps. |
| `ArithmeticError` | An exact feasibility or certificate check failed. |

`CertificationError` and `SolverFailure` are exported from `minipick`. If numerical dual recovery fails, the solver retains the feasible strategy and an independent noise-only lower bound; the resulting gap may fail the accuracy requirement.

The package does not provide a general variance estimator, nonresponse correction, learned uncertainty model, or survey-analysis integration.

## Develop and build the paper

```sh
uv run make check
uv build
make note
```

`make check` runs lint, formatting checks, and tests of analytic optima, exact sampling moments, numerical failures, and corrupted certificates. `uv build` creates a wheel and source distribution. The paper build requires latexmk, pdfLaTeX, and BibTeX and writes `paper/note.pdf` beside its source.

| Path | Purpose |
|---|---|
| `src/minipick/` | Solver, sampler, and certificate verification |
| `tests/` | Solver and certificate regression tests |
| `paper/` | LaTeX source, bibliography, and compiled note |

The paper's comparison with *Conditional Optimal Sampling* uses the supplementary proofs available from the [publisher](https://doi.org/10.1093/jssam/smag027).
