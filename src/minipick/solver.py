"""Conditioned minimax sampling with exact post-solve risk certificates.

The optimizer is floating point. The returned distribution, weights and
certificate are rational and refer to the input floats interpreted exactly.
See README.md for the model, numerical contract and computational limits.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import comb
from numbers import Integral
from random import Random

import numpy as np

from ._certificates import (
    ONE,
    ZERO,
    DualWitness,
    dot,
    dual_witness,
    evaluate_dual,
    evaluate_noise_dual,
    noise_witness,
    rational,
    repair_primal,
    require_fractions,
    upper_values,
)


class SolverFailure(RuntimeError):
    """The numerical optimizer did not produce a usable candidate."""


class CertificationError(RuntimeError):
    """The recovered strategy is valid but the requested gap was not reached."""

    def __init__(self, result):
        self.result = result
        super().__init__(
            "Certified relaxation gap exceeds the requested accuracy. "
            "The feasible strategy and valid bounds are available as .result."
        )


@dataclass(frozen=True)
class SamplingModel:
    a: tuple
    noise: tuple
    r: tuple
    groups: tuple
    quotas: tuple
    head: tuple


@dataclass(frozen=True)
class RiskCertificate:
    lower: Fraction
    relaxation_upper: Fraction
    rounding_allowance: Fraction
    noise_risk: Fraction
    dual: DualWitness | None = None
    noise_dual: tuple = ()

    @property
    def risk_upper(self):
        return self.relaxation_upper + self.rounding_allowance

    @property
    def relaxation_gap(self):
        return self.relaxation_upper - self.lower

    @property
    def ratio_upper(self):
        if self.lower > 0:
            return self.risk_upper / self.lower
        return ONE if self.risk_upper == 0 else None

    def guarantees(self, relative_error):
        """Check a requested relative-risk guarantee in exact arithmetic."""
        error = _nonnegative(relative_error, "relative_error")
        return self.risk_upper <= (ONE + rational(error)) * self.lower


@dataclass(frozen=True)
class SamplingDraw:
    indices: tuple
    weights: tuple

    def estimate_exact(self, residuals, baseline=0):
        """Estimate from residuals in indices order, plus known a.T @ m0."""

        def exact(value):
            if isinstance(value, (Integral, np.bool_)):
                return Fraction(int(value))
            if isinstance(value, np.floating):
                return Fraction(*value.as_integer_ratio())
            return Fraction(value)

        values = tuple(exact(value) for value in residuals)
        if len(values) != len(self.indices):
            raise ValueError("Supply one residual per selected index, in indices order")
        return exact(baseline) + dot(self.weights, values)

    def estimate(self, residuals, baseline=0):
        """Return the rounded float representation of estimate_exact."""
        return float(self.estimate_exact(residuals, baseline))


@dataclass(frozen=True)
class SamplingSolution:
    model: SamplingModel
    cells: tuple
    masses: tuple
    inclusions: tuple
    contributions: tuple
    certificate: RiskCertificate
    solver_status: str
    raw_constraint_violation: float
    converged: bool

    def validate(self):
        """Check exact quotas, support, cell identity and unbiasedness."""
        n = len(self.model.a)
        _validate_model(self.model)
        require_fractions(self.masses)
        require_fractions(
            (
                self.certificate.lower,
                self.certificate.relaxation_upper,
                self.certificate.rounding_allowance,
                self.certificate.noise_risk,
            )
        )
        if not (
            len(self.cells)
            == len(self.masses)
            == len(self.inclusions)
            == len(self.contributions)
            > 0
        ):
            raise ArithmeticError("Mismatched cell arrays")
        if sum(self.masses) != 1 or any(q <= 0 for q in self.masses):
            raise ArithmeticError("Invalid cell masses")
        patterns = set()
        for fixed, p, x in zip(
            self.cells, self.inclusions, self.contributions, strict=True
        ):
            if len(fixed) != n or len(p) != n or len(x) != n:
                raise ArithmeticError("Invalid cell dimensions")
            require_fractions(p)
            require_fractions(x)
            if any(value not in (-1, 0, 1) for value in fixed):
                raise ArithmeticError("Invalid fixed indicator")
            pattern = tuple(fixed[i] for i in self.model.head)
            if pattern in patterns or any(value not in (0, 1) for value in pattern):
                raise ArithmeticError("Cells are not identified by head membership")
            patterns.add(pattern)
            for i in range(n):
                if not 0 <= p[i] <= 1 or (p[i] == 0 and x[i] != 0):
                    raise ArithmeticError("Invalid conditional inclusion or weight")
                if fixed[i] != -1 and p[i] != fixed[i]:
                    raise ArithmeticError("Changed a fixed indicator")
            for indices, quota in zip(
                self.model.groups, self.model.quotas, strict=True
            ):
                if sum(p[i] for i in indices) != quota:
                    raise ArithmeticError("Conditional quota is not exact")
        for i, a in enumerate(self.model.a):
            if sum(x[i] for x in self.contributions) != a:
                raise ArithmeticError("Recovered weights are not unbiased")
        if not 0 <= self.certificate.lower <= self.certificate.relaxation_upper:
            raise ArithmeticError("Inconsistent certificate endpoints")
        return True

    def draw(self, rng=None):
        """Draw with rational Bernoulli probabilities and exact-size rounding.

        Pass a random.Random instance for reproducibility. Sampling avoids
        float probability truncation; statistical guarantees use the usual
        ideal-randomness interpretation of the supplied generator.
        """
        rng = Random() if rng is None else rng
        remaining = ONE
        cell = len(self.masses) - 1
        for j, mass in enumerate(self.masses[:-1]):
            if _bernoulli(mass / remaining, rng):
                cell = j
                break
            remaining -= mass
        original = self.inclusions[cell]
        indicators = list(original)
        for group in self.model.groups:
            _round_group(indicators, group, rng)
        indices = tuple(i for i, value in enumerate(indicators) if value == 1)
        weights = tuple(
            self.contributions[cell][i] / (self.masses[cell] * original[i])
            for i in indices
        )
        return SamplingDraw(indices, weights)

    def verify_certificate(self):
        """Recheck the saved exact witnesses without calling an optimizer."""
        self.validate()
        certificate = self.certificate
        phi, allowance, noise = upper_values(
            self.model,
            self.masses,
            self.inclusions,
            self.contributions,
            certificate.relaxation_upper - certificate.noise_risk,
        )
        if (phi, allowance, noise) != (
            certificate.relaxation_upper,
            certificate.rounding_allowance,
            certificate.noise_risk,
        ):
            raise ArithmeticError("Risk upper certificate does not match the strategy")
        lower = ZERO
        if certificate.dual is not None:
            try:
                expected = _cells(
                    self.model,
                    len(self.cells),
                    len(self.cells)
                    * (4 * len(self.model.a) + (len(self.model.r[0]) + 1) ** 2),
                )
            except ValueError as error:
                raise ArithmeticError(
                    "Dual certificate does not cover all feasible head cells"
                ) from error
            if set(self.cells) != set(expected):
                raise ArithmeticError(
                    "Dual certificate uses incorrect cell constraints"
                )
            lower = max(lower, evaluate_dual(self.model, self.cells, certificate.dual))
        if certificate.noise_dual:
            lower = max(lower, evaluate_noise_dual(self.model, certificate.noise_dual))
        if lower != certificate.lower:
            raise ArithmeticError("Risk lower certificate does not match its witnesses")
        return True


def _bernoulli(probability, rng):
    return rng.randrange(probability.denominator) < probability.numerator


def _round_group(p, group, rng):
    fractional = [i for i in group if 0 < p[i] < 1]
    while len(fractional) >= 2:
        j, i = fractional.pop(), fractional.pop()
        up, down = min(1 - p[i], p[j]), min(p[i], 1 - p[j])
        if _bernoulli(down / (up + down), rng):
            p[i] += up
            p[j] -= up
        else:
            p[i] -= down
            p[j] += down
        fractional.extend(k for k in (i, j) if 0 < p[k] < 1)
    if fractional:
        raise ArithmeticError("Noninteger residual quota during rounding")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _validate_model(model):
    """Validate the assumptions used by the independent certificate checker."""
    n = len(model.a)
    if n == 0 or len(model.noise) != n or len(model.r) != n:
        raise ArithmeticError("Invalid model dimensions")
    dimension = len(model.r[0])
    if any(len(row) != dimension for row in model.r):
        raise ArithmeticError("Invalid feature dimensions")
    require_fractions(model.a)
    require_fractions(model.noise)
    require_fractions(value for row in model.r for value in row)
    if any(d <= 0 for d in model.noise):
        raise ArithmeticError("Noise variances must be positive")
    if len(model.groups) != len(model.quotas):
        raise ArithmeticError("Invalid quota dimensions")
    indices = [i for group in model.groups for i in group]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(n)):
        raise ArithmeticError("Strata must partition the population")
    for group, quota in zip(model.groups, model.quotas, strict=True):
        if type(quota) is not int or not 0 <= quota <= len(group):
            raise ArithmeticError("Invalid stratum quota")
        if quota == 0 and any(model.a[i] for i in group):
            raise ArithmeticError("No target coverage")
    if len(set(model.head)) != len(model.head) or any(
        type(i) is not int or not 0 <= i < n for i in model.head
    ):
        raise ArithmeticError("Invalid head indices")


def _model(a, features, noise, sample_size, rho, head, strata, quotas):
    for value in (a, noise, features):
        if value is not None and np.iscomplexobj(value):
            raise ValueError("Model inputs must be real")
    a, noise = np.asarray(a, dtype=float), np.asarray(noise, dtype=float)
    if a.ndim != 1 or not a.size or noise.shape != a.shape:
        raise ValueError("a and noise must be nonempty vectors of equal length")
    features = (
        np.empty((a.size, 0)) if features is None else np.asarray(features, dtype=float)
    )
    if features.ndim != 2 or features.shape[0] != a.size:
        raise ValueError("features must have shape (population_size, dimension)")
    if any(not np.isfinite(x).all() for x in (a, features, noise)):
        raise ValueError("Model inputs must be finite")
    if np.any(noise <= 0):
        raise ValueError("Noise variances must be strictly positive")
    rho = _nonnegative(rho, "rho")
    head = tuple(_integer(i, "head index") for i in head)
    if len(set(head)) != len(head) or any(i >= a.size for i in head):
        raise ValueError("head must contain distinct population indices")
    if strata is None:
        if quotas is not None:
            raise ValueError("quotas requires strata")
        size = _integer(sample_size, "sample_size")
        groups, sizes = (tuple(range(a.size)),), (size,)
    else:
        if len(strata) != a.size:
            raise ValueError("strata must have one integer label per unit")
        labels = tuple(_integer(i, "stratum label") for i in strata)
        unique = tuple(sorted(set(labels)))
        if not isinstance(quotas, Mapping) or set(quotas) != set(unique):
            raise ValueError("quotas must map every stratum label to its sample size")
        groups = tuple(
            tuple(i for i, label in enumerate(labels) if label == group)
            for group in unique
        )
        sizes = tuple(_integer(quotas[group], "quota") for group in unique)
        if sample_size is not None and _integer(sample_size, "sample_size") != sum(
            sizes
        ):
            raise ValueError("sample_size disagrees with the stratum quotas")
    for group, size in zip(groups, sizes, strict=True):
        if size > len(group):
            raise ValueError("Sample size exceeds population or stratum size")
        if size == 0 and any(a[i] != 0 for i in group):
            raise ValueError("A nonzero target has no sampling coverage")
    if rho == 0:
        features = np.empty((a.size, 0))
    else:
        features = features[:, np.any(features != 0, axis=0)]
    return SamplingModel(
        tuple(map(rational, a)),
        tuple(map(rational, noise)),
        tuple(tuple(rational(rho) * rational(v) for v in row) for row in features),
        groups,
        sizes,
        tuple(sorted(head)),
    )


def _cells(model, max_cells, max_cone_entries):
    parts, count = [], 1
    for group, quota in zip(model.groups, model.quotas, strict=True):
        head = tuple(i for i in group if i in model.head)
        tail = tuple(i for i in group if i not in model.head)
        lo, hi = max(0, quota - len(tail)), min(len(head), quota)
        count *= sum(comb(len(head), k) for k in range(lo, hi + 1))
        parts.append((group, head, tail, quota, lo, hi))
    if count > max_cells:
        raise ValueError(f"Conditioning requires {count} cells; max_cells={max_cells}")
    dimension = len(model.r[0])
    entries = count * (len(model.a) * 4 + (dimension + 1) ** 2)
    if entries > max_cone_entries:
        raise ValueError("Conic representation exceeds max_cone_entries")
    choices = [
        tuple(subset for k in range(lo, hi + 1) for subset in combinations(head, k))
        for _, head, _, _, lo, hi in parts
    ]
    cells = []
    for selections in product(*choices):
        fixed = [-1] * len(model.a)
        for selected, (_, head, tail, quota, _, _) in zip(
            selections, parts, strict=True
        ):
            for i in head:
                fixed[i] = int(i in selected)
            remaining = quota - len(selected)
            if remaining in (0, len(tail)):
                for i in tail:
                    fixed[i] = int(remaining != 0)
        cells.append(tuple(fixed))
    return tuple(cells)


def _zero_solution(model):
    if any(
        sum(model.a[i] != 0 for i in group) > quota
        for group, quota in zip(model.groups, model.quotas, strict=True)
    ):
        return None
    chosen = set()
    for group, quota in zip(model.groups, model.quotas, strict=True):
        active = [i for i in group if model.a[i]]
        chosen.update(active)
        chosen.update([i for i in group if not model.a[i]][: quota - len(active)])
    p = tuple(Fraction(i in chosen) for i in range(len(model.a)))
    result = SamplingSolution(
        model,
        (tuple(int(v) for v in p),),
        (ONE,),
        (p,),
        (model.a,),
        RiskCertificate(ZERO, ZERO, ZERO, ZERO),
        "exact_zero_risk",
        0.0,
        True,
    )
    result.verify_certificate()
    return result


def _optimize(model, cells, solver_tolerance, max_iter):
    import cvxpy as cp

    count, n, dimension = len(cells), len(model.a), len(model.r[0])
    a = np.array(list(map(float, model.a)))
    noise = np.array(list(map(float, model.noise)))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            r = np.array([[float(x) for x in row] for row in model.r])
            scales = np.sqrt(noise)
            target_scale = np.max(np.abs(a * scales))
            b = a * scales / target_scale
            d, rbar = noise / scales**2, r / scales[:, None]
    except (OverflowError, FloatingPointError) as error:
        raise SolverFailure(
            "Input scaling is outside the floating-point solver range"
        ) from error
    if (
        not np.isfinite(target_scale)
        or target_scale <= 0
        or any(not np.isfinite(x).all() for x in (b, d, rbar))
        or np.any((a != 0) & (b == 0))
    ):
        raise SolverFailure("Input scaling is outside the floating-point solver range")
    q = cp.Variable(count, nonneg=True)
    t, x, e = (cp.Variable((count, n)) for _ in range(3))
    mean_constraint = cp.sum(x, axis=0) == b
    constraints = [cp.sum(q) == 1, mean_constraint]
    matrices, blocks = [], []
    for j, fixed in enumerate(cells):
        allowed = [i for i in range(n) if fixed[i] != 0]
        excluded = [i for i in range(n) if fixed[i] == 0]
        forced = [i for i in range(n) if fixed[i] == 1]
        if excluded:
            constraints += [
                t[j, excluded] == 0,
                x[j, excluded] == 0,
                e[j, excluded] == 0,
            ]
        if forced:
            constraints.append(t[j, forced] == q[j])
        if allowed:
            tj, ej = t[j, allowed], e[j, allowed]
            residual = x[j, allowed] - cp.multiply(tj, b[allowed])
            constraints += [
                tj >= 0,
                tj <= q[j],
                cp.SOC(ej + tj, cp.vstack([2 * residual, ej - tj]), axis=0),
            ]
        for group, quota in zip(model.groups, model.quotas, strict=True):
            constraints.append(cp.sum(t[j, list(group)]) == quota * q[j])
        if dimension:
            matrix = cp.Variable((dimension, dimension), symmetric=True)
            feature = cp.reshape(rbar.T @ (x[j] - q[j] * b), (dimension, 1), order="C")
            block = (
                cp.bmat(
                    [
                        [matrix, feature],
                        [feature.T, cp.reshape(q[j], (1, 1), order="C")],
                    ]
                )
                >> 0
            )
            constraints.append(block)
            matrices.append(matrix)
            blocks.append(block)
    objective = cp.sum(cp.multiply(e, d)) + (d * b**2) @ (1 - cp.sum(t, axis=0))
    spectral = None
    if dimension:
        tau = cp.Variable()
        spectral = sum(matrices) << tau * np.eye(dimension)
        constraints.append(spectral)
        objective += tau
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(
            solver="CLARABEL",
            canon_backend=cp.SCIPY_CANON_BACKEND,
            tol_gap_abs=solver_tolerance,
            tol_gap_rel=solver_tolerance,
            tol_feas=solver_tolerance,
            max_iter=max_iter,
        )
    except cp.error.SolverError as error:
        raise SolverFailure(str(error)) from error
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise SolverFailure(f"Clarabel returned {problem.status}")
    violation = max(
        -float(np.min(q.value)),
        max(float(np.max(c.violation(), initial=0)) for c in constraints),
    )
    if not np.isfinite(violation) or violation > 1e-5:
        raise SolverFailure(f"Scaled constraint violation is too large: {violation}")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            raw_x = x.value * target_scale / scales
    except FloatingPointError as error:
        raise SolverFailure("Candidate overflows when restoring input units") from error
    if any(not np.isfinite(value).all() for value in (q.value, t.value, raw_x)):
        raise SolverFailure("Nonfinite primal candidate")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            h_matrix = spectral.dual_value if dimension else np.empty((0, 0))
            raw_z = -mean_constraint.dual_value / 2 * target_scale * scales
            raw_h = (
                np.array([-block.dual_value[:dimension, dimension] for block in blocks])
                * target_scale
                if dimension
                else np.empty((count, 0))
            )
        if any(not np.isfinite(value).all() for value in (h_matrix, raw_z, raw_h)):
            raise ValueError("Nonfinite dual candidate")
    except (TypeError, ValueError, FloatingPointError):
        h_matrix, raw_z, raw_h = None, None, None
    values = (q.value, t.value, raw_x, h_matrix, raw_z, raw_h)
    return values, problem.status, violation


def solve(
    a,
    features,
    noise,
    sample_size=None,
    *,
    rho=1.0,
    head=(),
    strata=None,
    quotas=None,
    accuracy_rtol=1e-6,
    accuracy_atol=1e-9,
    solver_tolerance=1e-9,
    max_iter=300,
    max_cells=256,
    max_cone_entries=1_000_000,
    require_accuracy=True,
):
    """Solve and certify the conditioned relaxation, then expose a sampler.

    Accuracy applies to the relaxation gap, not to the actual risk ratio.
    Inspect certificate.ratio_upper or certificate.guarantees(eta) for the
    latter. A CertificationError retains a valid, less accurate .result.
    Only feasible head patterns are enumerated. Full conditioning can coincide
    with the complete sample space; resource limits apply before construction.
    """
    rtol = _nonnegative(accuracy_rtol, "accuracy_rtol")
    atol = _nonnegative(accuracy_atol, "accuracy_atol")
    solver_tolerance = _nonnegative(solver_tolerance, "solver_tolerance")
    if solver_tolerance == 0:
        raise ValueError("solver_tolerance must be positive")
    max_iter = _integer(max_iter, "max_iter", 1)
    max_cells = _integer(max_cells, "max_cells", 1)
    max_cone_entries = _integer(max_cone_entries, "max_cone_entries", 1)
    model = _model(a, features, noise, sample_size, rho, head, strata, quotas)
    exact = _zero_solution(model)
    if exact is not None:
        return exact
    cells = _cells(model, max_cells, max_cone_entries)
    values, status, violation = _optimize(model, cells, solver_tolerance, max_iter)
    raw_q, raw_t, raw_x, raw_matrix, raw_z, raw_h = values
    masses, inclusions, contributions = repair_primal(model, cells, raw_q, raw_t, raw_x)
    phi, allowance, noise_risk = upper_values(model, masses, inclusions, contributions)
    noise_dual = noise_witness(model)
    lower = evaluate_noise_dual(model, noise_dual)
    dual = None
    if raw_matrix is not None:
        try:
            proposal = dual_witness(model, raw_matrix, raw_z, raw_h)
            dual_lower = evaluate_dual(model, cells, proposal)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass  # The independently valid noise witness remains available.
        else:
            dual, lower = proposal, max(lower, dual_lower)
    certificate = RiskCertificate(lower, phi, allowance, noise_risk, dual, noise_dual)
    converged = certificate.relaxation_gap <= rational(atol) + rational(rtol) * lower
    result = SamplingSolution(
        model,
        cells,
        masses,
        inclusions,
        contributions,
        certificate,
        status,
        violation,
        converged,
    )
    result.verify_certificate()
    if require_accuracy and not converged:
        raise CertificationError(result)
    return result
