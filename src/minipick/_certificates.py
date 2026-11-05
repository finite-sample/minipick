"""Exact postprocessing for the conditioned sampling relaxation.

All model floats are interpreted as their exact binary rational values.
Floating-point linear algebra only proposes witnesses; rational arithmetic
checks or evaluates every certificate.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

ZERO = Fraction(0)
ONE = Fraction(1)
GRID = 2**44


@dataclass(frozen=True)
class DualWitness:
    matrix: tuple
    z: tuple
    nodes: tuple


def rational(value):
    return Fraction(float(value))


def require_fractions(values):
    if any(not isinstance(value, Fraction) for value in values):
        raise ArithmeticError("Certificate data must contain exact Fraction values")


def dot(left, right):
    return sum((x * y for x, y in zip(left, right, strict=True)), ZERO)


def matvec(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def zeros(size):
    return [[ZERO for _ in range(size)] for _ in range(size)]


def psd(matrix):
    """Test positive semidefiniteness by exact symmetric elimination."""
    work = [list(row) for row in matrix]
    for k in range(len(work)):
        pivot = work[k][k]
        if pivot < 0:
            return False
        if pivot == 0:
            if any(work[i][k] != 0 for i in range(k + 1, len(work))):
                return False
            continue
        for i in range(k + 1, len(work)):
            for j in range(i, len(work)):
                work[i][j] -= work[i][k] * work[j][k] / pivot
                work[j][i] = work[i][j]
    return True


def inverse(matrix):
    """Invert a small rational matrix; singular proposals are rejected."""
    size = len(matrix)
    work = [
        list(row) + [Fraction(i == j) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for k in range(size):
        pivot = next((i for i in range(k, size) if work[i][k]), None)
        if pivot is None:
            raise ArithmeticError("Singular certificate matrix")
        work[k], work[pivot] = work[pivot], work[k]
        scale = work[k][k]
        work[k] = [value / scale for value in work[k]]
        for i in range(size):
            if i != k:
                scale = work[i][k]
                work[i] = [x - scale * y for x, y in zip(work[i], work[k])]
    return tuple(tuple(row[size:]) for row in work)


def spectral_upper(matrix):
    """Upper-bound lambda_max with an exactly verified PSD witness."""
    size = len(matrix)
    if not size or not any(value for row in matrix for value in row):
        return ZERO
    trace = sum((matrix[i][i] for i in range(size)), ZERO)
    gershgorin = max(
        matrix[i][i] + sum(abs(matrix[i][j]) for j in range(size) if j != i)
        for i in range(size)
    )
    fallback = min(trace, gershgorin)  # The input is a covariance Gram matrix.
    normalized = np.array(
        [[float(value / fallback) for value in row] for row in matrix]
    )
    try:
        proposal = rational(np.linalg.eigvalsh(normalized)[-1]) * fallback
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        proposal = fallback
    margin = fallback / 10**12
    for _ in range(12):
        upper = min(fallback, proposal + margin)
        witness = [
            [upper * (i == j) - matrix[i][j] for j in range(size)] for i in range(size)
        ]
        if psd(witness):
            return upper
        margin *= 10
    if psd(
        [[fallback * (i == j) - matrix[i][j] for j in range(size)] for i in range(size)]
    ):
        return fallback
    raise ArithmeticError("Invalid covariance Gram matrix")


def repair_adversary(raw):
    """Construct an exactly positive definite, trace-one adversary."""
    raw = np.asarray(raw, dtype=float)
    eigenvalues, vectors = np.linalg.eigh((raw + raw.T) / 2)
    factor = vectors * np.sqrt(np.maximum(eigenvalues, 1e-10))
    factor = tuple(tuple(rational(x) for x in row) for row in factor)
    matrix = tuple(tuple(dot(x, y) for y in factor) for x in factor)
    trace = sum((matrix[i][i] for i in range(len(matrix))), ZERO)
    matrix = tuple(tuple(x / trace for x in row) for row in matrix)
    return matrix, inverse(matrix)


def quota_probabilities(raw, quota):
    """Project a proposal to exact grid probabilities with an integer sum."""
    size = len(raw)
    if quota == 0:
        return (ZERO,) * size
    if quota == size:
        return (ONE,) * size
    values = np.clip(np.asarray(raw, dtype=float), 0, 1)
    lower, upper = -1.0, 1.0
    for _ in range(64):
        mid = (lower + upper) / 2
        if np.clip(values + mid, 0, 1).sum() < quota:
            lower = mid
        else:
            upper = mid
    projected = np.clip(values + (lower + upper) / 2, 0, 1)
    counts = [min(GRID - 1, max(1, round(float(v) * GRID))) for v in projected]
    remaining = quota * GRID - sum(counts)
    for i in range(size):
        change = min(GRID - 1 - counts[i], max(1 - counts[i], remaining))
        counts[i] += change
        remaining -= change
    if remaining:
        raise ArithmeticError("Could not repair a quota")
    return tuple(Fraction(count, GRID) for count in counts)


def repair_primal(model, cells, raw_q, raw_t, raw_x):
    counts = [max(1, round(max(0.0, float(q)) * GRID)) for q in raw_q]
    total = sum(counts)
    masses = tuple(Fraction(count, total) for count in counts)
    inclusions, contributions = [], []
    for j, fixed in enumerate(cells):
        p = [ZERO] * len(model.a)
        for indices, quota in zip(model.groups, model.quotas, strict=True):
            forced = [i for i in indices if fixed[i] == 1]
            free = [i for i in indices if fixed[i] == -1]
            for i in forced:
                p[i] = ONE
            proposal = raw_t[j, free] / max(float(raw_q[j]), 1e-30)
            for i, value in zip(
                free, quota_probabilities(proposal, quota - len(forced)), strict=True
            ):
                p[i] = value
        inclusions.append(tuple(p))
        row = []
        for i, value in enumerate(raw_x[j]):
            old_joint = rational(raw_t[j, i])
            new_joint = masses[j] * p[i]
            # A smaller probability must not amplify the proposed sample weight.
            scale = min(ONE, new_joint / old_joint) if old_joint > 0 else ZERO
            row.append(rational(value) * scale)
        contributions.append(row)
    for i, target in enumerate(model.a):
        correction = target - sum((x[i] for x in contributions), ZERO)
        if correction:
            j = max(range(len(cells)), key=lambda j: masses[j] * inclusions[j][i])
            if inclusions[j][i] == 0:
                raise ArithmeticError("Repair lost target coverage")
            contributions[j][i] += correction
    return masses, tuple(inclusions), tuple(tuple(x) for x in contributions)


def upper_values(model, masses, inclusions, contributions, spectral_bound=None):
    """Return Phi upper bound, rounding allowance, and exact noise risk."""
    dimension = len(model.r[0])
    matrix = zeros(dimension)
    within, between, allowance = ZERO, ZERO, ZERO
    rt = transpose(model.r) if dimension else ()
    row_norms = [dot(row, row) for row in model.r]
    for mass, p, x in zip(masses, inclusions, contributions, strict=True):
        k = tuple(value / mass for value in x)
        centered = tuple(ki - ai for ki, ai in zip(k, model.a, strict=True))
        feature = matvec(rt, centered)
        for i in range(dimension):
            for j in range(dimension):
                matrix[i][j] += mass * feature[i] * feature[j]
        for i in range(len(k)):
            between += mass * model.noise[i] * centered[i] ** 2
            if p[i]:
                variance = mass * k[i] ** 2 * (1 - p[i]) / p[i]
                within += model.noise[i] * variance
                allowance += 2 * row_norms[i] * variance
    noise = within + between
    if spectral_bound is None:
        spectral_bound = spectral_upper(matrix)
    elif spectral_bound < 0 or not psd(
        [
            [spectral_bound * (i == j) - matrix[i][j] for j in range(dimension)]
            for i in range(dimension)
        ]
    ):
        raise ArithmeticError("Invalid spectral upper witness")
    return noise + spectral_bound, allowance, noise


def cell_score(model, fixed, z, h, h_inverse):
    residual = tuple(zi - dot(row, h) for zi, row in zip(z, model.r, strict=True))
    scores = tuple(value**2 / d for value, d in zip(residual, model.noise))
    score = dot(h, matvec(h_inverse, h))
    for indices, quota in zip(model.groups, model.quotas, strict=True):
        forced = [i for i in indices if fixed[i] == 1]
        free = [i for i in indices if fixed[i] == -1]
        score += sum((scores[i] for i in forced), ZERO)
        score += sum(
            sorted((scores[i] for i in free), reverse=True)[: quota - len(forced)]
        )
    return score


def dual_witness(model, raw_h_matrix, raw_z, raw_h):
    """Reconstruct z and h from centered dual coordinates after repairing H."""
    dimension = len(model.r[0])
    matrix, _ = repair_adversary(raw_h_matrix) if dimension else ((), ())
    target = matvec(transpose(model.r), model.a) if dimension else ()
    anchor = matvec(matrix, target)
    z = tuple(
        d * a + rational(value) + dot(row, anchor)
        for d, a, value, row in zip(model.noise, model.a, raw_z, model.r, strict=True)
    )
    h = tuple(
        tuple(
            rational(value) + center for value, center in zip(row, anchor, strict=True)
        )
        for row in raw_h
    )
    return DualWitness(matrix, z, h)


def evaluate_dual(model, cells, witness):
    dimension = len(model.r[0])
    matrix, z, h = witness.matrix, witness.z, witness.nodes
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ArithmeticError("Invalid dual matrix dimensions")
    if len(z) != len(model.a) or len(h) != len(cells):
        raise ArithmeticError("Invalid dual vector dimensions")
    if any(len(row) != dimension for row in h):
        raise ArithmeticError("Invalid node witness dimensions")
    require_fractions(z)
    require_fractions(value for row in matrix for value in row)
    require_fractions(value for row in h for value in row)
    if dimension:
        if (
            any(
                matrix[i][j] != matrix[j][i]
                for i in range(dimension)
                for j in range(dimension)
            )
            or not psd(matrix)
            or sum(matrix[i][i] for i in range(dimension)) != 1
        ):
            raise ArithmeticError("Invalid trace-one PSD adversary")
        inv = inverse(matrix)
    else:
        inv = ()
    scores = [
        cell_score(model, fixed, z, hj, inv) for fixed, hj in zip(cells, h, strict=True)
    ]
    feature_target = matvec(transpose(model.r), model.a) if dimension else ()
    constant = sum((a * a * d for a, d in zip(model.a, model.noise)), ZERO)
    constant += dot(feature_target, matvec(matrix, feature_target))
    return 2 * dot(model.a, z) - max(scores) - constant


def noise_witness(model):
    """A noise-dual vector proposed using clipped allocation."""
    z = [ZERO] * len(model.a)
    for indices, quota in zip(model.groups, model.quotas, strict=True):
        active = [i for i in indices if model.a[i]]
        if len(active) <= quota:
            for i in active:
                z[i] = model.a[i] * model.noise[i]
            continue
        sizes = {
            i: abs(model.a[i]) * rational(np.sqrt(float(model.noise[i])))
            for i in active
        }
        ordered = sorted(active, key=sizes.__getitem__, reverse=True)
        total = sum(sizes.values(), ZERO)
        remaining = quota
        for position, i in enumerate(ordered):
            if remaining * sizes[i] >= total:
                z[i] = model.a[i] * model.noise[i]
                remaining -= 1
                total -= sizes[i]
            else:
                for k in ordered[position:]:
                    p = remaining * sizes[k] / total
                    z[k] = model.a[k] * model.noise[k] / p
                break
    return tuple(z)


def evaluate_noise_dual(model, z):
    if len(z) != len(model.a):
        raise ArithmeticError("Invalid noise witness dimensions")
    require_fractions(z)
    score = ZERO
    for indices, quota in zip(model.groups, model.quotas, strict=True):
        score += sum(
            sorted((z[i] ** 2 / model.noise[i] for i in indices), reverse=True)[:quota]
        )
    constant = sum((a * a * d for a, d in zip(model.a, model.noise)), ZERO)
    return max(ZERO, 2 * dot(model.a, z) - score - constant)
