"""Analytic regression cases and exact arithmetic checks, not a simulation study."""

from dataclasses import replace
from fractions import Fraction
from math import sqrt
from random import Random

import numpy as np
import pytest

from minipick._certificates import (
    DualWitness,
    evaluate_noise_dual,
    inverse,
    noise_witness,
    psd,
    quota_probabilities,
    repair_adversary,
    repair_primal,
    spectral_upper,
)
from minipick.solver import (
    CertificationError,
    RiskCertificate,
    SamplingSolution,
    SolverFailure,
    _cells,
    _model,
    solve,
)


def assert_brackets(result, optimum, tolerance=1e-6):
    certificate = result.certificate
    assert float(certificate.lower) <= optimum + 1e-12 * max(1, abs(optimum))
    assert float(certificate.relaxation_upper) >= optimum - 1e-12 * max(1, abs(optimum))
    assert float(certificate.relaxation_gap) <= tolerance * max(1, abs(optimum))
    assert result.validate()
    assert result.verify_certificate()


def test_noise_optimum_and_actual_sampling():
    result = solve([1] * 4, None, [1] * 4, 2)
    assert_brackets(result, 4)
    assert result.certificate.rounding_allowance == 0
    assert result.certificate.guarantees(1e-6)
    draw = result.draw(Random(42))
    assert len(draw.indices) == 2
    assert draw == result.draw(Random(42))
    assert tuple(map(float, draw.weights)) == pytest.approx((2, 2))
    assert (
        draw.estimate_exact([3, 5], baseline=7)
        == 7 + 3 * draw.weights[0] + 5 * draw.weights[1]
    )
    assert draw.estimate([3, 5], baseline=7) == pytest.approx(23)
    with pytest.raises(ValueError, match="one residual"):
        draw.estimate([3])


def test_relaxation_convergence_is_not_a_near_optimality_claim():
    result = solve([1, 1], [[2], [-2]], [1, 1], 1)
    assert_brackets(result, 2)
    assert result.converged
    assert float(result.certificate.risk_upper) == pytest.approx(18)
    assert float(result.certificate.ratio_upper) == pytest.approx(9)
    assert not result.certificate.guarantees(0.1)


def test_conditioning_recovers_exact_two_unit_solution():
    result = solve([1, 1], [[2], [-2]], [1, 1], 1, head=[0])
    assert_brackets(result, 18)
    assert result.certificate.rounding_allowance == 0
    assert result.certificate.guarantees(1e-6)
    # The only two outcomes allow an independent exact moment calculation.
    mean = [Fraction(0), Fraction(0)]
    second_noise, second_feature = Fraction(0), Fraction(0)
    for q, p, x in zip(result.masses, result.inclusions, result.contributions):
        k = [value / q for value in x]
        for i in range(2):
            mean[i] += q * k[i]
        second_noise += q * sum(value**2 for value in k)
        second_feature += q * (2 * k[0] - 2 * k[1]) ** 2
        assert p in ((0, 1), (1, 0))
    assert mean == [1, 1]
    actual = second_noise - 2 + second_feature
    assert result.certificate.lower <= actual <= result.certificate.risk_upper


@pytest.mark.parametrize("scale", [2, 10, 100])
def test_head_supported_structure_matches_classical_effective_variance(scale):
    target = [scale**-2, 1, 1]
    result = solve(target, [[scale], [0], [0]], [1, 1, 1], 2, head=[0])
    h = target[0] * sqrt(1 + scale**2)
    optimum = 2 * h - h**2 / 2
    assert_brackets(result, optimum)
    assert result.certificate.rounding_allowance == 0
    assert result.certificate.guarantees(1e-5)


def test_spectral_objective_and_full_conditioning_match_isotropic_optimum():
    result = solve(np.ones(4), np.eye(4), np.ones(4), 2, head=range(4))
    assert_brackets(result, 4 + 4 / 3)
    assert result.certificate.rounding_allowance == 0
    assert result.certificate.guarantees(1e-6)


def test_negative_estimand_singleton_closed_form():
    a = np.array([-2, 1, 0.5])
    f = np.array([[1], [-2], [3]])
    noise = np.array([1, 2, 4])
    result = solve(a, f, noise, 1, head=range(3))
    sizes = np.abs(a) * np.sqrt(noise + f[:, 0] ** 2)
    optimum = sizes.sum() ** 2 - (a**2 @ noise) - (a @ f[:, 0]) ** 2
    assert_brackets(result, optimum)
    assert result.certificate.rounding_allowance == 0


def test_stratum_quotas_and_exact_unbiasedness():
    result = solve(
        [1] * 5,
        [[1], [-1], [0.1], [-0.1], [0]],
        [1] * 5,
        strata=[2, 2, 7, 7, 7],
        quotas={2: 1, 7: 2},
        head=[0],
    )
    assert result.validate()
    for seed in (0, 1, 2, 3):
        draw = result.draw(Random(seed))
        assert len(draw.indices) == 3
        assert sum(i < 2 for i in draw.indices) == 1
    assert all(
        sum(q * p[i] for q, p in zip(result.masses, result.inclusions)) <= 1
        for i in range(5)
    )


def test_zero_risk_uses_exact_census_of_target_without_solver(monkeypatch):
    from minipick import solver as conditioned_sampling

    def fail(*args, **kwargs):
        raise AssertionError("Zero-risk instances must bypass numerical optimization")

    monkeypatch.setattr(conditioned_sampling, "_optimize", fail)
    result = solve([0, -2, 0, 3], np.eye(4) * 1e20, [1e-20] * 4, 3)
    assert result.solver_status == "exact_zero_risk"
    assert result.certificate.ratio_upper == 1
    assert result.certificate.guarantees(0)
    draw = result.draw(Random(5))
    assert {1, 3}.issubset(draw.indices)
    assert draw.estimate_exact([1] * 3) == 1


def test_zero_sample_and_zero_quota_strata():
    assert solve([0, 0], None, [1, 1], 0).draw(Random(1)).indices == ()
    result = solve(
        [0, 1, 1],
        None,
        [1, 1, 1],
        strata=[0, 1, 1],
        quotas={0: 0, 1: 1},
    )
    assert result.validate()
    assert 0 not in result.draw(Random(1)).indices


def test_repair_preserves_zero_target_control_variate_contributions():
    model = _model([0, 1, 1], [[1], [2], [3]], [1] * 3, 2, 1, [1], None, None)
    cells = _cells(model, 4, 1000)
    q, p, x = repair_primal(
        model,
        cells,
        np.array([0.5, 0.5]),
        np.array([[0.5, 0, 0.5], [0.25, 0.5, 0.25]]),
        np.array([[0.2, 0, 0.4], [-0.2, 1, 0.6]]),
    )
    assert x[0][0] != 0 and x[1][0] != 0
    assert x[0][0] + x[1][0] == 0
    assert sum(q) == 1
    assert all(sum(row) == 2 for row in p)
    assert tuple(sum(row[i] for row in x) for i in range(3)) == model.a


def test_resource_limit_counts_only_feasible_cells():
    with pytest.raises(ValueError, match="70 cells"):
        solve(np.ones(8), np.eye(8), np.ones(8), 4, head=range(8), max_cells=16)
    with pytest.raises(ValueError, match="max_cone_entries"):
        solve(np.ones(4), np.eye(4), np.ones(4), 2, max_cone_entries=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise": [0, 1, 1]},
        {"a": [np.nan, 1, 1]},
        {"features": [1, 2, 3]},
        {"features": np.array([[1j], [2], [3]])},
        {"rho": -1},
        {"sample_size": 4},
        {"sample_size": 1.0},
        {"head": [0, 0]},
        {"head": [3]},
        {"sample_size": 0},
        {"accuracy_rtol": -1},
        {"solver_tolerance": 0},
        {"strata": [0, 1, 1], "quotas": {0: 0, 1: 1}, "sample_size": None},
        {"strata": [0, 1, 1], "quotas": [1, 1], "sample_size": None},
    ],
)
def test_invalid_inputs(kwargs):
    inputs = dict(a=[1, 1, 1], features=[[1], [2], [3]], noise=[1, 1, 1], sample_size=1)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        solve(**inputs)


def test_unachieved_certificate_accuracy_is_not_silently_accepted():
    with pytest.raises(CertificationError) as error:
        solve(
            [1, 2, 3],
            [[2], [-1], [1]],
            [1, 2, 1],
            1,
            head=[0, 1, 2],
            accuracy_rtol=0,
            accuracy_atol=0,
        )
    result = error.value.result
    assert not result.converged
    assert result.validate()
    assert result.certificate.lower >= 0


def test_iteration_limit_is_a_solver_failure():
    with pytest.warns(UserWarning):
        with pytest.raises(SolverFailure, match="user_limit"):
            solve([1] * 4, np.eye(4), [1] * 4, 2, head=[0, 1], max_iter=1)


def test_corrupt_distribution_is_rejected():
    result = solve([1, 1], None, [1, 1], 1)
    corrupt = replace(result, contributions=((Fraction(2), Fraction(1)),))
    with pytest.raises(ArithmeticError, match="not unbiased"):
        corrupt.validate()


def test_exact_psd_check_resolves_sub_float_negative_eigenvalue():
    tiny = Fraction(1, 10**40)
    matrix = ((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1) - tiny))
    assert np.linalg.eigvalsh(np.array(matrix, dtype=float))[0] == 0
    assert not psd(matrix)
    assert psd(((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))))
    assert not psd(((Fraction(0), Fraction(1)), (Fraction(1), Fraction(1))))


def test_exact_spectral_certificate_and_adversary_repair():
    covariance = ((Fraction(2), Fraction(1)), (Fraction(1), Fraction(2)))
    upper = spectral_upper(covariance)
    assert Fraction(3) <= upper < Fraction(3) + Fraction(1, 10**9)
    h, h_inverse = repair_adversary(np.array([[1, 0], [0, -1e-9]]))
    assert psd(h)
    assert sum(h[i][i] for i in range(2)) == 1
    for i in range(2):
        for j in range(2):
            assert sum(h[i][k] * h_inverse[k][j] for k in range(2)) == int(i == j)
    with pytest.raises(ArithmeticError, match="Singular"):
        inverse(((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))))


def test_exact_quota_repair_handles_bad_boundary_proposals():
    values = quota_probabilities([-1e-8, 1 + 1e-8, 0.7, 0.7], 2)
    assert sum(values) == 2
    assert all(0 < value < 1 for value in values)


def test_certificate_can_be_verified_without_the_optimizer(monkeypatch):
    result = solve([1, 1], [[2], [-2]], [1, 1], 1, head=[0])
    from minipick import solver as conditioned_sampling

    def fail(*args, **kwargs):
        raise AssertionError("Certificate verification called the optimizer")

    monkeypatch.setattr(conditioned_sampling, "_optimize", fail)
    assert result.verify_certificate()
    bad = replace(
        result,
        certificate=replace(
            result.certificate,
            lower=result.certificate.lower + Fraction(1, 10**12),
        ),
    )
    with pytest.raises(ArithmeticError, match="lower certificate"):
        bad.verify_certificate()


def test_zero_certificate_is_recheckable():
    result = solve([0, 2], [[100], [1]], [1, 1], 1)
    assert result.verify_certificate()


def test_unit_rescaling_over_twelve_orders_of_magnitude():
    units = np.array([1e-6, 1e6])
    a, f, d = 1 / units, units * np.array([1, -1]), units**2
    result = solve(a, f[:, None], d, 1, head=[0])
    sizes = np.abs(a) * np.sqrt(d + f**2)
    optimum = sizes.sum() ** 2 - (a**2 @ d) - (a @ f) ** 2
    assert_brackets(result, optimum)
    assert result.certificate.guarantees(1e-6)


@pytest.mark.parametrize("target_scale", [1e-8, 1e8])
def test_target_rescaling_preserves_relative_certificate(target_scale):
    result = solve(
        np.ones(2) * target_scale,
        [[2], [-2]],
        [1, 1],
        1,
        head=[0],
        accuracy_atol=0,
    )
    assert result.certificate.guarantees(1e-6)
    assert_brackets(result, 18 * target_scale**2)


def test_certainty_boundary_in_noise_allocation():
    result = solve([100, 1, 1], None, [1, 1, 1], 2)
    assert_brackets(result, 2, tolerance=1e-5)


def test_radius_zero_ignores_nonzero_feature_columns():
    result = solve([1, 1], [[1e100], [-1e100]], [1, 1], 1, rho=0)
    assert_brackets(result, 2)
    assert result.certificate.rounding_allowance == 0


@pytest.mark.parametrize(
    "a, features, noise, rho",
    [
        ([1e308, 1e308], None, [1e308, 1e308], 1),
        ([1e-300, 1e-300], None, [1e-300, 1e-300], 1),
        ([1, 1], [[1e308], [-1e308]], [1, 1], 1e308),
    ],
)
def test_unrepresentable_scaling_fails_explicitly(a, features, noise, rho):
    with pytest.raises(SolverFailure, match="scaling"):
        solve(a, features, noise, 1, rho=rho)


def test_verifier_rejects_a_false_upper_bound_inside_the_optimization_gap():
    result = solve([1, 1], [[2], [-2]], [1, 1], 1, head=[0])
    corrupt = replace(
        result,
        certificate=replace(
            result.certificate,
            relaxation_upper=result.certificate.relaxation_upper - Fraction(1, 10**9),
        ),
    )
    assert corrupt.certificate.lower < corrupt.certificate.relaxation_upper
    with pytest.raises(ArithmeticError, match="spectral"):
        corrupt.verify_certificate()


def test_verifier_rejects_invalid_dual_adversary():
    result = solve([1, 1], [[2], [-2]], [1, 1], 1, head=[0])
    corrupt = replace(
        result,
        certificate=replace(
            result.certificate,
            dual=replace(result.certificate.dual, matrix=((Fraction(-1),),)),
        ),
    )
    with pytest.raises(ArithmeticError, match="adversary"):
        corrupt.verify_certificate()


def test_verifier_rejects_an_incomplete_dual_sample_family():
    model = _model([0, 1, 1], None, [1, 1, 1], 1, 1, [0], None, None)
    result = SamplingSolution(
        model=model,
        cells=((0, -1, -1),),
        masses=(Fraction(1),),
        inclusions=((Fraction(0), Fraction(1, 2), Fraction(1, 2)),),
        contributions=(model.a,),
        certificate=RiskCertificate(
            Fraction(2),
            Fraction(2),
            Fraction(0),
            Fraction(2),
            DualWitness((), (Fraction(0), Fraction(2), Fraction(2)), ((),)),
        ),
        solver_status="test",
        raw_constraint_violation=0,
        converged=True,
    )
    assert result.validate()
    with pytest.raises(ArithmeticError, match="all feasible head cells"):
        result.verify_certificate()


def test_verifier_rejects_float_data_and_overlapping_strata():
    result = solve([1, 1], None, [1, 1], 1)
    bad_weights = replace(result, contributions=((1.0, 1.0),))
    with pytest.raises(ArithmeticError, match="Fraction"):
        bad_weights.verify_certificate()
    bad_groups = replace(
        result,
        model=replace(result.model, groups=((0, 1), (1,)), quotas=(1, 0)),
    )
    with pytest.raises(ArithmeticError, match="partition"):
        bad_groups.verify_certificate()


def test_noise_witness_handles_extreme_allocation_ratios_exactly():
    model = _model([1, 1e-310], None, [1, 1], 1, 1, [], None, None)
    lower = evaluate_noise_dual(model, noise_witness(model))
    assert lower == 2 * model.a[1]


def test_probability_repair_does_not_inflate_tiny_cell_weights():
    model = _model([1, 1, 1], None, [1, 1, 1], 1, 1, [0], None, None)
    cells = _cells(model, 4, 1000)
    q, p, x = repair_primal(
        model,
        cells,
        np.array([0.5, 0.5]),
        np.array([[0, 1e-10, 0.500001], [0.5, 0, 0]]),
        np.array([[0, 1e-10, 1], [1, 0, 0]]),
    )
    # Unit 1 has no other cell coverage, so its required large weight is intrinsic.
    # Unit 2 retains ample coverage; the repair must not disturb its target.
    assert tuple(sum(row[i] for row in x) for i in range(3)) == model.a
    assert all(sum(row) == 1 for row in p)
    assert sum(q) == 1

    model = _model([1, 1, 1, 1], None, [1] * 4, 2, 1, [0], None, None)
    cells = _cells(model, 4, 1000)
    q, p, x = repair_primal(
        model,
        cells,
        np.array([0.5, 0.5]),
        np.array([[0, 1e-10, 0.5, 0.500001], [0.5, 0.25, 0.125, 0.125]]),
        np.array([[0, 1e-10, 0.5, 0.5], [1, 1 - 1e-10, 0.5, 0.5]]),
    )
    assert abs(x[0][1] / (q[0] * p[0][1])) <= 1
    assert tuple(sum(row[i] for row in x) for i in range(4)) == model.a


def test_spectral_certificate_survives_numerical_eigensolver_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("test failure")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    covariance = ((Fraction(2), Fraction(1)), (Fraction(1), Fraction(2)))
    assert spectral_upper(covariance) == 3


def test_failed_dual_repair_keeps_valid_strategy_and_noise_bound(monkeypatch):
    from minipick import solver as conditioned_sampling

    def fail(*args, **kwargs):
        raise ArithmeticError("test singular proposal")

    monkeypatch.setattr(conditioned_sampling, "dual_witness", fail)
    result = solve([1, 1], [[2], [-2]], [1, 1], 1, head=[0], require_accuracy=False)
    assert result.certificate.dual is None
    assert result.certificate.lower == 2
    assert not result.converged
    assert result.verify_certificate()
    with pytest.raises(CertificationError) as error:
        solve([1, 1], [[2], [-2]], [1, 1], 1, head=[0])
    assert error.value.result.verify_certificate()


def test_estimation_accepts_numpy_scalar_types():
    from minipick.solver import SamplingDraw

    draw = SamplingDraw((0, 1), (Fraction(2), Fraction(-3)))
    for dtype in (np.float32, np.float64, np.longdouble):
        values = np.array([0.1, 0.2], dtype=dtype)
        baseline = dtype(0.3)
        expected = Fraction(*baseline.as_integer_ratio()) + sum(
            w * Fraction(*v.as_integer_ratio()) for w, v in zip(draw.weights, values)
        )
        assert draw.estimate_exact(values, baseline) == expected


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64, np.bool_])
def test_estimation_with_numpy_integers_and_rational_weights(dtype):
    result = solve([0.25, 1, 1], [[2], [0], [0]], [1, 1, 1], 2, head=[0])
    draw = result.draw(Random(42))
    values = np.array([1, 2], dtype=dtype)
    baseline = dtype(1)
    expected = Fraction(int(baseline)) + sum(
        w * int(v) for w, v in zip(draw.weights, values)
    )
    assert draw.estimate_exact(values, baseline) == expected


@pytest.mark.parametrize("dtype,value", [(np.int64, 2**62), (np.uint64, 2**63)])
def test_exact_estimation_does_not_overflow_numpy_integers(dtype, value):
    draw = solve([1, 1], None, [1, 1], 1).draw(Random(42))
    assert draw.weights == (Fraction(2),)
    assert draw.estimate_exact(np.array([value], dtype=dtype)) == 2 * value
    assert draw.estimate_exact([0], baseline=dtype(value)) == value


def test_large_gap_preserves_recoverable_certification_error():
    with pytest.raises(CertificationError) as error:
        solve(
            [1e200, 2e200, 3e200],
            None,
            [1, 1, 1],
            1,
            accuracy_rtol=0,
            accuracy_atol=0,
        )
    result = error.value.result
    assert result.certificate.relaxation_gap > Fraction(10) ** 308
    assert result.verify_certificate()


def test_verifier_rejects_tampered_rounding_allowance():
    result = solve([1, 1], [[2], [-2]], [1, 1], 1)
    corrupt = replace(
        result,
        certificate=replace(result.certificate, rounding_allowance=Fraction(0)),
    )
    with pytest.raises(ArithmeticError, match="upper certificate"):
        corrupt.verify_certificate()


@pytest.mark.parametrize("case", ["sharp", "reordered", "conditioned"])
def test_exact_enumeration_of_rounding_tree_verifies_actual_risk(monkeypatch, case):
    from minipick import solver as conditioned_sampling
    from minipick._certificates import upper_values

    if case != "conditioned":
        features = [[2], [-2], [2], [-2]] if case == "sharp" else [[2], [2], [-2], [-2]]
        model = _model([1] * 4, features, [1] * 4, 2, 1, [], None, None)
        masses, p, x = (Fraction(1),), ((Fraction(1, 2),) * 4,), (model.a,)
        phi, allowance, noise = upper_values(model, masses, p, x)
        result = SamplingSolution(
            model,
            _cells(model, 1, 1000),
            masses,
            p,
            x,
            RiskCertificate(
                Fraction(4), phi, allowance, noise, noise_dual=noise_witness(model)
            ),
            "exact_test_fixture",
            0.0,
            True,
        )
    else:
        result = solve([1, -2, 1, 1], [[3], [-2], [1], [0]], [1] * 4, 2, head=[0])
    assert result.verify_certificate()

    class BranchNeeded(Exception):
        def __init__(self, probability):
            self.probability = probability

    path = []

    def branch(probability, rng):
        if not path:
            raise BranchNeeded(probability)
        return path.pop(0)

    monkeypatch.setattr(conditioned_sampling, "_bernoulli", branch)
    pending = [((), Fraction(1))]
    outcomes = []
    while pending:
        choices, mass = pending.pop()
        path[:] = choices
        try:
            draw = result.draw(Random(0))
        except BranchNeeded as needed:
            for choice, probability in (
                (True, needed.probability),
                (False, 1 - needed.probability),
            ):
                if probability:
                    pending.append((choices + (choice,), mass * probability))
        else:
            assert not path
            assert len(draw.indices) == 2
            outcomes.append((mass, draw))
    assert sum(mass for mass, _ in outcomes) == 1
    mean = [Fraction(0)] * 4
    risk = Fraction(0)
    for mass, draw in outcomes:
        w = [Fraction(0)] * 4
        for i, weight in zip(draw.indices, draw.weights):
            w[i] = weight
        error = [wi - ai for wi, ai in zip(w, result.model.a)]
        risk += mass * (
            sum(d * e**2 for d, e in zip(result.model.noise, error))
            + sum(row[0] * e for row, e in zip(result.model.r, error)) ** 2
        )
        mean = [m + mass * wi for m, wi in zip(mean, w)]
    assert tuple(mean) == result.model.a
    assert result.certificate.lower <= risk <= result.certificate.risk_upper
    assert result.certificate.rounding_allowance > 0
    if case == "sharp":
        assert risk == result.certificate.risk_upper == 36
    elif case == "reordered":
        assert risk == result.certificate.lower == 4
        assert result.certificate.risk_upper == 36
