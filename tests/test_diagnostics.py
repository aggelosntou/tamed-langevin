import numpy as np

from tamed_langevin.diagnostics import (
    active_fraction,
    divergence_time,
    relative_drift_distortion,
    running_second_moment,
    second_moment_error,
)


def test_active_fraction():
    active = np.array([True, False, True, False])
    assert active_fraction(active) == 0.5


def test_active_fraction_empty():
    active = np.array([])
    assert active_fraction(active) == 0.0


def test_running_second_moment_vector_samples():
    samples = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
    ])

    result = running_second_moment(samples)

    expected = np.array([
        1.0,
        2.5,
        3.0,
    ])

    assert np.allclose(result, expected)


def test_running_second_moment_scalar_samples():
    samples = np.array([1.0, 2.0, 3.0])
    result = running_second_moment(samples)

    expected = np.array([
        1.0,
        2.5,
        14.0 / 3.0,
    ])

    assert np.allclose(result, expected)


def test_relative_drift_distortion():
    original = np.array([2.0, 0.0])
    tamed = np.array([1.0, 0.0])

    assert relative_drift_distortion(original, tamed) == 0.5


def test_relative_drift_distortion_zero_original():
    original = np.array([0.0, 0.0])
    tamed = np.array([1.0, 0.0])

    assert relative_drift_distortion(original, tamed) == 0.0


def test_divergence_time_none():
    path = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    assert divergence_time(path) is None


def test_divergence_time_detects_nan():
    path = np.array([
        [1.0, 2.0],
        [np.nan, 4.0],
        [5.0, 6.0],
    ])

    assert divergence_time(path) == 1


def test_divergence_time_detects_inf():
    path = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [np.inf, 6.0],
    ])

    assert divergence_time(path) == 2


def test_second_moment_error():
    samples = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ])

    error = second_moment_error(samples, reference_second_moment=2.0)

    assert error == 0.5
