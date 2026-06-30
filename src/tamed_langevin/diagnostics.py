from __future__ import annotations

import numpy as np


Array = np.ndarray


def active_fraction(active: Array) -> float:
    active = np.asarray(active, dtype=bool)

    if active.size == 0:
        return 0.0

    return float(np.mean(active))


def running_second_moment(samples: Array) -> Array:
    samples = np.asarray(samples, dtype=float)

    if samples.ndim == 1:
        squared_norms = samples**2
    else:
        squared_norms = np.sum(samples**2, axis=-1)

    cumulative_sum = np.cumsum(squared_norms)
    counts = np.arange(1, squared_norms.shape[0] + 1)

    return cumulative_sum / counts


def relative_drift_distortion(original_drift: Array, tamed_drift: Array) -> float:
    original_drift = np.asarray(original_drift, dtype=float)
    tamed_drift = np.asarray(tamed_drift, dtype=float)

    numerator = np.linalg.norm(original_drift - tamed_drift)
    denominator = np.linalg.norm(original_drift)

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def divergence_time(path: Array) -> int | None:
    path = np.asarray(path, dtype=float)

    if path.ndim == 1:
        finite = np.isfinite(path)
    else:
        finite = np.all(np.isfinite(path), axis=tuple(range(1, path.ndim)))

    bad_indices = np.where(~finite)[0]

    if bad_indices.size == 0:
        return None

    return int(bad_indices[0])


def second_moment_error(samples: Array, reference_second_moment: float) -> float:
    samples = np.asarray(samples, dtype=float)

    if samples.ndim == 1:
        empirical_second_moment = float(np.mean(samples**2))
    else:
        empirical_second_moment = float(np.mean(np.sum(samples**2, axis=-1)))

    return abs(empirical_second_moment - float(reference_second_moment))
