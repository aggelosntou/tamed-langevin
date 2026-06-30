import numpy as np

from tamed_langevin.taming import adaptive_tamed_drift, check_g_c1, g_switch


def test_g_switch_values():
    values = g_switch(np.array([0.0, 0.5, 1.0, 2.0, 3.0]))
    assert np.allclose(values[[0, 1, 2]], [0.0, 0.0, 0.0])
    assert np.isclose(values[3], 2.0)
    assert np.isclose(values[4], 3.0)


def test_g_switch_c1():
    check_g_c1()


def test_adaptive_taming_inactive_for_small_drift():
    drift = np.array([0.1, -0.2])
    state = np.array([1.0, -1.0])
    tamed, active = adaptive_tamed_drift(drift, state, step_size=0.01, a_tame=0.05)
    assert not active.any()
    assert np.allclose(tamed, drift)


def test_adaptive_taming_shape():
    drift = np.array([10.0, -20.0, 0.1])
    state = np.array([1.0, -1.0, 0.5])
    tamed, active = adaptive_tamed_drift(drift, state, step_size=0.1, a_tame=0.05)
    assert tamed.shape == drift.shape
    assert active.shape == drift.shape
