import numpy as np

from tamed_langevin import (
    KTULASampler,
    active_fraction,
    divergence_time,
    running_second_moment,
)


def drift(x):
    return x**3 - x


def main():
    x0 = np.zeros(10)
    x0[0] = 100.0

    sampler = KTULASampler(
        drift=drift,
        step_size=0.01,
        beta=1.0,
        a_tame=0.05,
    )

    samples, active = sampler.sample(
        x0,
        n_steps=2000,
        burn_in=500,
        seed=1,
        return_active=True,
    )

    second_moment_path = running_second_moment(samples)

    print("samples shape:", samples.shape)
    print("active fraction:", active_fraction(active))
    print("divergence time:", divergence_time(samples))
    print("final running second moment:", second_moment_path[-1])


if __name__ == "__main__":
    main()
