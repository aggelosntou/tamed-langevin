#!/usr/bin/env python3
"""
Aggressive-regime sampling experiment for adaptive-tamed Langevin schemes.

This script compares:
    ULA,
    adaptive kTULA,
    adaptive tRLMC,

on the high-dimensional double-well target

    U(x) = x^4 / 4 - x^2 / 2,
    h(x) = grad U(x) = x^3 - x.

The reusable kTULA/tRLMC implementations are imported from src/tamed_langevin.
This file keeps only experiment-specific code: ULA baseline, target density,
plotting, statistics, and experiment orchestration.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tamed_langevin import KTULASampler, TRLMCSampler
from tamed_langevin.taming import check_g_c1


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class Config:
    beta: float = 1.0
    dim: int = 100

    lambdas: Tuple[float, ...] = (0.1, 0.01, 0.001)

    n_steps: int = 200_000
    burn_in: int = 50_000
    n_reps: int = 30

    seed: int = 1234
    illustrative_seed_offset: int = 999

    a_tame: float = 0.05

    density_xmin: float = -5.0
    density_xmax: float = 5.0
    density_grid_n: int = 3000
    hist_alpha: float = 0.35

    out_dir: str = "./figures/Sampling_Experiments_Adaptive_rLMC"

    def quick(self) -> "Config":
        return replace(
            self,
            lambdas=(0.1, 0.01),
            n_steps=5_000,
            burn_in=1_000,
            n_reps=3,
            density_grid_n=1000,
        )


METHODS = ("ULA", "kTULA", "tRLMC")


# ============================================================
# Potential and drift
# ============================================================
def potential_1d(x: np.ndarray) -> np.ndarray:
    return 0.25 * x**4 - 0.5 * x**2


def drift(x: np.ndarray) -> np.ndarray:
    return x**3 - x


# ============================================================
# Target density and second moment
# ============================================================
def true_density(x: np.ndarray, cfg: Config) -> np.ndarray:
    logp = -cfg.beta * potential_1d(x)
    logp -= np.max(logp)

    p = np.exp(logp)
    return p / np.trapezoid(p, x)


def true_second_moment(cfg: Config) -> float:
    x = np.linspace(-4.0, 4.0, 200_000)
    p = true_density(x, cfg)
    return float(np.trapezoid(x**2 * p, x))


# ============================================================
# Samplers
# ============================================================
def initial_state(cfg: Config) -> np.ndarray:
    x = np.zeros(cfg.dim)
    x[0] = 200.0
    return x


def make_sampler(method: str, lam: float, cfg: Config):
    if method == "kTULA":
        return KTULASampler(
            drift=drift,
            step_size=lam,
            beta=cfg.beta,
            a_tame=cfg.a_tame,
        )

    if method == "tRLMC":
        return TRLMCSampler(
            drift=drift,
            step_size=lam,
            beta=cfg.beta,
            a_tame=cfg.a_tame,
        )

    raise ValueError(f"Unknown sampler method: {method}")


def step_ula(x: np.ndarray, lam: float, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    noise = rng.normal(size=x.shape)
    return x - lam * drift(x) + np.sqrt(2.0 * lam / cfg.beta) * noise


# ============================================================
# Chain runners
# ============================================================
def run_chain(
    method: str,
    lam: float,
    seed: int,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    rng = np.random.default_rng(seed)
    x = initial_state(cfg)

    trajectory = []
    norm2 = []

    if method == "ULA":
        for step in range(cfg.n_steps):
            with np.errstate(over="ignore", invalid="ignore"):
                x = step_ula(x, lam, rng, cfg)

            if not np.isfinite(x).all():
                return None, None, step

            norm2.append(float(np.sum(x**2)))

            if step >= cfg.burn_in:
                trajectory.append(float(x[0]))

        return np.asarray(trajectory), np.asarray(norm2), None

    sampler = make_sampler(method, lam, cfg)

    for step in range(cfg.n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            x, _ = sampler.step(x, rng)

        if not np.isfinite(x).all():
            return None, None, step

        norm2.append(float(np.sum(x**2)))

        if step >= cfg.burn_in:
            trajectory.append(float(x[0]))

    return np.asarray(trajectory), np.asarray(norm2), None


def explosion_time_ula(lam: float, seed: int, cfg: Config) -> int:
    rng = np.random.default_rng(seed)
    x = initial_state(cfg)

    for step in range(cfg.n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            x = step_ula(x, lam, rng, cfg)

        if not np.isfinite(x).all():
            return step

    return cfg.n_steps + 1


def moment_error(
    method: str,
    lam: float,
    seed: int,
    true_m2: float,
    cfg: Config,
) -> float:
    rng = np.random.default_rng(seed)
    x = initial_state(cfg)
    sampler = make_sampler(method, lam, cfg)

    samples = []

    for step in range(cfg.n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            x, _ = sampler.step(x, rng)

        if not np.isfinite(x).all():
            return float("nan")

        if step >= cfg.burn_in:
            samples.append(float(x[0]))

    samples_array = np.asarray(samples)
    return float(abs(np.mean(samples_array**2) - true_m2))


# ============================================================
# Plot helpers
# ============================================================
def save_moment_growth_plot(
    lam: float,
    ula_norm2: Optional[np.ndarray],
    ktula_norm2: Optional[np.ndarray],
    trlmc_norm2: Optional[np.ndarray],
    cfg: Config,
) -> None:
    plt.figure()

    if ula_norm2 is not None:
        plt.plot(ula_norm2, label="ULA")

    if ktula_norm2 is not None:
        plt.plot(ktula_norm2, label="kTULA")

    if trlmc_norm2 is not None:
        plt.plot(trlmc_norm2, label="tRLMC")

    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(r"$\|X_n\|^2$")
    plt.title(f"Second-moment growth, d={cfg.dim}, λ={lam}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, f"moment_d{cfg.dim}_lam_{lam}.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_trajectory_plot(
    trajectory: np.ndarray,
    lam: float,
    name: str,
    cfg: Config,
) -> None:
    file_name = name.lower().replace(" ", "_")

    plt.figure()
    plt.plot(trajectory[:5000])
    plt.xlabel("post burn-in iteration")
    plt.ylabel(r"$X_n^{(1)}$")
    plt.title(f"{name} trajectory, λ={lam}")
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, f"traj_{file_name}_d{cfg.dim}_lam_{lam}.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_density_plot(
    trajectory: np.ndarray,
    lam: float,
    name: str,
    cfg: Config,
) -> None:
    file_name = name.lower().replace(" ", "_")
    grid = np.linspace(cfg.density_xmin, cfg.density_xmax, cfg.density_grid_n)

    plt.figure()
    plt.hist(
        trajectory,
        bins=120,
        range=(cfg.density_xmin, cfg.density_xmax),
        density=True,
        alpha=cfg.hist_alpha,
        label=name,
    )
    plt.plot(grid, true_density(grid, cfg), lw=2, label="Target density")
    plt.xlim(cfg.density_xmin, cfg.density_xmax)
    plt.xlabel("x, first coordinate")
    plt.ylabel("density")
    plt.title(f"Empirical density, d={cfg.dim}, λ={lam}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, f"density_{file_name}_d{cfg.dim}_lam_{lam}.png")
    plt.savefig(path, dpi=300)
    plt.close()


def boxplot_by_lambda(
    data_by_lambda: Dict[float, List[float]],
    title: str,
    file_name: str,
    ylabel: str,
    cfg: Config,
    logy: bool = False,
) -> None:
    data = [data_by_lambda[lam] for lam in cfg.lambdas]

    plt.figure(figsize=(6.0, 4.0))
    plt.boxplot(data)
    plt.xticks(
        range(1, len(cfg.lambdas) + 1),
        [f"λ={lam}" for lam in cfg.lambdas],
    )

    if logy:
        plt.yscale("log")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, file_name)
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Experiment
# ============================================================
def run_experiment(cfg: Config):
    check_g_c1()
    os.makedirs(cfg.out_dir, exist_ok=True)

    true_m2 = true_second_moment(cfg)

    ula_explosion_times: Dict[float, List[float]] = {}
    ktula_second_moment_errors: Dict[float, List[float]] = {}
    trlmc_second_moment_errors: Dict[float, List[float]] = {}

    illustrative_seed = cfg.seed + cfg.illustrative_seed_offset

    print("Adaptive sampling experiment")
    print("----------------------------")
    print(f"dimension d = {cfg.dim}")
    print(f"beta = {cfg.beta}")
    print(f"a_tame = {cfg.a_tame}")
    print(f"n_steps = {cfg.n_steps}")
    print(f"burn_in = {cfg.burn_in}")
    print(f"n_reps = {cfg.n_reps}")
    print(f"true E[X_1^2] = {true_m2:.8f}")
    print(f"output directory: {cfg.out_dir}")

    for lam_index, lam in enumerate(cfg.lambdas):
        print(f"\nRunning λ={lam}")

        ula_traj, ula_norm2, ula_expl = run_chain("ULA", lam, illustrative_seed, cfg)
        ktula_traj, ktula_norm2, _ = run_chain("kTULA", lam, illustrative_seed, cfg)
        trlmc_traj, trlmc_norm2, _ = run_chain("tRLMC", lam, illustrative_seed, cfg)

        save_moment_growth_plot(lam, ula_norm2, ktula_norm2, trlmc_norm2, cfg)

        if ktula_traj is not None:
            save_trajectory_plot(ktula_traj, lam, "kTULA", cfg)
            save_density_plot(ktula_traj, lam, "kTULA", cfg)

        if trlmc_traj is not None:
            save_trajectory_plot(trlmc_traj, lam, "tRLMC", cfg)
            save_density_plot(trlmc_traj, lam, "tRLMC", cfg)

        ula_times = []
        ktula_errors = []
        trlmc_errors = []

        for rep in range(cfg.n_reps):
            rep_seed = cfg.seed + 10_000 * lam_index + rep

            ula_times.append(float(explosion_time_ula(lam, rep_seed, cfg)))
            ktula_errors.append(moment_error("kTULA", lam, rep_seed, true_m2, cfg))
            trlmc_errors.append(moment_error("tRLMC", lam, rep_seed, true_m2, cfg))

        ula_explosion_times[lam] = ula_times
        ktula_second_moment_errors[lam] = ktula_errors
        trlmc_second_moment_errors[lam] = trlmc_errors

        n_exploded = sum(t < cfg.n_steps for t in ula_times)
        print(f"  ULA exploded in {n_exploded}/{cfg.n_reps} runs")

    return ula_explosion_times, ktula_second_moment_errors, trlmc_second_moment_errors


# ============================================================
# Summary
# ============================================================
def mean_std(values: List[float]) -> Tuple[float, float]:
    xs = np.asarray(values, dtype=float)
    return float(np.nanmean(xs)), float(np.nanstd(xs, ddof=1))


def print_summary(
    ula_explosion_times: Dict[float, List[float]],
    ktula_second_moment_errors: Dict[float, List[float]],
    trlmc_second_moment_errors: Dict[float, List[float]],
    cfg: Config,
) -> None:
    print("\nSummary table values:\n")

    for lam in cfg.lambdas:
        mean, std = mean_std(ula_explosion_times[lam])
        print(f"ULA explosion time, lambda={lam}: {mean:.2f} ± {std:.2f}")

    for lam in cfg.lambdas:
        mean, std = mean_std(ktula_second_moment_errors[lam])
        print(f"kTULA second-moment error, lambda={lam}: {mean:.6f} ± {std:.6f}")

    for lam in cfg.lambdas:
        mean, std = mean_std(trlmc_second_moment_errors[lam])
        print(f"tRLMC second-moment error, lambda={lam}: {mean:.6f} ± {std:.6f}")


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggressive-regime sampling experiment for adaptive-tamed Langevin schemes."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller quick test.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of chain steps.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=None,
        help="Override burn-in.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=None,
        help="Override number of repetitions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()

    if args.quick:
        cfg = cfg.quick()

    if args.out_dir is not None:
        cfg = replace(cfg, out_dir=args.out_dir)

    if args.n_steps is not None:
        cfg = replace(cfg, n_steps=args.n_steps)

    if args.burn_in is not None:
        cfg = replace(cfg, burn_in=args.burn_in)

    if args.n_reps is not None:
        cfg = replace(cfg, n_reps=args.n_reps)

    if cfg.burn_in >= cfg.n_steps:
        raise ValueError("burn_in must be smaller than n_steps.")

    (
        ula_explosion_times,
        ktula_second_moment_errors,
        trlmc_second_moment_errors,
    ) = run_experiment(cfg)

    boxplot_by_lambda(
        ula_explosion_times,
        "ULA stability: explosion time vs step size",
        "ula_explosion_time_boxplot.png",
        "Explosion time, iterations",
        cfg,
        logy=True,
    )

    boxplot_by_lambda(
        ktula_second_moment_errors,
        "kTULA accuracy: second-moment error vs step size",
        "ktula_second_moment_error_boxplot.png",
        r"$|\widehat{\mathbb{E}}[X_1^2] - \mathbb{E}_\pi[X_1^2]|$",
        cfg,
    )

    boxplot_by_lambda(
        trlmc_second_moment_errors,
        "tRLMC accuracy: second-moment error vs step size",
        "trlmc_second_moment_error_boxplot.png",
        r"$|\widehat{\mathbb{E}}[X_1^2] - \mathbb{E}_\pi[X_1^2]|$",
        cfg,
    )

    print_summary(
        ula_explosion_times,
        ktula_second_moment_errors,
        trlmc_second_moment_errors,
        cfg,
    )

    print(f"\nSaved figures to: {cfg.out_dir}")


if __name__ == "__main__":
    main()