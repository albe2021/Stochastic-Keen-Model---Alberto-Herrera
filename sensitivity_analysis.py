# sensitivity_analysis.py
#
# Revised sensitivity analysis for the positive-part stochastic Keen model.
#
# Main methodological changes:
#   1. The analysis is executed in the order OAT -> pairwise -> global.
#   2. Terminal non-crisis classification imposes both lower and upper
#      bounds on wage share and employment:
#          0.4 <= omega_T <= 0.99,
#          0.4 <= lambda_T <= 0.99,
#          ell_T <= 2.7.
#   3. The stochastic equations use the positive-part share diffusions and
#      no projection, clipping, or reflection.
#   4. OAT resolution is increased to 41 points in final mode.
#   5. OAT and pairwise experiments use common random numbers across the
#      parameter grid to reduce Monte Carlo roughness.
#   6. Pointwise uncertainty intervals are saved and displayed for the OAT
#      curves. No undocumented smoothing is applied.
#   7. Pairwise grids use 21 x 21 points in final mode and retain raw Monte
#      Carlo estimates; contour lines are added only as visual guides.
#   8. Post-burn-in employment is the discrete arithmetic mean of the
#      simulated observations, matching the quantity used in the code.
#   9. Terminal crises, finite divergent crises, numerical failures, and
#      negative-debt safety exits are recorded separately.
#  10. Conditional employment uses only non-crisis paths whose complete
#      post-burn share histories remain in [0,1]^2.
#  11. The global crisis model is fitted with a converged statsmodels
#      binomial GLM using classifiable trajectories only.
#  12. Cache filenames contain a methodology version so older outputs are
#      not silently reused after the classification changes.
#
# Expected dependency:
#   keen_model_functions.py must be available in the working directory or
#   on the Python path. Only KeenModel's parameter initialization is used;
#   the vectorized batch update below reproduces the equations explicitly
#   for computational efficiency.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import qmc, rankdata
from tqdm import tqdm

from keen_model_functions import KeenModel


# ============================================================
# 0. Paths and run configuration
# ============================================================

METHODOLOGY_VERSION = "v4_outcomes_admissible_employment_glm"


def get_project_dir() -> Path:
    """Return the script directory, or the notebook working directory."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = get_project_dir()
FIG_DIR = PROJECT_DIR / "Figures"
DATA_DIR = PROJECT_DIR / "SensitivityData"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Current working directory:", PROJECT_DIR)
print("Figures saved to:", FIG_DIR.resolve())
print("Data saved to:", DATA_DIR.resolve())


@dataclass
class RunConfig:
    """Central configuration for draft and final runs."""

    mode: str = "final"

    # Local one-at-a-time experiment.
    n_grid_oat: int = 41
    n_replications_oat: int = 100

    # Pairwise local experiments.
    n_grid_pairwise: int = 21
    n_replications_pairwise: int = 100

    # Global experiment.
    n_samples_global: int = 1024
    n_replications_global: int = 3
    n_boot_prcc: int = 1000

    # Simulation design.
    T: float = 100.0
    dt: float = 0.05
    burn_in_fraction: float = 0.50

    # Numerical safety limits. Finite positive-debt/share threshold
    # crossings are recorded as divergent crises. Non-finite updates are
    # recorded separately as numerical failures.
    numerical_abs_share_limit: float = 1.0e6
    numerical_abs_debt_limit: float = 1.0e8

    # Minimum admissible non-crisis sample needed to display a conditional
    # employment estimate at one OAT grid point.
    min_employment_eligible: int = 10

    # Recompute cached CSV files?
    force_recompute: bool = True


def make_config(mode: str = "final") -> RunConfig:
    """Create a computationally light draft or thesis-quality final run."""
    if mode == "draft":
        return RunConfig(
            mode="draft",
            n_grid_oat=17,
            n_replications_oat=30,
            n_grid_pairwise=11,
            n_replications_pairwise=30,
            n_samples_global=256,
            n_replications_global=1,
            n_boot_prcc=250,
            T=100.0,
            dt=0.05,
            burn_in_fraction=0.50,
            force_recompute=True,
        )

    if mode == "final":
        return RunConfig(
            mode="final",
            n_grid_oat=41,
            n_replications_oat=100,
            n_grid_pairwise=21,
            n_replications_pairwise=100,
            n_samples_global=1024,
            n_replications_global=3,
            n_boot_prcc=1000,
            T=100.0,
            dt=0.05,
            burn_in_fraction=0.50,
            force_recompute=True,
        )

    raise ValueError("mode must be either 'draft' or 'final'.")


# ============================================================
# 1. Parameter ranges, labels, and baseline values
# ============================================================

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "m": (1.2, 2.5),
    "eta_p": (0.05, 0.5),
    "sigma_ell": (0.01, 0.4),
    "kappa1": (0.2, 1.0),
    "r": (0.01, 0.1),
    "delta": (0.01, 0.1),
    "phi1": (0.2, 1.0),
    "nu": (1.5, 4.0),
}

PARAM_NAMES = list(PARAM_RANGES.keys())
N_PARAMS = len(PARAM_NAMES)

PARAM_LABELS = {
    "m": r"$m$",
    "eta_p": r"$\eta_p$",
    "sigma_ell": r"$\sigma_\ell$",
    "kappa1": r"$\kappa_1$",
    "r": r"$r$",
    "delta": r"$\delta$",
    "phi1": r"$\Phi_1$",
    "nu": r"$\nu$",
}

# Canonical positive-part-model calibration. The dividend keys use the
# canonical names adopted in the revised model implementation.
BASELINE_PARAMS = {
    "alpha": 0.02,
    "beta": 0.031,
    "delta": 0.04,
    "nu": 2.7,
    "r": 0.02,
    "eta_p": 0.192,
    "m": 1.875,
    "gamma": 0.9,
    "phi0": -0.292,
    "phi1": 0.469,
    "kappa0": 0.0318,
    "kappa1": 0.575,
    "kappa_min": 0.0,
    "kappa_max": 0.3,
    "dividend0": -0.078,
    "dividend1": 0.553,
    "dividend_min": 0.0,
    "dividend_max": 0.3,
    "sigma_omega": 0.05,
    "sigma_lambda": 0.05,
    "sigma_ell": 0.05,
}

DEFAULT_INITIAL_CONDITION = {
    "omega": 0.578,
    "lambda": 0.675,
    "ell": 1.53,
}

# Adapted GBH-style terminal classification.
TERMINAL_OMEGA_MIN = 0.40
TERMINAL_OMEGA_MAX = 0.99
TERMINAL_LAMBDA_MIN = 0.40
TERMINAL_LAMBDA_MAX = 0.99
TERMINAL_ELL_MAX = 2.70


# ============================================================
# 2. Plot settings and output helpers
# ============================================================

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def versioned_name(stem: str, mode: str, suffix: str) -> str:
    """Return a cache-safe output filename."""
    return f"{stem}_{mode}_{METHODOLOGY_VERSION}.{suffix}"


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path.resolve()}")
    return path


def save_data(df: pd.DataFrame, filename: str) -> Path:
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved data: {path.resolve()}")
    return path


def save_json(payload: dict, filename: str) -> Path:
    path = DATA_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Saved metadata: {path.resolve()}")
    return path


# ============================================================
# 3. Terminal crisis / non-crisis classification
# ============================================================


def terminal_noncrisis_condition(
    omega: np.ndarray | float,
    lam: np.ndarray | float,
    ell: np.ndarray | float,
) -> np.ndarray:
    """
    Adapted GBH-style terminal classification.

    Non-crisis if the terminal state satisfies

        0.40 <= omega_T <= 0.99,
        0.40 <= lambda_T <= 0.99,
        ell_T <= 2.70,

    and all terminal coordinates are finite.
    """
    omega_array = np.asarray(omega, dtype=float)
    lambda_array = np.asarray(lam, dtype=float)
    ell_array = np.asarray(ell, dtype=float)

    finite = (
        np.isfinite(omega_array)
        & np.isfinite(lambda_array)
        & np.isfinite(ell_array)
    )

    return (
        finite
        & (omega_array >= TERMINAL_OMEGA_MIN)
        & (omega_array <= TERMINAL_OMEGA_MAX)
        & (lambda_array >= TERMINAL_LAMBDA_MIN)
        & (lambda_array <= TERMINAL_LAMBDA_MAX)
        & (ell_array <= TERMINAL_ELL_MAX)
    )


def post_burn_employment_eligibility(
    *,
    noncrisis: np.ndarray,
    numerical_failure: np.ndarray,
    divergent_crisis: np.ndarray,
    negative_debt_safety_exit: np.ndarray,
    post_burn_omega_min: np.ndarray,
    post_burn_omega_max: np.ndarray,
    post_burn_lambda_min: np.ndarray,
    post_burn_lambda_max: np.ndarray,
    n_post_burn: np.ndarray,
    expected_post_burn_observations: int,
) -> np.ndarray:
    """
    Identify paths eligible for the conditional-employment analysis.

    Eligibility requires:
      1. terminal non-crisis classification;
      2. no numerical failure or finite safety exit;
      3. all post-burn wage-share and employment observations in [0, 1];
      4. the complete expected post-burn sample.
    """
    complete_sample = n_post_burn == expected_post_burn_observations

    admissible_post_burn = (
        np.isfinite(post_burn_omega_min)
        & np.isfinite(post_burn_omega_max)
        & np.isfinite(post_burn_lambda_min)
        & np.isfinite(post_burn_lambda_max)
        & (post_burn_omega_min >= 0.0)
        & (post_burn_omega_max <= 1.0)
        & (post_burn_lambda_min >= 0.0)
        & (post_burn_lambda_max <= 1.0)
    )

    return (
        noncrisis
        & ~numerical_failure
        & ~divergent_crisis
        & ~negative_debt_safety_exit
        & complete_sample
        & admissible_post_burn
    )


# ============================================================
# 4. Vectorized positive-part stochastic simulation
# ============================================================


def make_parameter_dict(varied_params: Dict[str, float]) -> Dict[str, float]:
    """Merge the baseline calibration with one varied parameter vector."""
    model = KeenModel({**BASELINE_PARAMS, **varied_params})
    return dict(model.params)


def dividend_values(profit: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Evaluate the canonical truncated-linear dividend function."""
    intercept = params.get("dividend0", params.get("delta0"))
    slope = params.get("dividend1", params.get("delta1"))
    lower = params.get("dividend_min", params.get("delta_min", 0.0))
    upper = params.get("dividend_max", params.get("delta_max", 0.3))

    if intercept is None or slope is None:
        raise KeyError(
            "Dividend parameters were not found. Expected dividend0/1 "
            "or legacy delta0/1 keys."
        )

    return np.clip(intercept + slope * profit, lower, upper)


def generate_brownian_increments(
    *,
    n_steps: int,
    n_replications: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """Generate one reproducible common-random-number shock array."""
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=(n_steps, n_replications, 3),
    )


def simulate_batch(
    params: Dict[str, float],
    brownian_increments: np.ndarray,
    *,
    T: float,
    dt: float,
    burn_in_fraction: float,
    initial_condition: Optional[Dict[str, float]] = None,
    numerical_abs_share_limit: float = 1.0e6,
    numerical_abs_debt_limit: float = 1.0e8,
) -> pd.DataFrame:
    """
    Simulate a batch of independent trajectories using Euler--Maruyama.

    Outcome categories are kept distinct:

      noncrisis
          The trajectory reaches the horizon and satisfies the terminal
          non-crisis criterion.

      terminal_crisis
          The trajectory reaches the horizon but fails the terminal
          non-crisis criterion.

      divergent_crisis
          A finite update crosses the positive-debt safety threshold or a
          share coordinate crosses the finite share-divergence threshold.

      numerical_failure
          A proposed update contains NaN or Inf.

      negative_debt_safety_exit
          A finite path reaches the large negative-debt safety threshold.
          This is reported separately and excluded from crisis regression,
          because a large net-savings position is not a positive-debt crisis.

    No projection, clipping, or reflection is imposed.
    """
    if initial_condition is None:
        initial_condition = DEFAULT_INITIAL_CONDITION

    n_steps = int(round(T / dt))

    if brownian_increments.shape[0] != n_steps:
        raise ValueError("Brownian increments have the wrong time dimension.")
    if brownian_increments.shape[2] != 3:
        raise ValueError("Brownian increments must have three shock columns.")

    n_replications = brownian_increments.shape[1]
    p = make_parameter_dict(params)

    omega = np.full(
        n_replications,
        float(initial_condition["omega"]),
        dtype=float,
    )
    lam = np.full(
        n_replications,
        float(initial_condition["lambda"]),
        dtype=float,
    )
    ell = np.full(
        n_replications,
        float(initial_condition["ell"]),
        dtype=float,
    )

    active = np.ones(n_replications, dtype=bool)
    numerical_failure = np.zeros(n_replications, dtype=bool)
    divergent_crisis = np.zeros(n_replications, dtype=bool)
    negative_debt_safety_exit = np.zeros(n_replications, dtype=bool)

    divergence_reason = np.full(n_replications, "", dtype=object)

    min_lambda = lam.copy()
    min_omega = omega.copy()
    max_ell = ell.copy()
    max_abs_ell = np.abs(ell)

    burn_in_steps = int(round(burn_in_fraction * n_steps))
    expected_post_burn_observations = n_steps - burn_in_steps

    lambda_sum_post_burn = np.zeros(n_replications, dtype=float)
    n_post_burn = np.zeros(n_replications, dtype=int)

    post_burn_omega_min = np.full(n_replications, np.inf)
    post_burn_omega_max = np.full(n_replications, -np.inf)
    post_burn_lambda_min = np.full(n_replications, np.inf)
    post_burn_lambda_max = np.full(n_replications, -np.inf)

    for step in range(n_steps):
        if not np.any(active):
            break

        idx = np.flatnonzero(active)
        omega_a = omega[idx]
        lambda_a = lam[idx]
        ell_a = ell[idx]

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            profit = 1.0 - omega_a - p["r"] * ell_a
            investment = np.clip(
                p["kappa0"] + p["kappa1"] * profit,
                p.get("kappa_min", 0.0),
                p["kappa_max"],
            )
            dividends = dividend_values(profit, p)
            inflation = p["eta_p"] * (p["m"] * omega_a - 1.0)
            phillips = p["phi0"] + p["phi1"] * lambda_a

            drift_omega = omega_a * (
                phillips
                - p["alpha"]
                - (1.0 - p["gamma"]) * inflation
            )
            drift_lambda = lambda_a * (
                investment / p["nu"]
                - p["delta"]
                - p["alpha"]
                - p["beta"]
            )
            drift_ell = (
                ell_a
                * (
                    p["r"]
                    - investment / p["nu"]
                    + p["delta"]
                    - inflation
                )
                + omega_a
                + investment
                - 1.0
                + dividends
            )

            diffusion_omega = p["sigma_omega"] * np.sqrt(
                np.maximum(omega_a * (1.0 - omega_a), 0.0)
            )
            diffusion_lambda = p["sigma_lambda"] * np.sqrt(
                np.maximum(lambda_a * (1.0 - lambda_a), 0.0)
            )
            diffusion_ell = p["sigma_ell"] * np.abs(ell_a)

            dW = brownian_increments[step, idx, :]

            omega_new = (
                omega_a
                + drift_omega * dt
                + diffusion_omega * dW[:, 0]
            )
            lambda_new = (
                lambda_a
                + drift_lambda * dt
                + diffusion_lambda * dW[:, 1]
            )
            ell_new = (
                ell_a
                + drift_ell * dt
                + diffusion_ell * dW[:, 2]
            )

        finite_local = (
            np.isfinite(omega_new)
            & np.isfinite(lambda_new)
            & np.isfinite(ell_new)
        )

        share_divergence_local = finite_local & (
            (np.abs(omega_new) >= numerical_abs_share_limit)
            | (np.abs(lambda_new) >= numerical_abs_share_limit)
        )
        positive_debt_divergence_local = finite_local & (
            ell_new >= numerical_abs_debt_limit
        )
        negative_debt_exit_local = finite_local & (
            ell_new <= -numerical_abs_debt_limit
        )

        divergent_local = (
            share_divergence_local
            | positive_debt_divergence_local
        )
        numerical_failure_local = ~finite_local

        continuing_local = ~(
            divergent_local
            | negative_debt_exit_local
            | numerical_failure_local
        )

        continuing_idx = idx[continuing_local]

        if len(continuing_idx) > 0:
            omega[continuing_idx] = omega_new[continuing_local]
            lam[continuing_idx] = lambda_new[continuing_local]
            ell[continuing_idx] = ell_new[continuing_local]

            min_lambda[continuing_idx] = np.minimum(
                min_lambda[continuing_idx],
                lam[continuing_idx],
            )
            min_omega[continuing_idx] = np.minimum(
                min_omega[continuing_idx],
                omega[continuing_idx],
            )
            max_ell[continuing_idx] = np.maximum(
                max_ell[continuing_idx],
                ell[continuing_idx],
            )
            max_abs_ell[continuing_idx] = np.maximum(
                max_abs_ell[continuing_idx],
                np.abs(ell[continuing_idx]),
            )

            # Include X_{n dt} for n=N_b+1,...,N.
            if step + 1 > burn_in_steps:
                lambda_sum_post_burn[continuing_idx] += lam[continuing_idx]
                n_post_burn[continuing_idx] += 1

                post_burn_omega_min[continuing_idx] = np.minimum(
                    post_burn_omega_min[continuing_idx],
                    omega[continuing_idx],
                )
                post_burn_omega_max[continuing_idx] = np.maximum(
                    post_burn_omega_max[continuing_idx],
                    omega[continuing_idx],
                )
                post_burn_lambda_min[continuing_idx] = np.minimum(
                    post_burn_lambda_min[continuing_idx],
                    lam[continuing_idx],
                )
                post_burn_lambda_max[continuing_idx] = np.maximum(
                    post_burn_lambda_max[continuing_idx],
                    lam[continuing_idx],
                )

        divergent_idx = idx[divergent_local]
        if len(divergent_idx) > 0:
            divergent_crisis[divergent_idx] = True
            active[divergent_idx] = False

            # Preserve the finite threshold-crossing state for diagnostics.
            omega[divergent_idx] = omega_new[divergent_local]
            lam[divergent_idx] = lambda_new[divergent_local]
            ell[divergent_idx] = ell_new[divergent_local]

            share_mask = share_divergence_local[divergent_local]
            positive_debt_mask = positive_debt_divergence_local[divergent_local]

            reasons = np.where(
                share_mask & positive_debt_mask,
                "share_and_positive_debt_divergence",
                np.where(
                    share_mask,
                    "share_divergence",
                    "positive_debt_divergence",
                ),
            )
            divergence_reason[divergent_idx] = reasons

        negative_idx = idx[negative_debt_exit_local]
        if len(negative_idx) > 0:
            negative_debt_safety_exit[negative_idx] = True
            active[negative_idx] = False
            omega[negative_idx] = omega_new[negative_debt_exit_local]
            lam[negative_idx] = lambda_new[negative_debt_exit_local]
            ell[negative_idx] = ell_new[negative_debt_exit_local]
            divergence_reason[negative_idx] = "negative_debt_safety_exit"

        failure_idx = idx[numerical_failure_local]
        if len(failure_idx) > 0:
            numerical_failure[failure_idx] = True
            active[failure_idx] = False
            omega[failure_idx] = np.nan
            lam[failure_idx] = np.nan
            ell[failure_idx] = np.nan
            divergence_reason[failure_idx] = "nonfinite_update"

    completed_horizon = active.copy()

    terminal_noncrisis = (
        completed_horizon
        & terminal_noncrisis_condition(omega, lam, ell)
    )
    terminal_crisis = completed_horizon & ~terminal_noncrisis

    crisis = terminal_crisis | divergent_crisis
    valid_for_logistic = (
        terminal_noncrisis
        | terminal_crisis
        | divergent_crisis
    )

    lambda_mean_raw = np.divide(
        lambda_sum_post_burn,
        n_post_burn,
        out=np.full(n_replications, np.nan),
        where=n_post_burn > 0,
    )

    employment_eligible = post_burn_employment_eligibility(
        noncrisis=terminal_noncrisis,
        numerical_failure=numerical_failure,
        divergent_crisis=divergent_crisis,
        negative_debt_safety_exit=negative_debt_safety_exit,
        post_burn_omega_min=post_burn_omega_min,
        post_burn_omega_max=post_burn_omega_max,
        post_burn_lambda_min=post_burn_lambda_min,
        post_burn_lambda_max=post_burn_lambda_max,
        n_post_burn=n_post_burn,
        expected_post_burn_observations=expected_post_burn_observations,
    )

    lambda_mean_eligible = np.where(
        employment_eligible,
        lambda_mean_raw,
        np.nan,
    )

    outcome_status = np.full(
        n_replications,
        "unclassified",
        dtype=object,
    )
    outcome_status[terminal_noncrisis] = "noncrisis"
    outcome_status[terminal_crisis] = "terminal_crisis"
    outcome_status[divergent_crisis] = "divergent_crisis"
    outcome_status[numerical_failure] = "numerical_failure"
    outcome_status[negative_debt_safety_exit] = (
        "negative_debt_safety_exit"
    )

    return pd.DataFrame(
        {
            "replication": np.arange(n_replications, dtype=int),
            "outcome_status": outcome_status,
            "crisis": crisis.astype(int),
            "noncrisis": terminal_noncrisis.astype(int),
            "terminal_crisis": terminal_crisis.astype(int),
            "divergent_crisis": divergent_crisis.astype(int),
            "numerical_failure": numerical_failure.astype(int),
            "negative_debt_safety_exit": (
                negative_debt_safety_exit.astype(int)
            ),
            "valid_for_logistic": valid_for_logistic.astype(int),
            "employment_eligible": employment_eligible.astype(int),
            "divergence_reason": divergence_reason,
            "omega_final": omega,
            "lambda_final": lam,
            "ell_final": ell,
            "lambda_mean_post_burn_raw": lambda_mean_raw,
            "lambda_mean_post_burn": lambda_mean_eligible,
            "lambda_min": min_lambda,
            "omega_min": min_omega,
            "ell_max": max_ell,
            "ell_max_abs": max_abs_ell,
            "post_burn_omega_min": post_burn_omega_min,
            "post_burn_omega_max": post_burn_omega_max,
            "post_burn_lambda_min": post_burn_lambda_min,
            "post_burn_lambda_max": post_burn_lambda_max,
            "n_post_burn": n_post_burn,
            "expected_post_burn_observations": (
                expected_post_burn_observations
            ),
        }
    )


# ============================================================
# 5. Monte Carlo uncertainty summaries
# ============================================================


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a 95% Wilson interval for a binomial proportion."""
    if trials <= 0:
        return np.nan, np.nan

    p_hat = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (p_hat + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            p_hat * (1.0 - p_hat) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )

    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def summarize_batch(
    batch: pd.DataFrame,
    *,
    min_employment_eligible: int = 10,
) -> dict:
    """
    Summarize crisis frequency among classifiable trajectories and
    conditional employment among economically admissible non-crisis paths.
    """
    n_replications = len(batch)
    valid = batch["valid_for_logistic"] == 1

    n_valid = int(valid.sum())
    n_crisis = int(batch.loc[valid, "crisis"].sum())
    n_noncrisis = int(batch["noncrisis"].sum())
    n_terminal_crisis = int(batch["terminal_crisis"].sum())
    n_divergent_crisis = int(batch["divergent_crisis"].sum())
    n_numerical_failure = int(batch["numerical_failure"].sum())
    n_negative_debt_exit = int(
        batch["negative_debt_safety_exit"].sum()
    )

    if n_valid > 0:
        crisis_rate = n_crisis / n_valid
    else:
        crisis_rate = np.nan

    crisis_ci_low, crisis_ci_high = wilson_interval(
        n_crisis,
        n_valid,
    )

    eligible_values = batch.loc[
        batch["employment_eligible"] == 1,
        "lambda_mean_post_burn",
    ].dropna().to_numpy(dtype=float)

    n_employment_eligible = len(eligible_values)

    if n_employment_eligible >= min_employment_eligible:
        employment_mean = float(np.mean(eligible_values))
    else:
        employment_mean = np.nan

    if n_employment_eligible >= max(2, min_employment_eligible):
        employment_sd = float(np.std(eligible_values, ddof=1))
        employment_se = (
            employment_sd / np.sqrt(n_employment_eligible)
        )
        employment_ci_low = (
            employment_mean - 1.96 * employment_se
        )
        employment_ci_high = (
            employment_mean + 1.96 * employment_se
        )
    else:
        employment_sd = np.nan
        employment_se = np.nan
        employment_ci_low = np.nan
        employment_ci_high = np.nan

    return {
        "crisis_rate": crisis_rate,
        "crisis_ci_low": crisis_ci_low,
        "crisis_ci_high": crisis_ci_high,
        "n_crisis": n_crisis,
        "n_valid_classification": n_valid,
        "n_replications": n_replications,
        "n_noncrisis": n_noncrisis,
        "n_terminal_crisis": n_terminal_crisis,
        "n_divergent_crisis": n_divergent_crisis,
        "n_numerical_failure": n_numerical_failure,
        "n_negative_debt_safety_exit": n_negative_debt_exit,
        "noncrisis_lambda_mean": employment_mean,
        "noncrisis_lambda_sd": employment_sd,
        "noncrisis_lambda_se": employment_se,
        "noncrisis_lambda_ci_low": employment_ci_low,
        "noncrisis_lambda_ci_high": employment_ci_high,
        "n_employment_eligible": n_employment_eligible,
        "employment_minimum_sample": min_employment_eligible,
    }


# ============================================================
# 6. Sobol parameter sampling
# ============================================================


def generate_sobol_parameter_samples(n_samples: int, seed: int = 123) -> np.ndarray:
    """Generate scrambled Sobol samples over PARAM_RANGES."""
    sampler = qmc.Sobol(d=N_PARAMS, scramble=True, seed=seed)
    m_power = int(np.ceil(np.log2(n_samples)))
    unit_samples = sampler.random_base2(m=m_power)[:n_samples]

    samples = np.zeros((n_samples, N_PARAMS), dtype=float)

    for index, name in enumerate(PARAM_NAMES):
        low, high = PARAM_RANGES[name]
        samples[:, index] = low + unit_samples[:, index] * (high - low)

    return samples


def samples_to_dataframe(samples: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(samples, columns=PARAM_NAMES)


# ============================================================
# 7. Local one-at-a-time sensitivity with common random numbers
# ============================================================


def run_local_oat_sensitivity(
    *,
    config: RunConfig,
    seed: int = 1234,
    data_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Vary one parameter at a time while holding the others at baseline.

    For each parameter, one Brownian shock array is generated and reused at
    every grid point. This common-random-number design reduces Monte Carlo
    roughness without smoothing the estimated response curves.
    """
    if data_file is None:
        data_file = versioned_name("sensitivity_local_oat", config.mode, "csv")

    path = DATA_DIR / data_file

    if path.exists() and not config.force_recompute:
        print(f"Loading existing OAT data from {path.resolve()}")
        return pd.read_csv(path)

    n_steps = int(round(config.T / config.dt))
    rows = []

    for parameter_index, name in enumerate(PARAM_NAMES):
        low, high = PARAM_RANGES[name]
        grid = np.linspace(low, high, config.n_grid_oat)

        # Reuse exactly the same shock realizations at all values of this
        # parameter. Different parameters receive different shock sets.
        shock_seed = seed + 100000 * parameter_index
        common_dW = generate_brownian_increments(
            n_steps=n_steps,
            n_replications=config.n_replications_oat,
            dt=config.dt,
            seed=shock_seed,
        )

        print(
            f"Running OAT sensitivity for {name}: "
            f"{config.n_grid_oat} values x "
            f"{config.n_replications_oat} common-random-number replications"
        )

        for grid_index, value in enumerate(tqdm(grid, desc=name)):
            varied = {parameter: BASELINE_PARAMS[parameter] for parameter in PARAM_NAMES}
            varied[name] = float(value)

            batch = simulate_batch(
                varied,
                common_dW,
                T=config.T,
                dt=config.dt,
                burn_in_fraction=config.burn_in_fraction,
                numerical_abs_share_limit=config.numerical_abs_share_limit,
                numerical_abs_debt_limit=config.numerical_abs_debt_limit,
            )

            summary = summarize_batch(
                batch,
                min_employment_eligible=config.min_employment_eligible,
            )
            rows.append(
                {
                    "parameter": name,
                    "parameter_index": parameter_index,
                    "grid_index": grid_index,
                    "value": float(value),
                    "shock_set_seed": shock_seed,
                    "common_random_numbers": True,
                    **summary,
                }
            )

    out = pd.DataFrame(rows)
    save_data(out, data_file)
    return out


# ============================================================
# 8. Pairwise local sensitivity with common random numbers
# ============================================================


def run_pairwise_sensitivity(
    *,
    pair: Tuple[str, str],
    config: RunConfig,
    seed: int,
    data_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Vary two selected parameters over a regular grid.

    The same Brownian shock set is reused at every grid cell, providing a
    smooth comparison without applying an interpolation or smoothing filter.
    """
    p1, p2 = pair

    if p1 not in PARAM_NAMES or p2 not in PARAM_NAMES:
        raise ValueError("Both pairwise parameters must be in PARAM_NAMES.")
    if p1 == p2:
        raise ValueError("Pairwise parameters must be distinct.")

    if data_file is None:
        data_file = versioned_name(
            f"sensitivity_pair_{p1}_{p2}",
            config.mode,
            "csv",
        )

    path = DATA_DIR / data_file

    if path.exists() and not config.force_recompute:
        print(f"Loading existing pairwise data from {path.resolve()}")
        return pd.read_csv(path)

    grid1 = np.linspace(*PARAM_RANGES[p1], config.n_grid_pairwise)
    grid2 = np.linspace(*PARAM_RANGES[p2], config.n_grid_pairwise)
    n_steps = int(round(config.T / config.dt))

    common_dW = generate_brownian_increments(
        n_steps=n_steps,
        n_replications=config.n_replications_pairwise,
        dt=config.dt,
        seed=seed,
    )

    rows = []
    total_cells = len(grid1) * len(grid2)
    progress = tqdm(total=total_cells, desc=f"{p1} vs {p2}")

    for i, value1 in enumerate(grid1):
        for j, value2 in enumerate(grid2):
            varied = {parameter: BASELINE_PARAMS[parameter] for parameter in PARAM_NAMES}
            varied[p1] = float(value1)
            varied[p2] = float(value2)

            batch = simulate_batch(
                varied,
                common_dW,
                T=config.T,
                dt=config.dt,
                burn_in_fraction=config.burn_in_fraction,
                numerical_abs_share_limit=config.numerical_abs_share_limit,
                numerical_abs_debt_limit=config.numerical_abs_debt_limit,
            )

            summary = summarize_batch(
                batch,
                min_employment_eligible=config.min_employment_eligible,
            )
            rows.append(
                {
                    p1: float(value1),
                    p2: float(value2),
                    f"{p1}_grid_index": i,
                    f"{p2}_grid_index": j,
                    "shock_set_seed": seed,
                    "common_random_numbers": True,
                    **summary,
                }
            )
            progress.update(1)

    progress.close()

    out = pd.DataFrame(rows)
    save_data(out, data_file)
    return out


# ============================================================
# 9. Global Sobol sensitivity
# ============================================================


def run_global_sensitivity(
    *,
    config: RunConfig,
    seed: int = 720,
    data_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the global Sobol experiment.

    Each parameter vector receives its own Brownian shock set; common random
    numbers are deliberately not used across global samples.
    """
    if data_file is None:
        data_file = versioned_name("sensitivity_global_gbh", config.mode, "csv")

    path = DATA_DIR / data_file

    if path.exists() and not config.force_recompute:
        print(f"Loading existing global data from {path.resolve()}")
        return pd.read_csv(path)

    samples = generate_sobol_parameter_samples(
        config.n_samples_global,
        seed=seed,
    )
    sample_df = samples_to_dataframe(samples)
    n_steps = int(round(config.T / config.dt))

    rows = []
    print(
        "Running global sensitivity: "
        f"{config.n_samples_global} samples x "
        f"{config.n_replications_global} replications"
    )

    for sample_id in tqdm(range(config.n_samples_global), desc="Global samples"):
        params = sample_df.iloc[sample_id].to_dict()
        shock_seed = seed + 10000 * sample_id

        dW = generate_brownian_increments(
            n_steps=n_steps,
            n_replications=config.n_replications_global,
            dt=config.dt,
            seed=shock_seed,
        )

        batch = simulate_batch(
            params,
            dW,
            T=config.T,
            dt=config.dt,
            burn_in_fraction=config.burn_in_fraction,
            numerical_abs_share_limit=config.numerical_abs_share_limit,
            numerical_abs_debt_limit=config.numerical_abs_debt_limit,
        )

        for replication, result in batch.iterrows():
            row = {
                "sample_id": sample_id,
                "replication": int(replication),
                "shock_set_seed": shock_seed,
            }
            row.update(params)
            row.update(result.to_dict())
            rows.append(row)

    out = pd.DataFrame(rows)
    save_data(out, data_file)
    return out


# ============================================================
# 10. Logistic regression
# ============================================================


def standardize_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> Tuple[np.ndarray, dict]:
    X = df[list(columns)].to_numpy(dtype=float)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=1)
    stds[stds == 0] = 1.0

    return (X - means) / stds, {
        "means": means,
        "stds": stds,
        "columns": list(columns),
    }


def fit_logistic_regression(
    df: pd.DataFrame,
    *,
    outcome_col: str = "crisis",
    param_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Fit a standardized binomial GLM to classifiable trajectories only.

    Numerical failures and negative-debt safety exits are excluded through
    valid_for_logistic. Standard errors and confidence intervals are taken
    from the converged statsmodels GLM fit.
    """
    if param_cols is None:
        param_cols = PARAM_NAMES

    param_cols = list(param_cols)
    required = param_cols + [outcome_col, "valid_for_logistic"]

    data = df[required].dropna().copy()
    data = data[
        (data["valid_for_logistic"] == 1)
        & data[outcome_col].isin([0, 1])
    ]

    if len(data) <= len(param_cols) + 1:
        raise ValueError(
            "Not enough classifiable observations for logistic regression."
        )

    y = data[outcome_col].to_numpy(dtype=float)
    Xs, standardization = standardize_columns(
        data,
        param_cols,
    )
    X_design = sm.add_constant(
        Xs,
        prepend=True,
        has_constant="add",
    )

    model = sm.GLM(
        y,
        X_design,
        family=sm.families.Binomial(),
    )
    fitted = model.fit(
        maxiter=500,
        tol=1e-9,
    )

    if not bool(getattr(fitted, "converged", False)):
        raise RuntimeError(
            "The binomial GLM did not converge. "
            "Do not report inferential intervals."
        )

    coefficients = np.asarray(fitted.params, dtype=float)
    standard_errors = np.asarray(fitted.bse, dtype=float)
    confidence = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    terms = ["Intercept"] + param_cols

    table = pd.DataFrame(
        {
            "term": terms,
            "coef": coefficients,
            "se": standard_errors,
            "ci_low": confidence[:, 0],
            "ci_high": confidence[:, 1],
            "odds_ratio": np.exp(np.clip(coefficients, -700.0, 700.0)),
            "model_converged": True,
            "n_classifiable": len(data),
            "n_crisis": int(np.sum(y)),
            "n_noncrisis": int(len(y) - np.sum(y)),
        }
    )

    fitted.standardization = standardization
    fitted.classification_sample_size = len(data)

    return table, fitted


# ============================================================
# 11. Conditional PRCC
# ============================================================


def compute_prcc_residual_method(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute PRCC values from rank-regression residuals."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, p = X.shape
    X_rank = np.apply_along_axis(rankdata, 0, X)
    y_rank = rankdata(y)
    prcc_values = np.full(p, np.nan)

    for parameter_index in range(p):
        other_indices = [j for j in range(p) if j != parameter_index]
        controls = np.column_stack(
            [np.ones(n), X_rank[:, other_indices]]
        )

        x_target = X_rank[:, parameter_index]
        beta_x, *_ = np.linalg.lstsq(controls, x_target, rcond=None)
        beta_y, *_ = np.linalg.lstsq(controls, y_rank, rcond=None)

        residual_x = x_target - controls @ beta_x
        residual_y = y_rank - controls @ beta_y

        if np.std(residual_x) > 0 and np.std(residual_y) > 0:
            prcc_values[parameter_index] = np.corrcoef(
                residual_x,
                residual_y,
            )[0, 1]

    return prcc_values


def bootstrap_prcc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Bootstrap percentile intervals for PRCC values."""
    rng = np.random.default_rng(seed)
    n = len(y)
    p = X.shape[1]
    bootstrap_values = np.full((n_boot, p), np.nan)

    for bootstrap_index in range(n_boot):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_values[bootstrap_index] = compute_prcc_residual_method(
            X[sample_indices],
            y[sample_indices],
        )

    return {
        "ci_low": np.nanpercentile(bootstrap_values, 2.5, axis=0),
        "ci_high": np.nanpercentile(bootstrap_values, 97.5, axis=0),
    }


def compute_conditional_prcc(
    df: pd.DataFrame,
    *,
    employment_col: str = "lambda_mean_post_burn",
    param_cols: Optional[Iterable[str]] = None,
    n_boot: int = 1000,
    seed: int = 720,
) -> pd.DataFrame:
    """
    Compute employment PRCC using only economically admissible post-burn
    non-crisis trajectories.
    """
    if param_cols is None:
        param_cols = PARAM_NAMES

    param_cols = list(param_cols)
    stable = df[df["employment_eligible"] == 1].copy()
    stable = stable[param_cols + [employment_col]].dropna()

    if len(stable) < len(param_cols) + 5:
        raise ValueError(
            "Not enough employment-eligible observations to compute PRCC."
        )

    X = stable[param_cols].to_numpy(dtype=float)
    y = stable[employment_col].to_numpy(dtype=float)

    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError(
            "Employment-eligible PRCC sample contains values outside [0,1]."
        )

    prcc = compute_prcc_residual_method(X, y)
    intervals = bootstrap_prcc(
        X,
        y,
        n_boot=n_boot,
        seed=seed,
    )

    return pd.DataFrame(
        {
            "term": param_cols,
            "prcc": prcc,
            "ci_low": intervals["ci_low"],
            "ci_high": intervals["ci_high"],
            "n_employment_eligible": len(stable),
            "employment_col": employment_col,
        }
    )


# ============================================================
# 12. Plotting: local OAT
# ============================================================


def plot_local_oat_crisis(
    oat_df: pd.DataFrame,
    *,
    filename: str,
    config: RunConfig,
) -> plt.Figure:
    """Plot OAT crisis rates with Wilson uncertainty bands."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharey=True)
    axes = axes.ravel()

    for ax, name in zip(axes, PARAM_NAMES):
        data = oat_df[oat_df["parameter"] == name].sort_values("value")

        ax.fill_between(
            data["value"].to_numpy(dtype=float),
            data["crisis_ci_low"].to_numpy(dtype=float),
            data["crisis_ci_high"].to_numpy(dtype=float),
            alpha=0.18,
            linewidth=0.0,
        )
        ax.plot(
            data["value"],
            data["crisis_rate"],
            marker="o",
            linewidth=1.6,
            markersize=3.2,
        )
        ax.axvline(
            BASELINE_PARAMS[name],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        ax.set_title(PARAM_LABELS[name], fontweight="bold")
        ax.set_xlabel(PARAM_LABELS[name])
        ax.set_ylabel("Crisis rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Local One-at-a-Time Sensitivity: Crisis Rate\n"
        f"{config.n_grid_oat} grid points; "
        f"{config.n_replications_oat} common-random-number replications per point",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_figure(fig, filename)
    return fig


def plot_local_oat_employment(
    oat_df: pd.DataFrame,
    *,
    filename: str,
    config: RunConfig,
) -> plt.Figure:
    """Plot conditional employment means with Monte Carlo confidence bands."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharey=False)
    axes = axes.ravel()

    for ax, name in zip(axes, PARAM_NAMES):
        data = oat_df[oat_df["parameter"] == name].sort_values("value")
        x = data["value"].to_numpy(dtype=float)
        mean = data["noncrisis_lambda_mean"].to_numpy(dtype=float)
        lower = data["noncrisis_lambda_ci_low"].to_numpy(dtype=float)
        upper = data["noncrisis_lambda_ci_high"].to_numpy(dtype=float)
        n_eligible = data["n_employment_eligible"].to_numpy(dtype=int)

        sufficient = n_eligible >= config.min_employment_eligible
        mean = np.where(sufficient, mean, np.nan)
        lower = np.where(sufficient, lower, np.nan)
        upper = np.where(sufficient, upper, np.nan)

        valid_band = np.isfinite(lower) & np.isfinite(upper)
        if np.any(valid_band):
            ax.fill_between(
                x,
                lower,
                upper,
                where=valid_band,
                alpha=0.18,
                linewidth=0.0,
            )

        ax.plot(
            x,
            mean,
            marker="s",
            linestyle="--",
            linewidth=1.6,
            markersize=3.2,
        )
        ax.axvline(
            BASELINE_PARAMS[name],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        ax.set_title(PARAM_LABELS[name], fontweight="bold")
        ax.set_xlabel(PARAM_LABELS[name])
        ax.set_ylabel(r"Mean employment, admissible non-crisis")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Local One-at-a-Time Sensitivity: Employment Conditional on Admissible Non-Crisis\n"
        f"{config.n_grid_oat} grid points; "
        f"minimum eligible sample per point = {config.min_employment_eligible}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_figure(fig, filename)
    return fig


# ============================================================
# 13. Plotting: pairwise local surfaces
# ============================================================


def plot_pairwise_surface(
    pair_df: pd.DataFrame,
    *,
    pair: Tuple[str, str],
    value_col: str,
    filename: str,
) -> plt.Figure:
    """Plot one raw pairwise response surface and optional contour guides."""
    p1, p2 = pair
    pivot = pair_df.pivot(index=p2, columns=p1, values=value_col)

    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    response = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    if value_col == "crisis_rate":
        vmin, vmax = 0.0, 1.0
        colorbar_label = "Crisis rate"
    else:
        vmin, vmax = np.nanmin(response), np.nanmax(response)
        colorbar_label = "Mean employment, non-crisis"

    mesh = ax.pcolormesh(
        x_values,
        y_values,
        response,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )

    # Contours are visual guides through the raw grid values. No separate
    # interpolation or smoothing step is applied.
    if value_col == "crisis_rate":
        response_min = float(np.nanmin(response))
        response_max = float(np.nanmax(response))
        candidate_levels = np.array([0.2, 0.4, 0.6, 0.8])
        levels = candidate_levels[
            (candidate_levels > response_min)
            & (candidate_levels < response_max)
        ]
        if len(levels) > 0:
            ax.contour(
                x_values,
                y_values,
                response,
                levels=levels,
                colors="white",
                linewidths=0.7,
                alpha=0.65,
            )

    ax.scatter(
        [BASELINE_PARAMS[p1]],
        [BASELINE_PARAMS[p2]],
        marker="*",
        s=130,
        color="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=5,
        label="Baseline",
    )

    ax.set_xlabel(PARAM_LABELS[p1])
    ax.set_ylabel(PARAM_LABELS[p2])
    ax.set_title(
        rf"Pairwise Local Sensitivity: {PARAM_LABELS[p1]} vs {PARAM_LABELS[p2]}",
        fontweight="bold",
    )

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure(fig, filename)
    return fig


# ============================================================
# 14. Plotting: global logistic regression and PRCC
# ============================================================


def plot_global_logistic_and_prcc(
    logistic_table: pd.DataFrame,
    prcc_table: pd.DataFrame,
    *,
    filename: str,
) -> plt.Figure:
    """Plot standardized logistic coefficients and conditional PRCC values."""
    logistic = logistic_table[logistic_table["term"] != "Intercept"].copy()
    prcc = prcc_table.copy()
    order = PARAM_NAMES[::-1]

    logistic["term"] = pd.Categorical(
        logistic["term"],
        categories=order,
        ordered=True,
    )
    prcc["term"] = pd.Categorical(
        prcc["term"],
        categories=order,
        ordered=True,
    )

    logistic = logistic.sort_values("term")
    prcc = prcc.sort_values("term")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), sharey=True)

    y_positions = np.arange(len(logistic))
    logistic_left_error = np.maximum(
        logistic["coef"].to_numpy(dtype=float)
        - logistic["ci_low"].to_numpy(dtype=float),
        0.0,
    )
    logistic_right_error = np.maximum(
        logistic["ci_high"].to_numpy(dtype=float)
        - logistic["coef"].to_numpy(dtype=float),
        0.0,
    )
    axes[0].errorbar(
        logistic["coef"],
        y_positions,
        xerr=[logistic_left_error, logistic_right_error],
        fmt="o",
        markersize=5.5,
        capsize=3.5,
        color="black",
        ecolor="gray",
        linewidth=1.5,
    )
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(
        [PARAM_LABELS[name] for name in logistic["term"].astype(str)]
    )
    axes[0].set_xlabel("Standardized logistic coefficient")
    axes[0].set_title("(a) Crisis classification", fontweight="bold")
    axes[0].grid(True, axis="x", alpha=0.25)

    y_positions = np.arange(len(prcc))
    # Percentile bootstrap intervals do not mathematically have to contain
    # the original point estimate in very small samples. Clamp plotting
    # lengths at zero so Matplotlib always receives valid non-negative xerr.
    prcc_left_error = np.maximum(
        prcc["prcc"].to_numpy(dtype=float)
        - prcc["ci_low"].to_numpy(dtype=float),
        0.0,
    )
    prcc_right_error = np.maximum(
        prcc["ci_high"].to_numpy(dtype=float)
        - prcc["prcc"].to_numpy(dtype=float),
        0.0,
    )
    axes[1].errorbar(
        prcc["prcc"],
        y_positions,
        xerr=[prcc_left_error, prcc_right_error],
        fmt="o",
        markersize=5.5,
        capsize=3.5,
        color="black",
        ecolor="gray",
        linewidth=1.5,
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(
        [PARAM_LABELS[name] for name in prcc["term"].astype(str)]
    )
    axes[1].set_xlabel("PRCC")
    axes[1].set_title(
        "(b) Employment conditional on admissible non-crisis",
        fontweight="bold",
    )
    axes[1].grid(True, axis="x", alpha=0.25)

    n_eligible = int(prcc_table["n_employment_eligible"].iloc[0])
    fig.text(
        0.5,
        0.01,
        r"PRCC outcome: discrete mean employment; "
        rf"admissible non-crisis sample size $n={n_eligible}$.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    save_figure(fig, filename)
    return fig


# ============================================================
# 15. Execution and summaries
# ============================================================


def print_global_summary(
    global_df: pd.DataFrame,
    logistic_table: pd.DataFrame,
    prcc_table: pd.DataFrame,
) -> None:
    """Print separated global outcome counts and fitted summaries."""
    valid = global_df["valid_for_logistic"] == 1
    n_valid = int(valid.sum())
    n_crisis = int(global_df.loc[valid, "crisis"].sum())

    print("\n=== Global classification summary ===")
    print(f"Total trajectories: {len(global_df)}")
    print(f"Classifiable trajectories: {n_valid}")
    print(
        "Crisis rate among classifiable trajectories: "
        f"{(n_crisis / n_valid if n_valid else np.nan):.4f}"
    )
    print(f"Terminal crises: {int(global_df['terminal_crisis'].sum())}")
    print(f"Divergent crises: {int(global_df['divergent_crisis'].sum())}")
    print(f"Non-crisis count: {int(global_df['noncrisis'].sum())}")
    print(
        "Employment-eligible non-crisis count: "
        f"{int(global_df['employment_eligible'].sum())}"
    )
    print(
        "Numerical failures (NaN/Inf): "
        f"{int(global_df['numerical_failure'].sum())}"
    )
    print(
        "Negative-debt safety exits: "
        f"{int(global_df['negative_debt_safety_exit'].sum())}"
    )

    print("\n=== Binomial GLM coefficients ===")
    print(logistic_table.to_string(index=False))

    print("\n=== PRCC conditional on admissible non-crisis ===")
    print(prcc_table.to_string(index=False))


def save_methodology_metadata(config: RunConfig) -> Path:
    n_steps = int(round(config.T / config.dt))
    burn_in_steps = int(round(config.burn_in_fraction * n_steps))

    payload = {
        "methodology_version": METHODOLOGY_VERSION,
        "configuration": asdict(config),
        "parameter_names": PARAM_NAMES,
        "parameter_ranges": {
            name: list(bounds) for name, bounds in PARAM_RANGES.items()
        },
        "baseline_parameters": BASELINE_PARAMS,
        "initial_condition": DEFAULT_INITIAL_CONDITION,
        "terminal_noncrisis_definition": {
            "omega_min": TERMINAL_OMEGA_MIN,
            "omega_max": TERMINAL_OMEGA_MAX,
            "lambda_min": TERMINAL_LAMBDA_MIN,
            "lambda_max": TERMINAL_LAMBDA_MAX,
            "ell_max": TERMINAL_ELL_MAX,
        },
        "post_burn_employment": {
            "type": "discrete arithmetic mean",
            "total_steps": n_steps,
            "burn_in_steps": burn_in_steps,
            "included_indices": f"n={burn_in_steps + 1},...,{n_steps}",
            "eligibility": {
                "terminal_noncrisis": True,
                "post_burn_omega_range": [0.0, 1.0],
                "post_burn_lambda_range": [0.0, 1.0],
                "complete_post_burn_sample": True,
                "minimum_oat_sample": config.min_employment_eligible,
            },
        },
        "outcome_categories": {
            "noncrisis": "completed horizon and met terminal bounds",
            "terminal_crisis": "completed horizon and failed terminal bounds",
            "divergent_crisis": (
                "finite positive-debt or share-divergence safety threshold"
            ),
            "numerical_failure": "NaN or Inf proposed update",
            "negative_debt_safety_exit": (
                "finite large negative-debt cutoff; excluded from crisis GLM"
            ),
        },
        "logistic_model": {
            "implementation": "statsmodels binomial GLM",
            "sample": "valid_for_logistic only",
        },
        "oat_common_random_numbers": True,
        "pairwise_common_random_numbers": True,
        "global_common_random_numbers_across_samples": False,
        "projection_or_clipping": False,
        "positive_part_share_diffusions": True,
    }

    return save_json(
        payload,
        versioned_name("sensitivity_methodology", config.mode, "json"),
    )


def main(mode: str = "final") -> dict:
    """
    Run the revised analysis in the requested order:

        OAT -> pairwise -> global.
    """
    config = make_config(mode)

    print("\n========== Revised Sensitivity Analysis ==========")
    print(f"Methodology version: {METHODOLOGY_VERSION}")
    print(f"Run mode: {config.mode}")
    print(
        f"OAT: {config.n_grid_oat} points x "
        f"{config.n_replications_oat} replications"
    )
    print(
        f"Pairwise: {config.n_grid_pairwise} x "
        f"{config.n_grid_pairwise} cells x "
        f"{config.n_replications_pairwise} replications"
    )
    print(
        f"Global: {config.n_samples_global} Sobol samples x "
        f"{config.n_replications_global} replications"
    )
    print("Order: OAT -> pairwise -> global")
    print("===================================================\n")

    save_methodology_metadata(config)

    # --------------------------------------------------------
    # 1. Local one-at-a-time sensitivity
    # --------------------------------------------------------
    oat_df = run_local_oat_sensitivity(
        config=config,
        seed=1234,
    )

    plot_local_oat_crisis(
        oat_df,
        filename=f"sensitivity_local_oat_crisis_{config.mode}.png",
        config=config,
    )
    plot_local_oat_employment(
        oat_df,
        filename=f"sensitivity_local_oat_employment_{config.mode}.png",
        config=config,
    )

    # --------------------------------------------------------
    # 2. Selected pairwise local sensitivity
    # --------------------------------------------------------
    pair_m_eta = run_pairwise_sensitivity(
        pair=("m", "eta_p"),
        config=config,
        seed=3000,
    )
    plot_pairwise_surface(
        pair_m_eta,
        pair=("m", "eta_p"),
        value_col="crisis_rate",
        filename=f"sensitivity_pair_m_eta_p_crisis_{config.mode}.png",
    )

    pair_sigma_r = run_pairwise_sensitivity(
        pair=("sigma_ell", "r"),
        config=config,
        seed=4000,
    )
    plot_pairwise_surface(
        pair_sigma_r,
        pair=("sigma_ell", "r"),
        value_col="crisis_rate",
        filename=f"sensitivity_pair_sigma_ell_r_crisis_{config.mode}.png",
    )

    # --------------------------------------------------------
    # 3. Global sensitivity
    # --------------------------------------------------------
    global_df = run_global_sensitivity(
        config=config,
        seed=720,
    )

    logistic_table, logistic_fit = fit_logistic_regression(
        global_df,
        outcome_col="crisis",
        param_cols=PARAM_NAMES,
    )
    prcc_table = compute_conditional_prcc(
        global_df,
        employment_col="lambda_mean_post_burn",
        param_cols=PARAM_NAMES,
        n_boot=config.n_boot_prcc,
        seed=720,
    )

    save_data(
        logistic_table,
        versioned_name("logistic_coefficients", config.mode, "csv"),
    )
    save_data(
        prcc_table,
        versioned_name("prcc_employment_noncrisis", config.mode, "csv"),
    )

    print_global_summary(global_df, logistic_table, prcc_table)

    plot_global_logistic_and_prcc(
        logistic_table,
        prcc_table,
        filename=f"sensitivity_global_logistic_prcc_{config.mode}.png",
    )

    print("\nAll revised sensitivity outputs created.")
    print("Figures folder:", FIG_DIR.resolve())
    print("Data folder:", DATA_DIR.resolve())

    return {
        "config": config,
        "oat_df": oat_df,
        "pair_m_eta": pair_m_eta,
        "pair_sigma_r": pair_sigma_r,
        "global_df": global_df,
        "logistic_table": logistic_table,
        "logistic_fit": logistic_fit,
        "prcc_table": prcc_table,
    }


if __name__ == "__main__":
    # Use "draft" first to verify the installation and output paths.
    # Then change to "final" for thesis-quality results.
    results = main(mode="final")
    plt.show()
