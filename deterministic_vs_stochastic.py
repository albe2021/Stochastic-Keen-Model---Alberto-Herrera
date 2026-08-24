# correlations.py
#
# Persistence and lead-lag diagnostics for the baseline stochastic path.
#
# Positive lags in the cross-correlation panels mean that the first
# variable leads the second:
#
#     Corr(x_t, y_{t+h}),  h > 0.
#
# The calculations reproduce the same unprojected positive-part
# baseline realization directly in this standalone script.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from keen_model_functions import KeenModel


# ============================================================
# Baseline calibration and simulation settings
# ============================================================
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

BASELINE_SEED = 128
BASELINE_INITIAL_CONDITION = (0.9, 0.9, 0.3)
T_SIM = 200.0
DT = 0.01


# ============================================================
# Standalone thesis colour convention
# ============================================================
COLOR_OMEGA = "#1f77b4"       # blue
COLOR_LAMBDA = "#d62728"      # red
COLOR_ELL = "#2ca02c"         # green
COLOR_PROFIT = "#1f77b4"      # blue
COLOR_REFERENCE = "#666666"   # grey


def simulate_baseline_path(
    seed: int = BASELINE_SEED,
    T: float = T_SIM,
    dt: float = DT,
    x0: tuple[float, float, float] = BASELINE_INITIAL_CONDITION,
) -> tuple[KeenModel, np.ndarray, np.ndarray]:
    """
    Simulate the same unprojected positive-part baseline path used in
    the baseline-dynamics figure.

    This file is fully standalone and does not import settings, colours,
    or simulation functions from baseline_dynamics.py.
    """
    model = KeenModel(BASELINE_PARAMS)
    t, states = model.simulate_path(
        x0=x0,
        T=T,
        dt=dt,
        seed=seed,
    )

    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "The baseline simulation produced non-finite states."
        )

    return model, t, states


# ============================================================
# Analysis settings
# ============================================================
BURN_IN = 20.0
ANALYSIS_INTERVAL = 0.10
MAX_ACF_LAG = 30.0
MAX_CCF_LAG = 20.0
ACF_REFERENCE_LEVEL = 0.50

OUTPUT_DIRECTORY = Path("Figures")
DATA_DIRECTORY = Path("NumericalData")
OUTPUT_STEM = "correlations"


# Cross-correlation panels follow the colour of the leading variable.
COLOR_PROFIT_LAMBDA = COLOR_PROFIT
COLOR_LAMBDA_ELL = COLOR_ELL


# ============================================================
# Data preparation
# ============================================================
def downsample_after_burn_in(
    t: np.ndarray,
    values: np.ndarray,
    *,
    burn_in: float = BURN_IN,
    interval: float = ANALYSIS_INTERVAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the initial transient and sample at a fixed interval."""
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")
    if interval <= 0:
        raise ValueError("interval must be strictly positive.")
    if burn_in >= t[-1]:
        raise ValueError("burn_in must be smaller than the horizon.")

    base_dt = float(t[1] - t[0])
    step = int(round(interval / base_dt))

    if step < 1 or not np.isclose(
        step * base_dt,
        interval,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "ANALYSIS_INTERVAL must be an integer multiple of the "
            "simulation time step."
        )

    start = int(np.searchsorted(t, burn_in, side="left"))
    indices = np.arange(start, len(t), step, dtype=int)

    return t[indices], np.asarray(values)[indices]


# ============================================================
# Correlation estimators
# ============================================================
def autocorrelation(
    series: np.ndarray,
    max_lag_steps: int,
) -> np.ndarray:
    """
    Estimate the sample autocorrelation for lags 0,...,max_lag_steps.

    The denominator is the total centered sum of squares. This is the
    conventional bounded, biased estimator used for descriptive ACF plots.
    """
    x = np.asarray(series, dtype=float)

    if x.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if max_lag_steps < 0:
        raise ValueError("max_lag_steps must be nonnegative.")
    if max_lag_steps >= len(x):
        raise ValueError("max_lag_steps must be smaller than the sample.")

    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))

    if denominator <= 0:
        raise ValueError("The series has zero sample variance.")

    result = np.empty(max_lag_steps + 1, dtype=float)

    for lag in range(max_lag_steps + 1):
        if lag == 0:
            result[lag] = 1.0
        else:
            result[lag] = (
                np.dot(centered[:-lag], centered[lag:]) / denominator
            )

    return result


def first_level_crossing(
    lags: np.ndarray,
    correlations: np.ndarray,
    level: float = ACF_REFERENCE_LEVEL,
) -> float | None:
    """
    Return the first downward crossing of a reference level.

    Linear interpolation is used between the two surrounding lag points.
    This is a descriptive persistence measure, not a structural half-life.
    """
    lags = np.asarray(lags, dtype=float)
    correlations = np.asarray(correlations, dtype=float)

    for index in range(1, len(correlations)):
        previous = correlations[index - 1]
        current = correlations[index]

        if previous > level and current <= level:
            denominator = current - previous

            if denominator == 0:
                return float(lags[index])

            fraction = (level - previous) / denominator
            return float(
                lags[index - 1]
                + fraction * (lags[index] - lags[index - 1])
            )

    return None


def lagged_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
    max_lag_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Corr(first_t, second_{t+h}) over symmetric lags.

    Positive h means that the first variable leads the second.
    Negative h means that the second variable leads the first.
    """
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both series must be one-dimensional.")
    if len(x) != len(y):
        raise ValueError("The two series must have the same length.")
    if max_lag_steps < 0:
        raise ValueError("max_lag_steps must be nonnegative.")
    if max_lag_steps >= len(x) - 2:
        raise ValueError("max_lag_steps is too large for the sample.")

    integer_lags = np.arange(
        -max_lag_steps,
        max_lag_steps + 1,
        dtype=int,
    )
    correlations = np.empty(len(integer_lags), dtype=float)

    for index, lag in enumerate(integer_lags):
        if lag > 0:
            paired_x = x[:-lag]
            paired_y = y[lag:]
        elif lag < 0:
            shift = -lag
            paired_x = x[shift:]
            paired_y = y[:-shift]
        else:
            paired_x = x
            paired_y = y

        if np.std(paired_x) == 0 or np.std(paired_y) == 0:
            correlations[index] = np.nan
        else:
            correlations[index] = np.corrcoef(
                paired_x,
                paired_y,
            )[0, 1]

    return integer_lags, correlations


def strongest_nonnegative_lag(
    integer_lags: np.ndarray,
    correlations: np.ndarray,
) -> tuple[int, float]:
    """
    Return the nonnegative lag with the largest absolute correlation.

    The sign of the returned correlation is retained.
    """
    mask = integer_lags >= 0
    candidate_indices = np.flatnonzero(mask)
    candidate_values = correlations[mask]

    local_index = int(np.nanargmax(np.abs(candidate_values)))
    full_index = int(candidate_indices[local_index])

    return int(integer_lags[full_index]), float(correlations[full_index])


# ============================================================
# Analysis
# ============================================================
def calculate_correlation_diagnostics() -> dict[str, object]:
    """Simulate the baseline path and calculate all diagnostics."""
    model, t, states = simulate_baseline_path(
        seed=BASELINE_SEED,
        T=T_SIM,
        dt=DT,
    )

    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]
    profit = np.asarray(model.profit_share(omega, ell), dtype=float)

    analysis_time, analysis_values = downsample_after_burn_in(
        t,
        np.column_stack([omega, lam, ell, profit]),
    )

    omega_a = analysis_values[:, 0]
    lambda_a = analysis_values[:, 1]
    ell_a = analysis_values[:, 2]
    profit_a = analysis_values[:, 3]

    analysis_dt = float(analysis_time[1] - analysis_time[0])
    max_acf_steps = int(round(MAX_ACF_LAG / analysis_dt))
    max_ccf_steps = int(round(MAX_CCF_LAG / analysis_dt))

    acf_lag_steps = np.arange(max_acf_steps + 1, dtype=int)
    acf_lags = acf_lag_steps * analysis_dt

    acf_omega = autocorrelation(omega_a, max_acf_steps)
    acf_lambda = autocorrelation(lambda_a, max_acf_steps)
    acf_ell = autocorrelation(ell_a, max_acf_steps)

    omega_crossing = first_level_crossing(acf_lags, acf_omega)
    lambda_crossing = first_level_crossing(acf_lags, acf_lambda)
    ell_crossing = first_level_crossing(acf_lags, acf_ell)

    ccf_lag_steps, ccf_profit_lambda = lagged_cross_correlation(
        profit_a,
        lambda_a,
        max_ccf_steps,
    )
    _, ccf_lambda_ell = lagged_cross_correlation(
        lambda_a,
        ell_a,
        max_ccf_steps,
    )
    ccf_lags = ccf_lag_steps * analysis_dt

    profit_lambda_step, profit_lambda_corr = strongest_nonnegative_lag(
        ccf_lag_steps,
        ccf_profit_lambda,
    )
    lambda_ell_step, lambda_ell_corr = strongest_nonnegative_lag(
        ccf_lag_steps,
        ccf_lambda_ell,
    )

    return {
        "model": model,
        "time": t,
        "states": states,
        "analysis_time": analysis_time,
        "analysis_dt": analysis_dt,
        "acf_lags": acf_lags,
        "acf_omega": acf_omega,
        "acf_lambda": acf_lambda,
        "acf_ell": acf_ell,
        "omega_crossing": omega_crossing,
        "lambda_crossing": lambda_crossing,
        "ell_crossing": ell_crossing,
        "ccf_lags": ccf_lags,
        "ccf_profit_lambda": ccf_profit_lambda,
        "ccf_lambda_ell": ccf_lambda_ell,
        "profit_lambda_peak_lag": profit_lambda_step * analysis_dt,
        "profit_lambda_peak_correlation": profit_lambda_corr,
        "lambda_ell_peak_lag": lambda_ell_step * analysis_dt,
        "lambda_ell_peak_correlation": lambda_ell_corr,
    }


# ============================================================
# Output
# ============================================================
def save_outputs(results: dict[str, object]) -> tuple[Path, Path, Path]:
    """Save ACF, CCF, and summary statistics."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    acf_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_acf.csv"
    ccf_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_cross_correlations.csv"
    json_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_statistics.json"

    acf_table = np.column_stack(
        [
            results["acf_lags"],
            results["acf_omega"],
            results["acf_lambda"],
            results["acf_ell"],
        ]
    )
    np.savetxt(
        acf_path,
        acf_table,
        delimiter=",",
        header="lag_years,acf_omega,acf_lambda,acf_ell",
        comments="",
    )

    ccf_table = np.column_stack(
        [
            results["ccf_lags"],
            results["ccf_profit_lambda"],
            results["ccf_lambda_ell"],
        ]
    )
    np.savetxt(
        ccf_path,
        ccf_table,
        delimiter=",",
        header=(
            "lag_years,"
            "corr_profit_t_lambda_t_plus_lag,"
            "corr_lambda_t_ell_t_plus_lag"
        ),
        comments="",
    )

    summary = {
        "seed": BASELINE_SEED,
        "simulation_horizon": T_SIM,
        "simulation_dt": DT,
        "burn_in": BURN_IN,
        "analysis_interval": results["analysis_dt"],
        "analysis_observations": len(results["analysis_time"]),
        "acf_reference_level": ACF_REFERENCE_LEVEL,
        "omega_first_0_5_crossing": results["omega_crossing"],
        "lambda_first_0_5_crossing": results["lambda_crossing"],
        "ell_first_0_5_crossing": results["ell_crossing"],
        "profit_lambda_peak_nonnegative_lag": (
            results["profit_lambda_peak_lag"]
        ),
        "profit_lambda_peak_correlation": (
            results["profit_lambda_peak_correlation"]
        ),
        "lambda_ell_peak_nonnegative_lag": (
            results["lambda_ell_peak_lag"]
        ),
        "lambda_ell_peak_correlation": (
            results["lambda_ell_peak_correlation"]
        ),
        "positive_lag_definition": (
            "Corr(first_t, second_{t+h}); h>0 means first leads second."
        ),
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return acf_path, ccf_path, json_path


def create_correlations_figure(
    show: bool = True,
) -> tuple[plt.Figure, dict[str, object]]:
    """Generate the five-panel persistence and lead-lag figure."""
    results = calculate_correlation_diagnostics()

    fig = plt.figure(figsize=(14, 8))
    layout = gridspec.GridSpec(2, 6, figure=fig)

    ax_omega = fig.add_subplot(layout[0, 0:2])
    ax_lambda = fig.add_subplot(layout[0, 2:4])
    ax_ell = fig.add_subplot(layout[0, 4:6])
    ax_profit_lambda = fig.add_subplot(layout[1, 0:3])
    ax_lambda_ell = fig.add_subplot(layout[1, 3:6])

    acf_panels = [
        (
            ax_omega,
            results["acf_omega"],
            results["omega_crossing"],
            r"$\omega_t$",
            "(a) Wage-Share Autocorrelation",
            COLOR_OMEGA,
        ),
        (
            ax_lambda,
            results["acf_lambda"],
            results["lambda_crossing"],
            r"$\lambda_t$",
            "(b) Employment Autocorrelation",
            COLOR_LAMBDA,
        ),
        (
            ax_ell,
            results["acf_ell"],
            results["ell_crossing"],
            r"$\ell_t$",
            "(c) Net-Debt Autocorrelation",
            COLOR_ELL,
        ),
    ]

    for axis, values, crossing, label, title, color in acf_panels:
        axis.plot(
            results["acf_lags"],
            values,
            color=color,
            linewidth=1.6,
            label=label,
        )
        axis.axhline(
            0.0,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        axis.axhline(
            ACF_REFERENCE_LEVEL,
            color=COLOR_REFERENCE,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )
        if crossing is not None:
            axis.axvline(
                crossing,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
            )
            axis.scatter(
                [crossing],
                [ACF_REFERENCE_LEVEL],
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=38,
                zorder=4,
                label=f"First 0.5 crossing: {crossing:.2f} years",
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Lag (years)")
        axis.set_ylabel("Autocorrelation")
        axis.set_xlim(0.0, MAX_ACF_LAG)
        axis.set_ylim(-1.0, 1.05)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", frameon=True)

    ax_profit_lambda.plot(
        results["ccf_lags"],
        results["ccf_profit_lambda"],
        color=COLOR_PROFIT_LAMBDA,
        linewidth=1.6,
        label=r"$\mathrm{Corr}(\pi_t,\lambda_{t+h})$",
    )
    ax_profit_lambda.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax_profit_lambda.axvline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax_profit_lambda.axvline(
        results["profit_lambda_peak_lag"],
        color=COLOR_PROFIT_LAMBDA,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
    )
    ax_profit_lambda.scatter(
        [results["profit_lambda_peak_lag"]],
        [results["profit_lambda_peak_correlation"]],
        color=COLOR_PROFIT_LAMBDA,
        edgecolor="white",
        linewidth=0.6,
        s=42,
        zorder=4,
        label=(
            f"Largest |correlation| for $h\\geq0$: "
            f"{results['profit_lambda_peak_correlation']:.3f} "
            f"at {results['profit_lambda_peak_lag']:.1f} years"
        ),
    )
    ax_profit_lambda.set_title(
        "(d) Profit Share and Future Employment",
        fontweight="bold",
    )
    ax_profit_lambda.set_xlabel(
        "Lag $h$ (years; positive means profit share leads)"
    )
    ax_profit_lambda.set_ylabel("Cross-correlation")
    ax_profit_lambda.set_xlim(-MAX_CCF_LAG, MAX_CCF_LAG)
    ax_profit_lambda.set_ylim(-1.0, 1.0)
    ax_profit_lambda.grid(True, alpha=0.3)
    ax_profit_lambda.legend(loc="best", frameon=True)

    ax_lambda_ell.plot(
        results["ccf_lags"],
        results["ccf_lambda_ell"],
        color=COLOR_LAMBDA_ELL,
        linewidth=1.6,
        label=r"$\mathrm{Corr}(\lambda_t,\ell_{t+h})$",
    )
    ax_lambda_ell.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax_lambda_ell.axvline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax_lambda_ell.axvline(
        results["lambda_ell_peak_lag"],
        color=COLOR_LAMBDA_ELL,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
    )
    ax_lambda_ell.scatter(
        [results["lambda_ell_peak_lag"]],
        [results["lambda_ell_peak_correlation"]],
        color=COLOR_LAMBDA_ELL,
        edgecolor="white",
        linewidth=0.6,
        s=42,
        zorder=4,
        label=(
            f"Largest |correlation| for $h\\geq0$: "
            f"{results['lambda_ell_peak_correlation']:.3f} "
            f"at {results['lambda_ell_peak_lag']:.1f} years"
        ),
    )
    ax_lambda_ell.set_title(
        "(e) Employment and Future Net Debt",
        fontweight="bold",
    )
    ax_lambda_ell.set_xlabel(
        "Lag $h$ (years; positive means employment leads)"
    )
    ax_lambda_ell.set_ylabel("Cross-correlation")
    ax_lambda_ell.set_xlim(-MAX_CCF_LAG, MAX_CCF_LAG)
    ax_lambda_ell.set_ylim(-1.0, 1.0)
    ax_lambda_ell.grid(True, alpha=0.3)
    ax_lambda_ell.legend(loc="best", frameon=True)

    fig.suptitle(
        "Baseline Persistence and Lead-Lag Diagnostics\n"
        f"Seed {BASELINE_SEED}; burn-in {BURN_IN:g} years; "
        f"analysis interval {ANALYSIS_INTERVAL:g} years",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    acf_path, ccf_path, json_path = save_outputs(results)
    print_summary(results)
    print(f"\nSaved figure: {png_path}")
    print(f"Saved PDF:    {pdf_path}")
    print(f"Saved ACF:    {acf_path}")
    print(f"Saved CCF:    {ccf_path}")
    print(f"Saved stats:  {json_path}")

    if show:
        plt.show()

    return fig, results


def format_crossing(value: float | None) -> str:
    """Format an optional ACF crossing."""
    return "not observed" if value is None else f"{value:.3f} years"


def print_summary(results: dict[str, object]) -> None:
    """Print values needed for the revised Section 5.2."""
    print("\nBaseline correlation diagnostics")
    print("=" * 58)
    print(f"Seed: {BASELINE_SEED}")
    print(f"Burn-in: {BURN_IN:.2f} years")
    print(f"Analysis interval: {results['analysis_dt']:.4f} years")
    print(f"Observations: {len(results['analysis_time'])}")

    print("\nFirst ACF crossing of 0.5")
    print("-" * 58)
    print(f"omega:  {format_crossing(results['omega_crossing'])}")
    print(f"lambda: {format_crossing(results['lambda_crossing'])}")
    print(f"ell:    {format_crossing(results['ell_crossing'])}")

    print("\nLead-lag diagnostics")
    print("-" * 58)
    print(
        "Profit share -> future employment: "
        f"corr = {results['profit_lambda_peak_correlation']:.6f}, "
        f"lag = {results['profit_lambda_peak_lag']:.2f} years"
    )
    print(
        "Employment -> future net debt: "
        f"corr = {results['lambda_ell_peak_correlation']:.6f}, "
        f"lag = {results['lambda_ell_peak_lag']:.2f} years"
    )
    print(
        "\nPositive lag h means Corr(first_t, second_{t+h}); "
        "the first variable leads the second."
    )
    print(
        "The 0.5-crossing lags and cross-correlations are descriptive "
        "statistics for this realization, not causal estimates."
    )


if __name__ == "__main__":
    create_correlations_figure(show=True)
