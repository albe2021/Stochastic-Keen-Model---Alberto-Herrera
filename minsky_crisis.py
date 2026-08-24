# minsky_crisis.py
#
# High-initial-debt financial fragility under the stochastic
# dividend-inclusive Keen model.
#
# Figure structure:
#   (a) state variables over time
#   (b) lightly smoothed 3D state‑space phase path (ω, λ, ℓ)
#   (c) financial indicators with very light regime shading
#
# The simulation uses the unprojected positive‑part stochastic system
# implemented in keen_model_functions.py. All diagnostics are calculated
# from the raw path. Smoothing is used only in panel (b) for readability.

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from keen_model_functions import KeenModel


# ============================================================
# Baseline calibration
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


# ============================================================
# Scenario configuration
# ============================================================
HORIZON = 200.0
DT = 0.01
SEED = 484
INITIAL_STATE = (0.578, 0.675, 1.53)

# Plotting controls only. Raw data are used for all diagnostics.
TIME_SERIES_INTERVAL = 0.10
PHASE_INTERVAL = 0.10
PHASE_SMOOTH_WINDOW = 41  # 1.01 years at dt = 0.01

# Regime shading is deliberately faint and placed behind the lines.
SPECULATIVE_SHADE = "#E3B04B"
PONZI_SHADE = "#C94C4C"
SPECULATIVE_ALPHA = 0.075
PONZI_ALPHA = 0.11

OUTPUT_DIRECTORY = Path("Figures")
DATA_DIRECTORY = Path("NumericalData")
OUTPUT_STEM = "minsky_crisis"


# ============================================================
# Standalone thesis colour convention
# ============================================================
# State variables
COLOR_OMEGA = "#1f77b4"          # blue
COLOR_LAMBDA = "#d62728"         # red
COLOR_ELL = "#2ca02c"            # green

# Financial indicators
COLOR_PROFIT = "#1f77b4"         # blue
COLOR_DEBT_SERVICE = "#9467bd"   # purple
COLOR_FINANCING_GAP = "#ff7f0e"  # orange

# Plot annotations
COLOR_REFERENCE = "#666666"      # grey
COLOR_RAW_PATH = "#8c8c8c"
COLOR_START = COLOR_OMEGA
COLOR_END = COLOR_FINANCING_GAP
PHASE_CMAP = "viridis"


# ============================================================
# Utilities
# ============================================================
def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average with edge padding."""
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values.copy()
    if window > len(values):
        raise ValueError("The smoothing window exceeds the data length.")

    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def sample_indices(t: np.ndarray, interval: float) -> np.ndarray:
    """Return approximately equally spaced indices for plotting."""
    if interval <= 0:
        raise ValueError("interval must be strictly positive.")

    dt = float(t[1] - t[0])
    step = int(round(interval / dt))

    if step < 1 or not np.isclose(
        step * dt,
        interval,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "The plotting interval must be an integer multiple of dt."
        )

    indices = np.arange(0, len(t), step, dtype=int)

    if indices[-1] != len(t) - 1:
        indices = np.append(indices, len(t) - 1)

    return indices


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True regions as half-open intervals [start, end)."""
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if len(mask) == 0:
        return []

    changes = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1) + 1

    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    return list(zip(starts.tolist(), ends.tolist()))


def longest_duration(mask: np.ndarray, dt: float) -> float:
    """Return the longest contiguous True duration."""
    regions = contiguous_true_regions(mask)
    if not regions:
        return 0.0
    return float(max((end - start) * dt for start, end in regions))


def first_true_time(t: np.ndarray, mask: np.ndarray) -> float | None:
    """Return the first time for which mask is True."""
    indices = np.flatnonzero(mask)
    return float(t[indices[0]]) if len(indices) else None


def _pad_limits(*arrays, pad_frac: float = 0.05) -> tuple[float, float]:
    """Return (vmin, vmax) with symmetric padding for a collection of arrays."""
    arr = np.concatenate([np.ravel(a) for a in arrays])
    lo = np.min(arr)
    hi = np.max(arr)
    span = hi - lo
    if span <= 0:
        span = 1.0
    pad = pad_frac * span
    return lo - pad, hi + pad


def add_regime_shading(
    ax: plt.Axes,
    t: np.ndarray,
    speculative: np.ndarray,
    ponzi: np.ndarray,
) -> None:
    """
    Shade speculative and Ponzi intervals lightly behind the plotted lines.

    No hatching is used. The low opacity prevents the spans from obscuring
    the financial indicators.
    """
    for start, end in contiguous_true_regions(speculative):
        ax.axvspan(
            t[start],
            t[min(end, len(t) - 1)],
            facecolor=SPECULATIVE_SHADE,
            alpha=SPECULATIVE_ALPHA,
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )

    for start, end in contiguous_true_regions(ponzi):
        ax.axvspan(
            t[start],
            t[min(end, len(t) - 1)],
            facecolor=PONZI_SHADE,
            alpha=PONZI_ALPHA,
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )


# ============================================================
# Simulation and diagnostics
# ============================================================
def simulate_minsky_crisis(
    *,
    seed: int = SEED,
    horizon: float = HORIZON,
    dt: float = DT,
    initial_state: Sequence[float] = INITIAL_STATE,
) -> tuple[KeenModel, np.ndarray, np.ndarray]:
    """
    Simulate the high-initial-debt scenario without an additional shock.

    No projection, clipping, reflection, or epsilon floor is imposed.
    """
    model = KeenModel(BASELINE_PARAMS)

    t, states = model.simulate_path(
        x0=initial_state,
        T=horizon,
        dt=dt,
        seed=seed,
    )

    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "The high-initial-debt simulation produced non-finite states."
        )

    return model, t, states


def calculate_derived_variables(
    model: KeenModel,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculate financial indicators and Minsky-style regime classifications.

    Let
        P_t = omega_t + kappa(pi_t) - 1,
        F_t = P_t + Delta(pi_t).

    Classification:
        Hedge:       F_t <= 0
        Speculative: P_t < 0 < F_t
        Ponzi:       P_t >= 0
    """
    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]

    profit = np.asarray(model.profit_share(omega, ell), dtype=float)
    investment = np.asarray(model.kappa(profit), dtype=float)
    dividends = np.asarray(model.dividend(profit), dtype=float)

    pre_dividend_gap = omega + investment - 1.0
    financing_gap = pre_dividend_gap + dividends
    debt_service = model.params["r"] * ell
    inflation = np.asarray(model.inflation(omega), dtype=float)
    interest_growth_gap = np.asarray(
        model.interest_growth_gap(omega, ell),
        dtype=float,
    )

    hedge = financing_gap <= 0.0
    speculative = (pre_dividend_gap < 0.0) & (financing_gap > 0.0)
    ponzi = pre_dividend_gap >= 0.0

    outside_economic_region = (
        (omega < 0.0)
        | (omega > 1.0)
        | (lam < 0.0)
        | (lam > 1.0)
    )

    return {
        "profit": profit,
        "investment": investment,
        "dividends": dividends,
        "pre_dividend_gap": pre_dividend_gap,
        "financing_gap": financing_gap,
        "debt_service": debt_service,
        "inflation": inflation,
        "interest_growth_gap": interest_growth_gap,
        "hedge": hedge,
        "speculative": speculative,
        "ponzi": ponzi,
        "outside_economic_region": outside_economic_region,
    }


def summarize_scenario(
    t: np.ndarray,
    states: np.ndarray,
    derived: dict[str, np.ndarray],
) -> dict[str, float | int | bool | None]:
    """Calculate statistics needed for the revised Section 5.3."""
    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]
    dt = float(t[1] - t[0])

    return {
        "seed": SEED,
        "horizon": float(t[-1]),
        "dt": dt,
        "initial_omega": float(INITIAL_STATE[0]),
        "initial_lambda": float(INITIAL_STATE[1]),
        "initial_ell": float(INITIAL_STATE[2]),
        "omega_min": float(np.min(omega)),
        "omega_max": float(np.max(omega)),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "ell_min": float(np.min(ell)),
        "ell_max": float(np.max(ell)),
        "terminal_omega": float(omega[-1]),
        "terminal_lambda": float(lam[-1]),
        "terminal_ell": float(ell[-1]),
        "first_share_exit_time": first_true_time(
            t,
            derived["outside_economic_region"],
        ),
        "outside_economic_region_fraction": float(
            np.mean(derived["outside_economic_region"])
        ),
        "profit_min": float(np.min(derived["profit"])),
        "profit_max": float(np.max(derived["profit"])),
        "debt_service_min": float(np.min(derived["debt_service"])),
        "debt_service_max": float(np.max(derived["debt_service"])),
        "pre_dividend_gap_min": float(
            np.min(derived["pre_dividend_gap"])
        ),
        "pre_dividend_gap_max": float(
            np.max(derived["pre_dividend_gap"])
        ),
        "financing_gap_min": float(
            np.min(derived["financing_gap"])
        ),
        "financing_gap_max": float(
            np.max(derived["financing_gap"])
        ),
        "inflation_min": float(np.min(derived["inflation"])),
        "inflation_max": float(np.max(derived["inflation"])),
        "hedge_fraction": float(np.mean(derived["hedge"])),
        "speculative_fraction": float(np.mean(derived["speculative"])),
        "ponzi_fraction": float(np.mean(derived["ponzi"])),
        "longest_speculative_duration": longest_duration(
            derived["speculative"],
            dt,
        ),
        "longest_ponzi_duration": longest_duration(
            derived["ponzi"],
            dt,
        ),
    }


def save_numerical_outputs(
    t: np.ndarray,
    states: np.ndarray,
    derived: dict[str, np.ndarray],
    stats: dict[str, float | int | bool | None],
) -> tuple[Path, Path]:
    """Save the raw path and diagnostics."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_data.csv"
    json_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_statistics.json"

    table = np.column_stack(
        [
            t,
            states,
            derived["profit"],
            derived["debt_service"],
            derived["pre_dividend_gap"],
            derived["financing_gap"],
            derived["inflation"],
            derived["interest_growth_gap"],
            derived["hedge"].astype(int),
            derived["speculative"].astype(int),
            derived["ponzi"].astype(int),
            derived["outside_economic_region"].astype(int),
        ]
    )

    header = (
        "time,omega,lambda,ell,profit_share,debt_service,"
        "pre_dividend_gap,financing_gap,inflation,interest_growth_gap,"
        "hedge,speculative,ponzi,outside_economic_region"
    )

    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=header,
        comments="",
    )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    return csv_path, json_path


# ============================================================
# Figure
# ============================================================
def create_minsky_figure(
    show: bool = True,
) -> tuple[plt.Figure, dict[str, float | int | bool | None]]:
    """Generate the revised three-panel Minsky-style stress figure."""
    model, t, states = simulate_minsky_crisis()
    derived = calculate_derived_variables(model, states)
    stats = summarize_scenario(t, states, derived)

    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]

    time_indices = sample_indices(t, TIME_SERIES_INTERVAL)
    phase_indices = sample_indices(t, PHASE_INTERVAL)

    t_plot = t[time_indices]
    omega_plot = omega[time_indices]
    lambda_plot = lam[time_indices]
    ell_plot = ell[time_indices]

    profit_plot = derived["profit"][time_indices]
    debt_service_plot = derived["debt_service"][time_indices]
    financing_gap_plot = derived["financing_gap"][time_indices]

    # ---- Smoothed state variables for the 3D phase path (panel b) ----
    omega_phase_raw = omega[phase_indices]
    lambda_phase_raw = lam[phase_indices]
    ell_phase_raw = ell[phase_indices]
    t_phase = t[phase_indices]

    omega_smoothed_full = moving_average(omega, PHASE_SMOOTH_WINDOW)
    lambda_smoothed_full = moving_average(lam, PHASE_SMOOTH_WINDOW)
    ell_smoothed_full = moving_average(ell, PHASE_SMOOTH_WINDOW)

    omega_phase = omega_smoothed_full[phase_indices]
    lambda_phase = lambda_smoothed_full[phase_indices]
    ell_phase = ell_smoothed_full[phase_indices]

    # ---- Figure layout with a 3D axis for panel (b) ----
    fig = plt.figure(figsize=(18, 6.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.12, 1.08, 1.28],
        left=0.05, right=0.98,
        top=0.92, bottom=0.08,
    )
    ax1 = fig.add_subplot(gs[0, 0])               # panel (a)
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")  # panel (b)
    ax3 = fig.add_subplot(gs[0, 2])               # panel (c)

    # --------------------------------------------------------
    # Panel (a): state variables over time
    # --------------------------------------------------------
    wage_line = ax1.plot(
        t_plot,
        omega_plot,
        color=COLOR_OMEGA,
        linewidth=1.7,
        label=r"Wage share $(\omega_t)$",
        zorder=3,
    )[0]
    employment_line = ax1.plot(
        t_plot,
        lambda_plot,
        color=COLOR_LAMBDA,
        linewidth=1.7,
        label=r"Employment $(\lambda_t)$",
        zorder=3,
    )[0]

    ax1.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax1.axhline(
        1.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax1.set_xlim(0.0, HORIZON)
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel(r"Share variables $(\omega_t,\lambda_t)$")
    ax1.set_title("(a) State Variables", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax1_debt = ax1.twinx()
    debt_line = ax1_debt.plot(
        t_plot,
        ell_plot,
        color=COLOR_ELL,
        linewidth=1.6,
        label=r"Net debt $(\ell_t)$",
        zorder=3,
    )[0]
    ax1_debt.set_ylabel(
        r"Net debt $(\ell_t)$",
        color=COLOR_ELL,
    )
    ax1_debt.tick_params(axis="y", colors=COLOR_ELL)
    ax1_debt.spines["right"].set_color(COLOR_ELL)

    ax1.legend(
        [wage_line, employment_line, debt_line],
        [
            wage_line.get_label(),
            employment_line.get_label(),
            debt_line.get_label(),
        ],
        loc="best",
        frameon=True,
    )

    # --------------------------------------------------------
    # Panel (b): 3D state-space phase path
    # --------------------------------------------------------
    # Raw path (faint)
    ax2.plot(
        omega_phase_raw,
        lambda_phase_raw,
        ell_phase_raw,
        linewidth=0.75,
        alpha=0.20,
        color=COLOR_RAW_PATH,
        zorder=1,
    )

    # Smoothed path coloured by time
    phase_points = np.column_stack(
        [omega_phase, lambda_phase, ell_phase]
    )
    phase_segments = np.stack(
        [phase_points[:-1], phase_points[1:]],
        axis=1,
    )

    phase_norm = Normalize(vmin=t_phase[0], vmax=t_phase[-1])
    phase_line = Line3DCollection(
        phase_segments,
        cmap=PHASE_CMAP,
        norm=phase_norm,
        linewidth=2.0,
    )
    phase_line.set_array(t_phase[:-1])
    ax2.add_collection3d(phase_line)

    # Start and end markers
    ax2.scatter(
        [omega_phase[0]],
        [lambda_phase[0]],
        [ell_phase[0]],
        s=65,
        marker="o",
        color=COLOR_START,
        zorder=4,
    )
    ax2.scatter(
        [omega_phase[-1]],
        [lambda_phase[-1]],
        [ell_phase[-1]],
        s=80,
        marker="X",
        color=COLOR_END,
        zorder=4,
    )

    ax2.set_xlim(*_pad_limits(omega_phase_raw, omega_phase))
    ax2.set_ylim(*_pad_limits(lambda_phase_raw, lambda_phase))
    ax2.set_zlim(*_pad_limits(ell_phase_raw, ell_phase))

    ax2.set_xlabel(r"Wage share $(\omega_t)$")
    ax2.set_ylabel(r"Employment $(\lambda_t)$")
    ax2.set_zlabel(r"Net debt $(\ell_t)$")
    ax2.set_title(
        "(b) Phase-Space Trajectory",
        fontweight="bold",
    )
    ax2.view_init(elev=24, azim=-58)
    ax2.grid(True, alpha=0.3)

    # Custom legend for panel (b)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_RAW_PATH,
            lw=1.2,
            alpha=0.6,
            label="Raw path",
        ),
        Line2D(
            [0],
            [0],
            color=plt.get_cmap(PHASE_CMAP)(0.70),
            lw=2.0,
            label="Smoothed path",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLOR_START,
            linestyle="None",
            markersize=7,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color=COLOR_END,
            linestyle="None",
            markersize=8,
            label="End",
        ),
    ]
    ax2.legend(handles=legend_handles, loc="best", frameon=True)

    colour_bar = fig.colorbar(
        phase_line,
        ax=ax2,
        fraction=0.047,
        pad=0.04,
    )
    colour_bar.set_label("Time (years)")

    # --------------------------------------------------------
    # Panel (c): financial indicators with light shading
    # --------------------------------------------------------
    add_regime_shading(
        ax3,
        t,
        derived["speculative"],
        derived["ponzi"],
    )

    profit_line = ax3.plot(
        t_plot,
        profit_plot,
        color=COLOR_PROFIT,
        linewidth=1.9,
        label=r"Profit share $(\pi_t)$",
        zorder=3,
    )[0]

    ax3.set_xlim(0.0, HORIZON)
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel(
        r"Profit share $(\pi_t)$",
        color=COLOR_PROFIT,
    )
    ax3.tick_params(axis="y", colors=COLOR_PROFIT)
    ax3.spines["left"].set_color(COLOR_PROFIT)
    ax3.set_title(
        "(c) Financial Indicators and Minsky Regimes",
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3, zorder=1)

    ax3_secondary = ax3.twinx()
    debt_service_line = ax3_secondary.plot(
        t_plot,
        debt_service_plot,
        color=COLOR_DEBT_SERVICE,
        linewidth=1.5,
        label=r"Debt service $(r\ell_t)$",
        zorder=3,
    )[0]
    financing_gap_line = ax3_secondary.plot(
        t_plot,
        financing_gap_plot,
        color=COLOR_FINANCING_GAP,
        linewidth=1.5,
        label=r"Financing gap $(F_t)$",
        zorder=3,
    )[0]

    ax3_secondary.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        zorder=2,
    )
    ax3_secondary.set_ylabel("Debt service / financing gap")
    ax3_secondary.spines["right"].set_color(COLOR_REFERENCE)

    regime_handles = [
        Patch(
            facecolor=SPECULATIVE_SHADE,
            edgecolor="none",
            alpha=0.30,
            label="Speculative-finance interval",
        ),
        Patch(
            facecolor=PONZI_SHADE,
            edgecolor="none",
            alpha=0.35,
            label="Ponzi-finance interval",
        ),
    ]

    ax3.legend(
        [
            profit_line,
            debt_service_line,
            financing_gap_line,
            *regime_handles,
        ],
        [
            profit_line.get_label(),
            debt_service_line.get_label(),
            financing_gap_line.get_label(),
            "Speculative-finance interval",
            "Ponzi-finance interval",
        ],
        loc="best",
        frameon=True,
    )

    fig.suptitle(
        "High-Initial-Debt Financial Fragility\n"
        f"Seed {SEED}; no additional shock; "
        rf"$X_0=({INITIAL_STATE[0]:.3f},"
        rf"{INITIAL_STATE[1]:.3f},"
        rf"{INITIAL_STATE[2]:.2f})$",
        fontsize=15,
        fontweight="bold",
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    csv_path, json_path = save_numerical_outputs(
        t,
        states,
        derived,
        stats,
    )

    print_summary(stats)
    print(f"\nSaved figure: {png_path}")
    print(f"Saved PDF:    {pdf_path}")
    print(f"Saved data:   {csv_path}")
    print(f"Saved stats:  {json_path}")

    if show:
        plt.show()

    return fig, stats


def format_optional_time(value: float | None) -> str:
    """Format an optional event time."""
    return "not observed" if value is None else f"{value:.2f} years"


def print_summary(
    stats: dict[str, float | int | bool | None],
) -> None:
    """Print the diagnostics required for the revised stress-test section."""
    print("\nHigh-initial-debt financial fragility")
    print("=" * 58)
    print(f"Seed: {stats['seed']}")
    print(f"Horizon: {stats['horizon']:.2f} years")
    print(f"Time step: {stats['dt']:.4f} years")
    print(
        "Initial state: "
        f"({stats['initial_omega']:.3f}, "
        f"{stats['initial_lambda']:.3f}, "
        f"{stats['initial_ell']:.3f})"
    )

    print("\nState-variable ranges")
    print("-" * 58)
    print(
        f"omega:  [{stats['omega_min']:.6f}, "
        f"{stats['omega_max']:.6f}]"
    )
    print(
        f"lambda: [{stats['lambda_min']:.6f}, "
        f"{stats['lambda_max']:.6f}]"
    )
    print(
        f"ell:    [{stats['ell_min']:.6f}, "
        f"{stats['ell_max']:.6f}]"
    )
    print(
        "First share-boundary exit: "
        f"{format_optional_time(stats['first_share_exit_time'])}"
    )

    print("\nFinancial ranges")
    print("-" * 58)
    print(
        f"Profit share: [{stats['profit_min']:.6f}, "
        f"{stats['profit_max']:.6f}]"
    )
    print(
        f"Debt service: [{stats['debt_service_min']:.6f}, "
        f"{stats['debt_service_max']:.6f}]"
    )
    print(
        f"Pre-dividend gap: [{stats['pre_dividend_gap_min']:.6f}, "
        f"{stats['pre_dividend_gap_max']:.6f}]"
    )
    print(
        f"Financing gap: [{stats['financing_gap_min']:.6f}, "
        f"{stats['financing_gap_max']:.6f}]"
    )

    print("\nMinsky-style regime diagnostics")
    print("-" * 58)
    print(
        f"Hedge share:       "
        f"{100.0 * stats['hedge_fraction']:.2f}%"
    )
    print(
        f"Speculative share: "
        f"{100.0 * stats['speculative_fraction']:.2f}%"
    )
    print(
        f"Ponzi share:       "
        f"{100.0 * stats['ponzi_fraction']:.2f}%"
    )
    print(
        "Longest speculative interval: "
        f"{stats['longest_speculative_duration']:.2f} years"
    )
    print(
        "Longest Ponzi interval: "
        f"{stats['longest_ponzi_duration']:.2f} years"
    )


if __name__ == "__main__":
    create_minsky_figure(show=True)
