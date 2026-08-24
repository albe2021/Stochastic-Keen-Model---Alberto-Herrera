# debt_deflation.py
#
# Deflationary stress under a temporary debt-volatility shock.
#
# This script uses the canonical unprojected positive-part Keen model.
# No clipping, projection, reflection, or epsilon floor is imposed.

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from keen_model_functions import KeenModel


# ============================================================
# Plot configuration
# ============================================================
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ============================================================
# Scenario configuration
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

SELECTED_SEED = 146
INITIAL_CONDITION = (0.60, 0.95, 1.80)
T_SIM = 300.0
DT = 0.01

SHOCK_START = 20.0
SHOCK_END = 25.0
SIGMA_ELL_SHOCK = 0.35

TIME_SERIES_INTERVAL = 0.10
PHASE_INTERVAL = 0.25
MIN_SUSTAINED_DEFLATION = 5.0

OUTPUT_DIRECTORY = Path("Figures")
DATA_DIRECTORY = Path("NumericalData")
OUTPUT_STEM = "debt_deflation"


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
COLOR_INFLATION = "#ff7f0e"      # orange

# Plot annotations
COLOR_REFERENCE = "#666666"      # grey
COLOR_DEFLATION_THRESHOLD = "#c76e00"
COLOR_SHOCK_SHADE = "#e76f51"    # light red when transparent
COLOR_DEFLATION_SHADE = "#f4a261"
COLOR_OUTSIDE_REGION = "#bdbdbd"
COLOR_START = COLOR_OMEGA
COLOR_END = COLOR_INFLATION
PHASE_CMAP = "viridis"


# ============================================================
# Helpers
# ============================================================
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
    """Return the first time at which mask is True, or None."""
    indices = np.flatnonzero(mask)
    return float(t[indices[0]]) if len(indices) else None


def sample_indices(t: np.ndarray, interval: float) -> np.ndarray:
    """Return approximately equally spaced indices for plotting."""
    if interval <= 0:
        raise ValueError("interval must be strictly positive.")

    base_dt = float(t[1] - t[0])
    step = int(round(interval / base_dt))

    if step < 1 or not np.isclose(
        step * base_dt,
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


def add_axis_padding(ax, values: np.ndarray, axis: str) -> None:
    """Add modest data-driven padding to one 3D axis."""
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = 0.05 * span if span > 0 else 0.05
    limits = (lower - padding, upper + padding)

    if axis == "x":
        ax.set_xlim(*limits)
    elif axis == "y":
        ax.set_ylim(*limits)
    elif axis == "z":
        ax.set_zlim(*limits)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")


# ============================================================
# Simulation
# ============================================================
def debt_volatility_schedule(
    time: float,
    state: np.ndarray,
) -> Mapping[str, float] | None:
    """Temporarily replace sigma_ell during the stress window."""
    del state
    if SHOCK_START <= time < SHOCK_END:
        return {"sigma_ell": SIGMA_ELL_SHOCK}
    return None


def simulate_debt_deflation_path(
    *,
    seed: int = SELECTED_SEED,
    T: float = T_SIM,
    dt: float = DT,
    x0: Sequence[float] = INITIAL_CONDITION,
) -> tuple[KeenModel, np.ndarray, np.ndarray]:
    """
    Simulate the unprojected positive-part stress path.

    The only exogenous intervention is a temporary increase in debt
    volatility during [SHOCK_START, SHOCK_END). The structural drift and
    the wage-share and employment volatilities remain unchanged.
    """
    model = KeenModel(BASELINE_PARAMS)
    t, states = model.simulate_path(
        x0=x0,
        T=T,
        dt=dt,
        seed=seed,
        sigma_schedule=debt_volatility_schedule,
    )

    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "The debt-volatility stress simulation produced non-finite states."
        )

    return model, t, states


# ============================================================
# Diagnostics
# ============================================================
def calculate_diagnostics(
    model: KeenModel,
    t: np.ndarray,
    states: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | bool | None]]:
    """Calculate full-path and post-shock diagnostics."""
    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]

    profit = np.asarray(model.profit_share(omega, ell), dtype=float)
    inflation = np.asarray(model.inflation(omega), dtype=float)
    debt_service = model.params["r"] * ell
    financing_gap = np.asarray(
        model.financing_gap(omega, lam, ell),
        dtype=float,
    )
    interest_growth_gap = np.asarray(
        model.interest_growth_gap(omega, ell),
        dtype=float,
    )

    deflation = inflation < 0.0
    omega_outside = (omega < 0.0) | (omega > 1.0)
    lambda_outside = (lam < 0.0) | (lam > 1.0)
    outside_economic_region = omega_outside | lambda_outside
    post = t >= SHOCK_END
    shock = (t >= SHOCK_START) & (t <= SHOCK_END)
    dt = float(t[1] - t[0])

    initial_profit = float(profit[0])
    initial_debt_service = float(debt_service[0])

    longest_post_deflation = longest_duration(deflation & post, dt)
    minimum_post_employment = float(np.min(lam[post]))
    minimum_post_profit = float(np.min(profit[post]))
    final_employment = float(lam[-1])

    conditions = {
        "deflation_threshold_crossed": bool(np.any(deflation[post])),
        "sustained_deflation": bool(
            longest_post_deflation >= MIN_SUSTAINED_DEFLATION
        ),
        "employment_weakens": bool(
            minimum_post_employment <= lam[0] - 0.10
        ),
        "persistent_employment_damage": bool(
            final_employment < lam[0] - 0.05
        ),
        "profit_deteriorates": bool(
            minimum_post_profit <= initial_profit - 0.08
        ),
        "debt_positive_and_rising": bool(
            np.min(ell[post]) > 0.0
            and np.max(ell[post]) >= ell[0] + 0.40
        ),
        "debt_service_rises": bool(
            np.max(debt_service[post]) >= initial_debt_service + 0.01
        ),
        "employment_remains_in_unit_interval": bool(
            not np.any(lambda_outside)
        ),
        "wage_share_remains_in_unit_interval": bool(
            not np.any(omega_outside)
        ),
    }

    arrays = {
        "profit": profit,
        "inflation": inflation,
        "debt_service": debt_service,
        "financing_gap": financing_gap,
        "interest_growth_gap": interest_growth_gap,
        "deflation": deflation,
        "omega_outside": omega_outside,
        "lambda_outside": lambda_outside,
        "outside_economic_region": outside_economic_region,
        "post_shock": post,
        "shock_window": shock,
    }

    statistics: dict[str, float | int | bool | None] = {
        "seed": seed,
        "T": float(t[-1]),
        "dt": dt,
        "shock_start": SHOCK_START,
        "shock_end": SHOCK_END,
        "sigma_ell_baseline": model.params["sigma_ell"],
        "sigma_ell_shock": SIGMA_ELL_SHOCK,
        "omega0": float(omega[0]),
        "lambda0": float(lam[0]),
        "ell0": float(ell[0]),
        "omega_min": float(np.min(omega)),
        "omega_max": float(np.max(omega)),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "ell_min": float(np.min(ell)),
        "ell_max": float(np.max(ell)),
        "profit_min": float(np.min(profit)),
        "profit_max": float(np.max(profit)),
        "inflation_min": float(np.min(inflation)),
        "inflation_max": float(np.max(inflation)),
        "debt_service_min": float(np.min(debt_service)),
        "debt_service_max": float(np.max(debt_service)),
        "deflation_fraction": float(np.mean(deflation)),
        "longest_deflation_duration": longest_duration(deflation, dt),
        "longest_post_shock_deflation_duration": longest_post_deflation,
        "first_share_exit_time": first_true_time(
            t,
            outside_economic_region,
        ),
        "outside_economic_region_fraction": float(
            np.mean(outside_economic_region)
        ),
        "min_post_shock_employment": minimum_post_employment,
        "final_employment": final_employment,
        "min_post_shock_profit": minimum_post_profit,
        "max_post_shock_profit": float(np.max(profit[post])),
        "min_post_shock_ell": float(np.min(ell[post])),
        "max_post_shock_ell": float(np.max(ell[post])),
        "min_post_shock_debt_service": float(
            np.min(debt_service[post])
        ),
        "max_post_shock_debt_service": float(
            np.max(debt_service[post])
        ),
        "min_shock_window_ell": float(np.min(ell[shock])),
        "max_shock_window_ell": float(np.max(ell[shock])),
        "min_post_shock_financing_gap": float(
            np.min(financing_gap[post])
        ),
        "max_post_shock_financing_gap": float(
            np.max(financing_gap[post])
        ),
        **conditions,
    }

    return arrays, statistics


def representative_score(stats: Mapping[str, float | int | bool | None]) -> float:
    """
    Score paths for the optional transparent seed search.

    The score targets a deflationary stress episode with employment and
    profit deterioration. Rising debt is recorded but is not required,
    because the scenario is not assumed to be a textbook Fisher spiral.
    """
    if not bool(stats["employment_remains_in_unit_interval"]):
        return -np.inf
    if abs(float(stats["ell_min"])) > 10.0 or abs(float(stats["ell_max"])) > 10.0:
        return -np.inf

    score = 0.0
    score += 3.0 * float(bool(stats["sustained_deflation"]))
    score += 2.0 * float(bool(stats["employment_weakens"]))
    score += 2.0 * float(bool(stats["persistent_employment_damage"]))
    score += 2.0 * float(bool(stats["profit_deteriorates"]))
    score += 0.05 * float(stats["longest_post_shock_deflation_duration"])
    score += 2.0 * max(
        0.0,
        float(stats["lambda0"])
        - float(stats["min_post_shock_employment"]),
    )
    return score


def search_representative_seed(
    n_candidates: int = 800,
) -> tuple[int, dict[str, float | int | bool | None]]:
    """
    Reproduce the seed-selection scan and return the highest-scoring path.

    The final figure uses SELECTED_SEED directly. This function is supplied
    for transparency and does not run when the figure is generated normally.
    """
    best_seed: int | None = None
    best_stats: dict[str, float | int | bool | None] | None = None
    best_score = -np.inf

    for seed in range(n_candidates):
        try:
            model, t, states = simulate_debt_deflation_path(seed=seed)
            _, stats = calculate_diagnostics(model, t, states, seed=seed)
        except (FloatingPointError, OverflowError):
            continue

        score = representative_score(stats)
        if score > best_score:
            best_seed = seed
            best_stats = stats
            best_score = score

    if best_seed is None or best_stats is None:
        raise RuntimeError("No numerically usable candidate path was found.")

    print(
        f"Best seed among 0,...,{n_candidates - 1}: "
        f"{best_seed} (score={best_score:.3f})"
    )
    return best_seed, best_stats


# ============================================================
# Output
# ============================================================
def save_outputs(
    t: np.ndarray,
    states: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    stats: Mapping[str, float | int | bool | None],
) -> tuple[Path, Path]:
    """Save the complete stress path and summary statistics."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_data.csv"
    json_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_statistics.json"

    table = np.column_stack(
        [
            t,
            states,
            arrays["profit"],
            arrays["debt_service"],
            arrays["inflation"],
            arrays["financing_gap"],
            arrays["interest_growth_gap"],
            arrays["outside_economic_region"].astype(int),
            arrays["shock_window"].astype(int),
        ]
    )
    header = (
        "time,omega,lambda,ell,profit_share,debt_service,inflation,"
        "financing_gap,interest_growth_gap,outside_economic_region,"
        "shock_window"
    )
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(stats), handle, indent=2, sort_keys=True)

    return csv_path, json_path


def create_debt_deflation_figure(
    show: bool = True,
) -> tuple[plt.Figure, dict[str, float | int | bool | None]]:
    """Generate the revised three-panel deflationary-stress figure."""
    model, t, states = simulate_debt_deflation_path(seed=SELECTED_SEED)
    arrays, stats = calculate_diagnostics(
        model,
        t,
        states,
        seed=SELECTED_SEED,
    )

    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]
    omega_deflation = 1.0 / model.params["m"]

    plot_indices = sample_indices(t, TIME_SERIES_INTERVAL)
    phase_indices = sample_indices(t, PHASE_INTERVAL)

    t_plot = t[plot_indices]
    omega_plot = omega[plot_indices]
    lambda_plot = lam[plot_indices]
    ell_plot = ell[plot_indices]
    debt_service_plot = arrays["debt_service"][plot_indices]
    profit_plot = arrays["profit"][plot_indices]
    inflation_plot = arrays["inflation"][plot_indices]

    t_phase = t[phase_indices]
    omega_phase = omega[phase_indices]
    lambda_phase = lam[phase_indices]
    ell_phase = ell[phase_indices]

    deflation_regions = contiguous_true_regions(arrays["deflation"])
    outside_regions = contiguous_true_regions(
        arrays["outside_economic_region"]
    )

    fig = plt.figure(figsize=(16, 5.8))
    layout = gridspec.GridSpec(
        1,
        3,
        width_ratios=[1.12, 1.25, 1.0],
    )

    # --------------------------------------------------------
    # Panel (a): shares and debt
    # --------------------------------------------------------
    ax1 = fig.add_subplot(layout[0])
    ax1_debt = ax1.twinx()

    line_omega = ax1.plot(
        t_plot,
        omega_plot,
        color=COLOR_OMEGA,
        linewidth=1.5,
        label=r"Wage share $(\omega_t)$",
        zorder=3,
    )[0]
    line_lambda = ax1.plot(
        t_plot,
        lambda_plot,
        color=COLOR_LAMBDA,
        linewidth=1.5,
        label=r"Employment $(\lambda_t)$",
        zorder=3,
    )[0]
    line_ell = ax1_debt.plot(
        t_plot,
        ell_plot,
        color=COLOR_ELL,
        linewidth=1.5,
        linestyle="--",
        label=r"Net debt $(\ell_t)$",
        zorder=3,
    )[0]

    ax1.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax1.axhline(
        1.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax1.axhline(
        omega_deflation,
        color=COLOR_DEFLATION_THRESHOLD,
        linestyle=":",
        linewidth=1.1,
        label=r"Deflation threshold $(\omega=1/m)$",
    )
    shock_patch = ax1.axvspan(
        SHOCK_START,
        SHOCK_END,
        facecolor=COLOR_SHOCK_SHADE,
        edgecolor="none",
        alpha=0.14,
        label="Debt-volatility shock",
        zorder=0,
    )

    first_outside_label = True
    for start, end in outside_regions:
        ax1.axvspan(
            t[start],
            t[min(end, len(t) - 1)],
            facecolor=COLOR_OUTSIDE_REGION,
            edgecolor=COLOR_REFERENCE,
            alpha=0.10,
            hatch="//",
            label=(
                "Outside economic region"
                if first_outside_label
                else None
            ),
            zorder=0,
        )
        first_outside_label = False

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_debt.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="best",
        frameon=True,
    )

    ax1.set_title("(a) State Variables", fontweight="bold")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel(r"Share variables $(\omega_t,\lambda_t)$")
    ax1_debt.set_ylabel(
        r"Net debt $(\ell_t)$",
        color=COLOR_ELL,
    )
    ax1_debt.tick_params(axis="y", colors=COLOR_ELL)
    ax1_debt.spines["right"].set_color(COLOR_ELL)
    ax1.set_xlim(0.0, T_SIM)
    ax1.grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Panel (b): raw downsampled 3D phase trajectory
    # --------------------------------------------------------
    ax2 = fig.add_subplot(layout[1], projection="3d")

    points = np.column_stack(
        [omega_phase, lambda_phase, ell_phase]
    ).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    normalization = plt.Normalize(float(t_phase.min()), float(t_phase.max()))
    phase_line = Line3DCollection(
        segments,
        cmap=PHASE_CMAP,
        norm=normalization,
    )
    phase_line.set_array(t_phase[:-1])
    phase_line.set_linewidth(2.0)
    ax2.add_collection3d(phase_line)

    ax2.scatter(
        omega_phase[0],
        lambda_phase[0],
        ell_phase[0],
        s=55,
        marker="o",
        color=COLOR_START,
        label="Start",
        depthshade=False,
        zorder=4,
    )
    ax2.scatter(
        omega_phase[-1],
        lambda_phase[-1],
        ell_phase[-1],
        s=65,
        marker="X",
        color=COLOR_END,
        label="End",
        depthshade=False,
        zorder=4,
    )

    add_axis_padding(ax2, omega_phase, "x")
    add_axis_padding(ax2, lambda_phase, "y")
    add_axis_padding(ax2, ell_phase, "z")

    ax2.set_title("(b) Phase-Space Trajectory", fontweight="bold", pad=12)
    ax2.set_xlabel(r"Wage share $(\omega)$", labelpad=8)
    ax2.set_ylabel(r"Employment $(\lambda)$", labelpad=8)
    ax2.set_zlabel(r"Net debt $(\ell)$", labelpad=8)
    ax2.view_init(elev=28, azim=-58)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True)

    colour_bar = fig.colorbar(
        phase_line,
        ax=ax2,
        ticks=np.linspace(0.0, T_SIM, 5),
        pad=0.08,
        shrink=0.8,
    )
    colour_bar.set_label("Time (years)")

    # --------------------------------------------------------
    # Panel (c): financial ratios
    # --------------------------------------------------------
    ax3 = fig.add_subplot(layout[2])

    ax3.plot(
        t_plot,
        debt_service_plot,
        color=COLOR_DEBT_SERVICE,
        linewidth=1.5,
        label=r"Debt service $(r\ell_t)$",
        zorder=3,
    )
    ax3.plot(
        t_plot,
        profit_plot,
        color=COLOR_PROFIT,
        linewidth=1.5,
        label=r"Profit share $(\pi_t)$",
        zorder=3,
    )
    ax3.plot(
        t_plot,
        inflation_plot,
        color=COLOR_INFLATION,
        linewidth=1.5,
        label=r"Inflation $(i(\omega_t))$",
        zorder=3,
    )
    ax3.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax3.axvspan(
        SHOCK_START,
        SHOCK_END,
        facecolor=COLOR_SHOCK_SHADE,
        edgecolor="none",
        alpha=0.14,
        label="Debt-volatility shock",
        zorder=0,
    )

    first_deflation_label = True
    for start, end in deflation_regions:
        ax3.axvspan(
            t[start],
            t[min(end, len(t) - 1)],
            facecolor=COLOR_DEFLATION_SHADE,
            edgecolor="none",
            alpha=0.12,
            label=(
                "Deflation episode"
                if first_deflation_label
                else None
            ),
            zorder=0,
        )
        first_deflation_label = False

    ax3.set_title("(c) Financial Ratios", fontweight="bold")
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Ratio")
    ax3.set_xlim(0.0, T_SIM)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", frameon=True)

    fig.tight_layout()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    csv_path, json_path = save_outputs(t, states, arrays, stats)
    print_summary(stats)
    print(f"\nSaved figure: {png_path}")
    print(f"Saved PDF:    {pdf_path}")
    print(f"Saved data:   {csv_path}")
    print(f"Saved stats:  {json_path}")

    if show:
        plt.show()

    return fig, stats


def format_optional_time(value: float | None) -> str:
    """Format an optional time value."""
    return "not observed" if value is None else f"{value:.2f} years"


def print_summary(stats: Mapping[str, float | int | bool | None]) -> None:
    """Print diagnostics used in the Section 5.3 rewrite."""
    print("\nDeflationary stress under debt-volatility shock")
    print("=" * 60)
    print(f"Seed: {stats['seed']}")
    print(
        "Initial state: "
        f"({stats['omega0']:.3f}, {stats['lambda0']:.3f}, "
        f"{stats['ell0']:.3f})"
    )
    print(
        "Debt volatility: "
        f"{stats['sigma_ell_baseline']:.3f} -> "
        f"{stats['sigma_ell_shock']:.3f} during "
        f"[{stats['shock_start']:.1f}, {stats['shock_end']:.1f})"
    )

    print("\nState ranges")
    print("-" * 60)
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
    print(
        "Fraction outside economic region: "
        f"{100.0 * float(stats['outside_economic_region_fraction']):.4f}%"
    )

    print("\nPost-shock diagnostics")
    print("-" * 60)
    print(
        "Longest post-shock deflation episode: "
        f"{stats['longest_post_shock_deflation_duration']:.2f} years"
    )
    print(
        f"Employment minimum: {stats['min_post_shock_employment']:.6f}"
    )
    print(f"Final employment:   {stats['final_employment']:.6f}")
    print(f"Profit minimum:     {stats['min_post_shock_profit']:.6f}")
    print(
        f"Net debt range:     [{stats['min_post_shock_ell']:.6f}, "
        f"{stats['max_post_shock_ell']:.6f}]"
    )
    print(
        f"Debt-service range: [{stats['min_post_shock_debt_service']:.6f}, "
        f"{stats['max_post_shock_debt_service']:.6f}]"
    )

    print("\nScenario classifications")
    print("-" * 60)
    for key in (
        "deflation_threshold_crossed",
        "sustained_deflation",
        "employment_weakens",
        "persistent_employment_damage",
        "profit_deteriorates",
        "debt_positive_and_rising",
        "debt_service_rises",
        "wage_share_remains_in_unit_interval",
        "employment_remains_in_unit_interval",
    ):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    create_debt_deflation_figure(show=True)
