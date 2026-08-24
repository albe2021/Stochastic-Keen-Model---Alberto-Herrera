from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from keen_model_functions import KeenModel

# -----------------------------------------------------------------------------
# Reproducible experiment settings
# -----------------------------------------------------------------------------

BASELINE_PARAMS = KeenModel.DEFAULT_PARAMS.copy()
INITIAL_CONDITION = (0.9, 0.9, 0.3)
T_SIM = 100.0
DT = 0.01
STOCHASTIC_SEED = 128
PLOT_INTERVAL = 0.2
OUTPUT_DIR = Path("Figures")
FIGURE_NAME = "deterministic_vs_stochastic"


def _time_grid(T: float, dt: float) -> np.ndarray:
    """Return a uniform time grid, validating that T / dt is integral."""
    if T <= 0.0:
        raise ValueError("T must be strictly positive.")
    if dt <= 0.0:
        raise ValueError("dt must be strictly positive.")

    ratio = T / dt
    n_steps = int(round(ratio))
    if not np.isclose(ratio, n_steps, rtol=0.0, atol=1e-10):
        raise ValueError("T / dt must be an integer within tolerance.")

    return np.arange(n_steps + 1, dtype=float) * dt


def simulate_deterministic(
    T: float = T_SIM,
    dt: float = DT,
    x0: tuple[float, float, float] = INITIAL_CONDITION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the deterministic drift system without projection.

    A high-accuracy adaptive ODE solver is used, with output evaluated on the
    same time grid as the stochastic Euler--Maruyama simulation.
    """
    params = BASELINE_PARAMS.copy()
    params.update(
        {
            "sigma_omega": 0.0,
            "sigma_lambda": 0.0,
            "sigma_ell": 0.0,
        }
    )
    model = KeenModel(params)
    t_eval = _time_grid(T, dt)

    solution = solve_ivp(
        fun=model.deterministic_rhs,
        t_span=(0.0, T),
        y0=np.asarray(x0, dtype=float),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )

    if not solution.success:
        raise RuntimeError(
            "Deterministic integration failed: " + solution.message
        )

    states = solution.y.T
    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "The deterministic integration produced a non-finite state."
        )

    return solution.t, states


def simulate_stochastic(
    T: float = T_SIM,
    dt: float = DT,
    seed: int = STOCHASTIC_SEED,
    x0: tuple[float, float, float] = INITIAL_CONDITION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the unprojected positive-part stochastic system.

    No clipping, reflection, projection, or epsilon floor is applied. If a
    share coordinate lies outside [0, 1], its direct Brownian coefficient is
    zero while its drift remains active.
    """
    model = KeenModel(BASELINE_PARAMS)
    return model.simulate_path(
        x0=x0,
        T=T,
        dt=dt,
        seed=seed,
    )


def _padded_limits(
    *series: np.ndarray,
    padding_fraction: float = 0.08,
    include: tuple[float, ...] = (),
) -> tuple[float, float]:
    """Return common finite plotting limits for a collection of series."""
    values = np.concatenate(
        [np.asarray(item, dtype=float).ravel() for item in series]
    )
    if include:
        values = np.concatenate([values, np.asarray(include, dtype=float)])

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot calculate limits from non-finite data.")

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    span = upper - lower
    padding = padding_fraction * span if span > 0.0 else 0.1
    return lower - padding, upper + padding


def _print_diagnostics(
    label: str,
    t: np.ndarray,
    states: np.ndarray,
) -> None:
    """Print share-boundary and range diagnostics for one trajectory."""
    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]

    print(f"\n{label}")
    print("-" * len(label))
    print(f"omega range:  [{omega.min():.6f}, {omega.max():.6f}]")
    print(f"lambda range: [{lam.min():.6f}, {lam.max():.6f}]")
    print(f"ell range:    [{ell.min():.6f}, {ell.max():.6f}]")

    for name, values in (("omega", omega), ("lambda", lam)):
        outside = (values < 0.0) | (values > 1.0)
        share = 100.0 * float(outside.mean())
        if outside.any():
            first_index = int(np.flatnonzero(outside)[0])
            print(
                f"{name} outside [0,1]: {share:.3f}% of observations; "
                f"first at t={t[first_index]:.4f}"
            )
        else:
            print(f"{name} outside [0,1]: 0.000% of observations")


def create_comparison_figure(
    T: float = T_SIM,
    dt: float = DT,
    seed: int = STOCHASTIC_SEED,
    x0: tuple[float, float, float] = INITIAL_CONDITION,
    output_dir: Path = OUTPUT_DIR,
) -> plt.Figure:
    """Create and save the deterministic-versus-stochastic comparison."""
    t_det, states_det = simulate_deterministic(T=T, dt=dt, x0=x0)
    t_stoch, states_stoch = simulate_stochastic(
        T=T,
        dt=dt,
        seed=seed,
        x0=x0,
    )

    if not np.array_equal(t_det, t_stoch):
        raise RuntimeError(
            "Deterministic and stochastic simulations use different grids."
        )

    _print_diagnostics("Deterministic trajectory", t_det, states_det)
    _print_diagnostics("Stochastic trajectory", t_stoch, states_stoch)

    plot_stride = max(1, int(round(PLOT_INTERVAL / dt)))
    t_plot = t_det[::plot_stride]
    det_plot = states_det[::plot_stride]
    stoch_plot = states_stoch[::plot_stride]

    omega_limits = _padded_limits(
        det_plot[:, 0],
        stoch_plot[:, 0],
        include=(0.0, 1.0),
    )
    lambda_limits = _padded_limits(
        det_plot[:, 1],
        stoch_plot[:, 1],
        include=(0.0, 1.0),
    )
    ell_limits = _padded_limits(
        det_plot[:, 2],
        stoch_plot[:, 2],
        include=(0.0,),
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    # Consistent thesis colors:
    # omega = blue, lambda = red, ell = green
    row_data = (
        (det_plot[:, 0], stoch_plot[:, 0], r"$\omega_t$", omega_limits, "#1f77b4"),
        (det_plot[:, 1], stoch_plot[:, 1], r"$\lambda_t$", lambda_limits, "#d62728"),
        (det_plot[:, 2], stoch_plot[:, 2], r"$\ell_t$", ell_limits, "#2ca02c"),
    )

    for row, (det_values, stoch_values, ylabel, limits, color) in enumerate(
        row_data
    ):
        axes[row, 0].plot(
            t_plot,
            det_values,
            color=color,
            linewidth=1.25,
        )
        axes[row, 1].plot(
            t_plot,
            stoch_values,
            color=color,
            linewidth=1.25,
        )

        axes[row, 0].set_ylabel(ylabel, fontsize=11)
        axes[row, 0].set_ylim(*limits)
        axes[row, 1].set_ylim(*limits)

        for column in range(2):
            axes[row, column].set_xlim(0.0, T)
            axes[row, column].grid(True, alpha=0.3)

    axes[0, 0].set_title("Deterministic drift system", fontweight="bold")
    axes[0, 1].set_title(
        f"Stochastic positive-part system (seed {seed})",
        fontweight="bold",
    )

    # Economically meaningful boundaries
    for row in (0, 1):
        for column in range(2):
            axes[row, column].axhline(
                0.0, linestyle="--", linewidth=0.8, color="0.4"
            )
            axes[row, column].axhline(
                1.0, linestyle="--", linewidth=0.8, color="0.4"
            )

    for column in range(2):
        axes[2, column].axhline(
            0.0, linestyle="--", linewidth=0.8, color="0.4"
        )
        axes[2, column].set_xlabel("Time (years)", fontsize=11)

    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for label, axis in zip(panel_labels, axes.ravel()):
        axis.text(
            0.015,
            0.94,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{FIGURE_NAME}.png"
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"
    csv_path = output_dir / f"{FIGURE_NAME}_data.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    output_data = np.column_stack((t_det, states_det, states_stoch))
    np.savetxt(
        csv_path,
        output_data,
        delimiter=",",
        header=(
            "time,omega_deterministic,lambda_deterministic,"
            "ell_deterministic,omega_stochastic,lambda_stochastic,"
            "ell_stochastic"
        ),
        comments="",
    )

    print(f"\nSaved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved data:   {csv_path}")

    return fig


if __name__ == "__main__":
    figure = create_comparison_figure()
    plt.show()
