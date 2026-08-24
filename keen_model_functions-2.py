from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


class KeenModel:
    """
    Numerical implementation of the stochastic dividend-inclusive Keen model.

    The share diffusions use the positive-part coefficients

        sigma_omega * sqrt([omega * (1 - omega)]_+)
        sigma_lambda * sqrt([lambda * (1 - lambda)]_+)

    where [x]_+ = max(x, 0).

    No projection, clipping, reflection, or epsilon floor is imposed on the
    wage-share or employment coordinates. If either share coordinate lies
    outside [0, 1], its direct Brownian coefficient is exactly zero while its
    drift remains active.
    """

    DEFAULT_PARAMS = {
        "alpha": 0.02,
        "beta": 0.031,
        "delta": 0.04,
        "nu": 2.7,
        "r": 0.02,
        "eta_p": 0.192,
        "m": 1.875,
        "gamma": 0.9,
        # Baseline volatilities used in Section 5.
        "sigma_omega": 0.05,
        "sigma_lambda": 0.05,
        "sigma_ell": 0.05,
        # Phillips curve: Phi(lambda) = phi0 + phi1 * lambda.
        "phi0": -0.292,
        "phi1": 0.469,
        # Numerical truncated-linear investment function.
        "kappa0": 0.0318,
        "kappa1": 0.575,
        "kappa_min": 0.0,
        "kappa_max": 0.3,
        # Numerical truncated-linear dividend function.
        "dividend0": -0.078,
        "dividend1": 0.553,
        "dividend_min": 0.0,
        "dividend_max": 0.3,
    }

    _LEGACY_ALIASES = {
        "delta0": "dividend0",
        "delta1": "dividend1",
        "delta_min": "dividend_min",
        "delta_max": "dividend_max",
    }

    def __init__(self, params: Mapping[str, float] | None = None) -> None:
        self.params = dict(self.DEFAULT_PARAMS)

        if params:
            normalized = dict(params)
            for old_name, new_name in self._LEGACY_ALIASES.items():
                if old_name in normalized:
                    if new_name in normalized:
                        raise ValueError(
                            f"Provide only one of {old_name!r} and {new_name!r}."
                        )
                    normalized[new_name] = normalized.pop(old_name)

            unknown = set(normalized) - set(self.DEFAULT_PARAMS)
            if unknown:
                raise KeyError(
                    "Unknown model parameter(s): "
                    + ", ".join(sorted(unknown))
                )

            self.params.update(normalized)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        p = self.params

        if p["nu"] <= 0:
            raise ValueError("nu must be strictly positive.")
        if p["r"] <= 0:
            raise ValueError("r must be strictly positive.")
        if not 0 <= p["gamma"] <= 1:
            raise ValueError("gamma must lie in [0, 1].")
        if p["eta_p"] <= 0:
            raise ValueError("eta_p must be strictly positive.")
        if p["m"] <= 1:
            raise ValueError("m must be greater than 1.")

        for name in ("sigma_omega", "sigma_lambda", "sigma_ell"):
            if p[name] < 0:
                raise ValueError(f"{name} must be nonnegative.")

        if p["kappa_min"] > p["kappa_max"]:
            raise ValueError("kappa_min cannot exceed kappa_max.")
        if p["dividend_min"] > p["dividend_max"]:
            raise ValueError("dividend_min cannot exceed dividend_max.")

    @staticmethod
    def positive_part(x):
        """Return [x]_+ = max(x, 0), supporting scalars and arrays."""
        return np.maximum(x, 0.0)

    def phi(self, lam):
        """Phillips curve Phi(lambda) = phi0 + phi1 * lambda."""
        return self.params["phi0"] + self.params["phi1"] * np.asarray(lam)

    def kappa(self, pi):
        """Numerical truncated-linear investment function."""
        raw = self.params["kappa0"] + self.params["kappa1"] * np.asarray(pi)
        return np.clip(
            raw,
            self.params["kappa_min"],
            self.params["kappa_max"],
        )

    def dividend(self, pi):
        """Numerical truncated-linear dividend function."""
        raw = (
            self.params["dividend0"]
            + self.params["dividend1"] * np.asarray(pi)
        )
        return np.clip(
            raw,
            self.params["dividend_min"],
            self.params["dividend_max"],
        )

    # Backward-compatible method name used by older scripts.
    def delta_func(self, pi):
        return self.dividend(pi)

    def inflation(self, omega):
        """Inflation i(omega) = eta_p * (m * omega - 1)."""
        return self.params["eta_p"] * (
            self.params["m"] * np.asarray(omega) - 1.0
        )

    def profit_share(self, omega, ell):
        """Pre-dividend profit share pi = 1 - omega - r * ell."""
        return 1.0 - np.asarray(omega) - self.params["r"] * np.asarray(ell)

    def drift_omega(self, omega, lam, ell):
        """Wage-share drift."""
        del ell
        return np.asarray(omega) * (
            self.phi(lam)
            - self.params["alpha"]
            - (1.0 - self.params["gamma"]) * self.inflation(omega)
        )

    def drift_lambda(self, omega, lam, ell):
        """Employment drift."""
        pi = self.profit_share(omega, ell)
        return np.asarray(lam) * (
            self.kappa(pi) / self.params["nu"]
            - self.params["delta"]
            - self.params["alpha"]
            - self.params["beta"]
        )

    def drift_ell(self, omega, lam, ell):
        """Net-debt drift."""
        del lam
        pi = self.profit_share(omega, ell)
        kap = self.kappa(pi)
        return (
            np.asarray(ell)
            * (
                self.params["r"]
                - kap / self.params["nu"]
                + self.params["delta"]
                - self.inflation(omega)
            )
            + np.asarray(omega)
            + kap
            - 1.0
            + self.dividend(pi)
        )

    def drift(self, omega, lam, ell) -> np.ndarray:
        """Return the three drift components as a NumPy array."""
        return np.asarray(
            [
                self.drift_omega(omega, lam, ell),
                self.drift_lambda(omega, lam, ell),
                self.drift_ell(omega, lam, ell),
            ],
            dtype=float,
        )

    def diffusion_terms(
        self,
        omega,
        lam,
        ell,
        sigma_override: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """
        Return the three scalar diffusion amplitudes.

        The positive-part coefficients are exactly zero at and outside the
        unit interval. No epsilon floor is used.
        """
        sigma_omega = self.params["sigma_omega"]
        sigma_lambda = self.params["sigma_lambda"]
        sigma_ell = self.params["sigma_ell"]

        if sigma_override:
            allowed = {"sigma_omega", "sigma_lambda", "sigma_ell"}
            unknown = set(sigma_override) - allowed
            if unknown:
                raise KeyError(
                    "Unknown volatility override(s): "
                    + ", ".join(sorted(unknown))
                )
            sigma_omega = sigma_override.get("sigma_omega", sigma_omega)
            sigma_lambda = sigma_override.get("sigma_lambda", sigma_lambda)
            sigma_ell = sigma_override.get("sigma_ell", sigma_ell)

        if min(sigma_omega, sigma_lambda, sigma_ell) < 0:
            raise ValueError("Volatility overrides must be nonnegative.")

        omega_variance = self.positive_part(
            np.asarray(omega) * (1.0 - np.asarray(omega))
        )
        lambda_variance = self.positive_part(
            np.asarray(lam) * (1.0 - np.asarray(lam))
        )

        return np.asarray(
            [
                sigma_omega * np.sqrt(omega_variance),
                sigma_lambda * np.sqrt(lambda_variance),
                sigma_ell * np.abs(np.asarray(ell)),
            ],
            dtype=float,
        )

    # Backward-compatible name used in keen_model_functions1.py.
    def sigma_terms(self, omega, lam, ell):
        return tuple(self.diffusion_terms(omega, lam, ell))

    def simulate_step(
        self,
        omega: float,
        lam: float,
        ell: float,
        dt: float,
        dW_omega: float,
        dW_lambda: float,
        dW_ell: float,
        *,
        sigma_override: Mapping[str, float] | None = None,
    ) -> tuple[float, float, float]:
        """
        Perform one unprojected Euler--Maruyama step.

        The dW inputs must be Brownian increments with variance dt, e.g.

            dW = rng.normal(0.0, np.sqrt(dt), size=3).

        This method does not clip the current state or the proposed update.
        """
        if dt <= 0:
            raise ValueError("dt must be strictly positive.")

        state = np.asarray([omega, lam, ell], dtype=float)
        dW = np.asarray([dW_omega, dW_lambda, dW_ell], dtype=float)

        if not np.all(np.isfinite(state)):
            raise ValueError("The current state must be finite.")
        if not np.all(np.isfinite(dW)):
            raise ValueError("The Brownian increments must be finite.")

        next_state = (
            state
            + self.drift(omega, lam, ell) * dt
            + self.diffusion_terms(
                omega,
                lam,
                ell,
                sigma_override=sigma_override,
            )
            * dW
        )

        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError(
                "The Euler--Maruyama step produced a non-finite state."
            )

        return tuple(float(value) for value in next_state)

    def simulate_path(
        self,
        x0: Sequence[float],
        T: float,
        dt: float,
        *,
        seed: int | None = None,
        sigma_schedule: Callable[
            [float, np.ndarray], Mapping[str, float] | None
        ]
        | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate an unprojected path with Euler--Maruyama.

        sigma_schedule(t, state) may return volatility overrides for stress
        windows. The returned state array has columns (omega, lambda, ell).
        """
        if T <= 0:
            raise ValueError("T must be strictly positive.")
        if dt <= 0:
            raise ValueError("dt must be strictly positive.")
        if len(x0) != 3:
            raise ValueError("x0 must contain (omega0, lambda0, ell0).")

        ratio = T / dt
        n_steps = int(round(ratio))
        if not np.isclose(ratio, n_steps, rtol=0.0, atol=1e-10):
            raise ValueError("T / dt must be an integer within tolerance.")

        times = np.arange(n_steps + 1, dtype=float) * dt
        states = np.empty((n_steps + 1, 3), dtype=float)
        states[0] = np.asarray(x0, dtype=float)

        if not np.all(np.isfinite(states[0])):
            raise ValueError("x0 must contain finite values.")

        rng = np.random.default_rng(seed)
        sqrt_dt = np.sqrt(dt)

        for index in range(n_steps):
            current = states[index]
            overrides = (
                sigma_schedule(times[index], current.copy())
                if sigma_schedule is not None
                else None
            )
            dW = rng.normal(0.0, sqrt_dt, size=3)
            states[index + 1] = self.simulate_step(
                *current,
                dt,
                *dW,
                sigma_override=overrides,
            )

        return times, states

    def deterministic_rhs(self, t: float, state: Sequence[float]) -> np.ndarray:
        """Right-hand side for deterministic ODE solvers."""
        del t
        if len(state) != 3:
            raise ValueError("state must contain (omega, lambda, ell).")
        return self.drift(*state)

    def financing_gap(self, omega, lam, ell):
        """Additive financing gap omega + kappa(pi) - 1 + Delta(pi)."""
        del lam
        pi = self.profit_share(omega, ell)
        return np.asarray(omega) + self.kappa(pi) - 1.0 + self.dividend(pi)

    def interest_growth_gap(self, omega, ell):
        """Coefficient multiplying ell in the deterministic debt drift."""
        pi = self.profit_share(omega, ell)
        return (
            self.params["r"]
            - self.kappa(pi) / self.params["nu"]
            + self.params["delta"]
            - self.inflation(omega)
        )

    def viability_diagnostics(self) -> dict[str, float | bool]:
        """Evaluate the sufficient upper-face viability inequalities."""
        wage_face = (
            self.phi(1.0)
            - self.params["alpha"]
            - (1.0 - self.params["gamma"]) * self.inflation(1.0)
        )
        employment_face = (
            self.params["kappa_max"] / self.params["nu"]
            - (
                self.params["delta"]
                + self.params["alpha"]
                + self.params["beta"]
            )
        )
        return {
            "wage_upper_face": float(wage_face),
            "employment_upper_face": float(employment_face),
            "wage_condition_satisfied": bool(wage_face <= 0.0),
            "employment_condition_satisfied": bool(employment_face <= 0.0),
        }

    @staticmethod
    def in_economic_region(omega, lam) -> bool:
        """Return True when both share coordinates lie in [0, 1]."""
        return bool(0.0 <= omega <= 1.0 and 0.0 <= lam <= 1.0)

    def summary_at_state(self, omega, lam, ell) -> dict[str, float | bool]:
        """Return a diagnostic summary at one state."""
        pi = float(self.profit_share(omega, ell))
        kap = float(self.kappa(pi))
        div = float(self.dividend(pi))
        infl = float(self.inflation(omega))
        b = self.drift(omega, lam, ell)
        s = self.diffusion_terms(omega, lam, ell)

        return {
            "omega": float(omega),
            "lambda": float(lam),
            "ell": float(ell),
            "in_economic_region": self.in_economic_region(omega, lam),
            "pi": pi,
            "kappa": kap,
            "Delta": div,
            "inflation": infl,
            "growth_rate": kap / self.params["nu"] - self.params["delta"],
            "financing_gap": float(self.financing_gap(omega, lam, ell)),
            "interest_growth_gap": float(
                self.interest_growth_gap(omega, ell)
            ),
            "b_omega": float(b[0]),
            "b_lambda": float(b[1]),
            "b_ell": float(b[2]),
            "sigma_omega": float(s[0]),
            "sigma_lambda": float(s[1]),
            "sigma_ell": float(s[2]),
        }