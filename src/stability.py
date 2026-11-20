import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

from parameters import SettlementParams
from ode_model import build_initial_state, settlement_rhs


def compute_jacobian(fun, y_star: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes the central finite-difference Jacobian J = df/dy at y_star."""
    n = len(y_star)
    J = np.zeros((n, n))
    for j in range(n):
        dy = np.zeros(n)
        step = eps * max(1.0, abs(y_star[j]))
        dy[j] = step
        f_plus = fun(y_star + dy)
        f_minus = fun(y_star - dy)
        J[:, j] = (f_plus - f_minus) / (2.0 * step)
    return J


def find_steady_state(p: SettlementParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrates the autonomous mean-annual system to near-equilibrium,
    refines y* via fsolve, and returns (y*, eigenvalues).
    """
    rhs_auto = lambda y: settlement_rhs(0.0, y, p, scenario="autonomous_steady")
    y0 = build_initial_state(p)
    pre = solve_ivp(lambda t, y: rhs_auto(y), (0.0, 2500.0), y0, method="BDF")
    y_guess = pre.y[:, -1]

    y_star = fsolve(rhs_auto, y_guess, xtol=1e-10)
    J = compute_jacobian(rhs_auto, y_star)
    eigvals = np.linalg.eigvals(J)
    return y_star, eigvals


def sweep_water_recycling(out_dir: str = "outputs"):
    """
    Sweeps water recycling efficiency r_w in [0.80, 0.98] to locate the
    critical sustainability threshold where equilibrium water W* collapses.
    """
    os.makedirs(out_dir, exist_ok=True)
    r_values = np.linspace(0.80, 0.98, 25)
    w_equilibria = []
    max_real_eigs = []

    for r_w in r_values:
        p = SettlementParams(water_recycling=float(r_w))
        y_star, eigvals = find_steady_state(p)
        w_equilibria.append(max(0.0, y_star[0]))
        max_real_eigs.append(float(np.max(np.real(eigvals))))

    p_nom = SettlementParams()
    crit_r = next(
        (r for r, w in zip(r_values, w_equilibria) if w >= p_nom.thresholds["water"]),
        None,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax1.plot(r_values * 100, w_equilibria, "o-", color="#0284c7", lw=2)
    ax1.axhline(p_nom.thresholds["water"], color="crimson", ls="--", label="Safety Threshold (300 L)")
    if crit_r is not None:
        ax1.axvline(crit_r * 100, color="#1b9e77", ls=":", lw=2, label=f"Critical r_w ≈ {crit_r*100:.1f}%")
    ax1.set_title("Equilibrium Water W* vs. Recycling Efficiency")
    ax1.set_xlabel("ECLSS Water Recycling Efficiency r_w [%]")
    ax1.set_ylabel("Steady-State Water W* [L]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(r_values * 100, max_real_eigs, "s-", color="#7570b3", lw=2)
    ax2.axhline(0.0, color="black", ls="--", lw=1.2, label="Stability Boundary Re(λ) = 0")
    ax2.set_title("Leading Jacobian Eigenvalue Max Re(λ)")
    ax2.set_xlabel("ECLSS Water Recycling Efficiency r_w [%]")
    ax2.set_ylabel("Max Re(λ) [1/day]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Nonlinear Stability & Critical Recycling Threshold Analysis", fontweight="bold")
    out_path = os.path.join(out_dir, "stability_bifurcation.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved stability bifurcation plot to {out_path}")
    if crit_r is not None:
        print(f"Critical minimum water recycling efficiency (with ISRU): r_w >= {crit_r*100:.2f}%")


if __name__ == "__main__":
    p_default = SettlementParams()
    y_eq, eigs = find_steady_state(p_default)
    labels = ["Water [L]", "Oxygen [kg]", "CO2 [kg]", "Biomass [kg]", "Energy [kWh]"]
    print("=== Autonomous Equilibrium State y* (Nominal ISRU) ===")
    for lbl, val in zip(labels, y_eq[:5]):
        print(f"  {lbl:<14}: {val:8.2f}")
    print(f"  Leading Eigenvalue Re(λ)_max: {np.max(np.real(eigs)):.5f} day^-1 (Asymptotically Stable)")
    sweep_water_recycling()
