import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from parameters import SettlementParams
from ode_model import build_initial_state, settlement_rhs


SCENARIOS = {
    "closed_no_isru": "Scenario A: Closed-Loop (No ISRU Resupply)",
    "sustainable_isru": "Scenario B: Sustainable ISRU + Nuclear/Solar Loop",
    "dust_storm": "Scenario C: ISRU + 150-Day Global Dust Storm (Days 320–470)",
}


def run_scenario(scenario: str, p: SettlementParams, num_points: int = 1001):
    """Integrates the coupled ODE/PDE system using RK45."""
    t_eval = np.linspace(p.t_span[0], p.t_span[1], num_points)
    y0 = build_initial_state(p)
    return solve_ivp(
        fun=lambda t, y: settlement_rhs(t, y, p, scenario),
        t_span=p.t_span,
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )


def compute_survival_metrics(sol, p: SettlementParams) -> dict:
    """Computes minimum values and first time-to-depletion below safety thresholds."""
    t = sol.t
    series = {
        "water": sol.y[0],
        "oxygen": sol.y[1],
        "food": sol.y[3],
        "energy": sol.y[4],
    }
    metrics = {}
    for name, arr in series.items():
        thresh = p.thresholds[name]
        breach_idx = np.where(arr < thresh)[0]
        depletion_day = float(t[breach_idx[0]]) if len(breach_idx) > 0 else None
        metrics[name] = {
            "min_value": float(np.min(arr)),
            "final_value": float(arr[-1]),
            "threshold": thresh,
            "depletion_day": depletion_day,
        }
    return metrics


def plot_all_scenarios(results: dict, p: SettlementParams, out_dir: str = "outputs"):
    """Generates comparative 2D time-series and 1D wall PDE thermal heatmap."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)

    panels = [
        (0, "Potable Water W(t) [L]", "water"),
        (1, "Breathable Oxygen O(t) [kg]", "oxygen"),
        (2, "Cabin CO2 C(t) [kg]", None),
        (3, "Crop Biomass F(t) [kg]", "food"),
        (4, "Stored Energy E(t) [kWh]", "energy"),
    ]

    colors = {
        "closed_no_isru": "#d95f02",
        "sustainable_isru": "#1b9e77",
        "dust_storm": "#7570b3",
    }

    for idx, (row_idx, ylabel, thresh_key) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        for key, label in SCENARIOS.items():
            sol = results[key]
            ax.plot(sol.t, sol.y[row_idx], label=label, color=colors[key], lw=2.0)

        if thresh_key is not None:
            ax.axhline(
                p.thresholds[thresh_key],
                color="crimson",
                linestyle="--",
                lw=1.2,
                label="Critical Threshold",
            )

        ax.axvspan(*p.storm_window, color="sandybrown", alpha=0.18, label="Dust Storm Window")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [Earth Days]")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    ax_pde = axes[2, 1]
    sol_storm = results["dust_storm"]
    T_field = sol_storm.y[5:, :]
    im = ax_pde.imshow(
        T_field - 273.15,
        aspect="auto",
        origin="lower",
        extent=[sol_storm.t[0], sol_storm.t[-1], 0.0, p.wall_thickness],
        cmap="coolwarm",
    )
    ax_pde.set_title("1D Wall PDE Temperature T(x, t) [°C] (Scenario C)")
    ax_pde.set_xlabel("Time [Earth Days]")
    ax_pde.set_ylabel("Wall Depth x [m] (0=Mars, 0.8=Cabin)")
    fig.colorbar(im, ax=ax_pde, label="Temp [°C]")

    fig.suptitle(
        "Mars Settlement Resource Dynamics (Coupled ODEs + 1D Wall PDE)",
        fontsize=14,
        fontweight="bold",
    )
    plot_path = os.path.join(out_dir, "mars_simulation_diagnostics.png")
    fig.savefig(plot_path, dpi=200)
    print(f"Saved diagnostic plot to {plot_path}")


if __name__ == "__main__":
    params = SettlementParams()
    all_results = {}

    for scen_key, scen_title in SCENARIOS.items():
        sol = run_scenario(scen_key, params)
        all_results[scen_key] = sol
        metrics = compute_survival_metrics(sol, params)

        print(f"\n=== {scen_title} ===")
        for res_name, info in metrics.items():
            status = (
                f"FAILED on Day {info['depletion_day']:.1f}"
                if info["depletion_day"] is not None
                else "SUSTAINED (1000 days)"
            )
            print(
                f"  {res_name.capitalize():<7}: min={info['min_value']:7.1f}, "
                f"final={info['final_value']:7.1f} | {status}"
            )

    plot_all_scenarios(all_results, params)
