import json
import os
import numpy as np

from parameters import SettlementParams
from analysis import SCENARIOS, run_scenario, compute_survival_metrics
from ode_model import solar_irradiance_factor


def export_simulation_json(out_path: str = "web/sim_data.json"):
    """Exports ODE/PDE trajectories so the Three.js 3D viewer is data-driven."""
    p = SettlementParams()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "meta": {
            "crew_size": p.crew_size,
            "t_max": p.t_span[1],
            "wall_thickness": p.wall_thickness,
            "thresholds": p.thresholds,
            "capacities": {
                "water": 5500.0,
                "oxygen": 900.0,
                "co2": 120.0,
                "food": p.crop_capacity,
                "energy": p.battery_capacity,
            },
        },
        "scenarios": {},
    }

    for key, title in SCENARIOS.items():
        sol = run_scenario(key, p, num_points=501)
        metrics = compute_survival_metrics(sol, p)
        phi_series = [float(solar_irradiance_factor(t, key, p)) for t in sol.t]

        payload["scenarios"][key] = {
            "title": title,
            "metrics": metrics,
            "t": np.round(sol.t, 2).tolist(),
            "water": np.round(sol.y[0], 2).tolist(),
            "oxygen": np.round(sol.y[1], 2).tolist(),
            "co2": np.round(sol.y[2], 2).tolist(),
            "food": np.round(sol.y[3], 2).tolist(),
            "energy": np.round(sol.y[4], 2).tolist(),
            "wall_temp_c": np.round(sol.y[5:] - 273.15, 2).tolist(),
            "solar_factor": np.round(phi_series, 3).tolist(),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Exported 3D simulation dataset to {out_path}")


if __name__ == "__main__":
    export_simulation_json()
