# Resource Management on Mars: Can a Settlement Remain Self-Sustaining?

A coupled Ordinary & Partial Differential Equation (ODE/PDE) simulation of a 6-person Martian outpost over 1,000 Earth days, featuring nonlinear equilibrium/Jacobian stability analysis and an interactive 3D WebGL (Three.js) visualization driven directly by simulation states.

## Repository Structure
- `src/parameters.py` — Physical constants, initial conditions, scenario settings, and safety thresholds.
- `src/ode_model.py` — Coupled 5-variable life-support ODEs ($W, O, C, F, E$) + 1D regolith wall heat conduction PDE solved via the Method of Lines.
- `src/analysis.py` — Numerical integration (`scipy.integrate.solve_ivp`), time-to-depletion metrics, and diagnostic plots.
- `src/stability.py` — Fixed-point equilibrium solver, numerical Jacobian eigenvalues, and water-recycling bifurcation curve.
- `src/export_web_data.py` — Packages ODE/PDE time series into `web/sim_data.json`.
- `web/index.html` — Interactive 3D Mars base with mouse orbit controls, dynamic resource tanks, thermal wall cutaway, and dust-storm optics.
- `tests/test_model.py` — Automated verification of mass balance, positivity, and PDE boundary fluxes.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run full simulation, stability analysis, diagnostic plots, and 3D JSON export
python run_all.py

# Launch the 3D simulation in your browser
python -m http.server 8000
# Open http://localhost:8000/web/
```
