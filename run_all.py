import subprocess
import sys


def main():
    steps = [
        ("Running Coupled ODE/PDE Scenario Simulation & Plots", [sys.executable, "src/analysis.py"]),
        ("Running Jacobian Stability & Bifurcation Analysis", [sys.executable, "src/stability.py"]),
        ("Exporting State Trajectories for 3D Three.js Viewer", [sys.executable, "src/export_web_data.py"]),
    ]
    for title, cmd in steps:
        print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
        subprocess.run(cmd, check=True)

    print("\nAll outputs generated in ./outputs/ and ./web/sim_data.json.")
    print("To view the 3D simulation, run: python -m http.server 8000")
    print("Then open: http://localhost:8000/web/")


if __name__ == "__main__":
    main()
