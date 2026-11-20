import numpy as np
from parameters import SettlementParams


def monod(resource: float, half_sat: float) -> float:
    """Smooth Michaelis-Menten / Monod saturation function in [0, 1)."""
    r_pos = max(resource, 0.0)
    return r_pos / (r_pos + half_sat)


def solar_irradiance_factor(t: float, scenario: str, p: SettlementParams) -> float:
    """Returns normalized solar factor phi(t) in [0, 1]."""
    if scenario == "autonomous_steady":
        return 0.92
    orbital = 0.92 + 0.08 * np.cos(2.0 * np.pi * t / 668.6)
    if scenario == "dust_storm":
        t_start, t_end = p.storm_window
        if t_start <= t <= t_end:
            return orbital * p.storm_solar_factor
    return orbital


def build_initial_state(p: SettlementParams) -> np.ndarray:
    """Constructs combined ODE + PDE initial state vector y0."""
    w0, o0, c0, f0, e0 = p.initial_resources
    t_wall0 = np.linspace(p.t_mars_mean, p.t_cabin_target, p.wall_nodes)
    return np.concatenate(([w0, o0, c0, f0, e0], t_wall0))


def settlement_rhs(t: float, y: np.ndarray, p: SettlementParams, scenario: str) -> np.ndarray:
    """
    Right-hand side dy/dt for the coupled Mars life-support ODEs and
    1D Method-of-Lines wall heat conduction PDE.
    """
    W, O, C, F, E = y[:5]
    T = y[5:]

    if scenario == "closed_no_isru":
        isru_water = 0.0
        moxie_o2 = 0.0
        moxie_co2 = 0.0
        isru_power = 0.0
        r_water = 0.86
    else:
        isru_water = p.isru_water_rate
        moxie_o2 = p.moxie_o2_rate
        moxie_co2 = p.moxie_co2_intake
        isru_power = p.isru_power_load
        r_water = p.water_recycling

    # 1. 1D Heat Conduction PDE via Method of Lines
    M = p.wall_nodes
    dx = p.wall_thickness / (M - 1)
    dTdt = np.zeros(M)

    seasonal = 0.0 if scenario == "autonomous_steady" else np.sin(2.0 * np.pi * t / 668.6)
    t_mars = p.t_mars_mean + p.t_mars_amp * seasonal
    t_interior_wall = T[-1]

    temp_deficit = max(0.0, p.t_cabin_target - t_interior_wall)
    hvac_power_kwh = p.hvac_gain * temp_deficit * monod(E, p.k_energy)

    dTdt[1:-1] = p.thermal_diffusivity * (T[2:] - 2.0 * T[1:-1] + T[:-2]) / (dx ** 2)

    flux_ext = (p.h_exterior / p.wall_conductivity) * (t_mars - T[0])
    dTdt[0] = p.thermal_diffusivity * (2.0 * (T[1] - T[0] + dx * flux_ext)) / (dx ** 2)

    flux_int = (p.h_interior / p.wall_conductivity) * (p.t_cabin_target - T[-1])
    dTdt[-1] = p.thermal_diffusivity * (2.0 * (T[-2] - T[-1] + dx * flux_int)) / (dx ** 2)

    # 2. Saturation factors
    m_w = monod(W, p.k_water)
    m_c = monod(C, p.k_co2)
    m_c_moxie = monod(max(0.0, C - 10.0), p.k_co2)
    m_e = monod(E, p.k_energy)

    # 3. Crop growth & harvest
    F_pos = max(F, 0.0)
    growth = p.crop_growth_rate * F_pos * (1.0 - F_pos / p.crop_capacity) * m_w * m_c * m_e
    harvest = min(p.food_use_per_person * p.crew_size, 0.5 * F_pos)
    dFdt = growth - harvest

    # 4. Water balance
    crew_water_demand = p.water_use_per_person * p.crew_size
    water_inflow = (r_water * crew_water_demand + isru_water * m_e) * max(0.0, 1.0 - (max(W, 0.0) / 6000.0) ** 4)
    dWdt = (
        water_inflow
        - crew_water_demand
        - p.crop_water_uptake * growth
    )

    # 5. Oxygen balance
    dOdt = (
        p.o2_per_crop * growth
        + moxie_o2 * m_e
        - p.o2_use_per_person * p.crew_size
        - p.o2_leak_rate * max(O, 0.0)
    )

    # 6. CO2 balance
    dCdt = (
        p.co2_prod_per_person * p.crew_size
        - p.co2_per_crop * growth
        - 0.0
        - p.co2_scrub_rate * max(C, 0.0)
    )

    # 7. Electrical energy balance
    phi = solar_irradiance_factor(t, scenario, p)
    generation = phi * p.solar_power_max + p.nuclear_power
    charge_taper = max(0.0, 1.0 - (max(E, 0.0) / p.battery_capacity) ** 4)
    consumption = p.base_load + isru_power * m_e + hvac_power_kwh
    dEdt = generation * charge_taper - consumption - p.battery_loss_rate * max(E, 0.0)

    # Clamp non-physical negative drift at zero boundary
    if W <= 0.0 and dWdt < 0.0:
        dWdt = 0.0
    if O <= 0.0 and dOdt < 0.0:
        dOdt = 0.0
    if C <= 0.0 and dCdt < 0.0:
        dCdt = 0.0
    if F <= 0.0 and dFdt < 0.0:
        dFdt = 0.0
    if E <= 0.0 and dEdt < 0.0:
        dEdt = 0.0

    return np.concatenate(([dWdt, dOdt, dCdt, dFdt, dEdt], dTdt))
