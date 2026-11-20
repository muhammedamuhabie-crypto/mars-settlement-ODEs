from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SettlementParams:
    """Physical and operational parameters for the Mars settlement model."""

    # Crew & time horizon
    crew_size: int = 6
    t_span: Tuple[float, float] = (0.0, 1000.0)

    # Water loop [L]
    water_recycling: float = 0.94          # r_w (0 < r_w < 1)
    water_use_per_person: float = 25.0     # c_w [L / person / day]
    isru_water_rate: float = 24.0          # I_w [L / day]
    crop_water_uptake: float = 1.8         # alpha_w [L water / kg biomass growth]

    # Atmosphere loop [kg]
    o2_use_per_person: float = 0.84        # c_o [kg / person / day]
    co2_prod_per_person: float = 1.00      # p_c [kg / person / day]
    o2_leak_rate: float = 0.0015           # lambda_o [1 / day]
    moxie_o2_rate: float = 4.2             # eta_MOXIE [kg O2 / day]
    moxie_co2_intake: float = 17.2          # mu_MOXIE [kg CO2 / day]
    co2_scrub_rate: float = 0.005           # r_c [1 / day]

    # Biomass / Greenhouse loop [kg]
    food_use_per_person: float = 0.62      # c_f [kg / person / day]
    crop_growth_rate: float = 0.085        # mu_f [1 / day]
    crop_capacity: float = 1200.0          # F_max [kg]
    o2_per_crop: float = 0.72              # gamma_FO [kg O2 / kg biomass]
    co2_per_crop: float = 1.05             # gamma_FC [kg CO2 / kg biomass]

    # Energy loop [kWh]
    solar_power_max: float = 140.0         # P_solar [kWh / day]
    nuclear_power: float = 240.0           # P_nuc [kWh / day]
    base_load: float = 210.0               # P_base [kWh / day]
    isru_power_load: float = 95.0          # P_ISRU [kWh / day]
    battery_capacity: float = 1500.0       # E_max [kWh]
    battery_loss_rate: float = 0.002       # lambda_e [1 / day]

    # Monod half-saturation constants
    k_water: float = 400.0                 # [L]
    k_co2: float = 2.5                    # [kg]
    k_energy: float = 120.0                # [kWh]

    # 1D Regolith Wall Heat Conduction PDE parameters
    wall_nodes: int = 12                   # Spatial discretization points M
    wall_thickness: float = 0.8            # L [m]
    thermal_diffusivity: float = 0.015     # alpha [m^2 / day]
    wall_conductivity: float = 0.85        # k [W / (m*K)]
    h_exterior: float = 4.5                # Exterior convection [W / (m^2*K)]
    h_interior: float = 8.0                # Interior cabin convection [W / (m^2*K)]
    t_mars_mean: float = 210.0             # Mean Mars surface temp [K] (-63 C)
    t_mars_amp: float = 18.0               # Seasonal swing [K]
    t_cabin_target: float = 294.0          # Target habitat temp [K] (21 C)
    hvac_gain: float = 1.6                 # Electrical load scaling [kWh/day per K deficit]

    # Dust storm modifier (active only in storm scenario)
    storm_window: Tuple[float, float] = (320.0, 470.0)
    storm_solar_factor: float = 0.12       # 88% drop in solar irradiance

    # Initial state: [W0, O0, C0, F0, E0]
    initial_resources: Tuple[float, float, float, float, float] = (
        4500.0,   # Water [L]
        650.0,    # Oxygen [kg]
        45.0,     # CO2 [kg]
        420.0,    # Crop biomass [kg]
        1100.0,   # Stored energy [kWh]
    )

    # Critical survival thresholds (below = life-support failure)
    thresholds: dict = field(default_factory=lambda: {
        "water": 300.0,    # [L]
        "oxygen": 120.0,   # [kg]
        "food": 40.0,      # [kg]
        "energy": 80.0,    # [kWh]
    })
