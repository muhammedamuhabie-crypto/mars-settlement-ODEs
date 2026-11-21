import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from parameters import SettlementParams
from ode_model import build_initial_state, settlement_rhs
from analysis import run_scenario


def test_state_vector_dimensions():
    p = SettlementParams(wall_nodes=10)
    y0 = build_initial_state(p)
    assert y0.shape == (5 + 10,)
    dy = settlement_rhs(0.0, y0, p, "sustainable_isru")
    assert dy.shape == y0.shape


def test_positivity_and_no_nans():
    p = SettlementParams(t_span=(0.0, 200.0))
    sol = run_scenario("closed_no_isru", p, num_points=201)
    assert sol.success
    assert not np.isnan(sol.y).any()
    assert sol.y[0].min() >= -1e-5  # Small numerical tolerance for water [L]
    assert np.all(sol.y[1:5] >= -1e-6)


def test_dust_storm_reduces_energy():
    p = SettlementParams(t_span=(0.0, 450.0))
    sol_clear = run_scenario("sustainable_isru", p, num_points=451)
    sol_storm = run_scenario("dust_storm", p, num_points=451)
    # At day 420 (inside storm window 320-470), energy must be lower than clear sky
    assert sol_storm.y[4, 420] < sol_clear.y[4, 420]
