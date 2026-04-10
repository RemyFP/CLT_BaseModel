"""
sir_demo.py — Minimal working example for generic_core.

Loads sir_config.json and runs a simple SIR simulation with one age group
and one risk group, using binom_deterministic transitions.

Prints daily S, I, R values for the first 20 days.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import clt_toolkit as clt

from generic_core.config_parser import parse_model_config
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel
import flu_core as flu

CONFIG_PATH = Path(__file__).parent / "sir_config.json"
NUM_DAYS = 30

A, R = 1, 1           # single age group, single risk group
N = 10000            # total population

# -----------------------------------------------------------------
# 1. Parse config
# -----------------------------------------------------------------
model_config = parse_model_config(CONFIG_PATH, schedules_input=None)

# -----------------------------------------------------------------
# 2. Build initial state and params
# -----------------------------------------------------------------
compartment_init = {
    "S": np.array([[N - 1]], dtype=float),
    "I": np.array([[1]],     dtype=float),
    "R": np.array([[0]],      dtype=float),
}
state_init = build_state_from_config(model_config, compartment_init, epi_metric_init={})
params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)

# -----------------------------------------------------------------
# 3. Build single-population model wrapped in a one-subpop metapop
#    (ConfigDrivenSubpopModel requires SimulationSettings with start_real_date)
# -----------------------------------------------------------------
simulation_settings = clt.make_dataclass_from_json(
    clt.utils.PROJECT_ROOT / "tests" / "test_input_files" / "simulation_settings.json",
    clt.SimulationSettings,
)
simulation_settings = clt.updated_dataclass(simulation_settings, {
    "transition_type":     clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND,
    "timesteps_per_day":   1,
    "save_daily_history":  True,
})

RNG = np.random.default_rng(42)

subpop = ConfigDrivenSubpopModel(
    model_config=model_config,
    state_init=state_init,
    params=params,
    simulation_settings=simulation_settings,
    RNG=RNG,
    schedules_input=None,
    name="sir_subpop",
)

# Wrap in a trivial single-location metapop
mixing_params = flu.FluMixingParams(travel_proportions=np.array([[1.0]]), num_locations=1)
s_to_i_tc = next(tc for tc in model_config.transitions if tc.name == "S_to_I")
metapop = ConfigDrivenMetapopModel(
    subpop_models=[subpop],
    mixing_params=mixing_params,
    model_config=model_config,
    travel_config={},   # no travel config needed for constant_param transitions
)

# -----------------------------------------------------------------
# 4. Simulate
# -----------------------------------------------------------------
metapop.simulate_until_day(NUM_DAYS)

# -----------------------------------------------------------------
# 5. Print results
# -----------------------------------------------------------------
print(f"{'Day':>4}  {'S':>8}  {'I':>8}  {'R':>8}")
for day in range(NUM_DAYS):
    S = int(np.sum(subpop.compartments["S"].history_vals_list[day]))
    I = int(np.sum(subpop.compartments["I"].history_vals_list[day]))
    R = int(np.sum(subpop.compartments["R"].history_vals_list[day]))
    print(f"{day + 1:>4}  {S:>8}  {I:>8}  {R:>8}")
