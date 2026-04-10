"""
SIHR validation test — Phase 2, task 2.6.

Runs ConfigDrivenSubpopModel from sihr_generic_config.json alongside
SIHRSubpopModel with identical initial conditions, RNG seed, and transition
type, then asserts that all compartment trajectories are exactly equal.

A custom rate template "freq_dependent_infection" is registered before
parsing to handle the S→I density-dependent infection rate
(state.I * beta / total_pop_age_risk) which is not expressible via the
built-in templates.
"""

import numpy as np
import pytest

import clt_toolkit as clt
from SIHR_core.SIHR_components import SIHRSubpopModel

from generic_core.rate_templates import RateTemplate, register_rate_template
from generic_core.config_parser import parse_model_config
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
CONFIG_PATH = BASE_PATH / "sihr_generic_config.json"

# ---------------------------------------------------------------------------
# Custom rate template for S→I (not part of built-in registry)
# ---------------------------------------------------------------------------

class FreqDependentInfectionRate(RateTemplate):
    """
    Rate = state.I * beta / total_pop_age_risk.

    Matches SusceptibleToInfected.get_current_rate() in SIHR_core.
    """

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        for key in ("beta_param", "infectious_compartment"):
            if key not in rate_config:
                raise ValueError(
                    f"FreqDependentInfectionRate: missing required key '{key}'"
                )
        if rate_config["beta_param"] not in param_names:
            raise ValueError(
                f"FreqDependentInfectionRate: param '{rate_config['beta_param']}' not in model params"
            )
        if rate_config["infectious_compartment"] not in compartment_names:
            raise ValueError(
                f"FreqDependentInfectionRate: compartment "
                f"'{rate_config['infectious_compartment']}' not in model compartments"
            )

    def numpy_rate(self, state, params, rate_config):
        I = state.compartments[rate_config["infectious_compartment"]]
        beta = params.params[rate_config["beta_param"]]
        return I * beta / params.total_pop_age_risk

    def torch_rate(self, state_dict, params_dict, rate_config):
        import torch
        I = state_dict[rate_config["infectious_compartment"]]
        beta = params_dict[rate_config["beta_param"]]
        total_pop = params_dict["total_pop_age_risk"]
        return I * beta / total_pop


# Register once at module load time
register_rate_template("freq_dependent_infection", FreqDependentInfectionRate())


# ---------------------------------------------------------------------------
# Shared initial conditions: 1×1 (1 age group, 1 risk group)
# ---------------------------------------------------------------------------

_S0 = np.array([[9990.0]])
_I0 = np.array([[10.0]])
_H0 = np.array([[0.0]])
_R0 = np.array([[0.0]])

_COMPARTMENTS_DICT = {"S": _S0, "I": _I0, "H": _H0, "R": _R0}

_PARAMS_DICT = {
    "beta":             0.3,
    "I_to_H_rate":      0.2,
    "I_to_R_rate":      0.1,
    "H_to_R_rate":      0.15,
    "I_to_H_prop":      np.array([[0.3]]),
    "num_age_groups":   1,
    "num_risk_groups":  1,
    "total_pop_age_risk": _S0 + _I0,
}

_SIM_SETTINGS = {
    "timesteps_per_day": 7,
    "transition_type": "binom_deterministic",
    "start_real_date": "2022-08-08",
    "save_daily_history": True,
    "transition_variables_to_save": [],
}

NUM_DAYS = 100
RNG_SEED = 987654321


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_sihr_model():
    RNG = np.random.Generator(np.random.MT19937(RNG_SEED))
    return SIHRSubpopModel(
        compartments_epi_metrics_dict=_COMPARTMENTS_DICT,
        params_dict=_PARAMS_DICT,
        simulation_settings_dict=_SIM_SETTINGS,
        RNG=RNG,
        name="sihr_ref",
    )


def _make_generic_model():
    model_config = parse_model_config(CONFIG_PATH)

    state_init = build_state_from_config(
        model_config,
        compartment_init={"S": _S0, "I": _I0, "H": _H0, "R": _R0},
        epi_metric_init={},
    )
    params = build_params_from_config(model_config, num_age_groups=1, num_risk_groups=1)
    settings = clt.make_dataclass_from_dict(clt.SimulationSettings, _SIM_SETTINGS)
    RNG = np.random.Generator(np.random.MT19937(RNG_SEED))

    return ConfigDrivenSubpopModel(
        model_config=model_config,
        state_init=state_init,
        params=params,
        simulation_settings=settings,
        RNG=RNG,
        schedules_input=None,
        name="sihr_generic",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSIHRValidation:

    @pytest.fixture(scope="class")
    def both_models(self):
        ref_model = _make_sihr_model()
        gen_model = _make_generic_model()
        ref_model.simulate_until_day(NUM_DAYS)
        gen_model.simulate_until_day(NUM_DAYS)
        return ref_model, gen_model

    def test_S_trajectory(self, both_models):
        ref, gen = both_models
        ref_hist = np.array(ref.compartments["S"].history_vals_list)
        gen_hist = np.array(gen.compartments["S"].history_vals_list)
        assert np.array_equal(ref_hist, gen_hist), \
            f"S mismatch: max diff = {np.max(np.abs(ref_hist - gen_hist))}"

    def test_I_trajectory(self, both_models):
        ref, gen = both_models
        ref_hist = np.array(ref.compartments["I"].history_vals_list)
        gen_hist = np.array(gen.compartments["I"].history_vals_list)
        assert np.array_equal(ref_hist, gen_hist)

    def test_H_trajectory(self, both_models):
        ref, gen = both_models
        ref_hist = np.array(ref.compartments["H"].history_vals_list)
        gen_hist = np.array(gen.compartments["H"].history_vals_list)
        assert np.array_equal(ref_hist, gen_hist)

    def test_R_trajectory(self, both_models):
        ref, gen = both_models
        ref_hist = np.array(ref.compartments["R"].history_vals_list)
        gen_hist = np.array(gen.compartments["R"].history_vals_list)
        assert np.array_equal(ref_hist, gen_hist)

    def test_population_conserved(self, both_models):
        _, gen = both_models
        total_pop = float(np.sum(_S0) + np.sum(_I0))
        for day_vals in zip(
            gen.compartments["S"].history_vals_list,
            gen.compartments["I"].history_vals_list,
            gen.compartments["H"].history_vals_list,
            gen.compartments["R"].history_vals_list,
        ):
            day_total = sum(np.sum(v) for v in day_vals)
            assert abs(day_total - total_pop) < 1e-6, \
                f"Population not conserved: {day_total} != {total_pop}"
