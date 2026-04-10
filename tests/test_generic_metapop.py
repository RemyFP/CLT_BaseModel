"""
Travel validation test — Phase 3, task 3.4.

Runs ConfigDrivenMetapopModel alongside FluMetapopModel on the same caseB
two-subpopulation data (5 age groups, 1 risk group), using identical initial
conditions, RNG seeds, simulation settings, and transition type
(binom_deterministic, timesteps_per_day=1).

Asserts that all 10 compartment trajectories are exactly equal for both
subpopulations.  This is the highest-risk test: any dict-lookup bug in the
travel functions will surface here.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import clt_toolkit as clt
import flu_core as flu
from conftest import subpop_inputs

from generic_core.config_parser import parse_model_config
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
CONFIG_PATH = BASE_PATH / "caseb_flu_generic_metapop_config.json"

NUM_DAYS = 50
RNG_SEED_1 = 88888
RNG_SEED_2 = 88888 + 1   # jumped(1) equivalent

COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_flu_metapop_model(settings):
    state1, params1, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2, params2, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop2")

    bit_gen1 = np.random.MT19937(RNG_SEED_1)
    bit_gen2 = bit_gen1.jumped(1)

    subpop1 = flu.FluSubpopModel(
        state1, params1, settings,
        np.random.Generator(bit_gen1), schedules_info, name="subpop1"
    )
    subpop2 = flu.FluSubpopModel(
        state2, params2, settings,
        np.random.Generator(bit_gen2), schedules_info, name="subpop2"
    )
    return flu.FluMetapopModel([subpop1, subpop2], mixing_params)


def _make_generic_metapop_model(settings):
    state1_flu, params1_flu, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2_flu, params2_flu, _, _, _ = subpop_inputs("caseB_subpop2")

    model_config = parse_model_config(CONFIG_PATH, schedules_input=schedules_info)

    # Extract travel_config from the S_to_E transition's rate_config
    s_to_e_tc = next(tc for tc in model_config.transitions if tc.name == "S_to_E")
    travel_config = s_to_e_tc.rate_config["travel_config"]

    A = params1_flu.num_age_groups
    R = params1_flu.num_risk_groups

    def _make_subpop(flu_state, flu_params, rng_seed, name):
        compartment_init = {c: getattr(flu_state, c) for c in COMPARTMENTS}
        epi_metric_init = {
            "M":  np.asarray(flu_state.M,  dtype=float),
            "MV": np.asarray(flu_state.MV, dtype=float),
        }
        state_init = build_state_from_config(model_config, compartment_init, epi_metric_init)
        params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)
        RNG = np.random.Generator(rng_seed)
        return ConfigDrivenSubpopModel(
            model_config=model_config,
            state_init=state_init,
            params=params,
            simulation_settings=settings,
            RNG=RNG,
            schedules_input=schedules_info,
            name=name,
        )

    bit_gen1 = np.random.MT19937(RNG_SEED_1)
    bit_gen2 = bit_gen1.jumped(1)

    subpop1 = _make_subpop(state1_flu, params1_flu, bit_gen1, "subpop1")
    subpop2 = _make_subpop(state2_flu, params2_flu, bit_gen2, "subpop2")

    return ConfigDrivenMetapopModel(
        subpop_models=[subpop1, subpop2],
        mixing_params=mixing_params,
        model_config=model_config,
        travel_config=travel_config,
    )


# ---------------------------------------------------------------------------
# Fixture: both models simulated for NUM_DAYS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_models():
    _state, _params, _mp, settings, _sched = subpop_inputs("caseB_subpop1")
    settings = clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC,
        "timesteps_per_day": 1,
        "save_daily_history": True,
    })
    ref_model = _make_flu_metapop_model(settings)
    gen_model = _make_generic_metapop_model(settings)

    ref_model.simulate_until_day(NUM_DAYS)
    gen_model.simulate_until_day(NUM_DAYS)

    return ref_model, gen_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subpop_name", ["subpop1", "subpop2"])
@pytest.mark.parametrize("compartment", COMPARTMENTS)
def test_compartment_trajectory_identical(both_models, subpop_name, compartment):
    ref_metapop, gen_metapop = both_models

    ref_subpop = ref_metapop.subpop_models[subpop_name]
    gen_subpop = gen_metapop.subpop_models[subpop_name]

    ref_hist = np.array(ref_subpop.compartments[compartment].history_vals_list)
    gen_hist = np.array(gen_subpop.compartments[compartment].history_vals_list)

    assert np.array_equal(ref_hist, gen_hist), (
        f"{subpop_name} compartment '{compartment}' mismatch; "
        f"max diff = {np.max(np.abs(ref_hist.astype(float) - gen_hist.astype(float))):.6g}; "
        f"first divergence at day "
        f"{int(np.argmax(np.any(ref_hist != gen_hist, axis=(1, 2))))}"
    )


@pytest.mark.parametrize("subpop_name", ["subpop1", "subpop2"])
def test_epi_metric_M_trajectory(both_models, subpop_name):
    ref_metapop, gen_metapop = both_models
    ref_hist = np.array(ref_metapop.subpop_models[subpop_name].epi_metrics["M"].history_vals_list)
    gen_hist = np.array(gen_metapop.subpop_models[subpop_name].epi_metrics["M"].history_vals_list)
    assert np.allclose(ref_hist, gen_hist, rtol=1e-14), (
        f"{subpop_name} M mismatch; max diff = {np.max(np.abs(ref_hist - gen_hist)):.6g}"
    )


@pytest.mark.parametrize("subpop_name", ["subpop1", "subpop2"])
def test_epi_metric_MV_trajectory(both_models, subpop_name):
    ref_metapop, gen_metapop = both_models
    ref_hist = np.array(ref_metapop.subpop_models[subpop_name].epi_metrics["MV"].history_vals_list)
    gen_hist = np.array(gen_metapop.subpop_models[subpop_name].epi_metrics["MV"].history_vals_list)
    assert np.allclose(ref_hist, gen_hist, rtol=1e-14), (
        f"{subpop_name} MV mismatch; max diff = {np.max(np.abs(ref_hist - gen_hist)):.6g}"
    )


def test_population_conserved(both_models):
    """Generic model conserves total population across both subpops combined."""
    _, gen_metapop = both_models
    total_ref_pop = 0.0
    for subpop in gen_metapop.subpop_models.values():
        total_ref_pop += float(np.sum(subpop.params.total_pop_age_risk))

    for day_idx in range(NUM_DAYS):
        day_total = 0.0
        for subpop in gen_metapop.subpop_models.values():
            for comp in COMPARTMENTS:
                day_total += float(np.sum(subpop.compartments[comp].history_vals_list[day_idx]))
        assert abs(day_total - total_ref_pop) < 1e-3, (
            f"Population not conserved on day {day_idx}: {day_total:.2f} != {total_ref_pop:.2f}"
        )
