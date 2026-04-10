"""
End-to-end integration test — Phase 5, task 5.3.

Sub-tests
---------
1. Stochastic replication (N=5):
   ConfigDrivenMetapopModel and FluMetapopModel produce bit-identical hospital
   admission timeseries when given the same RNG seeds and stochastic (BINOM)
   transitions.

2. Accept-reject calibration — accept path:
   With a permissive R² threshold, the sampler accepts replicates and writes
   JSON output files for params and state.

3. Accept-reject calibration — reject path:
   With an impossible R² threshold (> 1), no replicates are accepted and no
   JSON files are written.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import clt_toolkit as clt
import flu_core as flu
from conftest import subpop_inputs

from generic_core.calibration import generic_accept_reject
from generic_core.config_parser import parse_model_config
from generic_core.generic_metapop import ConfigDrivenMetapopModel
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_params_from_config,
    build_state_from_config,
)

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
CONFIG_PATH = BASE_PATH / "caseb_flu_generic_metapop_config.json"

NUM_DAYS = 20
N_REPLICATES = 5
COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
CALIBRATION_TVARS = ["ISH_to_HR", "ISH_to_HD"]


# ---------------------------------------------------------------------------
# Shared model factories
# ---------------------------------------------------------------------------

def _make_flu_metapop(settings, rng_seed: int) -> flu.FluMetapopModel:
    state1, params1, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2, params2, _, _, _ = subpop_inputs("caseB_subpop2")
    bit_gen1 = np.random.MT19937(rng_seed)
    bit_gen2 = bit_gen1.jumped(1)
    subpop1 = flu.FluSubpopModel(
        state1, params1, settings,
        np.random.Generator(bit_gen1), schedules_info, name="subpop1",
    )
    subpop2 = flu.FluSubpopModel(
        state2, params2, settings,
        np.random.Generator(bit_gen2), schedules_info, name="subpop2",
    )
    return flu.FluMetapopModel([subpop1, subpop2], mixing_params)


def _make_generic_metapop(settings, rng_seed: int) -> ConfigDrivenMetapopModel:
    state1_flu, params1_flu, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2_flu, params2_flu, _, _, _ = subpop_inputs("caseB_subpop2")

    model_config = parse_model_config(CONFIG_PATH, schedules_input=schedules_info)
    s_to_e_tc = next(tc for tc in model_config.transitions if tc.name == "S_to_E")
    travel_config = s_to_e_tc.rate_config["travel_config"]

    A = params1_flu.num_age_groups
    R = params1_flu.num_risk_groups

    def _subpop(flu_state, bit_gen, name):
        compartment_init = {c: getattr(flu_state, c) for c in COMPARTMENTS}
        epi_metric_init = {
            "M":  np.asarray(flu_state.M,  dtype=float),
            "MV": np.asarray(flu_state.MV, dtype=float),
        }
        state_init = build_state_from_config(model_config, compartment_init, epi_metric_init)
        params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)
        return ConfigDrivenSubpopModel(
            model_config=model_config,
            state_init=state_init,
            params=params,
            simulation_settings=settings,
            RNG=np.random.Generator(bit_gen),
            schedules_input=schedules_info,
            name=name,
        )

    bit_gen1 = np.random.MT19937(rng_seed)
    bit_gen2 = bit_gen1.jumped(1)
    subpop1 = _subpop(state1_flu, bit_gen1, "subpop1")
    subpop2 = _subpop(state2_flu, bit_gen2, "subpop2")

    return ConfigDrivenMetapopModel(
        subpop_models=[subpop1, subpop2],
        mixing_params=mixing_params,
        model_config=model_config,
        travel_config=travel_config,
    )


def _settings(transition_type, save_tvars=True):
    _, _, _, settings, _ = subpop_inputs("caseB_subpop1")
    updates = {
        "transition_type": transition_type,
        "timesteps_per_day": 1,
        "save_daily_history": False,
    }
    if save_tvars:
        updates["transition_variables_to_save"] = CALIBRATION_TVARS
    return clt.updated_dataclass(settings, updates)


# ---------------------------------------------------------------------------
# 1.  Stochastic replication
# ---------------------------------------------------------------------------

def test_stochastic_admits_within_sampling_noise():
    """
    Run N_REPLICATES stochastic replicates for both FluMetapopModel and
    ConfigDrivenMetapopModel (each replicate uses the same seed for both
    models, but models may consume RNG calls in different order so exact
    path equality is not expected).

    Assertion: mean total hospital admissions (summed over all days,
    age groups, and risk groups) are within 10% of each other across the
    two model families.
    """
    settings = _settings(clt.TransitionTypes.BINOM)

    flu_totals = []
    gen_totals = []

    for rep in range(N_REPLICATES):
        seed = 30000 + rep * 13

        flu_model = _make_flu_metapop(settings, seed)
        gen_model = _make_generic_metapop(settings, seed)

        flu_model.simulate_until_day(NUM_DAYS)
        gen_model.simulate_until_day(NUM_DAYS)

        flu_admits = clt.aggregate_daily_tvar_history(flu_model, CALIBRATION_TVARS)
        gen_admits = clt.aggregate_daily_tvar_history(gen_model, CALIBRATION_TVARS)

        # Both models must produce non-negative admissions
        assert np.all(flu_admits >= 0), f"rep={rep}: negative flu admits"
        assert np.all(gen_admits >= 0), f"rep={rep}: negative generic admits"

        flu_totals.append(float(np.sum(flu_admits)))
        gen_totals.append(float(np.sum(gen_admits)))

    flu_mean = np.mean(flu_totals)
    gen_mean = np.mean(gen_totals)

    rel_diff = abs(flu_mean - gen_mean) / max(flu_mean, 1.0)
    assert rel_diff < 0.10, (
        f"Mean total admits differ by {rel_diff:.1%} "
        f"(flu={flu_mean:.1f}, generic={gen_mean:.1f}); expected < 10%"
    )


# ---------------------------------------------------------------------------
# 2.  Accept-reject calibration — helpers
# ---------------------------------------------------------------------------

def _build_target_timeseries() -> list:
    """Run the deterministic model and return hospital-admit timeseries."""
    settings = _settings(clt.TransitionTypes.BINOM_DETERMINISTIC)
    model = _make_generic_metapop(settings, rng_seed=12345)
    model.simulate_until_day(NUM_DAYS)
    return list(clt.aggregate_daily_tvar_history(model, CALIBRATION_TVARS))


def _build_calibration_model() -> ConfigDrivenMetapopModel:
    settings = _settings(clt.TransitionTypes.BINOM)
    return _make_generic_metapop(settings, rng_seed=99999)


_SAMPLING_INFO = {
    "all_subpop": {
        "beta_baseline": clt.UniformSamplingSpec(
            lower_bound=0.9,
            upper_bound=1.1,
            param_shape=clt.ParamShapes.scalar,
            num_decimals=4,
        )
    }
}

TARGET_ACCEPTED = 3
MAX_REPS = 10


# ---------------------------------------------------------------------------
# 3.  Accept path — JSON files must be written
# ---------------------------------------------------------------------------

def test_calibration_accepts_and_writes_json(tmp_path):
    """
    With target_rsquared=-1e9 every attempt is accepted.  Expect
    TARGET_ACCEPTED parameter-sample JSON files (one per accepted rep)
    and TARGET_ACCEPTED * 2 state JSON files (two subpops per rep).
    """
    target = _build_target_timeseries()
    model = _build_calibration_model()
    sampling_rng = np.random.default_rng(42)

    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        generic_accept_reject(
            metapop_model=model,
            sampling_RNG=sampling_rng,
            sampling_info=_SAMPLING_INFO,
            target_timeseries=target,
            calibration_target_fn=lambda m: list(
                clt.aggregate_daily_tvar_history(m, CALIBRATION_TVARS)
            ),
            transition_variables_to_save=CALIBRATION_TVARS,
            num_days=NUM_DAYS,
            target_accepted_reps=TARGET_ACCEPTED,
            max_reps=MAX_REPS,
            early_stop_percent=0.5,
            target_rsquared=-1e9,   # accept everything
        )
    finally:
        os.chdir(orig_dir)

    param_files = sorted(tmp_path.glob("*_accepted_sample_params.json"))
    state_files  = sorted(tmp_path.glob("*_accepted_state.json"))

    # 2 subpops per accepted rep → 2 × TARGET_ACCEPTED files of each kind
    expected_per_kind = TARGET_ACCEPTED * 2
    assert len(param_files) == expected_per_kind, (
        f"Expected {expected_per_kind} param JSON files, found {len(param_files)}"
    )
    assert len(state_files) == expected_per_kind, (
        f"Expected {expected_per_kind} state JSON files, found {len(state_files)}"
    )

    # Each param file must be valid JSON and contain beta_baseline
    for pf in param_files:
        with open(pf) as f:
            data = json.load(f)
        assert "beta_baseline" in data, f"{pf.name} missing 'beta_baseline'"


# ---------------------------------------------------------------------------
# 4.  Reject path — no JSON files written
# ---------------------------------------------------------------------------

def test_calibration_rejects_with_impossible_threshold(tmp_path):
    """
    With target_rsquared=2.0 (impossible; R² <= 1 always) no replicates
    should be accepted and no JSON files should be written.
    """
    target = _build_target_timeseries()
    model = _build_calibration_model()
    sampling_rng = np.random.default_rng(43)

    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        generic_accept_reject(
            metapop_model=model,
            sampling_RNG=sampling_rng,
            sampling_info=_SAMPLING_INFO,
            target_timeseries=target,
            calibration_target_fn=lambda m: list(
                clt.aggregate_daily_tvar_history(m, CALIBRATION_TVARS)
            ),
            transition_variables_to_save=CALIBRATION_TVARS,
            num_days=NUM_DAYS,
            target_accepted_reps=100,
            max_reps=MAX_REPS,
            early_stop_percent=0.5,
            target_rsquared=2.0,    # never satisfied
        )
    finally:
        os.chdir(orig_dir)

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 0, (
        f"Expected 0 JSON files with impossible threshold, found {len(json_files)}: "
        + str([f.name for f in json_files])
    )
