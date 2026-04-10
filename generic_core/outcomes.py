"""
outcomes.py — Generic outcome computation utilities.

Mirrors flu_core/flu_outcomes.py but accepts string names instead of
hard-coded transition/compartment references, and operates on both
simulation model objects and on raw history dicts produced by the torch
simulation loop.

Functions
---------
daily_transition_sum      — sum named transition flows per day (from model history)
compartment_timeseries    — time series of a compartment's current value (from model history)
attack_rate               — cumulative infections / initial susceptibles (from model)
summarize_outcomes        — mean, median, CI across replicates
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import clt_toolkit as clt


def daily_transition_sum(
    metapop_model,
    transition_names: list[str],
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> np.ndarray:
    """
    Daily sum of one or more named transition flows, aggregated across subpops
    and timesteps-per-day.

    Mirrors ``_tvar_daily`` in flu_core/flu_outcomes.py but accepts any transition
    name(s) rather than hardcoded ISH_to_HR/ISH_to_HD.

    Parameters
    ----------
    metapop_model : MetapopModel
    transition_names : list[str]
        Names of transition variables to sum.
    subpop_name : str or None
        Restrict to one named subpopulation; None sums all.
    age_group : int or None
        Age-group index to slice; None sums all age groups.
    risk_group : int or None
        Risk-group index to slice; None sums all risk groups.

    Returns
    -------
    np.ndarray, shape (days,)
    """
    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    n_per_day = next(iter(subpops)).simulation_settings.timesteps_per_day

    arrays = []
    for subpop in subpops:
        for name in transition_names:
            tvar = subpop.transition_variables[name]
            if not tvar.history_vals_list:
                raise ValueError(
                    f"Transition variable '{name}' has no saved history. "
                    "Add it to SimulationSettings.transition_variables_to_save "
                    "before running the simulation."
                )
            arrays.append(np.asarray(tvar.history_vals_list))

    total = np.sum(np.stack(arrays, axis=0), axis=0)  # (T, A, R)
    daily = clt.utils.daily_sum_over_timesteps(total, n_per_day)   # (days, A, R)

    if age_group is not None:
        daily = daily[:, age_group : age_group + 1, :]
    if risk_group is not None:
        daily = daily[:, :, risk_group : risk_group + 1]
    return daily.sum(axis=(1, 2))


def compartment_timeseries(
    metapop_model,
    compartment_name: str,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> np.ndarray:
    """
    Time series of a compartment's current value across the simulation.

    Returns the day-end size of the compartment at each day — i.e. how many
    individuals are in that compartment on each day.  For absorbing states
    (e.g. "D") this equals the cumulative total; for transient states (e.g.
    "H") it is the current census, not cumulative admissions.  Use
    ``daily_transition_sum(...).cumsum()`` for cumulative flow-based counts.

    Parameters
    ----------
    metapop_model : MetapopModel
    compartment_name : str
        Name of the compartment (e.g. "D", "H").
    subpop_name : str or None
        Restrict to one subpopulation; None sums all.
    age_group : int or None
    risk_group : int or None

    Returns
    -------
    np.ndarray, shape (days,)
        Day-end compartment value, summed across the selected dimensions.
    """
    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    arrays = []
    for subpop in subpops:
        comp = subpop.compartments[compartment_name]
        if not comp.history_vals_list:
            raise ValueError(
                f"Compartment '{compartment_name}' has no saved history. "
                "Set save_daily_history=True in SimulationSettings."
            )
        arrays.append(np.asarray(comp.history_vals_list))

    total = np.sum(np.stack(arrays, axis=0), axis=0)  # (days, A, R)

    if age_group is not None:
        total = total[:, age_group : age_group + 1, :]
    if risk_group is not None:
        total = total[:, :, risk_group : risk_group + 1]
    return total.sum(axis=(1, 2))


def attack_rate(
    metapop_model,
    infection_transition_name: str | list[str] = "S_to_E",
    susceptible_compartment_names: str | list[str] = "S",
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> float:
    """
    Attack rate = cumulative infections / initial susceptible population.

    Parameters
    ----------
    metapop_model : MetapopModel
    infection_transition_name : str or list[str]
        Transition name(s) that move individuals into the exposed / infected
        state (default "S_to_E"). Pass a list when there are multiple
        infection transitions, e.g. one per susceptible compartment.
    susceptible_compartment_names : str or list[str]
        Compartment name(s) that hold susceptible individuals at time zero
        (default "S"). Pass a list when susceptibles are spread across
        multiple compartments, e.g. ["S_unvax", "S_vax"].
    subpop_name : str or None
    age_group : int or None
    risk_group : int or None

    Returns
    -------
    float
    """
    if isinstance(infection_transition_name, str):
        infection_transition_name = [infection_transition_name]
    if isinstance(susceptible_compartment_names, str):
        susceptible_compartment_names = [susceptible_compartment_names]

    infections = daily_transition_sum(
        metapop_model, infection_transition_name, subpop_name, age_group, risk_group
    ).sum()

    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    init_S_arrays = []
    for sp in subpops:
        compartment_sum = sum(
            np.asarray(sp.compartments[name].history_vals_list[0])
            for name in susceptible_compartment_names
        )
        init_S_arrays.append(compartment_sum)
    init_S = np.sum(np.stack(init_S_arrays, axis=0), axis=0)  # (A, R)

    if age_group is not None:
        init_S = init_S[age_group : age_group + 1, :]
    if risk_group is not None:
        init_S = init_S[:, risk_group : risk_group + 1]

    return float(infections / init_S.sum())


def summarize_outcomes(
    values,
    credible_interval: float = 0.95,
) -> dict:
    """
    Summarize a collection of scalar outcomes across replicates.

    Mirrors summarize_outcomes in flu_core/flu_outcomes.py.

    Parameters
    ----------
    values : array-like, shape (reps,)
    credible_interval : float
        Width of the central credible interval (default 0.95).

    Returns
    -------
    dict with keys: "mean", "median", "lower_ci", "upper_ci"
    """
    values = np.asarray(values, dtype=float)
    half = (1.0 - credible_interval) / 2.0
    return {
        "mean":     float(np.mean(values)),
        "median":   float(np.median(values)),
        "lower_ci": float(np.percentile(values, 100 * half)),
        "upper_ci": float(np.percentile(values, 100 * (1.0 - half))),
    }
