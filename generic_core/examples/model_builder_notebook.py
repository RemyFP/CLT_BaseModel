"""
model_builder_notebook.py
=========================

Interactive marimo notebook for building, visualising, and running
config-driven epidemic models.

Run with::

    marimo run generic_core/examples/model_builder_notebook.py

Supported rate templates
------------------------
- ``constant_param``
- ``param_product``
- ``immunity_modulated``
- ``force_of_infection``
- ``force_of_infection_travel``

Scope note
----------
This notebook is intentionally aggregate-first. "No age groups" is handled as
one aggregate age bucket internally, so force-of-infection schedules use 1x1
contact matrices and 1x1 age-risk arrays.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import sys
    import json
    import io
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    import clt_toolkit as clt
    import flu_core as flu

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import generic_core as gc
    from generic_core.config_parser import parse_model_config_from_dict
    from generic_core.generic_model import (
        ConfigDrivenSubpopModel,
        build_state_from_config,
        build_params_from_config,
    )
    from generic_core.generic_metapop import ConfigDrivenMetapopModel

    return (
        Path, SimpleNamespace, clt, flu, gc, io, json, mo, np, pd, plt,
        ConfigDrivenMetapopModel, ConfigDrivenSubpopModel,
        build_state_from_config, build_params_from_config,
        parse_model_config_from_dict,
    )


@app.cell
def _helpers(SimpleNamespace, json, np, pd):
    def parse_csv_list(text):
        return [_item.strip() for _item in text.split(",") if _item.strip()]

    def parse_infectious_mapping(text):
        _mapping = {}
        for _item in parse_csv_list(text):
            if ":" in _item:
                _name, _rel = _item.split(":", 1)
                _name = _name.strip()
                _rel = _rel.strip()
                if _name:
                    _mapping[_name] = _rel or None
            elif _item:
                _mapping[_item] = None
        return _mapping

    def build_scalar_array(value):
        return np.array([[float(value)]], dtype=float)

    def build_notebook_schedules_input(
        start_date,
        num_days,
        absolute_humidity,
        mobility_value,
        daily_vaccines_value,
    ):
        _horizon = max(int(num_days) + 14, 370)
        _dates = pd.date_range(start=start_date, periods=_horizon, freq="D").date

        _absolute_humidity_df = pd.DataFrame({
            "date": _dates,
            "absolute_humidity": [float(absolute_humidity)] * _horizon,
        })
        _school_work_calendar_df = pd.DataFrame({
            "date": _dates,
            "is_school_day": [1.0] * _horizon,
            "is_work_day": [1.0] * _horizon,
        })
        _mobility_payload = json.dumps([[float(mobility_value)]])
        _mobility_df = pd.DataFrame({
            "day_of_week": [
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
            ],
            "mobility_modifier": [_mobility_payload] * 7,
        })
        _daily_vaccines_payload = json.dumps([[float(daily_vaccines_value)]])
        _daily_vaccines_df = pd.DataFrame({
            "date": _dates,
            "daily_vaccines": [_daily_vaccines_payload] * _horizon,
        })
        return SimpleNamespace(
            absolute_humidity_df=_absolute_humidity_df,
            school_work_calendar_df=_school_work_calendar_df,
            mobility_df=_mobility_df,
            daily_vaccines_df=_daily_vaccines_df,
        )

    return build_notebook_schedules_input, build_scalar_array, parse_csv_list, parse_infectious_mapping


@app.cell
def _intro(mo):
    mo.md(
        """
        # Generic Epidemic Model Builder

        Build, visualise, and run a config-driven model without editing JSON.

        This notebook now supports all built-in rate templates, including the
        force-of-infection variants. Aggregate models with no age groups are
        treated as a single 1x1 age-risk block internally.
        """
    )
    return


@app.cell
def _compartments_ui(mo):
    compartments_text = mo.ui.text(
        value="S, E, I, R",
        placeholder="S, E, I, R",
        label="Compartments (comma-separated)",
        full_width=True,
    )
    return (compartments_text,)


@app.cell
def _compartments_parse(compartments_text):
    raw = [_c.strip() for _c in compartments_text.value.split(",") if _c.strip()]
    compartments = list(dict.fromkeys(raw))
    return (compartments,)


@app.cell
def _compartments_display(compartments, compartments_text, mo):
    if compartments:
        _body = mo.md("**Parsed:** " + "  ".join(f"`{_c}`" for _c in compartments))
    else:
        _body = mo.callout(mo.md("Enter at least one compartment name."), kind="warn")
    mo.vstack([
        mo.md("### Step 1 — Compartments"),
        compartments_text,
        _body,
    ])
    return


@app.cell
def _transition_count_ui(mo):
    n_transitions = mo.ui.number(
        start=1,
        stop=12,
        step=1,
        value=3,
        label="Number of transitions",
    )
    return (n_transitions,)


@app.cell
def _transition_forms_ui(compartments, mo):
    _max_t = 12
    _comps = compartments if compartments else ["?"]
    _templates = [
        "constant_param",
        "param_product",
        "immunity_modulated",
        "force_of_infection",
        "force_of_infection_travel",
    ]

    t_name = mo.ui.array([mo.ui.text(value=f"t{_i+1}", label="Name") for _i in range(_max_t)])
    t_origin = mo.ui.array([
        mo.ui.dropdown(options=_comps, value=_comps[_i] if _i < len(_comps) else _comps[0], label="Origin") for _i in range(_max_t)
    ])
    t_dest = mo.ui.array([
        mo.ui.dropdown(options=_comps, value=_comps[_i+1] if _i < len(_comps)-1 else _comps[-1], label="Destination") for _i in range(_max_t)
    ])
    t_template = mo.ui.array([
        mo.ui.dropdown(options=_templates, value="constant_param", label="Rate template")
        for _ in range(_max_t)
    ])

    t_param = mo.ui.array([
        mo.ui.text(value=f"param_{_i+1}", label="Param name") for _i in range(_max_t)
    ])
    t_factors = mo.ui.array([
        mo.ui.text(value="", label="Factors") for _ in range(_max_t)
    ])
    t_complements = mo.ui.array([
        mo.ui.text(value="", label="Complement factors") for _ in range(_max_t)
    ])

    t_base_rate = mo.ui.array([
        mo.ui.text(value="base_rate", label="Base rate param") for _ in range(_max_t)
    ])
    t_proportion = mo.ui.array([
        mo.ui.text(value="split_prop", label="Proportion param") for _ in range(_max_t)
    ])
    t_is_complement = mo.ui.array([
        mo.ui.checkbox(label="Use complement branch", value=False) for _ in range(_max_t)
    ])
    t_inf_reduce = mo.ui.array([
        mo.ui.text(value="inf_risk_reduce", label="Infection reduction param") for _ in range(_max_t)
    ])
    t_vax_reduce = mo.ui.array([
        mo.ui.text(value="vax_risk_reduce", label="Vaccine reduction param") for _ in range(_max_t)
    ])

    t_beta = mo.ui.array([
        mo.ui.text(value="beta_baseline", label="Beta param") for _ in range(_max_t)
    ])
    t_rel_sus = mo.ui.array([
        mo.ui.text(value="relative_suscept", label="Relative susceptibility param") for _ in range(_max_t)
    ])
    t_infectious = mo.ui.array([
        mo.ui.text(
            value="I",
            label="Infectious compartments",
            placeholder="IP:IP_relative_inf, IA:IA_relative_inf, ISR, ISH",
        )
        for _ in range(_max_t)
    ])
    t_use_humidity = mo.ui.array([
        mo.ui.checkbox(label="Include humidity modifier", value=False) for _ in range(_max_t)
    ])
    t_humidity_impact = mo.ui.array([
        mo.ui.text(value="humidity_impact", label="Humidity impact param") for _ in range(_max_t)
    ])
    t_use_foi_immunity = mo.ui.array([
        mo.ui.checkbox(label="Include immunity modifier", value=False) for _ in range(_max_t)
    ])
    t_immobile = mo.ui.array([
        mo.ui.text(value="", label="Immobile compartments") for _ in range(_max_t)
    ])

    return (
        t_name, t_origin, t_dest, t_template,
        t_param, t_factors, t_complements,
        t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
        t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
        t_use_foi_immunity, t_immobile,
    )


@app.cell
def _transition_show(
    mo,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
):
    _n = int(n_transitions.value)
    _rows = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _rate_ui = t_param[_i]
        elif _template == "param_product":
            _rate_ui = mo.vstack([t_factors[_i], t_complements[_i]])
        elif _template == "immunity_modulated":
            _rate_ui = mo.vstack([
                t_base_rate[_i],
                t_proportion[_i],
                t_is_complement[_i],
                t_use_foi_immunity[_i],
                t_inf_reduce[_i],
                t_vax_reduce[_i],
            ])
        elif _template == "force_of_infection":
            _rate_ui = mo.vstack([
                t_beta[_i],
                t_rel_sus[_i],
                t_infectious[_i],
                t_use_humidity[_i],
                t_humidity_impact[_i],
                t_use_foi_immunity[_i],
                t_inf_reduce[_i],
                t_vax_reduce[_i],
            ])
        else:
            _rate_ui = mo.vstack([
                t_beta[_i],
                t_use_humidity[_i],
                t_humidity_impact[_i],
                t_use_foi_immunity[_i],
                t_inf_reduce[_i],
                t_vax_reduce[_i],
                t_infectious[_i],
                t_rel_sus[_i],
                t_immobile[_i],
            ])

        _rows.append(mo.vstack([
            mo.md(f"**Transition {_i + 1}**"),
            mo.vstack([
                mo.hstack([t_name[_i], t_origin[_i], t_dest[_i]], justify="start"),
                t_template[_i]
                ]),
            _rate_ui,
            mo.md("---"),
        ]))

    mo.vstack([
        mo.md("### Step 2 — Transitions"),
        n_transitions,
        *_rows,
    ])
    return


@app.cell
def _template_requirements(
    n_transitions, t_template, t_use_humidity, t_use_foi_immunity,
):
    _n = int(n_transitions.value)
    _uses_contact_matrix = False
    _uses_absolute_humidity = False
    _uses_mobility = False
    _requires_immunity_metrics = False

    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "immunity_modulated":
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "force_of_infection":
            _uses_contact_matrix = True
            _uses_absolute_humidity = _uses_absolute_humidity or bool(t_use_humidity.value[_i])
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "force_of_infection_travel":
            _uses_contact_matrix = True
            _uses_absolute_humidity = _uses_absolute_humidity or bool(t_use_humidity.value[_i])
            _uses_mobility = True
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])

    uses_absolute_humidity = _uses_absolute_humidity
    uses_contact_matrix = _uses_contact_matrix
    uses_mobility = _uses_mobility
    requires_immunity_metrics = _requires_immunity_metrics
    return uses_absolute_humidity, uses_contact_matrix, uses_mobility, requires_immunity_metrics


@app.cell
def _collect_param_names(
    n_transitions, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact, t_use_foi_immunity,
    parse_csv_list, parse_infectious_mapping,
):
    _n = int(n_transitions.value)
    _names = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _p = t_param.value[_i].strip()
            if _p:
                _names.append(_p)
        elif _template == "param_product":
            _names.extend(parse_csv_list(t_factors.value[_i]))
            _names.extend(parse_csv_list(t_complements.value[_i]))
        elif _template == "immunity_modulated":
            for _p in (t_base_rate.value[_i], t_proportion.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
        elif _template == "force_of_infection":
            for _p in (
                t_beta.value[_i],
                t_rel_sus.value[_i],
            ):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_humidity.value[_i]:
                _p = t_humidity_impact.value[_i].strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
            _names.extend([_p for _p in parse_infectious_mapping(t_infectious.value[_i]).values() if _p])
        elif _template == "force_of_infection_travel":
            for _p in (
                t_beta.value[_i],
                t_rel_sus.value[_i],
            ):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_humidity.value[_i]:
                _p = t_humidity_impact.value[_i].strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
            _names.extend([_p for _p in parse_infectious_mapping(t_infectious.value[_i]).values() if _p])

    param_names = list(dict.fromkeys(_names))
    return (param_names,)


@app.cell
def _params_ui(param_names, mo):
    params_inputs = mo.ui.array([
        mo.ui.number(start=0.0, stop=10.0, step=0.1, value=1.0, label=_name)
        for _name in param_names
    ])
    return (params_inputs,)


@app.cell
def _params_show(param_names, params_inputs, mo):
    _body = (
        mo.hstack(list(params_inputs), wrap=True)
        if param_names
        else mo.callout(mo.md("No transition parameters found yet."), kind="warn")
    )
    mo.vstack([mo.md("### Step 3 — Parameters"), _body])
    return


@app.cell
def _schedule_and_immunity_ui(mo):
    include_inf_immunity = mo.ui.checkbox(
        label="Include infection-induced immunity metric (M)",
        value=False,
    )
    include_vax_immunity = mo.ui.checkbox(
        label="Include vaccine-induced immunity metric (MV)",
        value=False,
    )
    absolute_humidity_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.0001, value=0.006, label="Absolute humidity",
    )
    total_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=0.1, value=1.0, label="Total contact matrix value",
    )
    school_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=0.1, value=0.0, label="School contact subtraction",
    )
    work_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=0.1, value=0.0, label="Work contact subtraction",
    )
    mobility_input = mo.ui.number(
        start=0.0, stop=5.0, step=0.01, value=1.0, label="Mobility modifier",
    )
    daily_vaccines_input = mo.ui.number(
        start=0.0, stop=1e9, step=1.0, value=0.0, label="Daily vaccines",
    )
    return (
        include_inf_immunity,
        include_vax_immunity,
        absolute_humidity_input,
        total_contact_input,
        school_contact_input,
        work_contact_input,
        mobility_input,
        daily_vaccines_input,
    )


@app.cell
def _epi_metric_ui(n_transitions, t_name, mo):
    transition_names = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    opts = transition_names if transition_names else [""]
    r_to_s_picker = mo.ui.dropdown(
        options=opts,
        value=opts[-1],
        label="Transition used for R→S-style immunity update",
    )
    inf_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.01, value=0.0, label="inf_induced_saturation",
    )
    vax_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.01, value=0.0, label="vax_induced_saturation",
    )
    inf_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.001, value=0.01, label="inf_induced_immune_wane",
    )
    vax_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.001, value=0.0, label="vax_induced_immune_wane",
    )
    return r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input, vax_wane_input


@app.cell
def _schedule_and_immunity_show(
    mo,
    include_inf_immunity,
    include_vax_immunity,
    absolute_humidity_input,
    total_contact_input,
    school_contact_input,
    work_contact_input,
    mobility_input,
    daily_vaccines_input,
    r_to_s_picker,
    inf_sat_input,
    vax_sat_input,
    inf_wane_input,
    vax_wane_input,
    uses_absolute_humidity,
    uses_contact_matrix,
    uses_mobility,
    requires_immunity_metrics,
):
    _parts = [
        mo.md("### Step 4 — Schedules and Immunity"),
        mo.hstack([include_inf_immunity, include_vax_immunity], wrap=True),
    ]

    if uses_absolute_humidity or uses_contact_matrix or uses_mobility:
        _schedule_inputs = []
        if uses_absolute_humidity:
            _schedule_inputs.append(absolute_humidity_input)
        if uses_contact_matrix:
            _schedule_inputs.extend([total_contact_input, school_contact_input, work_contact_input])
        if uses_mobility:
            _schedule_inputs.append(mobility_input)
        _parts.append(mo.hstack(_schedule_inputs, wrap=True))
    else:
        _parts.append(mo.md("*No schedule-backed rate templates selected.*"))

    _immunity_active = include_inf_immunity.value or include_vax_immunity.value
    if requires_immunity_metrics:
        _parts.append(mo.callout(
            mo.md(
                "Selected rate templates can use `M` and/or `MV`. "
                "Enable whichever immunity metrics you want to track."
            ),
            kind="info",
        ))

    if _immunity_active:
        _metric_inputs = []
        if include_inf_immunity.value:
            _metric_inputs.extend([
                r_to_s_picker,
                inf_sat_input,
                vax_sat_input,
                inf_wane_input,
            ])
        if include_vax_immunity.value:
            _metric_inputs.extend([
                vax_wane_input,
                daily_vaccines_input,
            ])
        _parts.append(mo.hstack(_metric_inputs, wrap=True))
    else:
        _parts.append(mo.md("*Dynamic immunity metrics disabled.*"))

    mo.vstack(_parts)
    return


@app.cell
def _diagram(compartments, n_transitions, t_name, t_origin, t_dest, mo, plt):
    _n = int(n_transitions.value)
    _inner = None
    _graphviz_error = None
    try:
        import graphviz as gv  # type: ignore[import-untyped]
        _dot = gv.Digraph(
            graph_attr={"rankdir": "LR", "bgcolor": "white", "pad": "0.3"},
            node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#ddeeff"},
        )
        for _c in compartments:
            _dot.node(_c)
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            _label = t_name.value[_i]
            if _origin and _dest:
                _dot.edge(_origin, _dest, label=_label)
        _inner = mo.image(_dot.pipe(format="png"), width="100%")
    except Exception as _exc:
        _graphviz_error = f"{type(_exc).__name__}: {_exc}"

    if _inner is None:
        _fig, _ax = plt.subplots(figsize=(max(4, len(compartments) * 2), 2))
        _ax.set_xlim(-0.5, len(compartments) - 0.5)
        _ax.set_ylim(-0.5, 1.5)
        _ax.axis("off")
        _pos = {_c: (_i, 0.5) for _i, _c in enumerate(compartments)}
        for _c, (_x, _y) in _pos.items():
            _ax.text(_x, _y, _c, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ddeeff"))
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            if _origin in _pos and _dest in _pos:
                _x0, _y0 = _pos[_origin]
                _x1, _y1 = _pos[_dest]
                _ax.annotate(
                    "", xy=(_x1 - 0.15, _y1), xytext=(_x0 + 0.15, _y0),
                    arrowprops=dict(arrowstyle="->", color="#336699"),
                )
        plt.tight_layout()
        _fallback_parts = []
        if _graphviz_error is None:
            _fallback_parts.append(
                mo.callout(
                    mo.md("*Graphviz not available; using a simple fallback diagram.*"),
                    kind="info",
                )
            )
        else:
            _fallback_parts.append(
                mo.callout(
                    mo.md(
                        "**Graphviz rendering failed; using fallback diagram.**\n\n"
                        f"`{_graphviz_error}`"
                    ),
                    kind="warn",
                )
            )
        _fallback_parts.append(_fig)
        _inner = mo.vstack(_fallback_parts)

    mo.vstack([mo.md("### Step 5 — Model Diagram"), _inner])
    return


@app.cell
def _init_ui(compartments, mo):
    total_pop_input = mo.ui.number(
        start=1, stop=int(1e9), step=1, value=10000, label="Total population N",
    )
    seed_compartments = compartments[1:] if len(compartments) > 1 else []
    seed_inputs = mo.ui.array([
        mo.ui.number(start=0, stop=int(1e9), step=1, value=0 if _j > 0 else 50, label=f"Initial {_c}")
        for _j, _c in enumerate(seed_compartments)
    ])
    return total_pop_input, seed_inputs


@app.cell
def _init_show(compartments, total_pop_input, seed_inputs, mo):
    _N = int(total_pop_input.value)
    _seeded = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
    _remainder = _N - sum(_seeded.values())
    _first = compartments[0] if compartments else "?"
    _table_rows = {_first: _remainder, **_seeded}
    _rows_md = "\n".join(f"| `{_c}` | {_v:,} |" for _c, _v in _table_rows.items())
    _parts = [
        mo.md("### Step 6 — Initial Conditions"),
        total_pop_input,
        mo.hstack(list(seed_inputs), wrap=True) if seed_inputs.value else mo.md(""),
        mo.md(
            "| Compartment | Initial count |\n"
            "|---|---|\n"
            f"{_rows_md}"
        ),
    ]
    if _remainder < 0:
        _parts.append(mo.callout(mo.md("Seeded counts exceed total population N."), kind="danger"))
    mo.vstack(_parts)
    return


@app.cell
def _sim_settings_ui(mo):
    sim_days = mo.ui.number(start=10, stop=730, step=10, value=100, label="Simulation days")
    sim_mode = mo.ui.radio(
        options=["Deterministic", "Stochastic"],
        value="Deterministic",
        label="Simulation mode",
    )
    n_reps = mo.ui.number(start=1, stop=100, step=1, value=10, label="Replicates")
    rng_seed = mo.ui.number(start=0, stop=99999, step=1, value=42, label="RNG seed")
    timesteps = mo.ui.number(start=1, stop=24, step=1, value=7, label="Timesteps per day")
    return sim_days, sim_mode, n_reps, rng_seed, timesteps


@app.cell
def _sim_settings_show(mo, sim_days, sim_mode, n_reps, rng_seed, timesteps):
    mo.vstack([
        mo.md("### Step 7 — Simulation Settings"),
        mo.hstack([sim_days, sim_mode, timesteps, rng_seed], justify="start"),
        mo.hstack([n_reps, mo.md("*Ignored in deterministic mode.*") if sim_mode.value == "Deterministic" else mo.md("")]),
    ])
    return


@app.cell
def _build_config(
    compartments,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
    param_names, params_inputs,
    include_inf_immunity, include_vax_immunity,
    r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input, vax_wane_input,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, requires_immunity_metrics,
    parse_csv_list, parse_infectious_mapping,
    total_contact_input, school_contact_input, work_contact_input,
):
    _n = int(n_transitions.value)
    params_dict = {
        _name: float(params_inputs.value[_j])
        for _j, _name in enumerate(param_names)
    }

    _transitions = []
    _metapop_travel_config = {}
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _rate_config = {"param": t_param.value[_i].strip()}
        elif _template == "param_product":
            _factors = parse_csv_list(t_factors.value[_i])
            _complements = parse_csv_list(t_complements.value[_i])
            _rate_config = {"factors": _factors}
            if _complements:
                _rate_config["complement_factors"] = _complements
        elif _template == "immunity_modulated":
            _rate_config = {
                "base_rate": t_base_rate.value[_i].strip(),
                "proportion": t_proportion.value[_i].strip(),
                "is_complement": bool(t_is_complement.value[_i]),
            }
            if t_use_foi_immunity.value[_i]:
                _inf_reduce = t_inf_reduce.value[_i].strip()
                _vax_reduce = t_vax_reduce.value[_i].strip()
                if _inf_reduce:
                    _rate_config["inf_reduce_param"] = _inf_reduce
                if _vax_reduce:
                    _rate_config["vax_reduce_param"] = _vax_reduce
        elif _template == "force_of_infection":
            _rate_config = {
                "beta_param": t_beta.value[_i].strip(),
                "contact_matrix_schedule": "flu_contact_matrix",
                "infectious_compartments": parse_infectious_mapping(t_infectious.value[_i]),
                "relative_susceptibility_param": t_rel_sus.value[_i].strip(),
            }
            if t_use_humidity.value[_i]:
                _rate_config["humidity_impact_param"] = t_humidity_impact.value[_i].strip()
                _rate_config["humidity_schedule"] = "absolute_humidity"
            if t_use_foi_immunity.value[_i]:
                _inf_reduce = t_inf_reduce.value[_i].strip()
                _vax_reduce = t_vax_reduce.value[_i].strip()
                if _inf_reduce:
                    _rate_config["inf_reduce_param"] = _inf_reduce
                if _vax_reduce:
                    _rate_config["vax_reduce_param"] = _vax_reduce
        else:
            _travel_config = {
                "infectious_compartments": parse_infectious_mapping(t_infectious.value[_i]),
                "immobile_compartments": parse_csv_list(t_immobile.value[_i]),
                "relative_susceptibility_param": t_rel_sus.value[_i].strip(),
                "contact_matrix_schedule": "flu_contact_matrix",
                "mobility_schedule": "mobility_modifier",
            }
            _rate_config = {
                "beta_param": t_beta.value[_i].strip(),
                "travel_config": _travel_config,
            }
            if t_use_humidity.value[_i]:
                _rate_config["humidity_impact_param"] = t_humidity_impact.value[_i].strip()
                _rate_config["humidity_schedule"] = "absolute_humidity"
            if t_use_foi_immunity.value[_i]:
                _inf_reduce = t_inf_reduce.value[_i].strip()
                _vax_reduce = t_vax_reduce.value[_i].strip()
                if _inf_reduce:
                    _rate_config["inf_reduce_param"] = _inf_reduce
                if _vax_reduce:
                    _rate_config["vax_reduce_param"] = _vax_reduce
            if not _metapop_travel_config:
                _metapop_travel_config = _travel_config

        _transitions.append({
            "name": t_name.value[_i].strip(),
            "origin": t_origin.value[_i],
            "destination": t_dest.value[_i],
            "rate_template": _template,
            "rate_config": _rate_config,
        })

    if uses_contact_matrix:
        params_dict["total_contact_matrix"] = [[float(total_contact_input.value)]]
        params_dict["school_contact_matrix"] = [[float(school_contact_input.value)]]
        params_dict["work_contact_matrix"] = [[float(work_contact_input.value)]]

    _schedules = []
    if uses_absolute_humidity:
        _schedules.append({
            "name": "absolute_humidity",
            "schedule_template": "timeseries_lookup",
            "schedule_config": {
                "df_attribute": "absolute_humidity_df",
                "value_column": "absolute_humidity",
            },
        })
    if uses_contact_matrix:
        _schedules.append({
            "name": "flu_contact_matrix",
            "schedule_template": "contact_matrix",
            "schedule_config": {
                "school_work_day_df_attribute": "school_work_calendar_df",
                "total_contact_matrix_param": "total_contact_matrix",
                "school_contact_matrix_param": "school_contact_matrix",
                "work_contact_matrix_param": "work_contact_matrix",
            },
        })
    if uses_mobility:
        _schedules.append({
            "name": "mobility_modifier",
            "schedule_template": "mobility",
            "schedule_config": {
                "df_attribute": "mobility_df",
            },
        })

    _immunity_active = include_inf_immunity.value or include_vax_immunity.value
    _epi_metrics = []
    if include_vax_immunity.value:
        _schedules.append({
            "name": "daily_vaccines",
            "schedule_template": "vaccine_schedule",
            "schedule_config": {
                "df_attribute": "daily_vaccines_df",
            },
        })
    if include_inf_immunity.value:
        params_dict.update({
            "inf_induced_saturation": float(inf_sat_input.value),
            "vax_induced_saturation": float(vax_sat_input.value),
            "inf_induced_immune_wane": float(inf_wane_input.value),
        })
        _epi_metrics.append({
            "name": "M",
            "init_val": [[0.0]],
            "metric_template": "infection_induced_immunity",
            "update_config": {
                "r_to_s_transition": r_to_s_picker.value,
                "inf_induced_saturation_param": "inf_induced_saturation",
                "vax_induced_saturation_param": "vax_induced_saturation",
                "inf_induced_immune_wane_param": "inf_induced_immune_wane",
            },
        })
    if include_vax_immunity.value:
        params_dict["vax_induced_immune_wane"] = float(vax_wane_input.value)
        _epi_metrics.append({
            "name": "MV",
            "init_val": [[0.0]],
            "metric_template": "vaccine_induced_immunity",
            "update_config": {
                "daily_vaccines_schedule": "daily_vaccines",
                "vax_induced_immune_wane_param": "vax_induced_immune_wane",
            },
        })

    config_dict = {
        "compartments": compartments,
        "params": params_dict,
        "transitions": _transitions,
        "transition_groups": [],
        "epi_metrics": _epi_metrics,
        "schedules": _schedules,
    }
    immunity_active = _immunity_active
    metapop_travel_config = _metapop_travel_config
    return config_dict, immunity_active, metapop_travel_config


@app.cell
def _config_preview(config_dict, json, mo):
    json_str = json.dumps(config_dict, indent=2)
    mo.vstack([
        mo.md("### Step 8 — Config Preview"),
        mo.md(f"```json\n{json_str}\n```"),
        mo.download(
            data=json_str.encode(),
            filename="model_config.json",
            mimetype="application/json",
            label="Download config JSON",
        ),
    ])
    return


@app.cell
def _run_button(mo):
    run_button = mo.ui.run_button(label="Run simulation")
    return (run_button,)


@app.cell
def _run_section_display(run_button, mo):
    mo.vstack([mo.md("### Step 9 — Run"), run_button])
    return


@app.cell
def _run_sim(
    run_button,
    config_dict,
    metapop_travel_config,
    compartments,
    total_pop_input,
    seed_inputs,
    sim_days,
    sim_mode,
    n_reps,
    rng_seed,
    timesteps,
    absolute_humidity_input,
    mobility_input,
    daily_vaccines_input,
    build_notebook_schedules_input,
    build_scalar_array,
    parse_model_config_from_dict,
    ConfigDrivenSubpopModel,
    ConfigDrivenMetapopModel,
    build_state_from_config,
    build_params_from_config,
    clt,
    flu,
    np,
    mo,
):
    mo.stop(not run_button.value, mo.md(""))

    start_real_date = "2024-01-01"
    schedules_input = build_notebook_schedules_input(
        start_date=start_real_date,
        num_days=int(sim_days.value),
        absolute_humidity=float(absolute_humidity_input.value),
        mobility_value=float(mobility_input.value),
        daily_vaccines_value=float(daily_vaccines_input.value),
    )

    config_err = None
    model_config = None
    try:
        model_config = parse_model_config_from_dict(config_dict, schedules_input=schedules_input)
    except Exception as exc:
        config_err = str(exc)
    mo.stop(
        config_err is not None,
        mo.callout(mo.md(f"**Config error:** {config_err}"), kind="danger"),
    )
    assert model_config is not None

    _N = int(total_pop_input.value)
    _seed_vals = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
    _first_comp = compartments[0]
    _remainder = _N - sum(_seed_vals.values())
    mo.stop(
        _remainder < 0,
        mo.callout(mo.md("**Initial condition error:** seeded counts exceed total population."), kind="danger"),
    )

    compartment_init = {_first_comp: build_scalar_array(_remainder)}
    compartment_init.update({_c: build_scalar_array(_v) for _c, _v in _seed_vals.items()})
    for _c in compartments:
        compartment_init.setdefault(_c, build_scalar_array(0.0))

    _is_stochastic = sim_mode.value == "Stochastic"
    _transition_type = (
        clt.TransitionTypes.BINOM if _is_stochastic else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    )
    _reps = int(n_reps.value) if _is_stochastic else 1
    _num_days = int(sim_days.value)
    _seed = int(rng_seed.value)
    _ts_per_day = int(timesteps.value)

    def _run_once(seed_offset):
        _state = build_state_from_config(model_config, compartment_init, epi_metric_init={})
        _params = build_params_from_config(model_config, num_age_groups=1, num_risk_groups=1)
        _settings = clt.SimulationSettings(
            timesteps_per_day=_ts_per_day,
            transition_type=_transition_type,
            start_real_date=start_real_date,
            save_daily_history=True,
        )
        _rng = np.random.default_rng(_seed + seed_offset)
        _subpop = ConfigDrivenSubpopModel(
            model_config=model_config,
            state_init=_state,
            params=_params,
            simulation_settings=_settings,
            RNG=_rng,
            schedules_input=schedules_input,
            name="aggregate_pop",
        )
        _mixing = flu.FluMixingParams(
            travel_proportions=np.array([[1.0]]),
            num_locations=1,
        )
        _metapop = ConfigDrivenMetapopModel(
            subpop_models=[_subpop],
            mixing_params=_mixing,
            model_config=model_config,
            travel_config=metapop_travel_config,
        )
        _metapop.simulate_until_day(_num_days)
        return {
            _c: np.array(_subpop.compartments[_c].history_vals_list).squeeze()
            for _c in compartments
        }

    sim_err = None
    histories = []
    with mo.status.spinner("Running simulation..."):
        try:
            histories = [_run_once(_rep) for _rep in range(_reps)]
        except Exception as exc:
            sim_err = str(exc)
    mo.stop(
        sim_err is not None,
        mo.callout(mo.md(f"**Simulation error:** {sim_err}"), kind="danger"),
    )
    return (histories,)


@app.cell
def _plot_curves(histories, compartments, sim_days, sim_mode, np, plt, mo):
    _num_days = int(sim_days.value)
    _days = np.arange(1, _num_days + 1)
    _is_stochastic = sim_mode.value == "Stochastic"

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for _ci, _comp in enumerate(compartments):
        _color = _colors[_ci % len(_colors)]
        if _is_stochastic and len(histories) > 1:
            _mat = np.stack([_h[_comp] for _h in histories], axis=0)
            _median = np.median(_mat, axis=0)
            _lo = np.percentile(_mat, 2.5, axis=0)
            _hi = np.percentile(_mat, 97.5, axis=0)
            for _rep in range(len(histories)):
                _ax.plot(_days, _mat[_rep], color=_color, alpha=0.15, linewidth=0.8)
            _ax.plot(_days, _median, color=_color, linewidth=2, label=f"{_comp} (median)")
            _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)
        else:
            _ax.plot(_days, histories[0][_comp], color=_color, linewidth=2, label=_comp)

    _ax.set_xlabel("Day")
    _ax.set_ylabel("Count")
    _ax.set_title("Epidemic Curves")
    _ax.legend(loc="best")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.vstack([mo.md("### Results — Epidemic Curves"), _fig])
    return


@app.cell
def _summary_stats(histories, compartments, np, mo):
    _rows = []
    for _comp in compartments:
        _vals = np.stack([_h[_comp] for _h in histories], axis=0)
        _peak = np.median(np.max(_vals, axis=1))
        _peak_day = int(np.median(np.argmax(_vals, axis=1))) + 1
        _rows.append(f"| `{_comp}` | {_peak:,.0f} | {_peak_day} |")
    _table = "\n".join(_rows)
    mo.vstack([
        mo.md("### Results — Summary"),
        mo.md(
            "| Compartment | Peak value (median) | Peak day (median) |\n"
            "|---|---|---|\n"
            f"{_table}"
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
