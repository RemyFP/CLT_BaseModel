# generic_core — config-driven epidemic model

from .config_parser import parse_model_config, parse_model_config_from_dict, ModelConfig

from .generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)

from .generic_metapop import ConfigDrivenMetapopModel

from .torch_generic import (
    build_generic_torch_inputs,
    generic_torch_simulate_full_history,
    generic_torch_simulate_calibration_target,
)

from .rate_templates import (
    RATE_TEMPLATE_REGISTRY,
    register_rate_template,
)

from .metric_templates import (
    METRIC_TEMPLATE_REGISTRY,
    register_metric_template,
)

from .schedule_templates import (
    SCHEDULE_TEMPLATE_REGISTRY,
    register_schedule_template,
)

from .calibration import (
    compute_rsquared,
    generic_accept_reject,
)

from .outcomes import (
    daily_transition_sum,
    compartment_timeseries,
    attack_rate,
    summarize_outcomes,
)
