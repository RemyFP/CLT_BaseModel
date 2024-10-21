from BaseModel import BaseModel, SimulationParams
from PlotTools import create_basic_compartment_history_plot
import numpy as np
import matplotlib.pyplot as plt
import time
import json

from collections import namedtuple

from pathlib import Path

def get_change_in_immunity(current_val,
                           R_val,
                           immunity_increase_factor,
                           total_population_val,
                           saturation_constant,
                           waning_factor):
    return np.asarray(
        (immunity_increase_factor * R_val) / (total_population_val * (1 + saturation_constant * current_val)) \
        - waning_factor * current_val)


def get_new_exposed_rate(I_val,
                         immunity_against_inf,
                         efficacy_against_inf,
                         beta,
                         total_population_val):
    return np.asarray(beta * I_val / (total_population_val * (1 + efficacy_against_inf * immunity_against_inf)))


def get_new_hosp_rate(I_to_H_rate,
                      I_to_H_adjusted_proportion,
                      immunity_against_hosp,
                      efficacy_against_hosp):
    return np.asarray(I_to_H_rate * I_to_H_adjusted_proportion / (1 + efficacy_against_hosp * immunity_against_hosp))


def get_new_dead_rate(H_to_D_adjusted_proportion,
                      H_to_D_rate,
                      immunity_against_hosp,
                      efficacy_against_death):
    return np.asarray(H_to_D_adjusted_proportion * H_to_D_rate / (1 + efficacy_against_death * immunity_against_hosp))


class ImmunoSEIRSModel(BaseModel):

    def __init__(self,
                 epi_params_json_filename,
                 simulation_params_json_filename,
                 epi_compartments_json_filename,
                 state_variables_json_filename,
                 RNG_seed=np.random.SeedSequence()):
        super().__init__(RNG_seed)

        self.add_epi_params_from_json(epi_params_json_filename)
        self.add_simulation_params_from_json(simulation_params_json_filename)
        self.add_epi_compartments_from_json(epi_compartments_json_filename)
        self.add_state_variables_from_json(state_variables_json_filename)

    def update_change_in_state_variables(self):
        epi_params = self.epi_params

        self.population_immunity_hosp.change_in_current_val = get_change_in_immunity(
            current_val=self.population_immunity_hosp.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_hosp_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_constant=epi_params.immunity_hosp_saturation_constant,
            waning_factor=epi_params.waning_factor_hosp
        )

        self.population_immunity_inf.change_in_current_val = get_change_in_immunity(
            current_val=self.population_immunity_inf.current_val,
            R_val=self.R.current_val,
            immunity_increase_factor=epi_params.immunity_inf_increase_factor,
            total_population_val=epi_params.total_population_val,
            saturation_constant=epi_params.immunity_inf_saturation_constant,
            waning_factor=epi_params.waning_factor_inf
        )

    def update_transition_rates(self):
        epi_params = self.epi_params

        self.new_exposed.current_rate = get_new_exposed_rate(
            I_val=self.I.current_val,
            immunity_against_inf=self.population_immunity_inf.current_val,
            efficacy_against_inf=epi_params.efficacy_immunity_inf,
            total_population_val=epi_params.total_population_val,
            beta=epi_params.beta)

        self.new_hosp.current_rate = get_new_hosp_rate(
            I_to_H_rate=epi_params.I_to_H_rate,
            I_to_H_adjusted_proportion=epi_params.I_to_H_adjusted_proportion,
            immunity_against_hosp=self.population_immunity_hosp.current_val,
            efficacy_against_hosp=epi_params.efficacy_immunity_hosp
        )

        self.new_dead.current_rate = get_new_dead_rate(
            H_to_D_rate=epi_params.H_to_D_rate,
            H_to_D_adjusted_proportion=epi_params.H_to_D_adjusted_proportion,
            immunity_against_hosp=self.population_immunity_hosp.current_val,
            efficacy_against_death=epi_params.efficacy_immunity_death
        )

        num_age_groups = self.epi_params.num_age_groups
        num_risk_groups = self.epi_params.num_risk_groups

        self.new_infected.current_rate = np.full((num_age_groups, num_risk_groups), epi_params.E_to_I_rate)

        self.new_recovered_home.current_rate = np.full((num_age_groups, num_risk_groups),
                                                       (1 - epi_params.I_to_H_adjusted_proportion) * epi_params.I_to_R_rate)

        self.new_recovered_hosp.current_rate = np.full((num_age_groups, num_risk_groups),
                                                       (1 - epi_params.H_to_D_adjusted_proportion) * epi_params.H_to_R_rate)

        self.new_susceptible.current_rate = np.full((num_age_groups, num_risk_groups),
                                                    epi_params.R_to_S)

base_path = Path(__file__).parent

random_seed = np.random.SeedSequence()

model_1age_1risk = ImmunoSEIRSModel(base_path / "instance1_1age_1risk_test" / "epi_params.json",
                                    base_path / "instance1_1age_1risk_test" / "config.json",
                                    base_path / "instance1_1age_1risk_test" / "epi_compartments_state_vars_init_vals.json",
                                    base_path / "instance1_1age_1risk_test" / "state_variables_init_vals.json",
                                    random_seed)

import time

start = time.time()

model_1age_1risk.simulate_until_time_period(365)

print(time.time() - start)

create_basic_compartment_history_plot(model_1age_1risk)