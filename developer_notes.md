# Journal

## 2025 12 11
Added input checks for subpop and metapop models.
We check that humidity, vaccination, contact matrix, and initial compartment values are non-negative. All values at zero are possible but wouldn't make sense.
For transition rates we need strictly positive values.
For vaccination rates we check whether cumulative vaccination rates in each age-risk group are not exceeding 100% in any 365-day period. This only issues a warning.
The mobility matrix (or travel_proportions) should have rows that sum to 1: this ensures people either travel to another subpopulation or stay in their home location.

A new parameter called use_deterministic_softplus is added to the simulation settings. If the object oriented model is run with deterministic transitions this can be used to prevent softplus values instead of zeros in compartments, which leads to strange behaviors when epidemics occur in populations without any exposure.

Small fixes were made to the travel model equations in the file `flu_travel_functions.py`.

## 2025 11 17 - Adding ghost compartments
Updated website notation and made code updates in a lot of places.


# For future developers (from LP)

Technical notes
- After making changes, please make sure ALL tests in `tests/` folder pass, and add new tests for new code.
- Due to the highly complicated influenza model, there are a lot of input combinations and formats -- errors might arise due to incorrect inputs (e.g. dimensions, missing commas, etc...) -- if there is an error in running the model, the inputs should be checked first. Additionally, more work should be spent on writing input validators and error messages.

Tests to add
- Experiments
  - Make sure aggregating over subpopulation/age/risk is correct (e.g. in `get_state_var_df`).
  - Make sure all the ways to create different CSV files lead to consistent results!
- Accept-reject sampling
  - Reproducibility: running the algorithm twice (with the same RNG each time) should give the same result.
  - Make sure the sampling updates are applied correctly (e.g. to the correct subpopulation(s) and with the correct dimensions).

Features to add
- Would be nice to make the "checker" in `FluMetapopModel` `__init__` method more robust -- can check dimensions, check for nonnegativity, etc...