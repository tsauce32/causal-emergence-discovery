# causal-emergence-discovery

`causal-emergence-discovery` is an experimental toolkit for ranking candidate
macro representations of longitudinal tabular data.

Given a dataset with an entity ID, a time column, and user-described variables,
the library searches for higher-level representations using predictive fit,
validation specificity, score dispersion, and compression preferences.

The library does **not** establish causal emergence or identify causal truth
from arbitrary observational data. Its ranking score is a heuristic preference
over candidate representations. Predictive R² comparisons are descriptive and
are not causal effects or formal causal-emergence evidence.

## Input Shape

The MVP expects one CSV row per entity-time observation:

```text
entity_id,time,...open-ended columns...
```

The study spec is a small JSON file:

```json
{
  "dataset": {
    "id_column": "entity_id",
    "time_column": "time"
  },
  "outcomes": ["reading_score"],
  "interventions": ["program_hours"],
  "environments": ["site"],
  "columns": {
    "reading_score": {"role": "outcome", "type": "numeric"},
    "program_hours": {"role": "intervention", "type": "numeric"},
    "site": {"role": "environment", "type": "categorical"},
    "attendance": {"role": "measurement", "type": "numeric"},
    "stress": {"role": "context", "type": "numeric"}
  }
}
```

The roles are domain-agnostic. The same contract can describe children,
patients, schools, customers, machines, regions, sensors, or other longitudinal
entities.

## What The MVP Does

1. Validates the panel structure.
2. Builds lagged modeling rows, such as `variables_t -> outcome_t+1`.
3. Generates candidate macro variables with quantiles, composites, and
   deterministic k-means over numeric state/context features.
4. Runs branching greedy state-merge search over candidate macrostates.
5. Ranks candidates with a weighted heuristic: 0.45 times positive-clipped
   macro R², 0.25 times validation specificity, 0.20 times fold stability, and
   0.10 times compression. Separately, it reports raw macro and micro R² values
   and their signed mean paired-fold difference (`macro R² - micro R²`).
6. Locks the selected macro and evaluates it on a reserved outer holdout.
7. Reports exploratory intervention coefficients using an explicit adjustment
   declaration and compares them with macro-only adjustment on development rows.

## Validation and adjustment

Discovery defaults to `entity_holdout`: the outer holdout and inner folds keep
all modeling observations of each entity together. This estimates transfer to
new entities. Choose `forward_time` to estimate prediction in future periods
with expanding training windows, or `within_entity_interpolation` for interior
time blocks from already observed entities. These modes answer different
questions; their R² values should not be treated as interchangeable estimates.

```bash
ced discover panel.csv study.json --validation-mode entity_holdout --folds 3 --holdout-fraction 0.2 --seed 0 --output result.json
ced discover panel.csv study.json --validation-mode forward_time --folds 3 --output future-result.json
```

Quantile cutpoints, missing-value medians, composite scaling, k-means centers,
and regression categories are fitted on each inner training fold. State merges
are candidate recipes evaluated with those fitted transforms. Validation
specificity is the mean positive-clipped held-out R² of state-only predictions
whose state means were learned on each training fold. This named quantity is
the specificity input to the discovery ranking. The full-development
between-state outcome variance share is separately reported as descriptive
`outcome_specificity`; it is not a validation estimate. Search and
outcome-dependent ranking use development data only. The chosen recipe is
refitted on development data, frozen, and evaluated once on the outer holdout.
Merge labels identify rank positions in the fitted recipe; they do not certify
that clusters represent the same populations after refitting. A refit that
cannot support a requested merge is marked unsupported.

`ranking_score` is a heuristic preference. Its components and weights are
reported with the result. `predictive_r2_difference` is the signed mean paired
fold gap between macro and micro models, with their raw R² values and fold
values retained. A positive difference does not assess emergence. Each
macro score reports `emergence_evidence_status: "not_assessed"`; the former
`emergence_delta` field is removed.

`top_macros` and `searches` are development selection results, which can be
optimistic after searching many candidates. `outer_evaluation` reports the
selected macro's untouched holdout R² and the micro reference on the same rows
when enough reserved target labels are available. Otherwise it reports
`not_evaluable`, a reason, null metrics, and reserved-versus-eligible target
support without replacing the reserved cohort. It does not calculate holdout
specificity or rank other macros. Negative R² is preserved in the predictive
comparison and fold-dispersion calculation. The ranking formula is
experimental; predictive performance is separate from causal identification.

`validation` records the exact row positions, their source frame, entity and
time coverage, and purged/excluded rows. Forward splits purge training leads
that reach the first test predictor time. Interpolation purges overlapping
predictor-to-target windows in both directions. Lag counts observations within
an entity, not calendar intervals. Too few entities, usable timestamps, rows,
or nonconstant finite targets raises an error; folds are never silently reduced.
Rows contribute equally to R² within each split, and development fold R² values
are averaged equally; entity holdout keeps sampling units whole but does not
make a long trajectory and a short trajectory carry equal metric weight.

Column roles describe variables; they do not determine a valid adjustment set.
For example, a declared pre-treatment common cause can be included explicitly:

```json
"adjustment": {
  "columns": ["u"],
  "rationale": "u is a measured pre-treatment common cause of treatment and outcome",
  "include_macro": false
}
```

Without this declaration, pathway coefficients are treatment-only associations.
`include_macro: true` adds the selected discrete macro while retaining all
declared covariates. Context/environment roles never automatically add
covariates. The `adjustment` report compares macro-only and declared adjustment,
and flags declared covariates omitted or coarsened by the macro. A coefficient
change is a descriptive diagnostic; stability does not establish sufficiency.
Causal interpretation would require a justified pre-treatment adjustment set,
conditional exchangeability, consistency, positivity, appropriate interference
assumptions, and a correctly specified model. Current standard errors assume
independent rows and are unsuitable for inference on repeated trajectories.

See [validation design](docs/validation.md) for contracts and control
experiments. Standalone `score_macro()` calls without explicit validation
splits use the helper's row-fold evaluation and descriptive in-sample
specificity; they do not reserve a final holdout. `run_discovery()` supplies
nested training-only folds and performs the selected-model outer evaluation.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python examples/generate_synthetic_panel.py
ced validate examples/synthetic_panel.generated.csv examples/synthetic_study.generated.json
ced discover examples/synthetic_panel.generated.csv examples/synthetic_study.generated.json --outcome reading_score
python -m pytest
```

Or from Python:

```python
import pandas as pd

from causal_emergence_discovery import DiscoveryConfig, StudySpec, run_discovery

df = pd.read_csv("examples/synthetic_panel.generated.csv")
spec = StudySpec.from_dict({
    "dataset": {"id_column": "entity_id", "time_column": "time"},
    "outcomes": ["reading_score"],
    "interventions": ["program_hours"],
    "environments": ["site"]
})
result = run_discovery(df, spec, DiscoveryConfig(outcome="reading_score", lag=1))
print(result["top_macros"][0])
```

## Interpretation

The output should be read as:

```text
Under the declared validation design, this macro representation receives a
higher heuristic ranking for the selected predictive and descriptive criteria.
```

It should not be read as:

```text
The library detected causal emergence or proved the true causal graph.
```

Useful follow-up work includes sensitivity analysis, stronger causal discovery
backends, learned latent-state models, invariance checks across environments,
and prespecified tests of causal-emergence definitions.
