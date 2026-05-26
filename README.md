# causal-emergence-discovery

`causal-emergence-discovery` is an experimental toolkit for discovering
macro-level causal structure in longitudinal tabular data.

Given a dataset with an entity ID, a time column, and user-described variables,
the library searches for higher-level representations that make candidate causal
pathways cleaner, more stable, and more compressive than raw observed variables.

The library does **not** claim to identify causal truth from arbitrary
observational data. It produces ranked causal hypotheses under explicit
temporal, adjustment, and invariance assumptions.

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
5. Scores candidates by:
   - outcome clarity
   - macro-specificity
   - cross-fold stability
   - compression
   - difference from a raw micro-feature reference
6. Estimates simple intervention-to-outcome pathway coefficients adjusted for
   the selected macro state and declared environment columns.

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
Under the declared temporal order and adjustment choices, this macro
representation gives a cleaner candidate pathway than the raw table alone.
```

It should not be read as:

```text
The library proved the true causal graph.
```

Useful follow-up work includes sensitivity analysis, stronger causal discovery
backends, learned latent-state models, invariance checks across environments,
and richer causal-emergence scores.
