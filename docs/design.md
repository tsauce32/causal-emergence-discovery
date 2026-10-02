# Design Notes

## Goal

The project searches for higher-level descriptions of a longitudinal table that
make candidate causal pathways clearer.

The MVP is intentionally simple:

- no hidden causal claims
- no required domain vocabulary
- no heavyweight causal discovery dependency
- explicit metadata for time, entity, outcomes, interventions, and environments

## Pipeline

```text
CSV + study spec
  -> validated panel
  -> lagged table
  -> entity / time / interpolation validation plan and reserved outer holdout
  -> candidate macro variables
  -> branching greedy state-merge search
  -> macro emergence scores
  -> locked selected macro and untouched outer predictive evaluation
  -> explicitly adjusted development-only pathway diagnostics
  -> JSON/text report
```

## Score Components

The current macro score is a weighted combination of:

- `macro_r2`: development-fold predictive clarity of intervention + categorical macro state
- `specificity`: positive development-validation R² of training-fitted state means
- `stability`: inverse cross-fold score dispersion
- `compression`: fewer macro states relative to row count

`emergence_delta` subtracts the weighted positive raw micro-feature reference R2.
This is a pragmatic MVP score, not a final definition of causal emergence.
All these terms belong to development selection. The selected macro's
`outer_evaluation` contains separate predictive R² values with no holdout
utility calculation or candidate reranking. [Validation contracts](validation.md)
describe the split and fitted-transform APIs.

## Causal Caution

The MVP estimates intervention coefficients with linear adjustment models. These
are causal hypotheses under assumptions, not proof. Future versions should add
sensitivity analysis, doubly robust estimators, causal graph constraints, and
environmental invariance tests.
