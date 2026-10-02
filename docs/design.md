# Design Notes

## Goal

The project ranks candidate macro representations of a longitudinal table
under a declared predictive validation design.

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
  -> heuristic ranking scores and paired predictive summaries
  -> locked selected macro and untouched outer predictive evaluation
  -> explicitly adjusted development-only pathway diagnostics
  -> JSON/text report
```

## Score Components

`ranking_score` is a heuristic weighted combination of:

- positive-clipped `macro_r2`: predictive R² for intervention + categorical macro state
- `validation_specificity`: mean positive-clipped held-out R² from state-only models whose state means are fit on each inner training fold
- `stability`: inverse dispersion of the raw macro fold R² values
- `compression`: fewer macro states relative to row count

The weights are 0.45, 0.25, 0.20, and 0.10, respectively. The separately
reported `outcome_specificity` is the descriptive full-development
between-state outcome variance share; it is not the validation specificity
used for ranking. Raw `macro_r2`, raw `micro_r2`, and signed
`predictive_r2_difference` (`macro_r2 - micro_r2`, averaged over paired folds)
remain available for comparison. Fold-level raw values and differences are
retained. Negative R² values are not clipped for reporting or dispersion.

The score is a heuristic ranking preference, not a detector or measure of
causal emergence. Candidate score records carry
`emergence_evidence_status: "not_assessed"`; there is no `emergence_delta`
field. All candidate ranking uses development
data. The selected macro's `outer_evaluation` reports its separate predictive
performance on the reserved holdout and never enters candidate scoring or
reranking. [Validation contracts](validation.md) describe split construction,
training-only refits, and adjustment.

## Causal Caution

The MVP reports exploratory intervention associations with a declared
adjustment model. Causal interpretation requires a scientifically justified
adjustment set and assumptions the software cannot verify. The workflow does
not identify causal effects or establish causal emergence. Future versions
could add sensitivity analysis, doubly robust estimators, causal graph
constraints, and environmental invariance tests.
