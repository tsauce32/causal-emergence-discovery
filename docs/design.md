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

## Refit recipe identity

Merge IDs identify raw ranks under a declared training-only recipe. Quantile
IDs are ordered threshold-bin ranks. K-means IDs are lexicographic centroid
ranks in the declared feature order after canonical fitting. This rank contract
supports comparing the recipe across folds; it does not claim that states map
to the same latent or real-world populations. Each fold fits its own
preprocessing and centers or thresholds.

Merged recipes require complete raw topology in the source and every refit.
Missing clusters, collapsed cuts, empty bins, or missing operands in a chained
merge make the recipe unsupported. Unmerged recipes may refit with fewer
occupied states. The requested and observed fold topologies are reported in
`fold_refit_topologies`. If one required fold fails, `MacroScoreRefitError`
contains fold diagnostics and no partial average is emitted. Search excludes
rejected recipes from scores, paths, and ranking, and discovery fails clearly
when no candidate is supported on every required fold. Algorithm version 2
changes k-means initialization, reductions, and centroid labels, so earlier
k-means scores and rankings may change. Saved legacy merged recipes without
the rank/occupancy evidence remain usable for frozen application but cannot be
refit.

## Adjustment and causal caution

Adjustment variables are user-declared with a rationale; roles such as context
or environment never cause automatic adjustment. `adjustment.estimand` defaults
to `joint_conditional`: each declared treatment coefficient comes from one
regression containing all declared treatments and the explicit covariates, plus
the selected macro when `include_macro` is true. `marginal` instead fits one
regression per treatment, conditional on the declared covariates and optional
macro, while omitting the other treatments. “Marginal” here means separate
treatment regressions under that covariate set, not an unadjusted
population-average causal effect.

These are associational linear models. Joint adjustment can condition on a
mediator or collider if the specification is wrong; adding co-treatments does
not automatically yield a direct causal effect. The software does not select
extra adjustment variables. Rank and term-level diagnostics distinguish an
estimable term from one aliased with other columns. Categorical predictors use
reference-level indicator contrasts. Numeric missing and non-finite predictors
use the training median; categorical missing values have a distinct level,
even when a literal category is named `__missing__`. Variable kinds follow a
declared `ColumnSpec.type` when supplied; otherwise the kind is inferred from
training values only and frozen. The study spec is loaded before CSV parsing so
declared categorical tokens such as `"01"` and `"1"` stay distinct. The result's
`feature_schema` reports each kind, its declared/inferred source, and outer
missing, invalid numeric, and unknown category counts.

IID OLS standard errors do not account for repeated entities or panel dependence.
Term status distinguishes estimable from aliased terms; saturated designs with
zero residual degrees of freedom have no standard errors. Per-term uncertainty
status reports whether IID uncertainty is available, degrees of freedom are
zero, uncertainty is non-finite, or the term is aliased. Fit errors mark
uncertainty unavailable. The audit is descriptive and does not establish
exchangeability, positivity, consistency, or correct temporal ordering.
