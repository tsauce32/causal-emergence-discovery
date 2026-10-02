# Longitudinal validation and adjustment contracts

## Evaluation

`DiscoveryConfig.validation_mode` accepts `entity_holdout` (default),
`forward_time`, or `within_entity_interpolation`. `holdout_fraction` reserves
20% by default; entity sampling is reproducible with `seed`. Holdout fractions
refer to entities, unique predictor timestamps, and an interior block per
entity, respectively; they need not yield exactly that fraction of rows.

`build_validation_plan()` reads only entity IDs and actual predictor/lead
timestamps. `DataSplit` exposes `train_indices`, `test_indices`,
`purged_indices`, and `dropped_indices`. Outer indices address the supplied
lagged table; inner indices address the reset-index outer training frame.
`ValidationPlan.to_dict()` exports the full plan and coverage. Entity splits
keep trajectories whole. Forward splits train on earlier target observations
than the next validation predictor boundary. Interpolation is bidirectional
within observed entities and quarantines intersecting outcome windows.
Forward evaluation conditions on predictors available at each held-out time,
including prior outcomes when the spec allows them as predictors. It describes
prediction of subsequent outcomes from those measurements, not a multi-step
forecast made without observing any future covariates at the initial boundary.

Candidate definitions, state-merge search, and selection use development data.
For every inner split, `refit_macro()` learns the recipe's imputation, scaling,
cutpoints, or centers from training rows, then `apply_macro()` applies the
frozen encoder to test rows. Merge IDs refer to training-fitted rank positions
in a recipe. They are structural recipe labels, not certified identities for
the same population across different fitted distributions. A refit that loses
a requested state or cutpoint fails as unsupported instead of silently
reassigning that merge. Quantile ties are never split by row position. State
IDs enter regression as categories, without an ordinal distance assumption.

Merged-recipe IDs have a stricter recipe contract: quantile IDs are ordered bin
ranks; k-means IDs are lexicographic centroid ranks in the declared feature
order after canonical training-only fitting. These ranks define the same
structural recipe across folds, not correspondence to the same real-world
population. Preprocessing and thresholds/centroids are fit only on each
training fold. A merged recipe is supported only when the source fit and every
refit retain all requested occupied ranks and the merge operands still exist
at each chained step. Missing clusters, collapsed quantile cuts, or empty bins
reject the recipe. Unmerged recipes may use a reduced state count; requested
and observed topologies are recorded in the encoder and each score's
`fold_refit_topologies` list.

`fit_linear_model()` retains a `DesignEncoder`; `predict_linear_model()` and
`predict_r2()` reuse its frozen variable kinds, medians, and category levels.
Kinds come from declared `ColumnSpec.type` values when available; otherwise
they are inferred from training values only. The schema is resolved on each
training partition before outcome filtering, then preprocessing is fit on its
usable-target rows. Unknown prediction categories map to the fitted reference
design and are counted; non-finite numeric values are treated as missing and
imputed from the training median. Metrics use finite target rows; fewer than
two such rows or a constant validation target fails with a clear error.
Insufficient fold support also fails rather than changing the declared
evaluation.
Input columns that collide with generated target or internal metadata names
are rejected so a preexisting lead outcome cannot silently become a predictor.

CSV commands load the study specification before reading the panel and pass its
column declarations to the loader. Declared categorical columns are read as
strings, preserving tokens such as `"01"` and `"1"` before pandas can coerce
them to one number. Discovery's `feature_schema` reports `kind`, whether the
kind is `declared` or `outer_training_inference`, and outer counts for missing
values, invalid numeric values, and unknown categories.

`top_macros` and `searches` are development rankings. In discovery,
`validation_specificity` (also exposed under the compatibility key
`specificity`) is the mean of each inner fold's positive-clipped held-out R²
for state-only predictions, with state means refit on that fold's training
rows. The result records the definition as
`mean_positive_clipped_validation_r2_of_training_fitted_state_means`.
`outcome_specificity` is a different quantity: the full-development
between-state outcome variance share, included as a descriptive statistic and
excluded from ranking. Standalone `score_macro()` use without validation splits
cannot claim held-out specificity and labels its descriptive fallback explicitly.

The schema-v2 `ranking_score` is a heuristic weighted preference over
positive-clipped macro R², validation specificity, raw-fold R² dispersion, and
compression. The output exposes its named components and weights, raw paired
macro/micro R² values, and signed `predictive_r2_difference` (macro minus micro,
averaged across the same folds). `emergence_evidence_status` is always
`not_assessed`; the removed `emergence_delta` is not a substitute definition
of emergence. Once ranked, one chosen recipe is refitted on development data
and applied to the untouched outer holdout. `outer_evaluation` reports the
selected macro and micro predictive R², their signed difference, state
coverage, and frozen-encoder audit. When the reserved cohort lacks enough
eligible target labels for evaluation, it reports `status: not_evaluable`, a
reason, null predictive metrics, and reserved-versus-eligible target support;
the cohort is not silently replaced. Outer scores never enter candidate
scoring or reranking. Holdout metrics describe this selected recipe and split;
they are not causal effects. Repeatedly tuning against the outer report
invalidates its untouched status; a new test sample is then needed.

If any required fold cannot refit a merged recipe, `score_macro()` raises
`MacroScoreRefitError` with fold diagnostics and emits no score or partial fold
average. Search places that recipe in `rejected_candidates` with
`status: unsupported_refit_topology` and `comparable: false`; rejected recipes
do not enter `sampled_macros`, search paths, or rankings. A search with some
rejections reports `partial_topology_rejections`. If its starting recipe is
unsupported, it returns `best: null` and `no_valid_candidate`. Discovery ranks
only valid searches and fails clearly if none have supported topology on every
required fold. Legacy merged recipes without saved raw-topology evidence remain
usable through their frozen encoders but are unsupported for refitting.
## Adjustment

`StudySpec.adjustment` is optional. When supplied, it names covariates, explains
its causal rationale, and states whether the selected macro is an additional
regressor. Descriptive roles (`context`, `environment`, etc.) do not choose the
set. Declared continuous covariates remain continuous; they are never replaced
automatically by macro states. Outcome, treatment, ID, and time adjustment is
rejected. Without declared covariates, pathways still condition on other
declared interventions under the default `joint_conditional` estimand; no
covariates are automatically selected.

The adjustment estimand defaults to `joint_conditional`: all declared
interventions enter one additive model with the explicit covariates and optional
macro. Set `adjustment.estimand` to `marginal` to fit a separate model for each
intervention with those covariates, omitting the other interventions. An
`estimand` keyword override is also supported by the adjustment audit and the
resolved choice is reported. Here marginal means omission of co-treatments; the
coefficient still conditions on the declared covariates. Conditioning on
another treatment can change the scientific question or introduce mediator or
collider bias. Both model choices report observational associations, not
identified causal effects.

`resolve_adjustment()` validates the declaration.
`adjustment_sufficiency_audit()` compares macro-only coefficients with declared
adjustment, reports source covariates absent from the macro and information
coarsened into fewer states, and returns `pathways`. It also measures remaining
within-state numeric variance and treatment–covariate residual correlations.
Discovery calls it on development rows only. The audit is descriptive and
cannot verify the assumptions needed for identification. The caller must
justify that covariates are measured before treatment and block the required
backdoor paths without introducing collider or mediator bias. Consistency,
conditional exchangeability, positivity, interference assumptions, and model
specification still need scientific/design review.

OLS reports design rank, rank deficiency, residual degrees of freedom
(`nobs - rank`), and individual term estimability. If an intervention equals a
covariate exactly, its separate coefficient and uncertainty are suppressed.
Rank loss in nuisance terms does not automatically suppress an estimable
treatment term. Saturated designs can report estimable coefficients but have no
residual standard errors. Categorical regressors use frozen reference-level
indicator contrasts, with level and reference metadata in their term records.
Actual missing categories remain distinct from the literal `__missing__`
category; unseen prediction categories map to the fitted reference design.
Non-finite numeric predictors are treated as missing and use training medians.
Non-finite fitted scales, estimates, or R² values fail clearly. Non-finite
uncertainty is reported unavailable with a status, or raises a fit error when
residual scaling fails. JSON serialization rejects non-finite values.

Standard errors are IID-row OLS; panel-robust uncertainty is unsupported, so
they do not support inference on repeated entities. Fold score dispersion is
descriptive and is not an uncertainty interval. The API does not certify causal
effects or confounder sufficiency.
## Reproducible controls

Run `python examples/audit_validation_controls.py` and
`python examples/audit_adjustment_controls.py` with their output-path options to
generate the control reports.
Default reports are written beside these scripts as ignored `*.generated.json`
files; `--output` selects a separate results directory.
The entity fingerprint control compares row interpolation with unseen-entity
evaluation. The drift control compares row
interpolation with future prediction. Lower scores under the latter estimands
expose why panel row folds are insufficient for those claims, not a promise
that one estimand must always be harder.

The known-confounder control sets treatment = u + noise and the true linear
effect to 1.5. It compares the preserved coarse-adjustment result with explicit
continuous-u adjustment and reports the new development/outer separation. The
toy data-generating mechanism supplies the causal assumptions for this control;
the software cannot establish them in observational input.

## Scope and interpretation

The validation plan describes a prediction estimand and the rows used to
evaluate it; it does not certify the caller's causal assumptions or a formal
emergence test. `predictive_evaluation_scope` and optional IDs identify the
producer-declared paired score sample, but do not by themselves prove that the
models used identical target rows or a valid split. Fold dispersion is a
descriptive stability preference, not an uncertainty interval. See the
[design notes](design.md) for score definitions and limitations.
