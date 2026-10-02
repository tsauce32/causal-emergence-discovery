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

`fit_linear_model()` retains a `DesignEncoder`; `predict_linear_model()` and
`predict_r2()` reuse its medians and category levels. New categories map to zero
in the fitted dummy design. Metrics use finite target rows; fewer than two such
rows or a constant validation target fails with a clear error. Insufficient
fold support also fails rather than changing the declared evaluation.
Input columns that collide with generated target or internal metadata names
are rejected so a preexisting lead outcome cannot silently become a predictor.

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

## Adjustment

`StudySpec.adjustment` is optional. When supplied, it names covariates, explains
their causal rationale, and states whether the selected macro is an additional
regressor. Descriptive roles (`context`, `environment`, etc.) do not choose the
set. Declared continuous covariates remain continuous; they are never replaced
automatically by macro states. Outcome, treatment, ID, and time adjustment is
rejected. Absence of a declaration produces unadjusted treatment associations.

`resolve_adjustment()` validates the declaration. `adjustment_sufficiency_audit()`
compares macro-only coefficients with the declared adjustment, reports source
covariates absent from the macro and information coarsened into fewer states,
and returns `pathways`. It also measures remaining within-state numeric variance
and treatment–covariate residual correlations. Discovery calls it on development
rows only. This
comparison cannot verify the unobserved assumptions needed for identification.
The caller must justify that covariates are measured before treatment and block
the required backdoor paths without introducing collider or mediator bias.
Consistency, conditional exchangeability, positivity, interference assumptions,
and model specification still need scientific/design review. IID standard
errors are explicitly labeled; repeated-entity inference requires a suitable
cluster or trajectory-based procedure.

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
