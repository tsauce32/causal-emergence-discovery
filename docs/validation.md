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
frozen encoder to test rows. Validation labels retain training state identities
regardless of row ordering. Quantile ties are never split by row position.
State IDs enter regression as categories, without an ordinal distance assumption.

`fit_linear_model()` retains a `DesignEncoder`; `predict_linear_model()` and
`predict_r2()` reuse its medians and category levels. New categories map to zero
in the fitted dummy design. Metrics use finite target rows; fewer than two such
rows or a constant validation target fails with a clear error. Insufficient
fold support also fails rather than changing the declared evaluation.
Input columns that collide with generated target or internal metadata names
are rejected so a preexisting lead outcome cannot silently become a predictor.

`top_macros` and `searches` are development rankings. Specificity now measures
validation predictions of training-fitted state means, not full-sample target
separation. Once ranked, one chosen recipe is refitted on development data and
applied to the untouched outer holdout. `outer_evaluation` reports macro R²,
micro R², their signed predictive difference, state coverage, and the frozen
encoder audit. No outer outcome influences selection or specificity. Holdout
metrics describe this selected recipe and this split; they are not identified
causal effects. Repeatedly tuning against the outer report invalidates its
untouched status; a new test sample is then needed.

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

## Integration with the parallel scoring priority

Base: upstream commit `22538091e1b44941ee750e96c5d8afe048fe9ae0`.
Checkout: `work/discovery-validation`, branch `codex/discovery-validation`.
The preserved review checkout remains unchanged.

Potential overlap with scoring work:

- `scoring.py`: `score_macro(..., validation_splits=None)` refits transforms per
  fold, uses categorical states, and supplies validation specificity. The
  `MacroScore` fields, serialized keys, weighted score formula, and asymmetric
  `emergence_delta` formula are left for the scoring priority to repair.
- `search.py`: forwards `validation_splits`; ranking semantics are unchanged.
- `discovery.py`: reserves the outer split before candidate creation, scopes
  ranking to development, and adds `validation`, `outer_evaluation`, and
  `adjustment` results. Scoring-specific output descriptions must be reconciled.
- `models.py`: adds training-fitted `DesignEncoder` to `LinearFit`; CV accepts
  explicit splits. Calls omitting splits retain legacy row interpolation and
  have no selected-model holdout guarantee.
- `macro.py` and `panel.py`: fitted macro recipes/merges and explicit target-time
  metadata. `spec.py` adds the independent adjustment declaration.
- `cli.py`, `README.md`, and `docs/design.md`: evaluation and adjustment wording
  may overlap with updated score/API descriptions.

New independent modules: `validation.py` and `adjustment.py`. Regression tests
are separated in `test_validation.py`, `test_encoders_adjustment.py`, and
`test_nested_evaluation.py`. Integrate the fold/outer evaluation changes with the
other chat's score formula; do not restore row-based validation while resolving
conflicts. `outer_evaluation.predictive_r2_difference` is explicitly a predictive
comparison, not a replacement definition for that chat's emergence delta.
