# Frozen selection-aware benchmark protocol

This directory defines the simulation budget and the summaries for the
selection-aware Discovery benchmark. The protocol is frozen before inspecting
outer benchmark results. The full profile plans 32 independent complete
pipeline replicates in each of 12 cells (384 runs); the quick profile plans two
replicates per cell (24 runs) for plumbing and smoke checks only. The quick
profile is not evidence for performance, error rates, or coverage.

## Cells and estimands

The ten generated cases are `iid_null`, `additive_signal`,
`nonlinear_threshold`, `sign_interaction`, `entity_fingerprint`,
`temporal_drift`, `rare_absent_states`, `known_confounding`,
`correlated_treatments`, and `population_shift`. All use an
`entity_holdout` split except `temporal_drift` and `population_shift`, which
use `forward_time`. Two additional `interpolation` cells compare the
entity-fingerprint and temporal-drift cases against seen-entity interpolation.
The resulting cells are:

| Case | Split | Target population / purpose |
| --- | --- | --- |
| iid_null | entity_holdout | New-entity null calibration illustration |
| additive_signal | entity_holdout | New-entity prediction under additive signal |
| nonlinear_threshold | entity_holdout | New-entity threshold signal and candidate-recovery stress |
| sign_interaction | entity_holdout | New-entity interaction and matched-information stress |
| entity_fingerprint | entity_holdout | Transfer to unseen entities |
| entity_fingerprint | interpolation | Conditional interpolation among observed entities |
| temporal_drift | forward_time | Rolling conditional prediction after temporal change |
| temporal_drift | interpolation | Seen-time interpolation comparison |
| rare_absent_states | entity_holdout | New-entity prediction with sparse or absent states |
| known_confounding | entity_holdout | Predictive and declared-adjustment stress under a known mechanism |
| correlated_treatments | entity_holdout | Declared treatment-estimand stress |
| population_shift | forward_time | Forward prediction under shifted population composition |

Forward-time evaluation is rolling conditional prediction: at each held-out
time the generated predictor is observed. It does not evaluate a fixed-origin,
recursive multi-step forecast. The confounding cell's known-effect adjustment
summary is distinct from the predictive comparison and from a causal claim in
observational data.

## Frozen settings and seeds

| Setting | Full | Quick |
| --- | ---: | ---: |
| Independent replicate seeds per cell | 32 | 2 |
| Entities | 48 | 24 |
| Periods | 16 | 12 |
| Inner folds | 3 | 2 |
| Outcome lag | 1 | 1 |
| Outer holdout fraction | 0.25 | 0.25 |
| Maximum macro states | 3 | 3 |
| Candidate paths | 2 | 1 |
| Branching | 2 | 2 |

Seed families derive from NumPy `SeedSequence` entropy `20261002`. The quick
profile uses the first two seeds of the full profile. A replicate seed is
shared across cells as a paired simulation index; each cell remains a separate
data-generating run. A failed or unsupported run is recorded under its
assigned seed and is never rerun with a replacement seed. Freeze and hash the
serialized profile before starting the full run. Summary bootstrap draws use
1,999 fixed-seed resamples (`20261003`) of complete replicate outcomes.

## Selection, evaluation, and metrics

Each replicate runs the complete declared Discovery procedure. Reserve the
outer cohort first; perform candidate generation, encoding, state selection,
merge/search choices and any tuning within development data; refit the frozen
selected recipe on development data; and evaluate that recipe once on the
reserved cohort. Do not rank or tune candidates from outer results. Macro and
micro predictions use the same held-out rows and target. The benchmark record
retains paired outer R² for the selected macro, restricted micro, no-macro
intercept, prespecified macro, flexible micro, and matched micro comparators
when the comparator is available, plus the raw outer R² for each comparator.

The primary predictive contrast is selected macro minus restricted micro
held-out R². Also summarize contrasts against no macro, the prespecified macro,
flexible micro, and matched micro. Preserve signed negative and positive R²
values. Report compression, selection score, selected family and state-count
frequencies, target and per-state row/entity support, test coverage, failed or
unsupported refits, and explicit failure reasons when the runner supplies
them. `selected_family` counts describe the choices made by the full search;
they are descriptive stability summaries.
If the generator supplies an oracle micro basis, summarize its held-out R²
across independent replicates under the label “generator-known oracle basis;
excluded from Discovery selection.” It is a diagnostic comparator and never a
candidate or selection input.

An initial support flag marks a state below 20 development rows across at
least 5 development entities. This is a stress diagnostic, not a universal
validity boundary. It must not be used to silently remove difficult runs from
the denominator. Keep unsupported state coverage and failed refits visible.
Summaries retain target training/test row and entity support, test coverage,
and per-replicate per-state row/entity support so sparse and absent states can
be inspected without assuming state labels align across refits.

The output must always retain `emergence_evidence_status: not_assessed`. A
positive predictive contrast, a selected family frequency, or a recovery
count does not declare emergence. This bounded design is not the 1,000-null
replicate gate recommended for calibrated false-positive claims; it does not
support a calibrated false-positive rate, power, detection rule, or causal
emergence claim.

## Uncertainty and denominators

The independent complete-pipeline replicate is the unit of uncertainty.
Calculate Monte Carlo standard errors and two-sided 95% Student-t intervals
from replicate-level paired contrasts. For a single usable replicate, report
the point estimate with no interval. A fold standard deviation is split
dispersion and is never an interval or standard error. Positive-contrast
fractions use Wilson intervals over usable independent replicates. Also report
a deterministic percentile interval from 1,999 resamples of the independent
complete-pipeline replicate outcomes. Each observation in this resampling
already reran the full candidate selection and refit on an independently
generated dataset, so this interval reflects selection variability in the
declared DGP and profile. It describes expected predictive behavior under that
simulation design, conditional on producing a finite score; it is not an
entity-level inferential interval or a claim about real-world emergence.
Report the raw held-out R² mean and uncertainty for each comparator alongside
the paired contrasts.

For every cell, retain the planned replicate count as the denominator for
failure rates, support-threshold rates, and counts. Report the number of
records received, missing records, successful runs, failed/unsupported runs,
and the number of runs contributing each score. Score intervals are
conditional on runs that produced finite paired scores; the accompanying
failure and missing counts make that conditioning visible. Do not treat folds
or rows as independent replicates.

For the known-confounding cell, report an independent entity-cluster bootstrap
interval for the prespecified treatment-plus-raw-`U` adjustment model when the
runner provides it. This model is fixed by the simulation estimand and does not
depend on the selected macro recipe. Report Wilson uncertainty for interval
coverage as constructed-SCM coefficient coverage only; it is not a Discovery
selection or emergence coverage claim. At 32 full replicates this coverage
summary is descriptive and low precision, not evidence of nominal interval
coverage.

## Runner contract

`benchmarks.pipeline.protocol` exposes:

* `get_profile(name: str) -> dict`: return the frozen JSON-serializable
  `full` or `quick` profile, including the cell matrix and all settings,
  including the seed and bootstrap seed/budget.
* `generate_seeds(profile: dict) -> list[int]`: return deterministic seeds for
  independent complete-pipeline replicates.
* `summarize_trials(records: list[dict], profile: dict) -> dict`: summarize
  one record per cell and seed. Each record contains `case`, `split`, `seed`,
  `status` (`success` for an evaluable run), `paired_r2` (the comparator names
  above, with unavailable values set to null), `compression`,
  `selected_family`, `selected_state_count`, `support`, and `failure_reason`.
  An optional `oracle_micro` object contains finite `r2`, generator-known
  `features`, and its label; it is excluded from Discovery selection.
  Optional `selection_score`, truth-recovery diagnostics, and adjustment
  cluster-bootstrap fields may be included.

The runner must write the serialized profile and its hash before full
execution, then preserve every planned cell/seed outcome, including failures.
Summarization rejects duplicate records and seeds outside the frozen family.

