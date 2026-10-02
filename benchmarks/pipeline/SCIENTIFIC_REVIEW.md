# Scientific review of the selection-aware Discovery benchmark

## Review scope and source identity

This is an adversarial review of the selection-aware simulation design and its
final execution. It is not a causal-emergence test, a CE2/PyMergence
conformance study, or an analysis of real observational effects. The reviewed
Discovery implementation must be the ready, pinned integrated checkout named
by `research/discovery-integrated/status.json`; the separate merge simulation
and the validation-only base are not eligible experiment sources. The
benchmark manifest must identify the integrated commit and hash the generator,
runner, baseline, protocol, and written protocol files.

The frozen full profile plans 32 independent generated panels in each of 12
case/split cells. Seeds index independent data-generating runs; folds, rows,
and repeated observations within one panel are not independent replicates.
The same seed family is reused across cells, so cells are paired by seed; the
fingerprint and drift panels are each evaluated under two split modes using
the same generated panel for a given seed. Thus 384 pipeline runs do not mean
384 mutually independent generated datasets. Per-cell Monte Carlo summaries
use that cell's 32 independent seed draws and must not be pooled as though all
run records were independent.
The selected macro, restricted micro, intercept-only, prespecified macro,
flexible micro, and basis-matched micro are evaluated on the same reserved
rows and targets where each is available. Failures, unsupported recipes, and
missing records remain visible against the planned denominator; an unavailable
paired comparator has its own usable-score count.

## Acceptance criteria

The benchmark supports an interpretable predictive result only if all of these
checks pass:

1. The outer cohort is chosen independently of held-out outcomes and held-out
   predictor dtype. Making every reserved target missing may make outer scoring
   unavailable, but it must not cause a new cohort to be selected or alter
   development candidate scores and the selected recipe.
2. Declared numeric features remain numeric under the frozen schema when a
   malformed string appears only in a held-out row. That row may become
   unusable or fail evaluation; the feature and development ranking remain
   intact.
3. Rare and absent states are reported by both row and independent-entity
   support. Unsupported state/refit topology is flagged or rejected; numeric
   cluster IDs must not silently acquire a different population meaning after
   refitting. The 20-row/5-entity threshold is a stress diagnostic specific to
   this study, not a universal rule.
4. Macro and comparator scores use identical outer observations. The
   basis-matched comparator is independently implemented and checks that the
   frozen macro basis itself can be fit directly; equality is expected by
   construction and is only a representational-equivalence audit. Formula
   oracle features are separate comparators, labeled as generator-known and
   excluded from candidate search.
5. Source, protocol settings, seeds, and benchmark code are auditable. Failed
   seeds are never replaced or silently rerun. A smoke run is not substituted
   for the frozen full matrix.
6. Uncertainty is summarized over complete independent simulation
   replicates. Fold standard deviations are split dispersion, not standard
   errors or confidence intervals. Score intervals condition on trials with
   finite paired scores and must be read with the failure/missing denominators.

Environment-gated adversarial regressions are in
`tests/test_pipeline_adversarial.py`. They load only the source named by
`DISCOVERY_BENCHMARK_SOURCE` and
`DISCOVERY_BENCHMARK_EXPECTED_COMMIT`; without both values they skip. The
validation base is explicitly rejected by the fixture.

## What this bounded design can and cannot establish

If the checks pass, the study can describe the selection behavior and
held-out predictive differences of this fixed candidate family under these
specified synthetic panel mechanisms, split rules, sample sizes, and random
seed family. The nonlinear and interaction controls can show when a selected
coarse representation predicts better than the specified restricted linear
micro reference. A generator-known oracle basis can show whether that
advantage disappears when the micro model receives the same formula-level
information. Entity fingerprints, temporal drift, rare support, confounding,
correlated treatments, and population shifts are targeted stress cases, not
estimates of their prevalence in real studies.

The integrated encoder drops a reference level for categorical inputs and
maps an unseen category to that reference-level contribution. For the
entity-fingerprint control under entity holdout, this is an explicit
extrapolation convention for a never-seen entity ID; its predicted effect is
the reference entity contribution, not a learned or identified new-entity
effect. Scores for that case must be interpreted with this convention and
unseen-category support in view. Independent score reproduction must use the
same frozen encoding convention; a redundant all-indicator basis with an
intercept has a different least-squares solution under rank deficiency.

Before any final outer results were inspected, the `correlated_treatments`
generator was explicitly amended to add a shared latent shock `W` to both
treatment equations. `W` is independent of `U` and all errors, and has no
direct outcome effect; it induces residual treatment correlation of about
0.69 after adjustment for `U`. The joint conditional coefficients remain
1.7 and -0.9, while separate single-treatment models adjusted for `U` omit a
correlated co-treatment and target a different coefficient. The amendment
preserved the cells, sample sizes, search budget, split rules, seed family,
primary comparisons, and uncertainty protocol. Its reason and old/new DGP
hashes are recorded in
`research/discovery-pipeline-benchmark/design-amendment-conditional-treatments.json`;
the original freeze is retained separately. This is a documented correction
to the planned stress mechanism, not a change based on benchmark outcomes.

This design uses 32 null replicates and does not define a calibrated rejection
rule. It cannot support a calibrated false-positive rate, power, sensitivity,
detection, or interval-coverage claim for Discovery's full search. The
research design's 1,000-null gate for an empirical 3–7% false-positive rate is
not met. Even if a null cell has a mostly positive or negative mean difference,
that result is a descriptive finite-simulation contrast and does not become an
emergence decision.

No predictive contrast, ranking score, compression value, selection
frequency, recovery count, or adjustment coefficient identifies causal
emergence. No result computes Hoel's causal emergence or CE2 estimand. The
known-confounding and correlated-treatment controls verify behavior only under
their generated equations and declared adjustment set; they do not establish
exchangeability, adjustment sufficiency, positivity, consistency, absence of
interference, or real-world causal identification. The matched oracle
comparisons are mechanism-informed benchmarks and are not blinded discovery
baselines.

The benchmark's Student-t and percentile-bootstrap summaries quantify
Monte Carlo variation across the chosen independent replicate seeds. The
known-confounder entity bootstrap estimates the prespecified treatment
coefficient conditional on the generated panel and raw-`U` adjustment model;
it does not depend on the selected macro. A coverage fraction over 32
independent panels is a low-precision descriptive check under the constructed
SCM, not validation of nominal 95% coverage for Discovery's full search.
Neither interval family is an inferential interval for real-world emergence.

## Results audit

The final matrix contains 384 unique planned records: 32 seeds in each of 12
frozen case/split cells. All 384 completed successfully, with no unsupported
trial, missing record, or replacement seed. The integrated source was clean
and pinned at `735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d`; the final DGP SHA-256
matches the amended freeze, and the manifest seed list matches the frozen
seed list. The parent reports 61 benchmark tests passing with zero skips,
including all four source-dependent adversarial checks. The independent
raw-record audit reproduced 132 score and paired-difference summaries and
all trial identities, denominators, family counts, and failure counts.

Every comparator had 32 finite held-out scores in every cell. The six score
columns and five selected-minus-reference contrasts therefore have complete
per-cell denominators. The independent basis-matched micro implementation
matched the selected macro exactly in all 384 trials (maximum absolute R²
difference 0). This equality is expected by design; it verifies basis and
capacity equivalence, not independent evidence for the macro. Held-out target
coverage was 100% for all cells. Entity-holdout cells used 540 development
and 180 held-out transition rows with 36 and 12 entities respectively and no
entity overlap. Forward-time cells used 480 development and 192 future rows
over the same 48 entities; this evaluates predictors observed at each future
time, not recursive fixed-origin forecasts.

The table reports the raw-recomputed selected-macro minus restricted-linear-
micro mean, Monte Carlo standard error, protocol two-sided nominal 95% t interval (the df=31 critical value uses its stated large-df approximation),
and selected family counts. The interval is conditional on these 32
successful panels and is descriptive for the frozen simulation cell; it is
not a calibrated search rejection rule, simultaneous interval, or emergence
test. The complete summary also reports replicate-bootstrap intervals and
positive-sign fractions.

| Cell | n | Mean ΔR² | MC SE | 95% t interval | Selected family counts |
|---|---:|---:|---:|---:|---|
| iid null / entity holdout | 32 | 0.001729 | 0.002097 | [-0.002548, 0.006006] | kmeans 6, q2 3, q3 23 |
| additive signal / entity holdout | 32 | -0.288047 | 0.008472 | [-0.305325, -0.270769] | q3 32 |
| nonlinear threshold / entity holdout | 32 | 0.135579 | 0.009610 | [0.115980, 0.155178] | q2 32 |
| sign interaction / entity holdout | 32 | 0.037631 | 0.005977 | [0.025442, 0.049820] | kmeans 14, q3 18 |
| entity fingerprint / entity holdout | 32 | 1.471942 | 0.440932 | [0.572695, 2.371190] | kmeans 5, q2 3, q3 24 |
| entity fingerprint / interpolation | 32 | -0.757992 | 0.009369 | [-0.777100, -0.738885] | kmeans 9, q2 9, q3 14 |
| temporal drift / forward time | 32 | 0.432987 | 0.022077 | [0.387963, 0.478011] | q3 32 |
| temporal drift / interpolation | 32 | -0.158417 | 0.015576 | [-0.190184, -0.126651] | kmeans 2, q2 2, q3 28 |
| rare absent states / entity holdout | 32 | -0.048168 | 0.005623 | [-0.059635, -0.036700] | q2 3, q3 29 |
| known confounding / entity holdout | 32 | -0.058891 | 0.001499 | [-0.061949, -0.055834] | q3 32 |
| correlated treatments / entity holdout | 32 | -0.028400 | 0.001668 | [-0.031802, -0.024999] | q3 32 |
| population shift / forward time | 32 | -0.632437 | 0.015718 | [-0.664492, -0.600381] | q3 32 |

The null-cell mean contrast is close to zero (0.0017; 16 of 32 replicate
differences positive; t interval [-0.0025, 0.0060]). This is a finite
descriptive null simulation only. With 32 replicates and no prespecified
calibrated rejection rule, it does not estimate the full-search empirical
false-positive rate. The nonlinear-threshold cell favors the selected macro
over the restricted linear micro reference (mean difference 0.1356), whereas
the additive and several shift/confounding cells favor the restricted micro.
For sign interaction, selected macro exceeds restricted micro by 0.0376 but
performs well below the prespecified mechanism label (mean R² 0.0240 versus
0.6853) and flexible micro (0.2664). This is evidence that these search
settings do not uniformly recover the designed representation.

The fingerprint/entity-holdout contrast must not be interpreted as a
generalization advantage. Twelve new entity IDs are unseen in every replicate.
The integrated categorical encoder drops a reference level and maps unknown
IDs to that level's contribution. In this generator, entity effects are
arbitrary random intercepts, so the reference effect is not a valid new-entity
prediction. This creates very poor restricted-micro R² in many trials and an
apparently favorable macro-minus-micro contrast, even though the macro R² is
still negative on average (-0.0865). The result diagnoses unsupported
categorical extrapolation, not a discovered or meaningful identity-level
representation. Interpolation has no unseen entities and instead favors the
restricted micro (ΔR² -0.7580).

The rare-state audit confirms the designed support limitation: the
generator-defined rare entity state (one held-out entity, 15 rows) and rare
temporal-tail state (one held-out transition row) are absent from development
for all 32 replicates. The selected macro's own fitted states met the
study-specific 20-row/5-entity support diagnostic in all runs, but that does
not mean the generator-defined rare states were learned or recovered. No
trial was silently excluded for either condition.

The development-only native adjustment checks reproduced independent joint
OLS for all 96 treatment-coefficient terms across 64 adjustment-control runs. Across replicates, the known-confounding
raw-U treatment coefficient averaged 1.5050 (constructed value 1.5). In the
correlated-treatment cell, native joint coefficients averaged 1.7297 for A
(constructed 1.7) and -0.9314 for B (constructed -0.9). These are software
and estimand checks under the generated equations, not causal identification
inferences. The prespecified raw-U entity bootstrap contained the constructed
known-confounding coefficient in 31/32 panels; its Wilson 95% interval for
this descriptive coverage fraction is approximately [0.843, 0.994]. This
low-precision coverage fraction is not nominal-coverage validation.

The main limitations remain: one panel size/noise setting, a three-state
candidate cap, 32 replicates per cell, same seed families across cells, and no
calibrated rejection rule. Positive signs and intervals cannot be read as
emergence discoveries. No result identifies causal effects, tests Hoel or
CE2, supports empirical sensitivity/false-positive-rate claims, or
generalizes beyond these specified simulations. The raw data, complete
summary, numerical export, and independent numerical-audit file are retained
under `research/discovery-pipeline-benchmark/full` and
`research/discovery-pipeline-benchmark`.

## Interpretation boundary

The strongest permissible conclusion is about predictive performance of a
specified selected representation against specified comparators for the
declared held-out target population in these simulations. Broader method
claims require a larger preregistered calibration, stronger external
reproducibility, and independently justified estimands. Causal-emergence
claims require a separately defined intervention/transition estimand and
appropriate assumptions and calibration; this benchmark does not provide
them.
