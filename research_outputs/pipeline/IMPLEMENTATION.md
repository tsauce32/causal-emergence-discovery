# Selection-aware Discovery pipeline benchmark

Completed on October 2, 2026: **384/384 full-profile runs succeeded**, all **61
benchmark tests passed**, and the independent numerical audit passed **132
summary comparisons**. The integrated source stayed pinned to
`735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d`. The fitted harness was committed as
`8473ec011ba76940f8e2da170f5043dc5991f1ef` before execution; later changes add
report interpretation and compact evidence. The verified quick profile passed
24/24 runs with four native coefficient checks. Historical development and
failed quick-run artifacts are retained and are not substituted for full results.

The benchmark reruns candidate construction, merge search, inner-fold ranking,
selection, training-only recipe refits, and one reserved outer evaluation for
each generated panel. Its comparisons concern held-out prediction and synthetic
software controls. No emergence rejection rule was specified, so every result
retains `emergence_evidence_status = not_assessed`.

## Frozen experiment

The full profile contains 32 independent generated panels per case/split cell,
12 cells, and 384 complete pipeline runs. Panels contain 48 entities and 16 raw
periods, with lag one, three inner folds, a 25% outer holdout, split seed 17,
three maximum states, two search paths, and branching two. The seed family is
20261002. The same panel seeds are reused across cells; fingerprint and drift
panels are each evaluated under two split modes. Independence applies within a
cell, rather than to all 384 records jointly.

The ten generators cover an IID null, additive signal, a nonlinear threshold,
sign interaction, arbitrary entity fingerprints, temporal drift, rare and
training-absent states, known confounding, conditionally correlated treatments,
and population shift. Entity holdout measures transfer to reserved entities.
Forward-time evaluation conditions on predictors observed at each later time;
it is a rolling one-step task, not a recursive fixed-origin forecast.
Interpolation is explicitly labeled and evaluates seen entities.

The original freeze is preserved. A documented amendment before final outer
results added a treatment-only latent shared shock to the correlated-treatment
generator. Correlation now remains after adjustment for the measured U, while
the joint conditional outcome coefficients remain 1.7 and -0.9. The final DGP
SHA-256 is `df52f739186d9c52f609846a3fd7ad504f8b693a017c8e799790372e451ea518`.
The protocol SHA-256 is
`5edef37fb1f8dca8ed81e16ea8d832d931911d9d6157e1e280d81110e8637257`.

## Comparisons and uncertainty

All evaluable comparators share the selected macro's outer cohort and finite
targets. Signed raw R-squared values and paired differences are retained without
clipping. Comparators include an intercept-only model, restricted additive raw
micro regression, fixed quadratic numeric/additive categorical micro
regression, a generator-prespecified macro, and a separately implemented micro
model on the selected recipe's indicator basis. The matched-basis model checks
representation and capacity equivalence. Formula-derived oracle controls are
excluded from Discovery's search and recorded separately.

Independent regressions use training-only reference-category contrasts and
column-scaled SVD. Unknown categorical values have zero contrasts and receive
the fitted reference contribution. This convention matters for unseen entity
IDs: the reference entity's arbitrary effect is an extrapolation convention,
not information about the reserved entities. The first integrated quick run
preserved two score-reproduction failures that exposed the previous independent
full-indicator convention. The comparator was repaired and checked before any
full-profile outer results were inspected; no generator, seed, or search-budget
setting changed in that repair.

Means weight complete generated panels equally. Monte Carlo standard errors,
approximate Student-t intervals, and 1,999-draw percentile bootstrap intervals
use independent complete-pipeline replicates within each cell. Fold dispersion
is not a confidence interval. Descriptive proportions retain planned
denominators and Wilson intervals. Failed or unsupported runs are retained;
there are no replacement seeds or repeated outer selection attempts. No
multiple-comparison or false-positive calibration claim is made.

Each run records the selected recipe and family, ranking score, compression,
training and outer state support, candidate refit rejections, fold topology,
development topology, encoder, and the package's native adjustment output.
Support thresholds (20 training rows and five entities) are stress diagnostics,
not a universal validity rule. Raw rare-state absence is recorded even if the
selected macro does not preserve that rare state.

Independent NumPy OLS reproduces package outer macro and micro scores. In both
adjustment controls, the package's native joint-conditional treatment
coefficients must match an independent joint fit using the measured U. The
known-confounder coefficient also receives a deterministic entity bootstrap
under that prespecified raw-U model. This interval does not include Discovery
selection uncertainty and does not establish adjustment sufficiency in real
observational data.

## Reproduction and evidence

`reproduce.ps1` requires the integrated source's readiness manifest, verifies
its pinned commit, and accepts explicit Python, Git, dependency, source, and
output paths. Output directories must be fresh. `runner.py` checkpoints each
finished replicate with a stable planned index and verifies source cleanliness,
commit identity, and harness byte fingerprints at both ends. Worker count is
runtime provenance and does not change the frozen design.

`report.py` independently recomputes raw means, paired differences, Monte Carlo
standard errors, seed identities, failure denominators, support rates, and
selection frequencies from the raw JSONL. It verifies the frozen protocol,
seed list, and DGP fingerprint before exporting the table and secondary
controls. The raw run directory retains the manifest, every trial, and the full
summary. The handoff includes a portable patch and preservation evidence for
the reviewed validation and scoring checkouts.

The source-dependent adversarial tests check reserved-label and dtype isolation
and topology rejection. Other tests check generators, baseline fitting,
protocol denominators, source gates, native coefficient reproduction, and
corruption detection in the independent numerical audit.

This is a bounded matrix: 32 replicates per cell, one panel size and noise
setting, and a search budget capped at three states. It does not meet the
proposed 1,000-null calibration gate, evaluate CE2/PyMergence, establish real
observational causal identification, or support a general emergence claim.

## Findings and practical limits

The full [numerical table](NUMERICAL_RESULTS.md)
and [CSV](full-results.csv) retain all 12 cells.
The nonlinear-threshold macro achieved mean held-out R-squared 0.4461 versus
0.3105 for restricted additive micro regression: paired difference 0.1356,
95% Monte Carlo interval [0.1160, 0.1552]. Its generator-known oracle reached
0.4860. Sign interaction improved over additive micro by 0.0376, but the macro
scored only 0.0240 versus 0.2664 for the fixed quadratic micro comparator and
0.6853 for the generator-known sign oracle. The three-state search budget does
not guarantee recovery of a four-quadrant sign representation.

The matched recipe-basis micro score equaled the selected macro in all 384
runs (maximum absolute difference zero). These results support a predictive
model-class/capacity comparison under this matrix. They do not show additional
information created by a coarse graining or identify causal emergence.

The IID null selected macros with mean heuristic ranking score 0.2890 and
compression 0.8898, while mean macro held-out R-squared was -0.0095. The macro
minus restricted-micro mean was 0.0017 with interval [-0.0025, 0.0060]; 16/32
differences were positive. These are descriptive signs without a rejection
rule, so they are not an empirical false-positive rate.

Two large relative gains were not useful prediction. Forward drift gave macro
R-squared -1.6611 versus micro -2.0941 and the intercept-only reference -0.0105.
Fingerprint entity holdout gave macro -0.0865 versus raw micro -1.5584, while
the intercept-only reference was -0.0794. The fingerprint gap is sensitive to
unsupported unseen categorical IDs receiving the arbitrary training reference
entity's effect. The same panel's seen-entity interpolation raw micro scored
0.7522; that result is not transfer to new entities.

Additive signal favored restricted micro (0.7331 versus macro 0.4451), and
population shift favored restricted micro (0.6763 versus macro 0.0439). Thus
the selected coarse representation did not supply a universal predictive gain.
All selected states passed the prespecified support diagnostic, but the rare
case still had a raw entity state absent from training (15 outer rows) and a
single-row tail state absent from training in every replicate. This benchmark
does not validate state-specific effects for those absent states. No candidate
refit rejection was observed in this finite matrix; topology rejection is
covered by the adversarial tests rather than claimed from these runs.

All 64 native package joint-conditional coefficient checks matched independent
OLS. Known confounding gave mean raw-U adjusted coefficient 1.5050 (truth 1.5),
while adjustment only for the selected coarse U macro gave 3.3337. Correlated
treatments gave joint means 1.7297 and -0.9314 (truths 1.7 and -0.9). These are
synthetic software controls. The known-U entity-bootstrap interval contained
the constructed coefficient in 31/32 replicates, Wilson interval
[0.8426, 0.9945]. At 32 replicates this is low-precision descriptive coverage
for a prespecified adjustment model, not calibrated post-selection coverage.

The independent [scientific review](scientific-review.md) accompanies these results. Preservation
checks confirm the reviewed validation HEAD and clean state, scoring HEAD and
staged state, exact reviewed staged-patch SHA-256, and original review/manifest
hashes all remain unchanged.

## Reproduce the completed experiment

From the dedicated checkout, select a fresh output directory:

```powershell
./benchmarks/pipeline/reproduce.ps1 -Profile full -Workers 4 `
  -PythonPath 'C:/Users/Owner/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' `
  -GitPath 'C:/Users/Owner/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe' `
  -DependenciesPath '../../research/review_dependencies' `
  -StatusPath '../../research/discovery-pipeline-benchmark/integrated-source-readiness.json' `
  -OutputPath '../../research/discovery-pipeline-benchmark/full-reproduction'
& 'C:/Users/Owner/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m benchmarks.pipeline.report `
  --run-dir '../../research/discovery-pipeline-benchmark/full-reproduction' `
  --output '../../research/discovery-pipeline-benchmark/reproduction-report' `
  --frozen '../../research/discovery-pipeline-benchmark/full-profile-frozen-final.json'
```

The report command requires the checkout and dependency directory on
`PYTHONPATH` and the same Python runtime. The original full raw manifest,
checkpointed trials, summary, final freeze, amendment, test logs, preservation
audit, numerical audit, and final status are under
`research/discovery-pipeline-benchmark` in the surrounding workspace. Compact
results and provenance are also included in this patch under
`research_outputs/pipeline`. The final status manifest identifies the portable
patch and final documentation/evidence commit.
