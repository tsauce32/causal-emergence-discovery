# Selection-aware Discovery benchmark

This harness evaluates the full candidate-generation, merge-search, ranking,
recipe-refit, and one-time outer evaluation process on fresh generated panels.
Each JSONL row is one independently seeded dataset and one frozen case/split
cell. A failed or unsupported pipeline run stays in the denominator; seeds are
never replaced. The protocol, seeds, source commit, runtime versions, and
SHA-256 fingerprints of the harness/DGP/protocol are written to `manifest.json`.

Reported full results require an integrated, ready source checkout, a matching
`--expected-commit`, clean Discovery source files at the beginning and end, and
a fresh output directory. The reserved cohort and its rows are used only for
the selected recipe's single evaluation. Candidate generation, score-based
selection, the macro encoder, and all baseline transformations are fit on
development/training rows. Outer outcomes do not change the selected recipe.
Macro and micro metrics use the same held-out rows and finite targets.

The result compares the selected macro to an intercept-only reference, raw
additive micro regression, a fixed quadratic numeric/additive categorical
micro model, a generator-prespecified macro where available, and a separately
implemented micro regression on the selected recipe's frozen state-indicator
basis. The matched-basis model has the same information by construction; its
purpose is to verify that any gap against a restricted raw-micro class is a
model-class difference. Generator-known nonlinear oracle features are recorded
separately and labeled as oracle references; they are excluded from Discovery
selection. The benchmark does not identify causal emergence.

The independent regressions use training-only reference-category contrasts and
column-scaled SVD. An unseen categorical value maps to zero contrasts, hence to
the fitted reference contribution. This matches the integrated package's
extrapolation rule and is recorded as a limitation for unseen entity IDs.

For `known_confounding` and `correlated_treatments`, the trial also records
independent marginal, joint raw-confounder, and coarse-macro adjusted OLS
diagnostics. The known-confounder raw-U coefficient receives a deterministic
entity bootstrap interval under that prespecified adjustment model. Its
coverage is conditional on this generated design and is not evidence about
adjustment sufficiency in observational data.

## Commands

From the benchmark project root, after integration readiness is recorded:

```powershell
python -m benchmarks.pipeline.runner `
  --profile full `
  --source ..\discovery-integrated `
  --expected-commit <ready-integrated-commit> `
  --workers 4 `
  --output ..\..\research\discovery-pipeline-benchmark\full-run
```

For a development-only smoke run against the reviewed validation base:

```powershell
python -m benchmarks.pipeline.runner `
  --profile quick `
  --source . `
  --expected-commit 5d6ab966deba6c705f4a869006b19b73736c9824 `
  --output ..\..\research\discovery-pipeline-benchmark\development-smoke `
  --development-smoke
```

Development smoke output is marked non-reportable. `trials.jsonl` is
checkpointed after every completed trial, including in multi-process runs.
Records carry a stable planned index and the final summary is sorted by that
index, independent of worker completion order. The worker count is recorded as
runtime provenance and does not change the frozen protocol. `summary.json`
aggregates paired outer R²
differences, replicate-level Monte Carlo uncertainty, selection frequencies,
support, and failures. Fold-to-fold standard deviations are not treated as
standard errors. The report contains no emergence decision flag.

## Reproduce from a GitHub clone

The recorded experiment used integrated source commit
`735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d`, which is an ancestor of this
publication branch. Keep that source in a separate worktree so the runner can
verify its exact commit while loading the benchmark from the publication
checkout. Python 3.12 and the NumPy/Pandas versions in
`research_outputs/pipeline/experiment-manifest.json` were used for the recorded
run. The following commands run from the publication checkout:

```sh
python -m pip install -e ".[dev]" jsonschema
git worktree add --detach ../discovery-integrated 735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d
python -m benchmarks.pipeline.runner --profile full --workers 4 --source ../discovery-integrated --expected-commit 735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d --readiness-status research_outputs/pipeline/integrated-source-readiness.json --output ../benchmark-results-full
python -m benchmarks.pipeline.report --run-dir ../benchmark-results-full --output ../benchmark-report --frozen research_outputs/pipeline/full-profile-frozen-final.json
```

Use fresh output directories. The readiness snapshot's historical workspace
paths are provenance; the runner verifies the source path supplied on the
command line against its recorded commit. Full numerical results, the frozen
profile and seeds, generator amendment, independent audits, and source/compact
artifact hashes are committed under `research_outputs/pipeline`. The original
384-run study is reported against its pinned source rather than relabeled as a
new experiment on the publication commit.
