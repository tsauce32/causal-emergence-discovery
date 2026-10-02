# Independent scientific review: selection-aware Discovery benchmark

## Audit basis

I independently read the frozen full-profile records in `research/discovery-pipeline-benchmark/full/trials.jsonl` and compared them with the summary and manifest. The reportable run contains exactly 384 unique planned records (12 cells x 32 replicate seeds), all successful. There are no missing, duplicate, or out-of-profile records, failures, unsupported trials, or replacement seeds. The source was clean at pinned integrated commit `735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d`; the final DGP SHA-256 and manifest seeds match `full-profile-frozen-final.json`. The parent reports 61 benchmark tests passed with zero skips, including all four source-dependent adversarial checks.

For each of the 12 cells, I recomputed all six held-out R2 means and sample-standard-error-over-square-root-n values from successful raw trial records. All six comparators had n=32 in each cell. I independently recomputed the five selected-minus-reference differences per cell (60 paired summaries); all n, means, and MCSEs agree with the saved summary to numerical tolerance. The table of protocol two-sided nominal 95% t intervals (using the frozen df=31 large-df critical-value approximation) is descriptive for that cell's 32 replicate draws, conditional on usable scores, and is not multiplicity-adjusted. The full summary also reports replicate-bootstrap intervals and positive-sign fractions.

The independent basis-matched micro control had exactly the same held-out R2 as the selected macro in all 384 runs (maximum absolute difference 0). This is a useful representation/capacity reproduction check; it is expected by construction and is not independent evidence for the macro. The selected macro and every reported baseline used the same held-out transition rows and targets within each trial. Test target coverage was 100% throughout. Entity-holdout cells split 48 entities into 36 development and 12 held-out entities, with no overlap. Forward-time cells use 480 development and 192 later transitions from all 48 entities.

The independent check of the package adjustment output also passed: all 96 native treatment-coefficient terms across 64 adjustment-control runs in the known-confounding and correlated-treatment cells match an independent joint OLS fit (largest absolute coefficient discrepancy 9.4e-15). This verifies the declared joint conditional regression implementation on these generated development samples; it does not validate causal identification.

## Descriptive primary contrasts

The contrast is selected-macro held-out R2 minus restricted linear micro held-out R2.

| Cell | n | Mean delta R2 | MC SE | 95% t interval | Selected families |
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

The null-cell mean contrast is close to zero (0.0017; 16 of 32 replicate differences positive; t interval [-0.0025, 0.0060]). This is a finite descriptive null simulation only. With 32 replicates and no calibrated rejection rule, it does not estimate the full-search empirical false-positive rate. The nonlinear-threshold cell favors the selected macro over the restricted linear micro reference (mean difference 0.1356). Additive signal, rare absent states, both interpolation cells, known confounding, correlated treatments, and population shift favor the restricted micro. Forward-time temporal drift has a positive relative difference (0.433), although both mean R2 values are negative (-1.661 macro, -2.094 micro), so the selected macro is less poor under this specified shift, not a strong forecaster. In sign interaction, selected macro exceeds restricted micro by 0.0376 but is far below the prespecified mechanism label (mean R2 0.024 versus 0.685) and flexible micro (0.266); the search did not recover the known sign representation.

## Material caveats and permitted claims

The apparently large entity-fingerprint/entity-holdout contrast (delta R2 1.472) is an encoding extrapolation artifact, not evidence that the selected macro transfers identity information. The test set contains 12 unseen entity IDs in every replicate. The integrated categorical encoder drops a reference level and assigns an unseen ID the reference-category contribution. Here that category effect is an arbitrary entity-specific random intercept and is not a meaningful new-entity prediction. For example, in one replicate macro R2 was -0.286 while restricted micro R2 was -7.243; the baseline's reference-effect prediction is extremely poor. Mean macro R2 is itself negative (-0.086). This stress case shows how an unsupported unknown-category policy can dominate the macro-vs-micro contrast. Interpolation, with observed entities, favors the restricted micro by -0.758. Readers should not summarize the entity-holdout result as a Discovery advantage.

The raw support audit confirms that the generator-defined rare-entity state (one held-out entity with 15 rows) and rare-tail state (one held-out transition row) were absent from development in all 32 runs. The candidate macro's own fitted states met the diagnostic threshold of 20 rows and five entities in each run. That candidate-support diagnostic does not imply that the true rare states were supported, recovered, or identified; the distinction matters here.

The known-confounding development-sample raw-U treatment coefficient averaged 1.505 for its constructed value 1.5. The correlated-treatment joint coefficients averaged 1.730 and -0.931 for constructed values 1.7 and -0.9. These are implementation and estimand checks in the generated equations. They do not establish exchangeability, adjustment sufficiency, positivity, consistency, no interference, or real-data causal identification. The known-confounding raw-U entity bootstrap included the constructed coefficient in 31 of 32 panels (Wilson 95% interval for that fraction about [0.843, 0.994]); this small descriptive check is not nominal coverage validation.

The strongest supported statement is that this fixed selection-and-refit pipeline produced the above predictive comparisons on these specified panels, split rules, sample size, and candidate budget. Same seeds are reused across cells, and fingerprint/drift data panels are assessed under more than one split, so 384 runs are not 384 independent generated panels. The intervals are not simultaneous across cells. The study does not compute Hoel's causal emergence or CE2 and supports no causal-emergence, causal-identification, empirical false-positive-rate, power, sensitivity, coverage, or out-of-matrix generalization claim.

Detailed outputs: `NUMERICAL_RESULTS.md`, `full-results.csv`, `secondary-controls.json`, `independent-numerical-audit.json`, and raw `full/trials.jsonl` plus `full/summary.json` in this directory.
