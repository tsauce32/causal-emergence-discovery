# Frozen full-profile numerical results

Pinned integrated commit: `735c6a8fa1111e2ac53026ec3cbe2c23e5f5625d`.

All 384 planned records are retained. 0 trials failed or were unsupported.

The table averages held-out R² equally across successful generated panels; each panel's R² weights its evaluation rows equally. Intervals measure Monte Carlo uncertainty over independent complete-pipeline replicates, conditional on successful evaluation; they are not fold dispersion or emergence evidence. Complete-pipeline replicate bootstrap intervals, individual baselines, state support, and selection frequencies are also saved in the full summary and CSV.

| Case / split | Successful / planned (paired n) | Macro R² | Restricted micro R² | Flexible micro R² | Macro − restricted micro (95% MC interval) |
|---|---:|---:|---:|---:|---:|
| additive_signal / entity_holdout | 32 / 32 (32) | 0.4451 | 0.7331 | 0.7299 | -0.2880 [-0.3053, -0.2708] |
| correlated_treatments / entity_holdout | 32 / 32 (32) | 0.8037 | 0.8321 | 0.8300 | -0.0284 [-0.0318, -0.0250] |
| entity_fingerprint / entity_holdout | 32 / 32 (32) | -0.0865 | -1.5584 | -1.5558 | 1.4719 [0.5727, 2.3712] |
| entity_fingerprint / interpolation | 32 / 32 (32) | -0.0058 | 0.7522 | 0.7501 | -0.7580 [-0.7771, -0.7389] |
| iid_null / entity_holdout | 32 / 32 (32) | -0.0095 | -0.0112 | -0.0205 | 0.0017 [-0.0025, 0.0060] |
| known_confounding / entity_holdout | 32 / 32 (32) | 0.9081 | 0.9670 | 0.9667 | -0.0589 [-0.0619, -0.0558] |
| nonlinear_threshold / entity_holdout | 32 / 32 (32) | 0.4461 | 0.3105 | 0.3033 | 0.1356 [0.1160, 0.1552] |
| population_shift / forward_time | 32 / 32 (32) | 0.0439 | 0.6763 | 0.6677 | -0.6324 [-0.6645, -0.6004] |
| rare_absent_states / entity_holdout | 32 / 32 (32) | 0.1384 | 0.1866 | 0.1766 | -0.0482 [-0.0596, -0.0367] |
| sign_interaction / entity_holdout | 32 / 32 (32) | 0.0240 | -0.0136 | 0.2664 | 0.0376 [0.0254, 0.0498] |
| temporal_drift / forward_time | 32 / 32 (32) | -1.6611 | -2.0941 | -2.0937 | 0.4330 [0.3880, 0.4780] |
| temporal_drift / interpolation | 32 / 32 (32) | -0.0122 | 0.1462 | 0.1233 | -0.1584 [-0.1902, -0.1267] |

Read positive differences alongside absolute R² and the intercept-only control in the full summary. The drift case can favor the macro while both predictors perform poorly. For entity-fingerprint holdout, every reserved ID is unseen by the raw categorical model: zero contrasts receive the arbitrary training reference entity's effect. Its large relative gap reflects unsupported category extrapolation and does not establish useful transfer to new entities.

The matched-recipe micro control uses the selected recipe's state-indicator basis and the same treatment terms; numerical equivalence audits representation/model capacity, rather than supplying independent evidence of emergence. Formula-derived oracle controls are excluded from selection. No rejection rule was specified; positive differences are descriptive signs, not discoveries or an empirical false-positive rate.

Limitations: 32 replicates per cell, one panel size/noise setting and a candidate budget capped at three states. The designed scenarios do not establish performance outside this matrix, simultaneous confidence guarantees, real-data causal identification, or Hoel/CE2 emergence. Forward-time tasks condition on predictors observed at each future time; they do not evaluate recursive fixed-origin forecasts.
