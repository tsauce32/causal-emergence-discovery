"""CSV ingestion must preserve declared category tokens before split planning."""

import pandas as pd

from causal_emergence_discovery.discovery import DiscoveryConfig, discover_from_csv
from causal_emergence_discovery.panel import LaggedDataset, load_panel_csv
from causal_emergence_discovery.spec import StudySpec


def _panel():
    rows = []
    for entity in range(16):
        site = "01" if entity % 2 else "1"
        previous_y = float(entity) / 10
        for time in range(8):
            x = float(entity) + time / 10
            z = float(time % 3) - entity / 20
            rows.append({"entity": entity, "time": time, "site": site, "x": x, "z": z, "y": previous_y})
            previous_y = 1.2 * x - 0.4 * z + (entity % 3) / 10
    return pd.DataFrame(rows)


def _spec():
    return StudySpec.from_dict(
        {
            "dataset": {"id_column": "entity", "time_column": "time"},
            "outcomes": ["y"],
            "columns": {
                "x": {"role": "context", "type": "numeric"},
                # `nominal` exercises the shared categorical type normalization.
                "site": {"role": "context", "type": "nominal"},
                "z": {"role": "context", "type": "numeric"},
                "y": {"role": "outcome", "type": "numeric", "allowed_as_cause": False, "allowed_as_adjustment": False},
            },
        }
    )


def test_spec_aware_csv_read_preserves_category_tokens_and_legacy_inference(tmp_path):
    source = _panel()
    path = tmp_path / "tokens.csv"
    source.to_csv(path, index=False)

    legacy = load_panel_csv(str(path))
    declared = load_panel_csv(str(path), spec=_spec())

    assert pd.api.types.is_integer_dtype(legacy["site"].dtype)
    assert legacy["site"].astype(str).unique().tolist() == ["1"]
    assert str(declared["site"].dtype) == "string"
    assert set(declared["site"].dropna()) == {"1", "01"}
    assert declared["site"].isna().sum() == legacy["site"].isna().sum()


def test_declared_types_on_ignored_metadata_do_not_affect_csv_loading(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"entity": ["001", "002"], "time": ["2024-01", "2024-02"], "unused": ["a", "b"]}).to_csv(path, index=False)
    spec = StudySpec.from_dict(
        {
            "dataset": {"id_column": "entity", "time_column": "time"},
            "columns": {
                "entity": {"type": "identifier"},
                "time": {"type": "datetime"},
                "unused": {"role": "exclude", "type": "unsupported-metadata-type"},
            },
        }
    )
    loaded = load_panel_csv(str(path), spec=spec)
    assert loaded.entity.tolist() == [1, 2]


def test_lagged_dataset_keeps_legacy_positional_constructor():
    legacy = LaggedDataset(pd.DataFrame(), (), {}, 1, "entity", "time")
    assert legacy.target_time_column == "__lead_time"
    assert legacy.target_eligibility == {}


def test_heldout_string_does_not_change_training_category_or_discovery(tmp_path):
    source = _panel()
    spec = _spec()
    baseline_path = tmp_path / "baseline.csv"
    poisoned_path = tmp_path / "heldout-category.csv"
    spec_path = tmp_path / "study.json"
    source.to_csv(baseline_path, index=False)

    # This fixture matches the deterministic entity holdout for seed 19.
    poisoned = source.copy()
    poisoned.loc[(poisoned.entity == 2) & (poisoned.time == 0), "site"] = "heldout-unknown"
    poisoned.to_csv(poisoned_path, index=False)
    spec_path.write_text(__import__("json").dumps(spec.to_dict()), encoding="utf-8")

    base_loaded = load_panel_csv(str(baseline_path), spec=spec)
    changed_loaded = load_panel_csv(str(poisoned_path), spec=spec)
    train_entities = set(range(16)) - {2, 4, 8, 12}
    baseline_levels = set(base_loaded.loc[base_loaded.entity.isin(train_entities), "site"])
    changed_levels = set(changed_loaded.loc[changed_loaded.entity.isin(train_entities), "site"])
    assert baseline_levels == changed_levels == {"1", "01"}

    config = DiscoveryConfig(
        outcome="y", validation_mode="entity_holdout", folds=2,
        max_states=3, paths=1, branching_factor=1, seed=19,
    )
    baseline = discover_from_csv(baseline_path, spec_path, config)
    changed = discover_from_csv(poisoned_path, spec_path, config)
    assert baseline["searches"] == changed["searches"]
    assert baseline["top_macros"] == changed["top_macros"]
    assert baseline["feature_schema"]["site"]["kind"] == "categorical"
    assert changed["feature_schema"]["site"]["kind"] == "categorical"
    assert changed["feature_schema"]["site"]["unknown_outer_categories"] == 1
