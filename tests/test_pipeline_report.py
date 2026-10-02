"""Independent reporting audit must detect lost, duplicated, or misreported trials."""
from copy import deepcopy

import pytest

from benchmarks.pipeline.protocol import get_profile, generate_seeds, summarize_trials
from benchmarks.pipeline.report import audit_records, moments


def fixture():
    profile = get_profile("quick")
    profile["cells"] = [{"case": "iid_null", "split": "entity_holdout"}]
    seeds = generate_seeds(profile)
    records = [
        {"case": "iid_null", "split": "entity_holdout", "seed": seeds[0], "status": "success",
         "paired_r2": {"selected_macro": .2, "restricted_micro": -.1},
         "support": {"states": [{"train_rows": 10, "train_entities": 3}]}},
        {"case": "iid_null", "split": "entity_holdout", "seed": seeds[1], "status": "failed", "paired_r2": {}},
    ]
    manifest = {"protocol": profile, "seeds": seeds, "reportable": True, "source": {"head": "abc"}}
    summary = {"reportable": True, "source_verified_at_end": {"head": "abc"}, "summary": summarize_trials(records, profile)}
    return records, manifest, summary


def test_known_moments_and_retained_failure():
    assert moments([1, 3])["mean"] == 2
    assert moments([1, 3])["mc_se"] == 1
    result = audit_records(*fixture())
    assert result["observed_trials"] == 2 and result["failed_trials"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nonreportable", "source", "mean", "failure", "support"])
def test_audit_rejects_corrupt_or_development_evidence(mutation):
    records, manifest, summary = deepcopy(fixture())
    cell = summary["summary"]["cells"]["iid_null::entity_holdout"]
    if mutation == "duplicate":
        records.append(records[0])
    elif mutation == "missing":
        records.pop()
    elif mutation == "nonreportable":
        manifest["reportable"] = False
    elif mutation == "source":
        summary["source_verified_at_end"]["head"] = "different"
    elif mutation == "mean":
        cell["heldout_r2"]["selected_macro"]["mean"] = 99
    elif mutation == "failure":
        cell["failure_rate_planned"] = 0
    else:
        cell["support_threshold"]["n_below_threshold"] = 0
    with pytest.raises(ValueError):
        audit_records(records, manifest, summary)
