"""Generate a small synthetic longitudinal dataset for the README workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def build_synthetic_panel(entity_count: int = 240, periods: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(entity_count):
        site = "north" if entity % 2 == 0 else "south"
        latent_support = rng.normal()
        reading = 45 + 4 * latent_support + rng.normal(scale=2)
        for time in range(periods):
            stress = rng.normal(loc=-latent_support + 0.15 * time, scale=0.8)
            attendance = 0.85 + 0.05 * latent_support - 0.04 * stress + rng.normal(scale=0.03)
            program_hours = max(0.0, 2.5 + rng.normal(scale=0.7))
            rows.append(
                {
                    "entity_id": entity,
                    "time": time,
                    "site": site,
                    "stress": stress,
                    "attendance": attendance,
                    "program_hours": program_hours,
                    "reading_score": reading,
                }
            )
            reading = (
                reading
                + 1.2 * program_hours
                + 1.3 * attendance
                - 0.55 * stress
                + 0.25 * latent_support
                + rng.normal(scale=1.2)
            )
    return pd.DataFrame(rows)


def build_spec() -> dict:
    return {
        "dataset": {
            "id_column": "entity_id",
            "time_column": "time",
        },
        "outcomes": ["reading_score"],
        "interventions": ["program_hours"],
        "environments": ["site"],
        "columns": {
            "site": {"role": "environment", "type": "categorical"},
            "stress": {"role": "context", "type": "numeric"},
            "attendance": {"role": "measurement", "type": "numeric"},
            "program_hours": {"role": "intervention", "type": "numeric"},
            "reading_score": {"role": "outcome", "type": "numeric"},
        },
    }


def main() -> None:
    panel = build_synthetic_panel()
    panel.to_csv(ROOT / "synthetic_panel.generated.csv", index=False)
    (ROOT / "synthetic_study.generated.json").write_text(
        json.dumps(build_spec(), indent=2),
        encoding="utf-8",
    )
    print("Wrote examples/synthetic_panel.generated.csv")
    print("Wrote examples/synthetic_study.generated.json")


if __name__ == "__main__":
    main()
