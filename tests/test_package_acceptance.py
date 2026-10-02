"""Build artifact installation smoke checks for release acceptance.

Run through ``packaging/run_acceptance.ps1`` so the wheel and sdist are first
built into a workspace-local artifact directory. Every import and CLI command
below runs from a temporary working directory with the installed target on
PYTHONPATH, which prevents the checkout's ``src`` tree from shadowing it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("CED_PACKAGE_ACCEPTANCE_ARTIFACT_DIR", REPOSITORY / "packaging" / "artifacts" / "dist"))
PYTHON = os.environ.get("CED_PYTHON_EXE", sys.executable)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, (
        f"Command failed ({completed.returncode}): {command!r}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def _installed_env(target: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    env.pop("PYTHONHOME", None)
    return env


def _assert_result(result: dict[str, object], installed_root: Path | None = None) -> None:
    assert result["library"] == "causal-emergence-discovery"
    assert result["schema_version"] == 2
    candidates = result["top_macros"]
    assert candidates
    best = candidates[0]
    assert "ranking_score" in best
    assert "predictive_r2_difference" in best
    assert best["emergence_evidence_status"] == "not_assessed"
    assert "emergence_delta" not in best
    assert best["evaluation_scope"] == "development_selection_only"

    evaluation = result["outer_evaluation"]
    assert evaluation["scope"] == "untouched_outer_holdout_selected_macro"
    assert "macro_r2" in evaluation and "micro_r2" in evaluation
    assert evaluation["predictive_r2_difference"] == pytest.approx(
        evaluation["macro_r2"] - evaluation["micro_r2"]
    )
    assert result["validation"]["outer"]["audit"]["test_rows"] == evaluation["test_rows"]

    if installed_root is not None:
        import_line = (
            "import causal_emergence_discovery as ced; "
            "from pathlib import Path; "
            "print(Path(ced.__file__).resolve())"
        )
        completed = _run([PYTHON, "-c", import_line], cwd=installed_root.parent, env=_installed_env(installed_root))
        imported_path = Path(completed.stdout.strip()).resolve()
        assert imported_path.is_relative_to(installed_root.resolve()), imported_path


def _fixture_files(directory: Path) -> tuple[Path, Path]:
    panel_path = directory / "panel.csv"
    spec_path = directory / "study.json"
    rng = __import__("random").Random(426)
    with panel_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("entity_id,time,x,z,outcome\n")
        for entity in range(40):
            level = rng.gauss(0, 1)
            prior = level
            for time in range(5):
                x = rng.gauss(level, 0.5)
                z = rng.gauss(0, 1)
                outcome = 0.65 * x + 0.2 * z + 0.3 * prior + rng.gauss(0, 0.25)
                stream.write(f"{entity},{time},{x},{z},{outcome}\n")
                prior = outcome
    spec_path.write_text(json.dumps({
        "dataset": {"id_column": "entity_id", "time_column": "time"},
        "outcomes": ["outcome"],
        "columns": {
            "x": {"role": "context", "type": "numeric"},
            "z": {"role": "context", "type": "numeric"},
            "outcome": {
                "role": "outcome", "type": "numeric",
                "allowed_as_cause": False, "allowed_as_adjustment": False,
            },
        },
    }), encoding="utf-8")
    return panel_path, spec_path


def _install_artifact(artifact: Path, target: Path, *, cwd: Path, env: dict[str, str]) -> None:
    _run([
        PYTHON, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
        "--target", str(target), str(artifact),
    ], cwd=cwd, env=env)


def test_wheel_public_api_and_console_command_use_installed_package(tmp_path: Path) -> None:
    wheels = list(ARTIFACTS.glob("causal_emergence_discovery-0.2.0-*.whl"))
    assert len(wheels) == 1, f"Expected one 0.2.0 wheel in {ARTIFACTS}; found {wheels}"
    install_root = tmp_path / "wheel-install"
    install_root.mkdir()
    _install_artifact(wheels[0], install_root, cwd=tmp_path, env=os.environ.copy())
    env = _installed_env(install_root)

    api_smoke = r'''
import importlib.metadata
import json
import random
import pandas as pd
from pathlib import Path
from causal_emergence_discovery import DiscoveryConfig, StudySpec, run_discovery
import causal_emergence_discovery as ced
assert importlib.metadata.version("causal-emergence-discovery") == "0.2.0"
assert Path(ced.__file__).resolve().is_relative_to(Path(__import__("sys").argv[1]).resolve())
rng = random.Random(426)
rows = []
for entity in range(40):
    level = rng.gauss(0, 1)
    prior = level
    for time in range(5):
        x, z = rng.gauss(level, .5), rng.gauss(0, 1)
        y = .65*x + .2*z + .3*prior + rng.gauss(0, .25)
        rows.append({"entity_id": entity, "time": time, "x": x, "z": z, "outcome": y})
        prior = y
spec = StudySpec.from_dict({"dataset":{"id_column":"entity_id","time_column":"time"},"outcomes":["outcome"],"columns":{"x":{"role":"context","type":"numeric"},"z":{"role":"context","type":"numeric"},"outcome":{"role":"outcome","type":"numeric","allowed_as_cause":False,"allowed_as_adjustment":False}}})
result = run_discovery(pd.DataFrame(rows), spec, DiscoveryConfig(outcome="outcome", max_states=3, paths=2, branching_factor=2, folds=2, top_k=2, seed=7))
print(json.dumps(result))
'''
    completed = _run([PYTHON, "-c", api_smoke, str(install_root)], cwd=tmp_path, env=env)
    _assert_result(json.loads(completed.stdout.strip().splitlines()[-1]), install_root)

    panel, spec = _fixture_files(tmp_path)
    scripts = install_root / "Scripts"
    cli = scripts / "ced.exe"
    assert cli.exists(), f"Installed console entry point missing: {cli}"
    cli_result = _run([
        str(cli), "discover", str(panel), str(spec), "--outcome", "outcome",
        "--max-states", "3", "--paths", "2", "--folds", "2", "--top-k", "2", "--json",
    ], cwd=tmp_path, env=env)
    _assert_result(json.loads(cli_result.stdout))


def test_sdist_module_command_uses_installed_package(tmp_path: Path) -> None:
    sdists = list(ARTIFACTS.glob("causal_emergence_discovery-0.2.0.tar.gz"))
    assert len(sdists) == 1, f"Expected one 0.2.0 sdist in {ARTIFACTS}; found {sdists}"
    install_root = tmp_path / "sdist-install"
    install_root.mkdir()
    _install_artifact(sdists[0], install_root, cwd=tmp_path, env=os.environ.copy())
    panel, spec = _fixture_files(tmp_path)
    result = _run([
        PYTHON, "-m", "causal_emergence_discovery", "discover", str(panel), str(spec),
        "--outcome", "outcome", "--max-states", "3", "--paths", "2", "--folds", "2",
        "--top-k", "2", "--json",
    ], cwd=tmp_path, env=_installed_env(install_root))
    _assert_result(json.loads(result.stdout), install_root)
