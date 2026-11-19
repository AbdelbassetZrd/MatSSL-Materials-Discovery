"""Choose the deployed model of every task and write app/models/registry.json.

For each SSL task the two candidate deployment models (unlabeled pool = same database, or
GNoME) were trained by run_ssl_experiments.py with the architecture selected by the
composite score. The one with the best test composite score is deployed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS, PRED = ROOT / "app" / "models", ROOT / "data" / "predictions"


def main() -> None:
    registry = {}
    for task in ("metallicity", "magnetism", "stability"):
        cands = []
        for pool in ("indist", "gnome"):
            s = PRED / task / pool / f"ssl_summary_{task}.json"
            m = MODELS / f"{task}_{pool}_model.pkl"
            if s.exists() and m.exists():
                S = json.loads(s.read_text())
                dep = S["deployment_model_test_metrics"]
                cands.append((dep["Composite_Score"], pool, S, dep))
        if not cands:
            print(f"{task}: no candidate model yet")
            continue
        _, pool, S, dep = max(cands, key=lambda c: c[0])
        shutil.copy(MODELS / f"{task}_{pool}_model.pkl", MODELS / f"{task}_model.pkl")
        shutil.copy(MODELS / f"{task}_{pool}_scaler.pkl", MODELS / f"{task}_scaler.pkl")
        registry[task] = {
            "model_file": f"{task}_model.pkl", "scaler_file": f"{task}_scaler.pkl",
            "method": S["best_ssl_model"]["Method"],
            "config": S["best_ssl_model"]["Model_Config"].split(" | ")[0],
            "selection_pool": pool, "test_metrics": dep,
            "candidates": {c[1]: c[0] for c in cands},
        }
        print(f"{task}: deployed {pool} model ({registry[task]['config']}) "
              f"test composite {dep['Composite_Score']:.4f}")
    bj = PRED / "band_gap" / "band_gap_summary.json"
    if bj.exists():
        B = json.loads(bj.read_text())
        registry["band_gap"] = {"model_file": "band_gap_model.pkl",
                                "scaler_file": "band_gap_scaler.pkl",
                                "method": "GA-selected descriptors + "
                                          f"{B['model_comparison'][0]['Model']}",
                                "config": f"{B['ga']['final_n_features']} GA descriptors",
                                "test_metrics": B["test_ga_descriptors"]}
    (MODELS / "registry.json").write_text(json.dumps(registry, indent=2, default=float))
    for f in MODELS.glob("*.pkl"):
        print(f"  {f.name:<45} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
