"""Attach to every measured-only score the number of molecules it actually rests on.

ChEMBL publishes what binds far more often than what does not, so for some proteins almost every
measured molecule is an active. NR3C1 is the clearest case here: 1,112 measured molecules, of
which **15** are inactive. A ROC-AUC over those molecules is a ranking metric with almost nothing
to rank against, and it duly comes out at 1.000 - a number that looks like a perfect model and
actually means "there was nothing to separate".

The response is not to quietly drop such targets, which would hide information the reader can
weigh perfectly well, but to make the denominator part of the claim:

* below MIN_MEASURED_MINORITY measured molecules of the rarer class, a ROC-AUC is noise and is
  withheld outright;
* at or above WELL_SUPPORTED_MINORITY the score is called well supported and is the only kind
  that enters the project-wide average;
* in between, the score is reported *with its denominator attached* and marked as thinly
  supported, so nobody quotes it as if it were measured on thousands of molecules.

`train_targets.py` applies these rules while training. This script applies the identical rules,
imported from the same module, to summaries written before the check existed - so every summary on
disk obeys one rule and a fresh training run reproduces exactly what is reported.

Usage:  uv run python pipeline/molecules/guard_target_metrics.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline" / "molecules"))
from train_targets import (  # noqa: E402  (one definition of the rules)
    MIN_MEASURED_MINORITY,
    WELL_SUPPORTED_MINORITY,
)

PROC = ROOT / "data" / "processed" / "targets"
OUT = ROOT / "data" / "predictions" / "targets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing anything")
    args = ap.parse_args()

    files = sorted(OUT.glob("*_summary.json"))
    if not files:
        print(f"no target summaries in {OUT}")
        return
    changed, well, thin, withheld = 0, [], [], []
    for f in files:
        s = json.loads(f.read_text(encoding="utf-8"))
        pq = PROC / f"{s['chembl_id']}.parquet"
        if not pq.exists():
            print(f"{s['symbol']:<9} no activity data at {pq.name} - left alone")
            continue
        d = pd.read_parquet(pq, columns=["label", "decoy"])
        meas = d[~d["decoy"]]
        n_pos = int((meas["label"] == 1).sum())
        n_neg = int((meas["label"] == 0).sum())
        minority = min(n_pos, n_neg)

        before = json.dumps(s, sort_keys=True)
        s["dataset"]["n_measured_active"] = n_pos
        s["dataset"]["n_measured_inactive"] = n_neg
        mo = s.get("metrics_measured_only") or {}
        mo.setdefault("n", int(len(meas)))
        mo["n_measured_minority"] = minority
        mo["well_supported_threshold"] = WELL_SUPPORTED_MINORITY
        row = (s["symbol"], mo.get("ROC-AUC"), minority, n_pos, n_neg)
        if mo.get("ROC-AUC") is None or minority < MIN_MEASURED_MINORITY:
            for k in ("ROC-AUC", "PR-AUC", "Balanced_Accuracy"):
                mo[k] = None
            mo["reportable"] = False
            mo["well_supported"] = False
            mo["withheld_because"] = (
                f"this target has only {minority} measured molecules of its minority class; at "
                f"least {MIN_MEASURED_MINORITY} are needed before a ROC-AUC is anything but noise")
            withheld.append(row)
        else:
            mo["reportable"] = True
            mo["well_supported"] = minority >= WELL_SUPPORTED_MINORITY
            mo.pop("withheld_because", None)
            (well if mo["well_supported"] else thin).append(row)
        s["metrics_measured_only"] = mo
        if json.dumps(s, sort_keys=True) != before:
            changed += 1
            if not args.check:
                f.write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")

    demoted = sum(1 for r in withheld if r[1] is not None)
    print(f"{len(files)} summaries | {len(well)} well supported (>= "
          f"{WELL_SUPPORTED_MINORITY} measured minority) | {len(thin)} thinly supported "
          f"({MIN_MEASURED_MINORITY}-{WELL_SUPPORTED_MINORITY - 1}) | {len(withheld)} with no "
          f"usable measured-only score, {demoted} of which had one before | {changed} file(s) "
          f"{'would change' if args.check else 'updated'}")
    for label, group in (("well supported", well), ("thinly supported", thin),
                         ("no usable measured-only score", withheld)):
        if not group:
            continue
        print(f"\n{label}:")
        print(f"  {'protein':<10} {'AUC':>7}  {'minority':>8}  measured actives / inactives")
        for sym, auc, minority, n_pos, n_neg in sorted(group, key=lambda r: -r[2]):
            shown = f"{auc:.3f}" if auc is not None else "   --"
            print(f"  {sym:<10} {shown:>7}  {minority:>8}  {n_pos:,} / {n_neg:,}")


if __name__ == "__main__":
    main()
