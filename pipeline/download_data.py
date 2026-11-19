"""Download the raw materials databases used in this project.

Sources (all open, no API key):
  * Alexandria PBE 3D (ICAMS, Ruhr-Universitat Bochum), release 2025.07.02, CC-BY 4.0
  * GNoME stable materials summary (Google DeepMind, Apache-2.0)
  * JARVIS-DFT 3D (NIST)            -> downloaded by jarvis-tools in the enhancing notebooks
  * Matbench mp_is_metal / mp_gap   -> downloaded by matminer in the enhancing notebooks

Usage:  python pipeline/download_data.py [--alexandria-files 5]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

ALEXANDRIA_URL = "https://alexandria.icams.rub.de/data/pbe/2025.07.02/alexandria_{:05d}.json.bz2"
GNOME_URL = (
    "https://storage.googleapis.com/gdm_materials_discovery/gnome_data/"
    "stable_materials_summary.csv"
)


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "materials-ssl-student-project"})
    for attempt in range(1, 8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
                shutil.copyfileobj(r, f, length=1 << 20)
            break
        except OSError as exc:  # timeouts / resets on slow mirrors
            print(f"  [retry {attempt}] {exc}")
            time.sleep(10 * attempt)
    else:
        raise RuntimeError(f"could not download {url}")
    tmp.replace(dest)
    print(f"  [done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alexandria-files", type=int, default=5,
                    help="number of ~100k-entry Alexandria chunks to download")
    args = ap.parse_args()

    print("=== GNoME (unlabeled pool) ===")
    fetch(GNOME_URL, RAW / "gnome" / "stable_materials_summary.csv")

    print(f"=== Alexandria PBE 3D ({args.alexandria_files} chunks) ===")
    for i in range(args.alexandria_files):
        fetch(ALEXANDRIA_URL.format(i), RAW / "alexandria" / f"alexandria_{i:05d}.json.bz2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
