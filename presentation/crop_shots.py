"""Crop the full-page app screenshots (report/figures/screenshots, 1440 px wide) into
slide-sized views for the deck. Run after report/screenshots.sh:
    uv run python presentation/crop_shots.py
"""

from pathlib import Path

from PIL import Image

SHOTS = Path(__file__).resolve().parents[1] / "report" / "figures" / "screenshots"
OUT = SHOTS / "deck"

# name: (source, top, bottom) in source pixels; full width is kept
CROPS = {
    "studio": ("02_editor.png", 70, 780),
    "results": ("05_analyze.png", 70, 1045),
    "explain": ("05_analyze.png", 1060, 1915),
    "lab_ssl": ("07_lab_ssl.png", 240, 925),
    "lab_models": ("08_lab_models.png", 240, 925),
    "compare": ("11_compare.png", 70, 770),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, top, bottom) in CROPS.items():
        f = SHOTS / src
        if not f.exists():
            print("missing", f.name)
            continue
        im = Image.open(f).convert("RGB")
        im.crop((0, top, im.width, min(bottom, im.height))).save(OUT / f"{name}.png")
        print("wrote", name)


if __name__ == "__main__":
    main()
