#!/usr/bin/env bash
# Regenerate result tables/macros from data/predictions and compile the report with Tectonic.
# Tectonic: https://tectonic-typesetting.github.io (single executable; set TECTONIC=path)
set -e
cd "$(dirname "$0")"
uv run --project .. python make_figures.py
uv run --project .. python make_results.py
"${TECTONIC:-tectonic}" -X compile main.tex --outdir .
echo "report/main.pdf ready"
