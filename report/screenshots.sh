#!/usr/bin/env bash
# Capture the web-app screenshots used in the report/presentation (server must run on :8000).
# Usage: bash report/screenshots.sh [out_dir] [width]
OUT="$(cygpath -aw "${1:-report/figures/screenshots}")"; W="${2:-1440}"
EDGE="${EDGE:-/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe}"
PROFILE="$(cygpath -aw "${TMPDIR:-/tmp}")\matssl-edge-profile"
BASE="http://127.0.0.1:8000/ui/#"
mkdir -p "$(cygpath -u "$OUT")"
shot () {  # name path height
  timeout 150 "$EDGE" --headless=new --disable-gpu --hide-scrollbars --window-size="$W,$3" \
    --user-data-dir="$PROFILE" --virtual-time-budget=8000 --screenshot="$OUT/$1.png" "$BASE$2" >/dev/null 2>&1
  echo "captured $1"
}
shot 01_start "/" 1500
shot 02_editor "/studio?formula=LiFePO4" 1250
shot 03_input "/studio?formula=CsPbI3" 1250
shot 04_viewer "/studio?formula=Fe2O3" 1250
shot 05_analyze "/results?formula=Nd2Fe14B" 2300
shot 06_results "/results?formula=LiFePO4" 2300
shot 07_lab_ssl "/lab?tab=ssl" 2300
shot 08_lab_models "/lab?tab=models" 1700
shot 09_lab_bandgap "/lab?tab=bandgap" 2000
shot 10_lab_unsupervised "/lab?tab=unsupervised" 1900
shot 11_compare "/compare?f=LiFePO4,LiMnPO4,LiCoO2" 1100
shot 12_lab_datasets "/lab?tab=datasets" 1100
