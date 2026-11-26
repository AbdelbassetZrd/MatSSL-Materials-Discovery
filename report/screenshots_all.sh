#!/usr/bin/env bash
# Capture every page of MatSSL Studio, materials and molecules alike, for the project
# description folder. The server must already be running on :8000.
#
# Usage: bash report/screenshots_all.sh [out_dir] [width]
OUT="$(cygpath -aw "${1:-project description/screenshots}")"; W="${2:-1440}"
EDGE="${EDGE:-/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe}"
PROFILE="$(cygpath -aw "${TMPDIR:-/tmp}")\matssl-edge-profile"
BASE="http://127.0.0.1:8000/ui/#"
mkdir -p "$(cygpath -u "$OUT")"

# A molecule the illness models have a lot to say about: donepezil, the standard Alzheimer's
# drug, whose target ACHE is one of the best-measured proteins in the set.
DONEPEZIL="COc1cc2c(cc1OC)C(%3DO)C(CC1CCN(Cc3ccccc3)CC1)C2"
# Benzo[a]pyrene, which correctly fires the aryl-hydrocarbon receptor on the toxicity page.
BAP="c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"

shot () {  # name path height
  timeout 180 "$EDGE" --headless=new --disable-gpu --hide-scrollbars --window-size="$W,$3" \
    --user-data-dir="$PROFILE" --virtual-time-budget=12000 \
    --screenshot="$OUT/$1.png" "$BASE$2" >/dev/null 2>&1
  if [ -f "$(cygpath -u "$OUT")/$1.png" ]; then echo "  captured $1"; else echo "  FAILED  $1"; fi
}

echo "materials pages"
shot 01_overview            "/"                                        1500
shot 02_composition_studio  "/studio?formula=LiFePO4"                   1250
shot 03_crystal_viewer      "/studio?formula=Fe2O3"                     1250
shot 04_results             "/results?formula=LiFePO4"                  2300
shot 05_results_magnet      "/results?formula=Nd2Fe14B"                 2300
shot 06_compare             "/compare?f=LiFePO4,LiMnPO4,LiCoO2"         1100

echo "research lab"
shot 07_lab_datasets        "/lab?tab=datasets"                         1100
shot 08_lab_ssl             "/lab?tab=ssl"                              2300
shot 09_lab_models          "/lab?tab=models"                           1700
shot 10_lab_bandgap         "/lab?tab=bandgap"                          2000
shot 11_lab_unsupervised    "/lab?tab=unsupervised"                     1900
shot 12_lab_accuracy        "/lab?tab=accuracy"                         1900

echo "molecules, illnesses and design"
shot 13_molecule_studio     "/molecules"                                1700
shot 14_toxicity            "/molecule-results?smiles=$BAP"             2300
shot 15_illnesses           "/diseases"                                 1800
shot 16_screen              "/screen?smiles=$DONEPEZIL"                 2000
shot 17_design_lab          "/design"                                   2000

echo
echo "-> $OUT"
