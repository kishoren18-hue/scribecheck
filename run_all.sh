#!/usr/bin/env bash
# Reproduce every number and figure. ~50 min on 2 CPU cores (most of it is 04 with 300 repeats).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d data/raw/aci-bench ]; then
  git clone --depth 1 https://github.com/wyim/aci-bench ../aci-bench
  git clone --depth 1 https://github.com/abachaa/MTS-Dialog ../MTS-Dialog
  mkdir -p data/raw/mts-dialog data/raw/aci-bench
  cp -r ../aci-bench/data/challenge_data ../aci-bench/data/challenge_data_json ../aci-bench/data/src_experiment_data ../aci-bench/data/src_experiment_data_json data/raw/aci-bench/
  cp ../MTS-Dialog/Main-Dataset/*.csv ../MTS-Dialog/LICENSE.txt data/raw/mts-dialog/
fi
python3 scripts/01_build_benchmark.py      # ScribeErr: 11,996 corrupted notes with labeled errors
python3 scripts/02_featurize.py            # sentence-level evidence features
python3 scripts/03_train_detector.py       # detectors + per-type AUROC
python3 scripts/04_experiments.py 300      # conformal review guarantees under deployment shift
python3 scripts/10_polarity_features.py    # negation-aware evidence, SC+N / SC+RN auditors
python3 scripts/09_shift_aware.py 200 SC   # type-conditional gates, trust dial, shift alarm
python3 scripts/09_shift_aware.py 200 SC+RN
python3 scripts/05_real_scribes.py         # report card on 10 real note writers (ACI-Bench outputs)
python3 scripts/07_figures.py
# 06_adjudication_sheet.py regenerates the sheet that results/hardcheck_labels.csv was labeled on;
# only re-run it if you intend to re-label.
