#!/bin/bash
set -euo pipefail
cd /home/user
bash redteam/stronger_train_selected/run.sh
bash redteam/shortcut_empty_results/run.sh
python redteam/rank.py
