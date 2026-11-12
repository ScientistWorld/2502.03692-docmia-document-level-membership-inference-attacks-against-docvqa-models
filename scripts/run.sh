#!/bin/bash
# Job 1 of the hardening iteration: plumbing smoke test + target-regime exploration.
set -uo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p tmp hardening/exploration

echo "[run] host $(hostname) nproc=$(nproc) python=$(python --version 2>&1)"
python -c "import torch, numpy, sklearn, PIL; print('[run] torch', torch.__version__, 'threads', torch.get_num_threads(), 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'PIL', PIL.__version__)"

python hardening/smoke.py 2>&1 | tee hardening/exploration/smoke.log
python hardening/explore.py P_es S_es P_long P_small 2>&1 | tee hardening/exploration/explore.log
echo "[run] done"
