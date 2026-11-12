#!/bin/bash
# Job 1 of the hardening iteration: plumbing smoke test + target-regime exploration.
# Resumable: finished regimes (hardening/exploration/<regime>.json) are skipped.
set -uo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
export OMP_NUM_THREADS=$DOCMIA_THREADS MKL_NUM_THREADS=$DOCMIA_THREADS OPENBLAS_NUM_THREADS=$DOCMIA_THREADS
mkdir -p tmp hardening/exploration

echo "[run] host $(hostname) nproc=$(nproc) cgroup=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null) python=$(python --version 2>&1)"
python -c "import torch, numpy, sklearn, PIL; print('[run] torch', torch.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'PIL', PIL.__version__)"

if [ ! -f hardening/exploration/smoke.done ]; then
  python hardening/smoke.py 2>&1 | tee hardening/exploration/smoke.log && touch hardening/exploration/smoke.done
fi
python hardening/explore.py P_es P_weak P_long S_es 2>&1 | tee -a hardening/exploration/explore.log
echo "[run] done"
