#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u - <<'AUDIT_PY'
import json, subprocess
from pathlib import Path
for name in json.loads(Path('redteam/cases.json').read_text()):
    status=Path('redteam')/name/'status.json'
    if status.exists() and 'error' not in json.loads(status.read_text()):
        print('Already saved:',name,flush=True)
        continue
    print('Running:',name,flush=True)
    subprocess.run(['bash',str(Path('redteam')/name/'run.sh')],check=False)
AUDIT_PY
python redteam/rank.py
