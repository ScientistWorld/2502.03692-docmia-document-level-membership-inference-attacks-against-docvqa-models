#!/bin/bash
# Hardening job 3: build the final gym assets, tune the reference on the dev
# audit, score the reference on train and on the hidden test, then replay the
# variants and shortcuts through the hidden test.  Resumable: every finished
# step leaves hardening/runs/.done_<step> and is skipped after a restart.
set -uo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
export OMP_NUM_THREADS=$DOCMIA_THREADS MKL_NUM_THREADS=$DOCMIA_THREADS OPENBLAS_NUM_THREADS=$DOCMIA_THREADS
RUNS=hardening/runs
mkdir -p $RUNS/reference tmp/pristine eval/test/runs

log() { echo "[run $(date +%H:%M:%S)] $*"; }
step() {
  local name=$1; shift
  if [ -f "$RUNS/.done_$name" ]; then log "skip $name"; return 0; fi
  log "start $name"
  if "$@"; then touch "$RUNS/.done_$name"; log "done $name"; else log "FAILED $name"; fi
}
# Files some shortcuts tamper with are restored from pristine copies (also after a restart).
restore() {
  for f in eval/test/audit_labels.json scoring/scores_train.json scoring/scores_test.json; do
    if [ -f "tmp/pristine/$(basename $f)" ]; then cp "tmp/pristine/$(basename $f)" "$f"; fi
  done
}

eval_train() {
  DOCMIA_KEEP_OUTPUT=$RUNS/reference/train_attack_output.json bash scripts/evaluate_train.sh &&
  cp scoring/scores_train.json $RUNS/reference/scores_train.json &&
  cp scoring/scores_train.json scoring/scores.json &&
  cp scoring/scores_train.json tmp/pristine/scores_train.json
}
eval_test() {
  DOCMIA_KEEP_OUTPUT=eval/test/runs/reference.json bash scripts/evaluate_test.sh &&
  cp scoring/scores_test.json $RUNS/reference/scores_test.json &&
  cp scoring/scores_test.json tmp/pristine/scores_test.json
}
variant() {
  python eval/test/evaluate.py --attack-cmd "python hardening/variants.py $1" \
    --scores $RUNS/$1/scores_test.json --keep-output eval/test/runs/$1.json
}
shortcut() {
  python eval/test/evaluate.py --attack-cmd "python hardening/shortcuts.py $1" \
    --scores $RUNS/shortcut_$1/scores_test.json
  local rc=$?
  restore
  return $rc
}
bootstrap() {
  local args=()
  for v in loss_threshold loss_dev_threshold gradient_ua calibrated_loss weaker_no_opt weaker_one_step calibrated_kmeans stronger_calibrated; do
    [ -f eval/test/runs/$v.json ] && args+=(--candidate "$v=eval/test/runs/$v.json")
  done
  python eval/test/paired_bootstrap.py --reference eval/test/runs/reference.json "${args[@]}" \
    --out hardening/paired_bootstrap_test.json
}

log "host $(hostname) nproc=$(nproc)"
restore
step build_public python data/build_public_assets.py
step build_test python eval/test/build_test_audit.py
[ -f tmp/pristine/audit_labels.json ] || cp eval/test/audit_labels.json tmp/pristine/audit_labels.json
step tune python method/tune_dev.py --cache-dir tmp/tune_cache
step eval_train eval_train
step eval_test eval_test
for v in loss_threshold weaker_no_opt loss_dev_threshold gradient_ua calibrated_loss; do step var_$v variant $v; done
for s in empty_output empty_results omit_methods constant_one manifest_labels filename_labels \
         bruteforce_generator move_ground_truth forged_scores threshold_override nan_scores \
         nonbinary_predictions read_hidden_labels; do
  step sc_$s shortcut $s
done
step var_weaker_one_step variant weaker_one_step
step var_calibrated_kmeans variant calibrated_kmeans
step var_stronger_calibrated variant stronger_calibrated
restore
step bootstrap bootstrap
python hardening/rank.py
python validate.py --compare > hardening/validate.log 2>&1; tail -40 hardening/validate.log
log "all steps attempted"
