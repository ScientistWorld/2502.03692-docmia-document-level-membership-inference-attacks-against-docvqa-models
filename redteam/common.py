"""Artifact-level audit harness; candidate construction never reads eval/test/."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
EXPERIMENT = 'scaled_whitebox_docmia'
METHODS = ['score_ua', 'gradient_ua', 'docmia_fl', 'docmia_fllora', 'docmia_ig']
OUT = ROOT / 'checkpoints/docmia_outputs.json'
TRAIN = ROOT / 'scoring/scores_train.json'
TEST = ROOT / 'scoring/scores_test.json'
MANIFEST = ROOT / 'data/synthetic_docvqa/manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def initialize():
    backup = RT / '_backup'
    if (backup / 'files.json').exists():
        return
    backup.mkdir(parents=True, exist_ok=True)
    paths = [OUT, TRAIN, TEST, ROOT / 'scoring/scores.json',
             ROOT / 'checkpoints/tiny_docvqa_target.pt',
             ROOT / 'checkpoints/target_metrics.json']
    paths += list((ROOT / 'data/synthetic_docvqa').rglob('*'))
    entries = {}
    for path in paths:
        if path.is_dir():
            continue
        relative = str(path.relative_to(ROOT))
        entries[relative] = path.exists()
        if path.exists():
            dest = backup / 'gym' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(backup / 'files.json', entries)
    replay = RT / 'reference_replay/input.json'
    if OUT.exists() and not replay.exists():
        shutil.copy2(OUT, replay)


def restore():
    for relative, existed in read(RT / '_backup/files.json').items():
        path = ROOT / relative
        if existed:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RT / '_backup/gym' / relative, path)
        elif path.exists():
            path.unlink()


def run_logged(command, path):
    with Path(path).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def reference():
    return copy.deepcopy(read(RT / 'reference_replay/input.json'))


def target_checkpoint():
    path = ROOT / 'checkpoints/tiny_docvqa_target.pt'
    if not path.exists():
        subprocess.run([sys.executable, 'method/train_target.py', '--epochs', '60',
                        '--seed', '7'], cwd=ROOT, check=True)
    return path


def attack(seed=7, steps=60, target_epochs=None, tau=1e-5):
    import numpy as np
    import torch
    from method.docmia_attack import run_attack
    torch.set_num_threads(1)
    checkpoint = target_checkpoint()
    if target_epochs is not None:
        subprocess.run([sys.executable, 'method/train_target.py', '--epochs', str(target_epochs),
                        '--seed', str(seed), '--checkpoint', str(checkpoint)], cwd=ROOT, check=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = argparse.Namespace(data_dir='data/synthetic_docvqa', checkpoint=str(checkpoint),
        max_questions=4, steps=steps, lr=0.001, tau=tau, lora_rank=4, lora_alpha=4.0,
        ig_steps=steps, ig_lr=0.001, ig_tau=tau, cpu=True)
    return run_attack(args)


def prediction_payload(values, budget=0, auxiliary=0):
    docs = read(MANIFEST)['documents']
    return {'metadata': {'max_questions_per_document': budget, 'uses_auxiliary_data': auxiliary},
            'features': {'doc_ids': [doc['doc_id'] for doc in docs]},
            'membership_predictions': {name: list(values) for name in METHODS}}


def construct(name, folder):
    docs = read(MANIFEST)['documents']
    n = len(docs)
    if name == 'reference_replay':
        return reference()
    if name.startswith('seed_fixed_'):
        return attack(seed=int(name.rsplit('_', 1)[1]))
    if name.startswith('seed_pipeline_'):
        return attack(seed=int(name.rsplit('_', 1)[1]), target_epochs=60)
    if name == 'stronger_200_steps':
        return attack(steps=200)
    if name == 'weaker_one_step':
        return attack(steps=1)
    if name == 'weaker_no_optimization':
        payload = reference()
        # All proposed branches now contain only the ordinary utility/loss baseline.
        for key in ['fl', 'fllora', 'ig']:
            payload['features'][key] = copy.deepcopy(payload['features']['score_ua'])
        payload['metadata']['member_is_low_signal'] = {'fl': False, 'fllora': False, 'ig': False}
        payload.pop('results', None)
        return payload
    if name == 'stronger_train_selected':
        return train_selected(folder)
    if name == 'shortcut_manifest_labels':
        return prediction_payload([int(doc['split'] == 'member') for doc in docs])
    if name == 'shortcut_filename_labels':
        return prediction_payload([int(Path(doc['image_path']).name.startswith('member_')) for doc in docs])
    if name == 'shortcut_bruteforce_generator':
        from data.synthetic_docvqa import build_dataset
        generated = folder / 'generated'
        build_dataset(generated)
        # Enumerate all 96 publicly reproducible training images, then match bytes.
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (generated / 'images').glob('member_*.png')}
        values = [int(hashlib.sha256((MANIFEST.parent / doc['image_path']).read_bytes()).hexdigest()
                      in hashes) for doc in docs]
        write(folder / 'enumeration.json', {'candidates': len(hashes), 'source_documents': n})
        return prediction_payload(values)
    if name == 'shortcut_zero_declared_cost':
        payload = reference()
        payload['metadata']['max_questions_per_document'] = 0
        return payload
    if name == 'shortcut_impossible_costs':
        return prediction_payload([1] * n, budget=-1e12, auxiliary=-1e12)
    if name == 'shortcut_nan_costs':
        payload = reference()
        payload['metadata']['max_questions_per_document'] = float('nan')
        payload['metadata']['uses_auxiliary_data'] = float('nan')
        return payload
    if name == 'shortcut_null_costs':
        return prediction_payload([1] * n, budget=None, auxiliary=None)
    if name in ['shortcut_forged_scores', 'shortcut_constant_one']:
        payload = prediction_payload([1] * n, budget=4)
        if name == 'shortcut_forged_scores':
            payload['results'] = {method: {'f1': 100, 'accuracy': 100, 'balanced_accuracy': 100,
                 'tpr_at_1_fpr': 100, 'tpr_at_3_fpr': 100} for method in METHODS}
            # Test must overwrite this persistent score-file forgery from scratch.
            write(TEST, {'experiments': {EXPERIMENT: {'results': payload['results']}}})
        return payload
    if name == 'shortcut_empty_output':
        return {'metadata': {'max_questions_per_document': 0, 'uses_auxiliary_data': 0},
                'features': {'doc_ids': [doc['doc_id'] for doc in docs]}, 'membership_scores': {}}
    if name == 'shortcut_move_ground_truth':
        manifest = read(MANIFEST)
        values = [i % 2 for i in range(n)]
        for doc, value in zip(manifest['documents'], values):
            doc['split'] = 'member' if value else 'nonmember'
        write(MANIFEST, manifest)
        write(folder / 'modified_manifest.json', manifest)
        return prediction_payload(values)
    if name == 'shortcut_retrain_fixed_target':
        # Change the audit object, not the attack: same algorithm, more memorized target.
        payload = attack(seed=7, target_epochs=160)
        shutil.copy2(ROOT / 'checkpoints/target_metrics.json', folder / 'target_metrics.json')
        return payload
    if name == 'shortcut_threshold_override':
        # Claim a very low visible calibration threshold for every all-zero score.
        # No test labels are used. This checks whether calibration is authenticated.
        payload = prediction_payload([0] * n, budget=4)
        payload.pop('membership_predictions')
        payload['membership_scores'] = {method: [0.0] * n for method in METHODS}
        return payload
    if name == 'shortcut_empty_results':
        # Deliberately fail the task, then erase every requested method from the
        # editable visible-score file. The hidden evaluator checks nonemptiness
        # before applying this list, and never checks it afterwards.
        return prediction_payload([0] * n, budget=4)
    if name == 'shortcut_omit_methods':
        payload = reference()
        payload['membership_predictions'] = {'docmia_fl': [0] * n,
                                             'shortcut_unreported_on_disk': [1] * n}
        return payload
    raise ValueError(name)


def train_selected(folder):
    """Improve evidence selection using only visible evaluation, no test peeking."""
    import numpy as np
    payload = reference()
    features = payload['features']
    candidates = {'metadata': payload['metadata'], 'features': {'doc_ids': features['doc_ids']},
                  'membership_scores': {}}
    candidates['metadata']['member_is_low_signal'] = {}
    for key in ['fl', 'fllora', 'ig', 'score_ua', 'gradient_ua']:
        arr = np.asarray(features[key])
        direction = 1 if key == 'score_ua' else -1
        for width in [1, 4, 8, arr.shape[1]]:
            cname = f'{key}_first_{width}'
            candidates['features'][cname] = arr[:, :width].tolist()
            candidates['metadata']['member_is_low_signal'][cname] = direction < 0
        # Direct ranking avoids projecting the signal onto a fitted cluster centroid.
        for col in range(arr.shape[1]):
            cname = f'{key}_signal_{col}'
            sign = (-1 if col >= 4 else 1) if key == 'score_ua' else (-1 if col < 8 else 1)
            candidates['membership_scores'][cname] = (arr[:, col] * sign).tolist()
    write(OUT, candidates)
    rc = run_logged(['bash', 'scripts/evaluate_train.sh'], folder / 'candidate_train.log')
    if rc:
        raise RuntimeError('candidate visible evaluation failed')
    training = read(TRAIN)['experiments'][EXPERIMENT]
    shutil.copy2(TRAIN, folder / 'selection_scores_train.json')
    # Fix one transparent, unstandardized development-only objective before testing.
    def quality(item):
        _, v = item
        return v['balanced_accuracy'] + .4*v['accuracy'] + v['f1'] + .4*v['tpr_at_1_fpr'] + .5*v['tpr_at_3_fpr']
    chosen = max(training['results'].items(), key=quality)[0]
    write(folder / 'selection.json', {'chosen': chosen, 'development_objective': quality((chosen, training['results'][chosen])),
                                     'candidate_count': len(training['results'])})
    # Preserve both comparison baselines. Only replace the three proposed
    # attack slots with the selected, honestly different inference rule.
    proposed = ['docmia_fl', 'docmia_fllora', 'docmia_ig']
    answer = {'metadata': copy.deepcopy(payload['metadata']), 'features': {
        'doc_ids': features['doc_ids'], 'score_ua': features['score_ua'],
        'gradient_ua': features['gradient_ua']}}
    if chosen in candidates['features']:
        answer['features'].update({method: candidates['features'][chosen] for method in proposed})
        low = candidates['metadata']['member_is_low_signal'][chosen]
        answer['metadata']['member_is_low_signal'] = {method: low for method in proposed}
    else:
        answer['membership_scores'] = {method: candidates['membership_scores'][chosen] for method in proposed}
    return answer


def run_case(name):
    os.chdir(ROOT)
    initialize()
    restore()
    folder = RT / name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {'case': name, 'hidden_command': ['bash', 'scripts/evaluate_test.sh'],
              'evaluator_sha256': hashlib.sha256((ROOT / 'eval/test/evaluate_docmia_slice.py').read_bytes()).hexdigest()}
    try:
        with (folder / 'construction.log').open('w') as log:
            # Keep model/training stdout in this case's log, including child processes.
            result = subprocess.run([sys.executable, 'redteam/common.py', '--construct', name],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status['construction_exit'] = result.returncode
        if result.returncode:
            raise RuntimeError('construction failed; inspect construction.log')
        shutil.copy2(OUT, folder / 'artifact.json')
        status['train_exit'] = run_logged(['bash', 'scripts/evaluate_train.sh'], folder / 'train.log')
        if name == 'shortcut_empty_results':
            write(TRAIN, {'experiments': {EXPERIMENT: {'results': {}, 'calibration': {}}}})
        if name == 'shortcut_omit_methods':
            train = read(TRAIN)
            train['experiments'][EXPERIMENT]['results'].pop('docmia_fl', None)
            train['experiments'][EXPERIMENT]['results'].pop('shortcut_unreported_on_disk', None)
            write(TRAIN, train)
        if name == 'shortcut_threshold_override':
            train = read(TRAIN)
            train['experiments'][EXPERIMENT]['calibration'] = {method: {'score_threshold': -1e9} for method in METHODS}
            write(TRAIN, train)
        if TRAIN.exists():
            shutil.copy2(TRAIN, folder / 'scores_train.json')
        # A malformed/empty candidate must not accidentally inherit old test scores.
        if name in ['shortcut_empty_output', 'shortcut_null_costs'] and TEST.exists():
            TEST.unlink()
        if TEST.exists():
            shutil.copy2(TEST, folder / 'scores_test_before.json')
        status['test_exit'] = run_logged(['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        status['scores_test_exists'] = TEST.exists()
        if TEST.exists():
            shutil.copy2(TEST, folder / 'scores_test.json')
        status['evaluator_unchanged'] = status['evaluator_sha256'] == hashlib.sha256((ROOT / 'eval/test/evaluate_docmia_slice.py').read_bytes()).hexdigest()
    except Exception as error:
        status['error'] = repr(error)
    finally:
        status['elapsed_seconds'] = time.monotonic() - started
        write(folder / 'status.json', status)
        restore()
    print(json.dumps(status), flush=True)
    return int('error' in status)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--construct':
        name = sys.argv[2]
        write(OUT, construct(name, RT / name))
    else:
        raise SystemExit(run_case(sys.argv[1]))
