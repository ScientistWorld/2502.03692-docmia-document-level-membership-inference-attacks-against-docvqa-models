"""Lightweight checks of saved audit evidence; no model computation."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / 'redteam'
EXPERIMENT = 'scaled_whitebox_docmia'


def read(path):
    return json.loads(path.read_text())


def rows(case):
    return read(RT / case / 'scores_test.json')['experiments'][EXPERIMENT]['results']


def main():
    cases = read(RT / 'cases.json')
    expected_failures = {'shortcut_null_costs', 'shortcut_empty_output'}
    evaluator_hash = hashlib.sha256((ROOT / 'eval/test/evaluate_docmia_slice.py').read_bytes()).hexdigest()
    for case in cases:
        folder = RT / case
        status = read(folder / 'status.json')
        assert (folder / 'run.sh').exists(), case
        assert (folder / 'artifact.json').exists(), case
        assert status['construction_exit'] == 0, case
        assert status['hidden_command'] == ['bash', 'scripts/evaluate_test.sh'], case
        assert status['evaluator_unchanged'], case
        assert status['evaluator_sha256'] == evaluator_hash, case
        if case in expected_failures:
            assert status['test_exit'] != 0 and not status['scores_test_exists'], case
            assert not (folder / 'scores_test.json').exists(), case
        else:
            assert status['test_exit'] == 0 and status['scores_test_exists'], case
    reference = rows('reference_replay')
    for method in ['score_ua', 'gradient_ua']:
        assert rows('stronger_train_selected')[method] == reference[method]
        assert rows('weaker_one_step')[method] == reference[method]
    benefits = ['balanced_accuracy', 'accuracy', 'f1', 'tpr_at_1_fpr', 'tpr_at_3_fpr']
    for method in ['docmia_fl', 'docmia_fllora', 'docmia_ig']:
        assert all(rows('stronger_train_selected')[method][key] > reference[method][key] for key in benefits)
    for case in ['shortcut_manifest_labels', 'shortcut_filename_labels',
                 'shortcut_bruteforce_generator', 'shortcut_move_ground_truth']:
        assert all(row[key] == 100 for row in rows(case).values() for key in benefits), case
    assert rows('shortcut_empty_results') == {}
    assert 'docmia_fl' not in rows('shortcut_omit_methods')
    assert 'shortcut_unreported_on_disk' not in rows('shortcut_omit_methods')
    assert all(row['balanced_accuracy'] == 50 for row in rows('shortcut_impossible_costs').values())
    assert all(row['max_questions_per_document'] == -1e12 for row in rows('shortcut_impossible_costs').values())
    assert all(math.isnan(row['uses_auxiliary_data']) for row in rows('shortcut_nan_costs').values())
    ranking = read(RT / 'ranking.json')
    assert ranking['ranks']['shortcut_impossible_costs']['rank_sum'] > 0
    assert ranking['ranks']['shortcut_forged_scores']['rank_sum'] < 0
    assert ranking['ranks']['shortcut_empty_results']['rank_sum'] == 0
    fixed_cases = [f'seed_fixed_{seed}' for seed in [7, 8, 9]]
    fixed_ranks = [ranking['ranks'][case]['rank_sum'] for case in fixed_cases]
    seed_ranges = {
        method: {key: max(rows(case)[method][key] for case in fixed_cases)
                      - min(rows(case)[method][key] for case in fixed_cases)
                 for key in reference[method]}
        for method in reference
    }
    summary = {
        'cases_checked': len(cases),
        'successful_test_runs': len(cases) - len(expected_failures),
        'expected_test_failures': sorted(expected_failures),
        'evaluator_sha256': evaluator_hash,
        'reference_test': reference,
        'fixed_target_seed_cases': fixed_cases,
        'fixed_target_seed_metric_ranges': seed_ranges,
        'fixed_target_seed_rank_min': min(fixed_ranks),
        'fixed_target_seed_rank_max': max(fixed_ranks),
        'fixed_target_seed_rank_spread': max(fixed_ranks) - min(fixed_ranks),
        'ranks': {case: values['rank_sum'] for case, values in ranking['ranks'].items()},
        'unranked': ranking['excluded'],
    }
    (RT / 'audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # The harness also restores ignored checkpoints and scores. Verify those
    # byte-for-byte, in addition to checking the tracked gym tree with git.
    restored = {}
    backup_index = RT / '_backup/files.json'
    for relative, existed in (read(backup_index) if backup_index.exists() else {}).items():
        path = ROOT / relative
        if existed:
            original_hash = hashlib.sha256((RT / '_backup/gym' / relative).read_bytes()).hexdigest()
            assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == original_hash, relative
            restored[relative] = original_hash
        else:
            assert not path.exists(), relative
            restored[relative] = None
    start = (RT / 'start_commit.txt').read_text().strip()
    checked_paths = ['baseline', 'briefing', 'data', 'environment', 'eval',
                     'method', 'scoring', 'scripts', 'MILESTONES.md', 'PROGRESS.md', '.gitignore']
    subprocess.run(['git', 'diff', '--exit-code', '--quiet', start, '--', *checked_paths],
                   cwd=ROOT, check=True)
    if backup_index.exists():
        (RT / 'restoration.json').write_text(json.dumps({
            'start_commit': start,
            'tracked_paths_matching_start': checked_paths,
            'restored_file_sha256': restored,
            'exceptions': ['Required final action.yaml signal',
                           'Pre-existing system-owned README.md and manifest.json device entries',
                           'System-managed trajectory/job records'],
        }, indent=2) + '\n')
    else:
        print('Ignored checkpoint backups are absent; saved restoration.json records the original audit check.')
    print(json.dumps({key: summary[key] for key in [
        'cases_checked', 'successful_test_runs', 'expected_test_failures',
        'fixed_target_seed_rank_min', 'fixed_target_seed_rank_max',
        'fixed_target_seed_rank_spread']}, indent=2))


if __name__ == '__main__':
    main()
