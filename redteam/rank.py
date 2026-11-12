"""Transparent audit-population ranking, never used by candidate construction.

The platform's later attempt population is unavailable. We therefore publish
raw values and compute the stated directional, absolute-coefficient objective
using population standard deviations across our successful finite attempts,
separately for each method/metric. Zero-spread metrics contribute zero. A common
division by five makes the suite score an average over reference method slots;
the unaveraged sum is also saved. This positive constant cannot change ranks.
Nonfinite and failed runs have no assigned rank: the external grader's handling
is not specified and cannot be inferred from a retained score file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
EXPERIMENT = 'scaled_whitebox_docmia'


def read(path):
    return json.loads(path.read_text())


def main():
    specs = read(ROOT / 'scoring/reference.json')['experiments'][EXPERIMENT]
    metrics = {k: v for k, v in specs['metrics'].items() if abs(v['coefficient']) > 0}
    reference = read(RT / 'reference_replay/scores_test.json')['experiments'][EXPERIMENT]['results']
    data = {}
    excluded = {}
    for case in read(RT / 'cases.json'):
        folder = RT / case
        if not (folder / 'status.json').exists():
            excluded[case] = 'not run'
            continue
        status = read(folder / 'status.json')
        if status.get('test_exit') != 0 or not (folder / 'scores_test.json').exists():
            excluded[case] = 'test failed or no output'
            continue
        rows = read(folder / 'scores_test.json')['experiments'][EXPERIMENT]['results']
        if any(value is not None and not math.isfinite(float(value))
               for name in reference for key in metrics
               if (value := rows.get(name, {}).get(key)) is not None):
            excluded[case] = 'nonfinite scored metric; external rank behavior unknown'
            continue
        data[case] = rows
    scales = {}
    for method in reference:
        scales[method] = {}
        for metric in metrics:
            values = [float(rows[method][metric]) for rows in data.values()
                      if method in rows and rows[method].get(metric) is not None]
            scales[method][metric] = statistics.pstdev(values) if len(values) > 1 else 0.0
    ranked = {}
    for case, rows in data.items():
        contributions = {}
        for method, ref in reference.items():
            parts = {}
            for key, spec in metrics.items():
                value = rows.get(method, {}).get(key)
                spread = scales[method][key]
                if value is None or spread == 0:
                    continue
                direction = 1 if spec['higher_is_better'] else -1
                parts[key] = abs(spec['coefficient']) * direction * (float(value) - ref[key]) / spread * specs['weight']
            contributions[method] = parts
        by_method = {method: sum(parts.values()) for method, parts in contributions.items()}
        total = sum(by_method.values())
        ranked[case] = {'rank_sum': total, 'rank_mean': total / len(reference),
                        'method_gains': by_method, 'contributions': contributions,
                        'returned_methods': list(rows)}
    output = {'normalization': 'population std across successful finite audit attempts, per reference method and metric',
              'reference': 'reference_replay/scores_test.json',
              'warning': 'Future attempt population can change tradeoff ranks; strict dominance is population independent.',
              'population': list(data), 'excluded': excluded, 'scales': scales, 'ranks': ranked}
    (RT / 'ranking.json').write_text(json.dumps(output, indent=2) + '\n')
    for case, values in sorted(ranked.items(), key=lambda x: -x[1]['rank_mean']):
        print(f"{case:40s} {values['rank_mean']:+.6f}")
    for case, reason in excluded.items():
        print(f'{case:40s} UNRANKED: {reason}')


if __name__ == '__main__':
    main()
