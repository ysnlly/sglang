"""Recompute the frozen six-pair performance statistics from audited metrics.

This does not verify raw requests, source identity, task accuracy or GPU ownership.
The input must be exported from separately audited role artifacts. Pair, not
request or GPU, is the sampling unit. No optional stopping or missing-pair drop.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

ORDERS = ['AB', 'BA', 'BA', 'BA', 'AB', 'AB']
PRIMARY = 'large_multi_c4_rps'
CONTROLS = [
    'small_single_c1_rps', 'small_single_c4_rps',
    'small_multi_c1_rps', 'small_multi_c4_rps',
    'large_single_c1_rps', 'large_single_c4_rps',
    'large_multi_c1_rps', 'mmmu64_rps',
    'large_multi_c4_p50_ms', 'large_multi_c4_p95_ms',
]
SEED = 20261110
DRAWS = 10000
CONFIDENCE = .95


def interval(values, seed):
    generator = random.Random(seed)
    draws = sorted(statistics.median(generator.choices(values, k=6)) for _ in range(DRAWS))

    def percentile(q):
        position = (DRAWS - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)

    tail = (1 - CONFIDENCE) / 2
    return [percentile(tail), percentile(1 - tail)]


def analyze(data):
    pairs = data['pairs']
    if len(pairs) != 6:
        raise ValueError('Exactly six complete pairs are required')
    for index, pair in enumerate(pairs):
        if type(pair['index']) is not int or pair['index'] != index or pair['order'] != ORDERS[index]:
            raise ValueError('Pair identity/order differs from the frozen plan')
        for role in ['reference', 'candidate']:
            metrics = pair[role]['metrics']
            if set(metrics) != {PRIMARY, *CONTROLS}:
                raise ValueError('Metric coverage mismatch')
            for value in metrics.values():
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Metrics must be positive finite numbers')
    metrics = []
    for index, name in enumerate([PRIMARY, *CONTROLS]):
        values = []
        for pair in pairs:
            reference = pair['reference']['metrics'][name]
            candidate = pair['candidate']['metrics'][name]
            # Primary is improvement; controls are regression (lower is better).
            delta = candidate - reference if index == 0 or name.endswith('_ms') else reference - candidate
            values.append(delta / abs(reference) * 100)
        low, high = interval(values, SEED + index)
        passed = low >= 5 if index == 0 else high <= 5
        metrics.append(dict(name=name, statistic='improvement_pct' if index == 0 else 'regression_pct',
                            paired_values_pct=values, median_pct=statistics.median(values),
                            ci_low_pct=low, ci_high_pct=high, passed=passed))
    return dict(pairs=6, confidence=CONFIDENCE, bootstrap_draws=DRAWS, seed=SEED,
                statistic='median_of_paired_percentage_changes',
                performance_statistical_gates_passed=all(item['passed'] for item in metrics),
                metrics=metrics, full_target_verdict_not_established=True,
                limitations='Requires separate raw/source/correctness/ownership audits; intervals are per metric, not simultaneous.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
