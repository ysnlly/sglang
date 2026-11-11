"""Run one complete quality crossover round against eight prestarted servers.

This client preserves the measured scheduler/scorer but does not own servers or
establish source identity, isolation, cleanup, or final statistical acceptance.
"""
import argparse
import json
from pathlib import Path

from portable_mmmu import PortableEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Checkout containing the pinned MMMU scorer')
    parser.add_argument('--round', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--urls', nargs=8, required=True)
    args = parser.parse_args()
    evaluator = PortableEvaluator(args.data, args.source, max_tokens=16384, urls=dict(enumerate(args.urls)))
    evaluator.load_scorer()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'client_plan.json').write_text(json.dumps(dict(
        round=args.round, max_tokens=16384, questions_per_arm=900,
        reference_gpus=[gpu for gpu in range(8) if gpu % 2 == args.round % 2],
        candidate_gpus=[gpu for gpu in range(8) if gpu % 2 != args.round % 2],
        labels={'default': 'original', 'split': 'candidate'},
        urls=args.urls, server_ownership_not_established_by_client=True,
    ), indent=2) + '\n')
    evaluator.warmup(list(range(8)), args.output / 'warmup.json')
    evaluator.run(args.round, args.output)
    completed = {}
    for arm in ('default', 'split'):
        rows = [json.loads(line) for line in (args.output / (arm + '.jsonl')).read_text().splitlines()]
        assert len(rows) == len({row['id'] for row in rows}) == 900
        correct = lambda row: row['correct'] and not row['error'] and row['finish_reason'] != 'length'
        completed[arm] = dict(
            completed_correct=sum(bool(correct(row)) for row in rows),
            multi_completed_correct=sum(bool(correct(row)) for row in rows if row['image_count'] > 1),
            request_errors=sum(bool(row['error']) for row in rows),
            truncated=sum(row['finish_reason'] == 'length' for row in rows),
        )
    (args.output / 'completed_summary.json').write_text(json.dumps(completed, indent=2) + '\n')
    print(json.dumps(dict(round=args.round, completed=completed,
                          final_qualification_not_computed=True)), flush=True)


if __name__ == '__main__':
    main()
