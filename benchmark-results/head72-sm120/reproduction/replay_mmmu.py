"""Complete 900-question fixed-64-token control; uses the measured replay code."""
import argparse
import json
from pathlib import Path

from mmmu_fixed_replay import prepare, replay, validate_trace
from portable_mmmu import PortableEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--urls', nargs=8)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and not args.urls:
        parser.error('--urls is required for inference')
    evaluator = PortableEvaluator(args.data)
    manifest, payloads = prepare(evaluator, seed=20261110, output_tokens=64)
    validate_trace(manifest, payloads)
    expected = json.loads(Path(__file__).with_name('mmmu_fixed64_trace.json').read_text())['trace_sha256']
    assert manifest['trace_sha256'] == expected, 'Inputs differ from the measured 900-question control'
    if args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'trace_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(dict(questions=900, trace_sha256=expected, exact_trace_identity_passed=True,
                              requests_sent=0, gpu_work_started=False)), flush=True)
        return
    urls = dict(enumerate(args.urls))
    assert len(set(urls.values())) == 8
    result = replay(manifest, payloads, args.output, urls)
    print(json.dumps(dict(requests=result['successful'], errors=result['errors'],
                          fleet_rps=result['fleet_success_rps'], accuracy_not_measured=True)), flush=True)


if __name__ == '__main__':
    main()
