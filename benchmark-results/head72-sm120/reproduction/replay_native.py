"""Replay the eight native strata using the unchanged formal client functions.

Starts no model servers. Does not replace the separate 900-question accuracy or
fixed-token performance controls, ownership checks, or six-pair statistics.
"""
import argparse
import json
from pathlib import Path

from mmmu_fixed_replay import digest, encoded, replay
from native_stress_trace import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--urls', nargs=8, metavar='URL',
                        help='Eight /v1/chat/completions endpoints, in GPU order')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and not args.urls:
        parser.error('--urls is required unless --prepare-only is set')
    if args.urls and len(set(args.urls)) != 8:
        parser.error('--urls must contain eight distinct endpoints')
    cases = json.loads(Path(__file__).with_name('native_cases.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for case in cases:
        manifest, payloads = prepare(
            args.images, width=case['width'], height=case['height'],
            image_count=case['images'], count_per_gpu=16,
            concurrency=case['concurrency'], output_tokens=64, seed=42,
        )
        if manifest['trace_sha256'] != case['trace_sha256']:
            raise ValueError('Payloads differ from the measured workload: ' + case['id'])
        (args.output / (case['id'] + '-manifest.json')).write_bytes(encoded(manifest))
        print(json.dumps(dict(case=case['id'], requests=len(payloads),
                              trace_sha256=manifest['trace_sha256'], identity_passed=True)), flush=True)
        # Preparation-only verification need not retain all large payloads.
        if not args.prepare_only:
            prepared[case['id']] = manifest, payloads
    if args.prepare_only:
        (args.output / 'preparation.json').write_text(json.dumps(dict(
            native_cases=8, all_trace_identities_passed=True,
            requests_sent=0, gpu_work_started=False,
            scope='CPU preparation check; not a new inference experiment.',
        ), indent=2) + '\n')
        return
    urls = dict(enumerate(args.urls))
    results = {}
    for case in cases:
        manifest, payloads = prepared[case['id']]
        warm = {k: v for k, v in manifest.items() if k != 'trace_sha256'}
        warm.update(requests=manifest['requests'][:32], request_count=32,
                    selection='First four requests per GPU of the fixed native trace')
        warm['trace_sha256'] = digest(encoded(warm))
        replay(warm, payloads[:32], args.output / (case['id'] + '-warmup'), urls)
        result = replay(manifest, payloads, args.output / (case['id'] + '-repeat0'), urls)
        results[case['id']] = result
        print(json.dumps(dict(case=case['id'], requests=result['successful'],
                              errors=result['errors'], fleet_rps=result['fleet_success_rps'])), flush=True)
    (args.output / 'native_summary.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
