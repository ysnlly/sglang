"""Finite fixed-token replay of the frozen complete MMMU validation corpus.

This is a client, not a formal evaluator or accuracy gate. It starts no services.
Image preparation and evidence writes stay outside the one fleet wall timer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time

from frozen_fixed_trace import validate_response


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def prepare(evaluator, seed=20261110, output_tokens=64):
    if type(output_tokens) is not int or output_tokens <= 0:
        raise ValueError('positive fixed output token count required')
    samples = evaluator.samples
    if len(samples) != 900 or len({s['id'] for s in samples}) != 900:
        raise ValueError('complete 900-question corpus required')
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    entries, payloads = [], []
    for position, sample_index in enumerate(order):
        sample = samples[sample_index]
        body = evaluator.body(sample)
        body.update(max_tokens=output_tokens, ignore_eos=True,
                    return_token_ids=True, return_prompt_token_ids=True)
        blob = encoded(body)
        payloads.append(blob)
        entries.append(dict(id=sample['id'], sample_index=sample_index,
                            gpu=position % 8, image_count=len(sample['images']),
                            image_sha256=[im['sha256'] for im in sample['images']],
                            request_sha256=digest(blob)))
    manifest = dict(format_version=1, seed=seed, output_tokens=output_tokens,
                    requests=entries, gpu_ids=list(range(8)), concurrency_per_gpu=4,
                    dataset='MMMU/MMMU validation', request_count=len(entries),
                    request_semantics='Pinned complete quality prompts and images; fixed-token generation, not task-accuracy scoring')
    manifest['trace_sha256'] = digest(encoded(manifest))
    return manifest, payloads


def validate_trace(manifest, payloads):
    core = {k: v for k, v in manifest.items() if k != 'trace_sha256'}
    if digest(encoded(core)) != manifest['trace_sha256']:
        raise ValueError('trace manifest identity changed')
    if (manifest['gpu_ids'] != list(range(8)) or
            type(manifest['concurrency_per_gpu']) is not int or
            manifest['concurrency_per_gpu'] not in (1, 4)):
        raise ValueError('expected eight TP1 replicas at declared concurrency one or four')
    entries = manifest['requests']
    if len(entries) != len(payloads) or len(entries) != manifest['request_count']:
        raise ValueError('request count mismatch')
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('duplicate question in trace')
    for i, (entry, payload) in enumerate(zip(entries, payloads)):
        if entry['gpu'] != i % 8 or digest(payload) != entry['request_sha256']:
            raise ValueError('payload identity or GPU assignment mismatch')
        body = json.loads(payload)
        if (body.get('max_tokens') != manifest['output_tokens'] or
                body.get('ignore_eos') is not True or body.get('stream') is not False or
                body.get('return_token_ids') is not True or
                body.get('return_prompt_token_ids') is not True):
            raise ValueError('fixed-token request contract mismatch')


def replay(manifest, payloads, output, urls=None):
    import requests
    validate_trace(manifest, payloads)
    urls = urls or {g: f'http://127.0.0.1:{19400+g}/v1/chat/completions' for g in range(8)}
    if set(urls) != set(range(8)):
        raise ValueError('exactly eight server endpoints required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'trace_manifest.json').write_bytes(encoded(manifest))

    def request_one(index):
        entry = manifest['requests'][index]
        row = dict(entry, index=index, started_unix_s=time.time(), error=None)
        begin = time.monotonic()
        try:
            response = requests.post(urls[entry['gpu']], data=payloads[index],
                                     headers={'Content-Type': 'application/json'}, timeout=(10, 600))
            row['http_status'] = response.status_code
            if not response.ok:
                row['error_response'] = response.text
            response.raise_for_status()
            row['response'] = response.json()
            row.update(validate_response(row['response'], manifest['output_tokens']))
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        row.update(finished_unix_s=time.time(), latency_ms=(time.monotonic()-begin)*1000)
        return row

    # One shared timer for all GPUs: never sum/average individual replica RPS.
    pools = {g: ThreadPoolExecutor(max_workers=manifest['concurrency_per_gpu']) for g in range(8)}
    rows = []
    try:
        started_unix = time.time()
        start = time.monotonic()
        pending = [pools[e['gpu']].submit(request_one, i)
                   for i, e in enumerate(manifest['requests'])]
        for future in as_completed(pending):
            rows.append(future.result())
        elapsed = time.monotonic()-start
        finished_unix = time.time()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(key=lambda r: r['index'])
    # Retain failed responses as well. No BOS write is included in the timer.
    with (output/'responses.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    successful = [r for r in rows if r['error'] is None]
    result = dict(trace_sha256=manifest['trace_sha256'], started_unix_s=started_unix,
                  finished_unix_s=finished_unix, elapsed_s=elapsed,
                  requested=len(payloads), successful=len(successful), errors=len(rows)-len(successful),
                  fleet_success_rps=len(successful)/elapsed,
                  output_tokens=sum(r['output_tokens'] for r in successful),
                  latency_ms=[r['latency_ms'] for r in successful],
                  prompt_identity={r['id']: dict(sha256=r['prompt_ids_sha256'], tokens=r['prompt_tokens'])
                                   for r in successful},
                  gpu_request_counts={str(g): sum(r['gpu'] == g for r in rows) for g in range(8)},
                  interpretation='Finite closed-loop fleet trace, includes queue fill/drain, HTTP transfer and response validation. Not steady-state capacity, TTFT, or task-accuracy evidence.')
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    if result['errors'] or len(rows) != len(payloads):
        raise RuntimeError('fixed-work response contract failed; raw evidence retained')
    if result['output_tokens'] != len(payloads)*manifest['output_tokens']:
        raise RuntimeError('fixed-work token accounting failed')
    return result


def main():
    parser = argparse.ArgumentParser(description='CPU-only preparation and identity audit; starts no server or requests')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from frozen_quality_requests import Evaluator
    manifest, payloads = prepare(Evaluator(args.data, args.source))
    validate_trace(manifest, payloads)
    result = dict(manifest=manifest, total_serialized_bytes=sum(map(len, payloads)),
                  multi_image_requests=sum(e['image_count'] > 1 for e in manifest['requests']),
                  preparation_only=True, gpu_work_started=False,
                  source_files={p.name: digest(p.read_bytes()) for p in
                                [Path(__file__), Path(__file__).with_name('frozen_quality_requests.py'),
                                 Path(__file__).with_name('frozen_fixed_trace.py')]})
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'manifest'}))


if __name__ == '__main__':
    main()
