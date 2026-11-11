"""Prepare the existing single/multi-image native workload for one fleet timer.

Uses the previously validated canonical requests; changes no image content,
model prompt or token policy. No servers are started here.
"""
from frozen_fixed_trace import prepare as prepare_native
from mmmu_fixed_replay import digest, encoded, replay, validate_trace


def prepare(image_root, *, width, height, image_count, count_per_gpu, concurrency,
            output_tokens=64, seed=42):
    canonical, canonical_payloads = prepare_native(image_root, width, height, image_count,
                                                  count_per_gpu, output_tokens, seed)
    entries, payloads = [], []
    for request_index, payload_index in enumerate(canonical['order']):
        for gpu in range(8):
            payload = canonical_payloads[payload_index]
            entries.append(dict(id=f'gpu{gpu}/request{request_index}', gpu=gpu,
                                payload_index=payload_index, image_count=image_count,
                                request_sha256=digest(payload)))
            payloads.append(payload)
    manifest = dict(format_version=1, seed=seed, output_tokens=output_tokens,
                    requests=entries, gpu_ids=list(range(8)), concurrency_per_gpu=concurrency,
                    dataset='Existing native sample-image workload', request_count=len(entries),
                    canonical=canonical,
                    request_semantics='Same canonical fixed-token native requests on each of eight TP1 replicas; finite trace including fill/drain')
    manifest['trace_sha256'] = digest(encoded(manifest))
    validate_trace(manifest, payloads)
    return manifest, payloads


def validate_prompt_identity(result, reference):
    expected = {'sha256': reference['sha256'], 'tokens': reference['tokens']}
    values = list(result['prompt_identity'].values())
    if len(values) != result['requested'] or not values or any(v != expected for v in values):
        raise ValueError('native prompt IDs/count differ from the frozen original reference')
    return True
