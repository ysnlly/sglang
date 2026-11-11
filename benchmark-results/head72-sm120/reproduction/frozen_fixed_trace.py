"""Fixed-work VLM request replay; preparation is outside the measured interval.

This client is diagnostic infrastructure. It does not establish task accuracy,
server/backend identity, resource isolation, or a complete paired experiment.
"""
import base64
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import time


IMAGE_NAMES = [
    'chelsea.png', 'coffee.png', 'rocket.jpg', 'astronaut.png',
    'hubble_deep_field.jpg', 'motorcycle_left.png', 'motorcycle_right.png',
    'camera.png', 'clock_motion.png', 'coins.png', 'grass.png', 'gravel.png',
    'ihc.png', 'cell.png', 'page.png', 'moon.png',
]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def prepare(image_root, width, height, images_per_request, count, output_tokens=64, seed=42):
    from PIL import Image, ImageOps
    assert width > 0 and height > 0 and 1 <= images_per_request <= len(IMAGE_NAMES)
    assert count > 0 and output_tokens > 0
    sources, urls = [], []
    for name in IMAGE_NAMES:
        path = Path(image_root)/name
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            original_size = list(im.size)
            resized = ImageOps.fit(im.convert('RGB'), (width, height))
        buffer = io.BytesIO()
        resized.save(buffer, 'PNG')
        blob = buffer.getvalue()
        sources.append(dict(name=name, original_sha256=sha(raw), original_size=original_size,
                            png_sha256=sha(blob), size=[width, height]))
        urls.append('data:image/png;base64,' + base64.b64encode(blob).decode())
    payloads = []
    for image_id in range(len(urls)):
        content = [{'type': 'text', 'text': 'Describe these images.'}]
        content.extend({'type': 'image_url', 'image_url': {'url': urls[(image_id+k)%len(urls)]}}
                       for k in range(images_per_request))
        body = dict(model='qwen35-9b', messages=[{'role': 'user', 'content': content}],
                    temperature=0, seed=seed, max_tokens=output_tokens, ignore_eos=True,
                    stream=False, return_token_ids=True, return_prompt_token_ids=True,
                    chat_template_kwargs={'enable_thinking': False})
        payloads.append(encoded(body))
    order = [i % len(payloads) for i in range(count)]
    random.Random(seed).shuffle(order)
    manifest = dict(sources=sources, image_count=images_per_request, width=width, height=height,
                    request_count=count, output_tokens=output_tokens, seed=seed,
                    payload_sha256=[sha(b) for b in payloads], order=order,
                    provenance='Existing local sample images, fitted to existing diagnostic shapes; execution stress, not image-quality evaluation')
    manifest['trace_sha256'] = sha(encoded(manifest))
    return manifest, payloads


def validate_response(data, expected_output_tokens):
    choices = data['choices']
    assert len(choices) == 1, 'expected exactly one choice'
    choice = choices[0]
    prompt_ids = choice['prompt_token_ids']
    output_ids = choice['response_token_ids']
    assert isinstance(prompt_ids, list) and prompt_ids, 'missing prompt IDs'
    assert isinstance(output_ids, list), 'missing output IDs'
    assert all(type(v) is int for v in prompt_ids + output_ids), 'non-integer token ID'
    usage = data['usage']
    assert usage['prompt_tokens'] == len(prompt_ids), 'prompt usage/ID count mismatch'
    assert usage['completion_tokens'] == len(output_ids) == expected_output_tokens, 'output count mismatch'
    assert usage['total_tokens'] == len(prompt_ids) + len(output_ids), 'total usage mismatch'
    assert choice['finish_reason'] == 'length', 'fixed-token request ended unexpectedly'
    message = choice['message']
    assert (message.get('content') or message.get('reasoning_content')), 'empty output'
    return dict(prompt_ids_sha256=sha(encoded(prompt_ids)), prompt_tokens=len(prompt_ids),
                output_tokens=len(output_ids), finish_reason=choice['finish_reason'])


def replay(url, manifest, payloads, concurrency, output, barrier=None):
    import requests
    assert concurrency > 0
    assert [sha(b) for b in payloads] == manifest['payload_sha256']
    original_manifest = {k: v for k, v in manifest.items() if k != 'trace_sha256'}
    assert sha(encoded(original_manifest)) == manifest['trace_sha256']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'trace_manifest.json').write_text(json.dumps(manifest, indent=2))

    def request_one(index, payload_index):
        start = time.monotonic()
        row = dict(index=index, payload_index=payload_index,
                   request_sha256=manifest['payload_sha256'][payload_index],
                   started_unix_s=time.time(), error=None)
        try:
            response = requests.post(url, data=payloads[payload_index],
                                     headers={'Content-Type': 'application/json'}, timeout=(10, 600))
            row['http_status'] = response.status_code
            if not response.ok:
                row['error_response'] = response.text
            response.raise_for_status()
            data = response.json()
            # Store the actual response, including token IDs, even on contract failure.
            row['response'] = data
            row.update(validate_response(data, manifest['output_tokens']))
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        row.update(latency_ms=(time.monotonic()-start)*1000, finished_unix_s=time.time())
        return row

    rows = []
    # Pool construction and payload serialization are excluded; HTTP transfer,
    # server preprocessing, inference, and response parsing remain included.
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        if barrier is not None:
            barrier.wait(timeout=60)
        started_unix = time.time()
        start = time.monotonic()
        futures = [pool.submit(request_one, i, payload_index)
                   for i, payload_index in enumerate(manifest['order'])]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
        elapsed = time.monotonic()-start
        finished_unix = time.time()
    # BOS writes may block unpredictably. Preserve evidence after the timer.
    with (output/'requests.jsonl').open('x') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False)+'\n')
    successful = [r for r in rows if not r['error']]
    result = dict(trace_sha256=manifest['trace_sha256'], concurrency=concurrency,
                  started_unix_s=started_unix, finished_unix_s=finished_unix,
                  elapsed_s=elapsed, requested=len(manifest['order']), successful=len(successful),
                  errors=len(rows)-len(successful), success_rps=len(successful)/elapsed,
                  completion_tokens=sum(r['output_tokens'] for r in successful),
                  latencies_ms=[r['latency_ms'] for r in successful],
                  limits='Finite canonical trace including queue fill/drain; elapsed includes response parsing and token-ID validation, excludes evidence-file writes. Token-ID response fields are enabled in both arms. No accuracy verdict or paired effect by itself.')
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    assert result['errors'] == 0 and result['successful'] == result['requested'], 'failed request contract'
    assert result['completion_tokens'] == result['successful']*manifest['output_tokens']
    return result
