"""Path/endpoint adaptation for the preserved measured MMMU evaluator.

Request-body construction, scoring, warmup and crossover scheduling are inherited
from its unchanged source. Scorer loading is deferred for CPU input-only checks.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import requests

from frozen_quality_requests import Evaluator


class PortableEvaluator(Evaluator):
    def __init__(self, data_root, source=None, max_tokens=16384, urls=None):
        assert isinstance(max_tokens, int) and max_tokens > 0
        self.max_tokens = max_tokens
        self.root = Path(data_root).resolve()
        self.source = Path(source).resolve() if source is not None else None
        self.urls = urls or {gpu: f'http://127.0.0.1:{19400+gpu}/v1/chat/completions' for gpu in range(8)}
        assert set(self.urls) == set(range(8)) and len(set(self.urls.values())) == 8
        contract = json.loads(Path(__file__).with_name('mmmu_contract.json').read_text())
        manifest = json.loads((self.root / 'manifest.json').read_text())
        assert all(manifest[key] == contract[key] for key in contract)
        blob = (self.root / 'samples.json').read_bytes()
        assert hashlib.sha256(blob).hexdigest() == manifest['samples_sha256']
        self.samples = json.loads(blob)
        semantic = json.dumps(self.samples, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        assert hashlib.sha256(semantic).hexdigest() == contract['portable_semantic_sha256']
        assert len(self.samples) == len({row['id'] for row in self.samples}) == 900
        for sample in self.samples:
            for image in sample['images']:
                expected = f"images/{sample['id']}_{image['index']}.png"
                assert image['path'] == expected
                image['path'] = str(self.root / expected)
        self.official = None

    def load_scorer(self):
        if self.official is not None:
            return
        if self.source is None:
            raise ValueError('A SGLang source checkout is required for scoring')
        folder = self.source / 'benchmark/mmmu'
        expected = {
            'eval_utils.py': '6937ca1a09dad6ef272a677bd30f5999a3c67c9dbe610d21f9da9c34d1d8aa1f',
            'data_utils.py': '2aa37d9848386b65972f67106a37f38cfa8438dd8a335336886f48abf74f5900',
        }
        for name, checksum in expected.items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == checksum, name
        spec = importlib.util.spec_from_file_location('portable_pinned_mmmu_scorer', folder / 'eval_utils.py')
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(folder))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        self.official = module

    def grade(self, sample, response):
        self.load_scorer()
        return super().grade(sample, response)

    def request(self, s, gpu, arm):
        # Same method as the measured evaluator, with only the endpoint replaced.
        body = self.body(s)
        result = dict(id=s['id'], subject=s['subject'], image_count=len(s['images']), gold=s['answer'],
                      question_type=s['question_type'], gpu=gpu, arm=arm,
                      request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                      started_unix_s=time.time())
        start = time.monotonic()
        try:
            response = requests.post(self.urls[gpu], json=body, timeout=(10, 900))
            result['http_status'] = response.status_code
            if not response.ok:
                result['error_response'] = response.text
            response.raise_for_status()
            data = response.json(); choice = data['choices'][0]; message = choice['message']
            answer = message.get('content') or ''
            result.update(text=answer, message=message, usage=data.get('usage'),
                          finish_reason=choice.get('finish_reason'), error=None if answer else 'empty final answer')
        except Exception as error:
            result.update(text='', error=f'{type(error).__name__}: {error}', finish_reason=None)
        result.update(latency_s=time.monotonic()-start, finished_unix_s=time.time())
        return s, result
