"""Frozen MMMU requests and serial scoring; no random-guess primary credit."""
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import requests


class Evaluator:
    def __init__(self, data_root, source, max_tokens=1024):
        assert isinstance(max_tokens,int) and max_tokens>0
        self.max_tokens=max_tokens
        self.root = Path(data_root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        assert manifest['dataset']=='MMMU/MMMU' and manifest['split']=='validation'
        assert manifest['revision']=='876ce5cb130f7f7e290ce4d9984357737d4db5cf'
        assert manifest['samples_sha256']=='1e4294801f379fa2071c21631a79d371d089dd1db6cf0a7c47586a3875ac0faf'
        blob = (self.root/'samples.json').read_bytes()
        assert hashlib.sha256(blob).hexdigest() == manifest['samples_sha256']
        self.samples = json.loads(blob)
        assert len(self.samples) == len({s['id'] for s in self.samples}) == 900
        scorer = Path(source)/'benchmark/mmmu/eval_utils.py'
        assert hashlib.sha256(scorer.read_bytes()).hexdigest() == '6937ca1a09dad6ef272a677bd30f5999a3c67c9dbe610d21f9da9c34d1d8aa1f'
        spec = importlib.util.spec_from_file_location('frozen_mmmu_scorer',scorer)
        self.official = importlib.util.module_from_spec(spec)
        sys.path.insert(0,str(scorer.parent))
        try:
            spec.loader.exec_module(self.official)
        finally:
            sys.path.pop(0)

    def body(self,s):
        content=[]
        for im in s['images']:
            blob=Path(im['path']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==im['sha256']
            content += [{'type':'text','text':f"Image {im['index']}:"},
                        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(blob).decode()}}]
        question=s['question']
        if s['question_type']=='multiple-choice':
            question+='\n\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(s['options']))
            question+="\n\nAnswer with the option's letter from the given choices directly."
        else:
            question+='\n\nAnswer the question using a single word or phrase.'
        content.append({'type':'text','text':question})
        return dict(model='qwen35-9b',messages=[{'role':'user','content':content}],temperature=0,
                    seed=42,max_tokens=self.max_tokens,stream=False,chat_template_kwargs={'enable_thinking':False})

    def request(self,s,gpu,arm):
        body=self.body(s)
        result=dict(id=s['id'],subject=s['subject'],image_count=len(s['images']),gold=s['answer'],
                    question_type=s['question_type'],gpu=gpu,arm=arm,
                    request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
                    started_unix_s=time.time())
        start=time.monotonic()
        try:
            response=requests.post(f'http://127.0.0.1:{19400+gpu}/v1/chat/completions',json=body,timeout=(10,900))
            result['http_status']=response.status_code
            if not response.ok:result['error_response']=response.text
            response.raise_for_status()
            data=response.json();choice=data['choices'][0];message=choice['message']
            answer=message.get('content') or ''
            result.update(text=answer,message=message,usage=data.get('usage'),
                          finish_reason=choice.get('finish_reason'),error=None if answer else 'empty final answer')
        except Exception as error:
            result.update(text='',error=f'{type(error).__name__}: {error}',finish_reason=None)
        result.update(latency_s=time.monotonic()-start,finished_unix_s=time.time())
        return s,result

    def grade(self,s,r):
        r.update(correct=False,parse_fallback=False)
        if r['error']:
            return r
        if s['question_type']=='multiple-choice':
            choices=[chr(65+i) for i in range(len(s['options']))]
            def fallback(options):
                r['parse_fallback']=True
                return random.Random(s['id']).choice(options)
            with patch.object(self.official.random,'choice',fallback):
                pred=self.official.parse_multi_choice_response(r['text'],choices,dict(zip(choices,s['options'])))
            r.update(predicted=pred,upstream_correct_with_seeded_fallback=bool(self.official.eval_multi_choice(s['answer'],pred)))
            r['correct']=r['upstream_correct_with_seeded_fallback'] and not r['parse_fallback']
        else:
            pred=self.official.parse_open_response(r['text'])
            r.update(predicted=pred,correct=bool(self.official.eval_open(s['answer'],pred)))
        return r

    def warmup(self,gpus,output,observe_truncation=False):
        samples=[next(s for s in self.samples if len(s['images'])==1),next(s for s in self.samples if len(s['images'])>1)]
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(self.request,s,gpu,'warmup') for gpu in gpus for s in samples]
            for future in concurrent.futures.as_completed(futures):
                s,r=future.result();rows.append(self.grade(s,r))
        Path(output).write_text(json.dumps(rows,indent=2))
        gate=dict(response_errors=sum(bool(r['error']) for r in rows),
                  truncated=sum(r['finish_reason']=='length' for r in rows),
                  observe_truncation=observe_truncation,
                  qualification_passed=not any(r['error'] or r['finish_reason']=='length' for r in rows))
        Path(output).with_name('warmup_gate.json').write_text(json.dumps(gate,indent=2))
        assert not gate['response_errors'],'warmup response error; see warmup.json'
        assert observe_truncation or not gate['truncated'],'warmup truncation; see warmup_gate.json'
        assert all((r.get('usage') or {}).get('prompt_tokens',0)>0 for r in rows)
        return gate

    def run(self,round_number,output):
        output=Path(output)
        handles={arm:open(output/f'{arm}.jsonl','x') for arm in ['default','split']}
        pools={gpu:concurrent.futures.ThreadPoolExecutor(max_workers=4) for gpu in range(8)}
        rows={arm:[] for arm in handles}
        try:
            futures=[]
            for index,sample in enumerate(self.samples):
                for arm in ['default','split']:
                    parity=(round_number%2) if arm=='default' else 1-(round_number%2)
                    gpu=2*(index%4)+parity
                    futures.append(pools[gpu].submit(self.request,sample,gpu,arm))
            for future in concurrent.futures.as_completed(futures):
                sample,r=future.result();r=self.grade(sample,r);arm=r['arm'];rows[arm].append(r)
                handles[arm].write(json.dumps(r,ensure_ascii=False)+'\n');handles[arm].flush()
                if len(rows[arm])%50==0 or r['error']:
                    print('ROUND',round_number,arm,len(rows[arm]),'correct',sum(x['correct'] for x in rows[arm]),
                          'errors',sum(bool(x['error']) for x in rows[arm]),flush=True)
        finally:
            for pool in pools.values():pool.shutdown(wait=True,cancel_futures=True)
            for f in handles.values():f.close()
        summary={}
        for arm,results in rows.items():
            assert len(results)==len({r['id'] for r in results})==900
            summary[arm]=dict(samples=len(results),correct=sum(r['correct'] for r in results),
                              errors=sum(bool(r['error']) for r in results),
                              truncated=sum(r['finish_reason']=='length' for r in results),
                              parse_fallback=sum(r['parse_fallback'] for r in results),
                              multi_total=sum(r['image_count']>1 for r in results),
                              multi_correct=sum(r['correct'] for r in results if r['image_count']>1))
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        assert all(s['errors']==0 for s in summary.values()),'request errors; round is not an accuracy pass'
        return summary
