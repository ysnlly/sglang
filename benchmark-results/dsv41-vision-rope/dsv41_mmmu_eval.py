import argparse,base64,collections,concurrent.futures,hashlib,json,pathlib,random,sys,time
from unittest.mock import patch
import requests
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--data-root',required=True);p.add_argument('--sglang-root',required=True);p.add_argument('--base-url',default='http://127.0.0.1:18990');p.add_argument('--model',default='deepseek-v41-flash');a=p.parse_args()
root=pathlib.Path(a.data_root)
sys.path.insert(0,str(pathlib.Path(a.sglang_root)/'benchmark/mmmu'))
import eval_utils as official
samples=json.loads((root/'samples.json').read_text());assert len(samples)==900
if a.smoke:
 samples=[next(x for x in samples if len(x['images'])==1),next(x for x in samples if len(x['images'])>1)]
output=root/f'{a.arm}.jsonl';assert not output.exists(),f'Would overwrite {output}'
def make_request(s):
 # Include every image exactly once, explicitly labeled to match the question's <image N> references.
 content=[]
 for im in s['images']:
  blob=pathlib.Path(im['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==im['sha256']
  content.extend([{'type':'text','text':f"Image {im['index']}:"},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(blob).decode()}}])
 question=s['question']
 if s['question_type']=='multiple-choice':
  question+='\n\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(s['options']))
  question+="\n\nAnswer with the option's letter from the given choices directly."
 else:question+='\n\nAnswer the question using a single word or phrase.'
 content.append({'type':'text','text':question})
 body={'model':a.model,'messages':[{'role':'user','content':content}],'temperature':0,'seed':42,'max_tokens':1024,'stream':False}
 t=time.monotonic();result={'id':s['id'],'subject':s['subject'],'image_count':len(s['images']),'gold':s['answer'],'question_type':s['question_type'],'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
 try:
  r=requests.post(a.base_url.rstrip('/')+'/v1/chat/completions',json=body,timeout=(10,300));r.raise_for_status();d=r.json();choice=d['choices'][0];msg=choice['message'];answer=msg.get('content') or msg.get('reasoning_content') or ''
  result.update(text=answer,usage=d.get('usage'),finish_reason=choice.get('finish_reason'),error=None if answer else 'empty output')
 except Exception as ex:result.update(text='',error=f'{type(ex).__name__}: {ex}',finish_reason=None)
 result['latency_s']=time.monotonic()-t;return s,result
def grade(s,r):
 r['correct']=False;r['parse_fallback']=False
 if r['error']:return r
 if s['question_type']=='multiple-choice':
  choices=[chr(65+i) for i in range(len(s['options']))];mapping=dict(zip(choices,s['options']))
  # Upstream parser randomly guesses when extraction fails. Track that explicitly;
  # never award random-guess credit in the primary score.
  def fallback(opts):r['parse_fallback']=True;return random.Random(s['id']).choice(opts)
  with patch.object(official.random,'choice',fallback):pred=official.parse_multi_choice_response(r['text'],choices,mapping)
  r['predicted']=pred;r['upstream_correct_with_seeded_fallback']=bool(official.eval_multi_choice(s['answer'],pred));r['correct']=r['upstream_correct_with_seeded_fallback'] and not r['parse_fallback']
 else:
  pred=official.parse_open_response(r['text']);r['predicted']=pred;r['correct']=bool(official.eval_open(s['answer'],pred))
 return r
results=[]
with output.open('x') as f, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for future in concurrent.futures.as_completed([pool.submit(make_request,s) for s in samples]):
  s,r=future.result();r=grade(s,r);results.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  if len(results)%25==0 or r['error']:print(a.arm,len(results),'correct',sum(v['correct'] for v in results),'errors',sum(bool(v['error']) for v in results),flush=True)
summary={'arm':a.arm,'samples':len(results),'errors':sum(bool(x['error']) for x in results),'truncated':sum(x['finish_reason']=='length' for x in results),'parse_fallback':sum(x['parse_fallback'] for x in results),'correct':sum(x['correct'] for x in results),'groups':{}}
for key,predicate in [('single',lambda x:x['image_count']==1),('multi',lambda x:x['image_count']>1)]:
 xs=[x for x in results if predicate(x)];summary['groups'][key]={'total':len(xs),'correct':sum(x['correct'] for x in xs)}
(root/f'{a.arm}_summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
if summary['errors']:sys.exit(2)
