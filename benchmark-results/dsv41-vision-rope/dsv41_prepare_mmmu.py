import concurrent.futures,hashlib,json,pathlib,time,ast,re,io
import requests,pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download
import argparse
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args()
out=pathlib.Path(a.output_dir).resolve();out.mkdir(exist_ok=True,parents=True)
rev='876ce5cb130f7f7e290ce4d9984357737d4db5cf'
r=requests.get(f'https://huggingface.co/api/datasets/MMMU/MMMU/revision/{rev}',timeout=30);r.raise_for_status()
files=sorted(x['rfilename'] for x in r.json()['siblings'] if '/validation-' in x['rfilename'] and x['rfilename'].endswith('.parquet'));assert len(files)==30,len(files)
def download(name):
 path=hf_hub_download('MMMU/MMMU',name,repo_type='dataset',revision=rev,cache_dir=str(out/'hf-cache'))
 print('downloaded',name,flush=True);return name,path
rows=[];sources={}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for name,path in pool.map(download,files):
  sources[name]=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
  for row in pq.read_table(path).to_pylist():
   images=[]
   for n in range(1,8):
    im=row.get(f'image_{n}')
    if im is None:continue
    assert isinstance(im,dict) and im.get('bytes'),(row['id'],n)
    folder=out/'images';folder.mkdir(exist_ok=True)
    target=folder/f"{row['id']}_{n}.png"
    Image.open(io.BytesIO(im['bytes'])).convert('RGB').save(target)
    images.append({'index':n,'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
   assert images,row['id']
   opts=ast.literal_eval(row['options']) if isinstance(row['options'],str) else row['options']
   assert isinstance(opts,list)
   rows.append({'id':row['id'],'subject':name.split('/')[0],'question':row['question'],'options':opts,'answer':row['answer'],'question_type':row['question_type'],'images':images})
assert len(rows)==900 and len({x['id'] for x in rows})==900,len(rows)
rows.sort(key=lambda x:x['id']); manifest=out/'samples.json';manifest.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
summary={'dataset':'MMMU/MMMU','revision':rev,'split':'validation','samples':len(rows),'subjects':30,'multi_image_samples':sum(len(x['images'])>1 for x in rows),'samples_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'parquet_sha256':sources}
(out/'manifest.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
