"""Bounded concurrent VLM load; record per-request latency and sampled GPU usage."""
import argparse, base64, concurrent.futures, io, json, statistics, subprocess, threading, time
from PIL import Image, ImageDraw, ImageOps
import requests

p=argparse.ArgumentParser()
p.add_argument("--image-dir", required=True)
p.add_argument("--model", default="deepseek-v41-flash")
p.add_argument("--ports",default="18990")
p.add_argument("--workers-per-port",type=int,default=8)
p.add_argument("--warmup",type=int,default=15)
p.add_argument("--duration",type=int,default=45)
p.add_argument("--size",type=int,default=448)
p.add_argument("--images-per-request",type=int,default=8)
p.add_argument("--max-tokens",type=int,default=1)
a=p.parse_args(); ports=[int(x) for x in a.ports.split(",")]
imgs=[]
files=[a.image_dir.rstrip("/")+"/"+name for name in ("chelsea.png","coffee.png","rocket.jpg","astronaut.png","hubble_deep_field.jpg","motorcycle_left.png","motorcycle_right.png","camera.png","clock_motion.png","coins.png","grass.png","gravel.png","ihc.png","cell.png","page.png","moon.png")]
for i in range(16):
 path=files[i]; im=Image.open(path).convert("RGB"); im=ImageOps.fit(im,(a.size,a.size)); buf=io.BytesIO();im.save(buf,"PNG");imgs.append("data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode())
start=time.monotonic(); measure_start=start+a.warmup; end=measure_start+a.duration; observations=[]; all_observations=[]; mutex=threading.Lock(); gpu_samples=[]

def worker(port,worker_id):
 session=requests.Session(); results=[]; all_results=[]; count=0
 while time.monotonic()<end:
  image_id=(count+worker_id)%len(imgs)
  content=[{"type":"text","text":"Describe these images."}]+[{"type":"image_url","image_url":{"url":imgs[(image_id+k)%len(imgs)]}} for k in range(a.images_per_request)]
  body={"model":a.model,"messages":[{"role":"user","content":content}],"temperature":0,"max_tokens":a.max_tokens,"stream":False}
  t=time.monotonic()
  try:
   response=session.post(f"http://127.0.0.1:{port}/v1/chat/completions",json=body,timeout=120); response.raise_for_status(); data=response.json(); message=data["choices"][0]["message"]; answer=message.get("content") or message.get("reasoning_content") or ""; tokens=data.get("usage",{}).get("completion_tokens"); error=None if answer else "empty output"
  except Exception as e: answer=None; tokens=None;error=f"{type(e).__name__}: {str(e)[:150]}"
  finish=time.monotonic(); count+=1
  entry={"port":port,"image_id":image_id,"latency_ms":(finish-t)*1000,"start_offset_s":t-start,"finish_offset_s":finish-start,"completion_tokens":tokens,"answer":answer,"error":error}
  all_results.append(entry)
  if measure_start<=finish<=end: results.append(entry)
 with mutex:
  observations.extend(results)
  all_observations.extend(all_results)

def monitor():
 while time.monotonic()<end:
  try:
   text=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True,timeout=3)
   if time.monotonic()>=measure_start:
    for line in text.splitlines():
     idx,util,mem=map(int,line.split(","))
     if idx in (0,1,2,3,4,5,6,7):gpu_samples.append({"gpu":idx,"util":util,"memory_mib":mem})
  except Exception: pass
  time.sleep(1)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)*a.workers_per_port) as pool:
 futures=[pool.submit(worker,port,j) for port in ports for j in range(a.workers_per_port)]
 for f in futures:f.result()
thread.join(timeout=5)
ok=[v for v in observations if not v["error"]]; lat=sorted(v["latency_ms"] for v in ok)
result={"images_per_request":a.images_per_request,"max_tokens":a.max_tokens,"ports":ports,"workers_per_port":a.workers_per_port,"warmup_s":a.warmup,"measurement_s":a.duration,"successful":len(ok),"failed":len(observations)-len(ok),"success_rps":len(ok)/a.duration,"latency_p50_ms":statistics.median(lat) if lat else None,"latency_p95_ms":lat[min(len(lat)-1,int(0.95*(len(lat)-1)))] if lat else None,"gpu_util_mean":{str(i):statistics.mean(x["util"] for x in gpu_samples if x["gpu"]==i) for i in (0,1,2,3,4,5,6,7) if any(x["gpu"]==i for x in gpu_samples)},"gpu_samples":gpu_samples,"requests":observations,"all_requests":all_observations,"all_failures":sum(1 for x in all_observations if x["error"]),"finished_after_window":sum(1 for x in all_observations if x["finish_offset_s"]>a.warmup+a.duration)}
print(json.dumps(result,indent=2))
