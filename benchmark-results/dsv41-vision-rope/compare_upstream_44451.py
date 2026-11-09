import torch,triton,importlib.util,json,random,pathlib
from sglang.kernels.ops.mm.dsv41_vit_rope import apply_vision_rotary as upstream
import argparse
p=argparse.ArgumentParser();p.add_argument('--candidate-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
s=importlib.util.spec_from_file_location('candidate',pathlib.Path(args.candidate_root)/'python/sglang/srt/models/dsv41_vision_rope.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def reference(q,k,c,s):
 def f(x):
  a,b=x.float().chunk(2,-1);return torch.cat((a*c-b*s,b*c+a*s),-1).to(x.dtype)
 return f(q),f(k)
rows=[]
for dtype in (torch.bfloat16,torch.float16):
 for strided in (False,True):
  torch.manual_seed(42);x=torch.randn(1521,3*16*64,device='cuda',dtype=dtype);q=x[:,:1024].view(1521,16,64);k=x[:,1024:2048].view_as(q)
  if not strided:q=q.contiguous();k=k.contiguous()
  q0=q.clone();k0=k.clone();a=torch.randn(1521,1,32,device='cuda');c,s=a.cos(),a.sin();expected=reference(q,k,c,s)
  for name,fn in [('upstream',lambda:upstream(q,k,(c,s))),('candidate',lambda:m.apply_fused_vision_rope(q,k,c,s))]:
   q.copy_(q0);k.copy_(k0);out=fn();torch.cuda.synchronize();row={'dtype':str(dtype),'strided':strided,'implementation':name,'diff_elements':sum((o.view(torch.int16)!=r.view(torch.int16)).sum().item() for o,r in zip(out,expected)),'max_abs_error':max((o.float()-r.float()).abs().max().item() for o,r in zip(out,expected)),'mutates_inputs':not(torch.equal(q,q0) and torch.equal(k,k0))};rows.append(row);print(row,flush=True)
  if not strided:
   funcs={'reference':lambda:reference(q,k,c,s),'upstream':lambda:upstream(q,k,(c,s)),'candidate':lambda:m.apply_fused_vision_rope(q,k,c,s)}
   for rep in range(3):
    order=list(funcs);random.Random(rep).shuffle(order)
    for name in order:
     def step():q.copy_(q0);k.copy_(k0);return funcs[name]()
     ms=triton.testing.do_bench_cudagraph(step,rep=150)
     row={'benchmark':'Q/K operator plus identical two input resets','dtype':str(dtype),'rep':rep,'implementation':name,'us':ms*1000};rows.append(row);print(row,flush=True)
pathlib.Path(args.output).write_text(json.dumps(rows,indent=2))
