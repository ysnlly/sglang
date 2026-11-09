import importlib.util,torch
import argparse,pathlib
p=argparse.ArgumentParser();p.add_argument('--sglang-root',required=True);a=p.parse_args()
path=str(pathlib.Path(a.sglang_root)/'python/sglang/srt/models/dsv41_vision_rope.py')
s=importlib.util.spec_from_file_location('isolated_rope',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def ref(x,c,s):
 a,b=x.float().chunk(2,-1);return torch.cat((a*c-b*s,b*c+a*s),-1).to(x.dtype)
n=0
for dtype in (torch.bfloat16,torch.float16):
 for tokens,heads in ((1,1),(37,7),(64,8),(256,15),(1521,16),(128,17),(4096,16)):
  for strided in (False,True):
   x=torch.randn(tokens,3*heads*64,device='cuda',dtype=dtype);q=x[:,:heads*64].view(tokens,heads,64);k=x[:,heads*64:2*heads*64].view_as(q)
   if not strided:q=q.contiguous();k=k.contiguous()
   angles=torch.randn(tokens,1,32,device='cuda');c,s=angles.cos(),angles.sin();u,v=m.apply_fused_vision_rope(q,k,c,s)
   assert torch.equal(u.view(torch.int16),ref(q,c,s).view(torch.int16));assert torch.equal(v.view(torch.int16),ref(k,c,s).view(torch.int16));n+=1
 torch.cuda.synchronize()
print('ISOLATED_KERNEL_CHECKS_PASSED',n,flush=True)
