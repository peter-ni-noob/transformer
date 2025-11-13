import torch
import re
import numpy as np
import os
import shutil
from typing import Union
import collections

tmp_dir = "./"


def save_npy(tmp_dir0: str,basename: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir0, basename), x)

def model_to_npy(md):
    path=os.path.join(tmp_dir,"P_weightXL")
    os.makedirs(path,exist_ok=True)
    md=md["model_state_dict"]

    for name,value in md.items():
        # print(name)
        if name.find("attention")!=-1:
            # print(name)
            if name.find("attention_mask")!=-1:
                 save_npy(path,"attention_mask.npy",value)
            ans=re.compile(r".*\.(\d+)\..*\.(weight|bias)").search(name)
            if ans is not None:
                newName="attn"+ans.group(1)+"out_proj_"+ans.group(2)
                save_npy(path,newName,value)
                continue
            ans=re.compile(r".*\.(\d+)\..*\.in_proj_(weight|bias)").search(name)
            if ans is not None:
                newName="attn"+ans.group(1)+"in_proj_"+ans.group(2)
                save_npy(path,newName,value)
                continue
        elif name.find("mlp")!=-1:
            ans=re.compile(r".*\.(\d+)\..*c_fc\.(weight|bias)").search(name)
            if ans is not None:
                newName="c_fcffw"+ans.group(1)+"_"+ans.group(2)
                save_npy(path,newName,value)
                continue
            ans=re.compile(r".*\.(\d+)\..*c_proj\.(weight|bias)").search(name)
            if ans is not None:
                newName="c_projffw"+ans.group(1)+"_"+ans.group(2)
                save_npy(path,newName,value)
                continue
        elif re.compile(r".*ln_[\d]+.*").search(name) is not None:
            ans=re.compile(r".*\.(\d+)\.ln_1\.(weight|bias)").search(name)
            if ans is not None:
                smap={"weight":"gamma","bias":"beta"}
                newName="ln1"+ans.group(1)+"_"+smap[ans.group(2)]
                save_npy(path,newName,value)
                continue
            ans=re.compile(r".*\.(\d+)\.ln_2\.(weight|bias)").search(name)
            if ans is not None:
                smap={"weight":"gamma","bias":"beta"}
                newName="ln2"+ans.group(1)+"_"+smap[ans.group(2)]
                save_npy(path,newName,value)
                continue
        elif name.find("transformer.wte.weight")!=-1:
                newName="wte_embedding"
                save_npy(path,newName,value)
                continue
        elif name.find("transformer.wpe.weight")!=-1:
                newName="wpe_embedding"
                save_npy(path,newName,value)
                continue
        elif re.compile(r".*ln_f.*").search(name) is not None:
                ans=re.compile(r".*ln_f\.(weight|bias)").search(name)
                if ans is not None:
                    smap={"weight":"gamma","bias":"beta"}
                    newName="ln_final_"+smap[ans.group(1)]
                    save_npy(path,newName,value)
                    continue

def _pfilename(name):
    smap={"gamma":"weight","beta":"bias"}
    if re.compile(r"attn(\d+)in_proj_(bias|weight)").search(name) is not None:
         ans=re.compile(r"attn(\d+)in_proj_(bias|weight)").search(name)
         newname="transformer.h."+ans.group(1)+".attention.attn.in_proj_"+ans.group(2)
         return newname
    if re.compile(r"attn(\d+)out_proj_(bias|weight)").search(name) is not None:
         ans=re.compile(r"attn(\d+)out_proj_(bias|weight)").search(name)
         newname="transformer.h."+ans.group(1)+".attention.attn.out_proj."+ans.group(2)
         return newname
    if re.compile(r"c_fcffw(\d+)_(bias|weight).npy").search(name) is not None:
         ans=re.compile(r"c_fcffw(\d+)_(bias|weight).npy").search(name)
         newname="transformer.h."+ans.group(1)+".mlp.c_fc."+ans.group(2)
         return newname
    if re.compile(r"c_projffw(\d+)_(bias|weight).npy").search(name) is not None:
         ans=re.compile(r"c_projffw(\d+)_(bias|weight).npy").search(name)
         newname="transformer.h."+ans.group(1)+".mlp.c_proj."+ans.group(2)
         return newname
    if re.compile(r"ln1(\d+)_(beta|gamma)").search(name) is not None:
         ans=re.compile(r"ln1(\d+)_(beta|gamma)").search(name)
         newname="transformer.h."+ans.group(1)+".ln_1."+smap[ans.group(2)]
         return newname
    if re.compile(r"ln2(\d+)_(beta|gamma)").search(name) is not None:
         ans=re.compile(r"ln2(\d+)_(beta|gamma)").search(name)
         newname="transformer.h."+ans.group(1)+".ln_2."+smap[ans.group(2)]
         return newname
    if name.find("wpe_embedding")!=-1:
         newname="transformer.wpe.weight"
         return newname
    if name.find("wte_embedding")!=-1:
         newname="transformer.wte.weight"
         return newname
    if name.find("fclogit_weight")!=-1:
         newname="lm_head.weight"
         return newname
    if re.compile(r"ln_final_(beta|gamma)").search(name) is not None:
         ans=re.compile(r"ln_final_(beta|gamma)").search(name)
         newname="transformer.ln_f."+smap[ans.group(1)]
         return newname 
    return "error"

def pfilename(name):
     return _pfilename(name)+".npy"

def npy_to_model(path,newname):
    dic=collections.OrderedDict()
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            data=np.load(filepath)
            data=torch.from_numpy(data)
            new_name=pfilename(filename)
            if new_name.find("error")!=-1:
                 continue
            new_name=re.sub(r".npy","",new_name)
            dic[new_name]=data
    ans={"model_state_dict":dic}
    torch.save(ans,newname)



npy_to_model(r"/data/Pweight/N1epc_weight/","/data/Pweight/1105N1epc.pth")





        
        # print(value)
# s="sdfs.10.dd5"
# ans=re.compile(r"[.]+").search(s)
# print(ans)


# model = torch.load("/root/workspace/nexus_gpt2/nexusnet/gpt2xl_weight0.pt")
# model_to_npy(model)
