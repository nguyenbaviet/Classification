import os
import torch

ckpt_path = "vitstr_small_patch16_224_aug.pth"

ckpt = torch.load(ckpt_path, "cpu")
new_ckpt = {"model": {}}

for k, v in ckpt.items():
    new_k = k.replace("module.vitstr.", "")

    if new_k.startswith("head."):
        new_k = new_k.replace("head.", "cls_head.")
        pad = 105 - v.shape[0]
        new_ckpt["model"][new_k] = torch.cat([v, v[-pad:]], 0)
        print(new_ckpt["model"][new_k].shape)
    elif new_k.startswith("patch_embed.proj.weight"):
        new_k = "backbone." + new_k
        new_ckpt["model"][new_k] = v.expand(-1, 3, -1, -1)
        print(new_ckpt["model"][new_k].shape)
    else:
        new_k = "backbone." + new_k
        new_ckpt["model"][new_k] = v

os.makedirs("pretrained/", exist_ok=True)
torch.save(new_ckpt, f"pretrained/{ckpt_path}")
