"""Unsloth FastLanguageModel LoRA steps on a tiny MoE with the router trainable. One arm per
process (UNSLOTH_MOE_FUSED_ROUTER set by the caller before import). Usage: REPO OUT.json"""

import json
import os
import sys

os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", os.path.join(os.environ.get("TMPDIR", "/tmp"), "k2_compiled_cache"))
import unsloth  # noqa: F401
from unsloth import FastLanguageModel
import torch
from unsloth_zoo.temporary_patches import moe_router_triton as mr

repo, out_path = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
model, tok = FastLanguageModel.from_pretrained(repo, max_seq_length = 128, load_in_4bit = False, dtype = torch.bfloat16)
model = FastLanguageModel.get_peft_model(
    model, r = 8, lora_alpha = 16, random_state = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
)
routers = []
for n, m in model.named_modules():
    if type(m).__name__.endswith(("TopKRouter", "TopkRouter")):
        m.weight.requires_grad_(True)
        f = type(m).forward
        routers.append([n, type(m).__module__, getattr(f, "__module__", None), getattr(f, "__qualname__", None),
                        bool(getattr(f, "_unsloth_fused_router", False))])
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr = 0.05)
g = torch.Generator().manual_seed(1)
model.train()
losses, grads0 = [], {}
mr.ROUTER_CALLS["fused"] = 0
for step in range(4):
    ids = torch.randint(0, model.config.vocab_size, (2, 48), generator = g).cuda()
    out = model(input_ids = ids, labels = ids, output_router_logits = True)
    losses.append(out.loss.item())
    opt.zero_grad(set_to_none = True)
    out.loss.backward()
    if step == 0:
        grads0 = {n: p.grad.float().cpu().tolist() for n, p in model.named_parameters()
                  if p.grad is not None and (".gate." in n or "router" in n)}
    opt.step()
res = {"repo": repo, "fused_env": os.environ.get("UNSLOTH_MOE_FUSED_ROUTER"), "losses": losses,
       "router_modules": routers, "router_forms": dict(mr._SUPPORTED), "engaged": mr.ROUTER_CALLS["fused"],
       "grads0": grads0}
json.dump(res, open(out_path, "w"))
print(json.dumps({k: v for k, v in res.items() if k != "grads0"}))
