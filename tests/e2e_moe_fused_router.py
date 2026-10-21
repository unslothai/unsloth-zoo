"""Tiny MoE LoRA steps with the router trainable, UNSLOTH_MOE_FUSED_ROUTER off vs on.
Prints one JSON line per model: per-step loss (+aux), router / LoRA grad deltas, engagement."""

import copy
import json
import os
import sys

os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = "1"   # install the class patch; arms toggle it per call
import torch
from unsloth_zoo.temporary_patches import moe_router_triton as mr
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

mr.patch_moe_fused_router()

MODELS = [
    ("trl-internal-testing/tiny-Qwen3MoeForCausalLM", {}, True),
    ("trl-internal-testing/tiny-Glm4MoeForCausalLM", {"first_k_dense_replace": 0}, False),
    ("trl-internal-testing/tiny-DeepseekV3ForCausalLM", {"first_k_dense_replace": 0}, False),
    ("trl-internal-testing/tiny-Qwen3MoeForCausalLM", {"num_experts": 60, "num_experts_per_tok": 8}, False),
]
STEPS = 4
dtype = getattr(torch, sys.argv[1]) if len(sys.argv) > 1 else torch.bfloat16


def build(repo, over, pretrained):
    torch.manual_seed(0)
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype = dtype)
    else:
        cfg = AutoConfig.from_pretrained(repo)
        for k, v in over.items():
            setattr(cfg, k, v)
        model = AutoModelForCausalLM.from_config(cfg, dtype = dtype)
        for m in model.modules():
            if hasattr(m, "e_score_correction_bias"):
                m.e_score_correction_bias.data = torch.randn_like(m.e_score_correction_bias.float()) * 0.05
            if type(m).__name__.endswith(("TopKRouter", "TopkRouter")):
                torch.nn.init.normal_(m.weight, std = 0.5)
    model = model.cuda()
    model = get_peft_model(model, LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]))
    routers = []
    for n, m in model.named_modules():
        if type(m).__name__.endswith(("TopKRouter", "TopkRouter")):
            m.weight.requires_grad_(True)
            routers.append((n, m))
    return model, routers


def run(model, routers, state, on):
    os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = "1" if on else "0"
    model.load_state_dict(state)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 0.05)
    mr.ROUTER_CALLS["fused"] = 0
    losses, grads0 = [], None
    g = torch.Generator().manual_seed(1)
    vocab = model.config.vocab_size
    aux_cfg = hasattr(model.config, "router_aux_loss_coef") and model.config.router_aux_loss_coef
    for step in range(STEPS):
        ids = torch.randint(0, vocab, (2, 48), generator = g).cuda()
        kw = {"output_router_logits": True} if aux_cfg else {}
        out = model(input_ids = ids, labels = ids, **kw)
        loss = out.loss
        losses.append([loss.item(), float(getattr(out, "aux_loss", 0.0) or 0.0)])
        opt.zero_grad(set_to_none = True)
        loss.backward()
        if step == 0:
            grads0 = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
        opt.step()
    return losses, grads0, mr.ROUTER_CALLS["fused"]


results = []
for repo, over, pretrained in MODELS:
    model, routers = build(repo, over, pretrained)
    forms = {type(m).__name__: mr._SUPPORTED.get(type(m).__name__) for _, m in routers}
    state = copy.deepcopy(model.state_dict())
    l0, g0, n0 = run(model, routers, state, False)
    l1, g1, n1 = run(model, routers, state, True)
    router_keys = [k for k in g0 if ".gate.weight" in k or "router" in k]
    lora_keys = [k for k in g0 if "lora_" in k]

    def rel(keys):
        num = sum((g1[k] - g0[k]).norm().item() ** 2 for k in keys) ** 0.5
        den = sum(g0[k].norm().item() ** 2 for k in keys) ** 0.5
        return num / max(den, 1e-30)
    r = {
        "model": repo, "override": over, "dtype": str(dtype), "router_forms": forms,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "loss_off": l0, "loss_on": l1,
        "max_abs_loss_delta": max(abs(a[0] - b[0]) for a, b in zip(l0, l1)),
        "router_grad_rel_delta_step0": rel(router_keys), "lora_grad_rel_delta_step0": rel(lora_keys),
        "router_grad_norm_step0": sum(g0[k].norm().item() ** 2 for k in router_keys) ** 0.5,
        "n_router_params_with_grad": len(router_keys),
        "engaged_off": n0, "engaged_on": n1,
    }
    results.append(r)
    print(json.dumps(r), flush = True)
    del model
    torch.cuda.empty_cache()
ok = all(r["engaged_off"] == 0 and r["engaged_on"] > 0 and r["router_grad_rel_delta_step0"] < 2e-2
         and r["max_abs_loss_delta"] < 2e-2 for r in results)
print("E2E_VERDICT", "PASS" if ok else "FAIL")
