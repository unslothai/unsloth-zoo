
## PRs Created

1. **unslothai/unsloth#5055** - Add Gemma-4 to FORCE_FLOAT32 list
   - `unsloth/models/loader.py`: adds `gemma4,` and `gemma4text` to FORCE_FLOAT32
   - Primary fix: auto-switches fp16 users to bfloat16 + float32 mixed precision

2. **unslothai/unsloth-zoo#598** - Add per-model logit_matmul_upcast toggle
   - `unsloth_zoo/rl_replacements.py`: threads `logit_matmul_upcast: bool` through the GRPO log-softmax path
   - Auto-detects Gemma-3/3n/4 and upcasts the hidden_states @ lm_head matmul to float32
   - Extensible: add model types to the inlined `_upcast_models` set
   - Zero overhead for non-Gemma models (default False)

## User direction
- Closed PR #5055 (FORCE_FLOAT32 for gemma4) -- too blunt
- User wants a new list in loader.py (like FORCE_FLOAT32 but for RL logit upcast) that sets an env var, and the zoo reads it
