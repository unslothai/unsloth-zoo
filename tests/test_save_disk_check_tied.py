import torch
def test_disk_check_does_not_refuse_a_tied_save_that_fits(tmp_path, monkeypatch):
    import shutil
    import transformers
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo import saving_utils

    cfg = transformers.Qwen2Config(
        vocab_size = 8192, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 1,
        num_attention_heads = 4, num_key_value_heads = 2, tie_word_embeddings = True,
    )
    model = transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.tie_weights()
    model = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"]))
    emb = model.base_model.model.model.embed_tokens.weight
    weights = sum(p.nbytes for p in model.base_model.model.state_dict().values()) - emb.nbytes
    free = int((weights + emb.nbytes // 2) / 0.95)  # room for the save, not for a second embedding
    monkeypatch.setattr(shutil, "disk_usage", lambda path: (2 * free, free, free))
    out = saving_utils.prepare_saving(model, str(tmp_path), output_dtype = torch.bfloat16)
    assert out[8] < free
