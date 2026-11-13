import torch

from unsloth_zoo.saving_utils import _save_size_in_bytes, get_torch_storage_size_new


def test_a_tied_embedding_counts_once():
    w = torch.zeros(4096, 1024, dtype = torch.bfloat16)
    sd = {"model.embed_tokens.weight": w, "lm_head.weight": w, "norm.weight": torch.zeros(1024, dtype = torch.bfloat16)}
    assert _save_size_in_bytes(sd, 2) == w.nbytes + 1024 * 2


def test_views_of_one_buffer_each_count():
    arena = torch.zeros(3 * 1024, 1024, dtype = torch.bfloat16)
    q, k, v = arena.chunk(3, 0)
    sd = {"q": q, "k": k, "v": v}
    assert _save_size_in_bytes(sd, 2) == sum(get_torch_storage_size_new(x, 2) for x in sd.values())


def test_meta_tensors_are_never_merged():
    a, b = torch.empty(64, 64, device = "meta"), torch.empty(64, 64, device = "meta")
    assert _save_size_in_bytes({"a": a, "b": b}, 2) >= 2 * 64 * 64
