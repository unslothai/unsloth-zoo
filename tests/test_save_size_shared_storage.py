import torch

from unsloth_zoo.saving_utils import get_torch_storage_size_new


def test_views_into_one_arena_count_their_own_bytes():
    arena = torch.empty(64 << 20, dtype = torch.uint8)
    views = [arena[i * 4096:(i + 1) * 4096].view(torch.float32) for i in range(8)]
    sizes = [get_torch_storage_size_new(v, 2) for v in views]
    assert sizes == [4096] * 8
    assert sum(sizes) < arena.nbytes


def test_plain_tensor_and_parameter():
    t = torch.zeros(3, 5, dtype = torch.bfloat16)
    assert get_torch_storage_size_new(t, 2) == 30
    assert get_torch_storage_size_new(torch.nn.Parameter(t), 2) == 30
