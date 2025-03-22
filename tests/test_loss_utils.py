import pytest
import torch
import os
from unsloth_zoo.loss_utils import (
    _unsloth_get_batch_samples,
    patch_loss_functions,
    post_patch_loss_function,
)

class MockModel:
    def __init__(self):
        self.args = type('Args', (), {'average_tokens_across_devices': False})()
        self.model = None
        self.forward = lambda *args, **kwargs: None
        self.__class__.__name__ = "CausalLM"

def test_unsloth_get_batch_samples():
    model = MockModel()
    
    # Create mock batch samples
    batch_samples = [
        {
            "labels": torch.tensor([[1, 2, -100, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        },
        {
            "labels": torch.tensor([[1, -100, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
    ]
    
    # Create a mock iterator
    class MockIterator:
        def __init__(self, samples):
            self.samples = samples
            self.current = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.current >= len(self.samples):
                raise StopIteration
            sample = self.samples[self.current]
            self.current += 1
            return sample

    iterator = MockIterator(batch_samples)
    
    # Test the function
    result_samples, num_items = _unsloth_get_batch_samples(model, iterator, 2)
    
    # Verify results
    assert len(result_samples) == 2
    assert isinstance(num_items, int)
    assert num_items == 5  # Total number of non-padding tokens in labels

def test_patch_loss_functions():
    # Test that patching doesn't raise errors
    patch_loss_functions(lambda **kwargs: 0.0)
    assert os.environ.get("UNSLOTH_PATCHED") == "1"

if __name__ == "__main__":
    pytest.main([__file__])
