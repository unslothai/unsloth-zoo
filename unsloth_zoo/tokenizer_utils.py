# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import gc
import numpy as np
import itertools
import datasets
import re

__all__ = [
    "mean_of_trained_tokens",
    "add_new_tokens",
    "fix_untrained_tokens",
    "patch_tokenizer",
    "patch_processor_call",
]


@torch.inference_mode
def mean_of_trained_tokens(model, eps = 1e-16):
    """
    Llama-3 etc have untrained vectors (<|eot_id|>, <|start_header_id|>, ...)
    in the base model. Reset them to the mean of the trained tokens.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    embedding_matrix = model.get_input_embeddings ().weight.clone()
    lm_head_matrix   = model.get_output_embeddings().weight.clone()

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    # if n_untrained != 0:
    #     print(
    #         f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
    #         "We shall set them to the mean of the other trained tokens."
    #     )
    # pass

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head   = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Remove bad tokens
    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype = torch.float32, axis = 0)
    sum_lm_head   -= torch.sum(lm_head_matrix  [where_untrained], dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained)
    mean_lm_head   = (sum_lm_head   / n_trained)

    return mean_embedding, mean_lm_head
pass


def add_new_tokens(
    model,
    tokenizer,
    new_tokens = [],
    method = "mean",
    interpolation = 0.5,
):
    """
    Resize the tokenizer and add new tokens to the model, excluding untrained
    tokens from the mean calculation.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    assert(isinstance(new_tokens, (list, tuple)))
    assert(len(new_tokens) > 0)
    assert(method == "mean" or method == "interpolation")
    assert(interpolation >= 0 and interpolation <= 1)

    # Check if tokens already exist
    overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
    if len(overlapping_tokens) != 0:
        print(
            f"Unsloth: You're adding new_tokens = {new_tokens}\n"\
            f"There are tokens which are overlapping = {list(overlapping_tokens)}\n"\
            f"We shall safely ignore these overlapping tokens."
        )
        new_tokens = [x for x in new_tokens if x not in overlapping_tokens]
    pass

    # Get mean of trained tokens
    # mean_embedding, mean_lm_head = fix_untrained_tokens(model)

    # Weirdly be careful reserved tokens can pop out
    mean_embedding, mean_lm_head = mean_of_trained_tokens(model)
    mean_embedding = mean_embedding.to(torch.float32)
    mean_lm_head   = mean_lm_head  .to(torch.float32)

    # Get old lengths
    old_input_embedding  = model.get_input_embeddings ().weight
    old_output_embedding = model.get_output_embeddings().weight
    old_input_length  = old_input_embedding .shape[0]
    old_output_length = old_output_embedding.shape[0]
    old_config_size   = model.config.vocab_size

    # Check for tied weights as well
    is_tied = (old_input_embedding.data_ptr() == old_output_embedding.data_ptr()) \
        or (model.config.tie_word_embeddings)

    # Add tokens!
    old_length = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    new_vocab_length = len(tokenizer)
    # Also resizes lm_head as well!
    model.resize_token_embeddings(new_vocab_length)

    # If we use interpolation, we interpolate between the mean embeddings and
    # the Word2Vec sum of the other vectors
    embedding_matrix = model.get_input_embeddings ().weight
    lm_head_matrix   = model.get_output_embeddings().weight

    # Confirm sizes are correct
    if embedding_matrix.shape[0] != new_vocab_length:
        raise RuntimeError(
            "Unsloth: Embedding matrix size did not get resized properly. Please file a bug report!"
        )
    if lm_head_matrix.shape[0]   != new_vocab_length:
        raise RuntimeError(
            "Unsloth: LM Head matrix size did not get resized properly. Please file a bug report!"
        )
    if model.config.vocab_size   != new_vocab_length:
        raise RuntimeError(
            "Unsloth: Model's config vocab_size did not get resized properly. Please file a bug report!"
        )
    pass

    if method == "interpolation":
        print(
            "Unsloth: You are using interpolation to add new tokens.\n"\
            f"We shall set new tokens = mean(embeddings)*{1-interpolation} + mean(new_tokens)*{interpolation}"
        )
        for j, token in enumerate(new_tokens):
            input_ids = tokenizer(token, add_special_tokens = False).input_ids
            mean_embedding_token = embedding_matrix[input_ids].mean(axis = 0, dtype = torch.float32)
            mean_lm_head_token   = lm_head_matrix  [input_ids].mean(axis = 0, dtype = torch.float32)

            # Interpolate
            mean_embedding_token = mean_embedding*(1-interpolation) + mean_embedding_token*interpolation
            mean_lm_head_token   = mean_lm_head  *(1-interpolation) + mean_lm_head_token  *interpolation

            # Set the new vector
            with torch.no_grad():
                embedding_matrix[old_length+j] = mean_embedding_token
                lm_head_matrix  [old_length+j] = mean_lm_head_token
        pass
    else:
        # Now set the new tokens to the mean!
        with torch.no_grad():
            embedding_matrix[old_length:] = mean_embedding
            lm_head_matrix  [old_length:] = mean_lm_head
    pass

    # We set a flag to say we need to train embeddings
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    pass
    internal_model._need_to_train_embeddings = True

    # Fix up all vocab sizes
    current_model = model
    while hasattr(current_model, "model") and hasattr(current_model, "config"):
        if hasattr(current_model.config, "vocab_size"):
            current_model.config.update({"vocab_size" : len(tokenizer)})
        current_model = current_model.model
    if hasattr(current_model, "model") and hasattr(current_model, "config"):
        if hasattr(current_model.config, "vocab_size"):
            current_model.config.update({"vocab_size" : len(tokenizer)})
    pass

    # Must tie lm_head and embed_tokens if they are tied!
    # Otherwise error will occur on saving models ie use save_model
    if is_tied: model.tie_weights()

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return
pass


@torch.inference_mode
def fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES = [], eps = 1e-16):
    """
    Llama-3 etc have untrained vectors (<|eot_id|>, <|start_header_id|>, ...)
    in the base model. Reset them to the mean of the trained tokens.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    embedding_matrix = model.get_input_embeddings ().weight
    lm_head_matrix   = model.get_output_embeddings().weight
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    # Ignore some model checks for now
    if model.config._name_or_path in IGNORED_TOKENIZER_NAMES:
        return
    pass

    # Sometimes the sizes can be different like in vision models
    # Ie <image> is in input, but not in output
    min_size = min(embedding_matrix.shape[0], lm_head_matrix.shape[0])
    embedding_matrix = embedding_matrix[:min_size]
    lm_head_matrix   = lm_head_matrix  [:min_size]
    
    # Get untrained tokens
    indicator_untrained1 = torch.amax(embedding_matrix, axis = 1) <= eps
    # Check lm_head as well

    # Does NOT work for Llama 3.1!!
    indicator_untrained2 = torch.amax(lm_head_matrix,   axis = 1) <= eps

    # We instead check for repeated vectors
    lm_head_where = torch.where(indicator_untrained1)[0]
    lm_head_bad = lm_head_matrix[lm_head_where.to(lm_head_matrix.device)]
    lm_head_bad = lm_head_bad.cpu().float().numpy().round(3)
    from collections import Counter
    counter = Counter()
    for row in lm_head_bad: counter[hash(row.data.tobytes())] += 1
    counter = Counter({k: c for k, c in counter.items() if c >= 2})

    lm_head_where = lm_head_where.cpu().numpy()
    final_bad_lm_head = []
    for j, row in enumerate(lm_head_bad):
        if hash(row.data.tobytes()) in counter:
            final_bad_lm_head.append(lm_head_where[j])
    indicator_untrained2 = indicator_untrained2 | torch.zeros_like(indicator_untrained2)
    indicator_untrained2[final_bad_lm_head] = True

    # Combine both checks
    indicator_untrained = indicator_untrained1.to("cpu") & indicator_untrained2.to("cpu")

    # Remove pad token and other important token possibilities
    special_tokens = (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    )
    for special_token in special_tokens:
        if hasattr(tokenizer, special_token + "_id"):
            token_id = eval(f"tokenizer.{special_token}_id")
            if token_id is not None and token_id < indicator_untrained.shape[0]:
                indicator_untrained[token_id] = False
        pass
    pass
    
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained

    # Get set and actual tokens
    where_untrained = where_untrained.tolist()
    if len(where_untrained) == 0: return

    # Remove untrained indices where it's longer
    
    where_untrained_set = frozenset(where_untrained)
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    # Remove None items in actual_bad_tokens
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    # Check if tokenizer and training datasets have bad tokens
    if_bad_first  = False
    if_bad_second = False
    # Check tokenizer's chat template for any untrained tokens
    if chat_template is not None:
        if_bad_first = any(x in chat_template for x in actual_bad_tokens)
    pass

    if isinstance(train_dataset, datasets.IterableDataset):
        # Skip the check, since the code below assumes
        # an indexable dataset
        return

    # Check the first 250, last 250 input_ids
    size_dataset = len(train_dataset)
    size = min(size_dataset, 250)
    for j in range(size):
        input_ids = train_dataset[j]
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
            if_bad = any(item in where_untrained_set for item in input_ids)
            if if_bad:
                if_bad_second = True
                break
            pass
        pass
    pass

    # Check last 250
    if not if_bad_second:
        left = max(size_dataset-250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                if_bad = any(item in where_untrained_set for item in input_ids)
                if if_bad:
                    if_bad_second = True
                    break
                pass
            pass
        pass
    pass

    # Check if bad tokens exists!
    if not if_bad_first and not if_bad_second: return

    # Check if lm_head / embed_token are trainable!
    bad_not_trainable = False
    if not embedding_matrix.requires_grad: bad_not_trainable = True
    if not lm_head_matrix  .requires_grad: bad_not_trainable = True

    if bad_not_trainable:

        final_bad_items = []
        which_locations = []

        # Re-check the first 250, last 250 input_ids
        size_dataset = len(train_dataset)
        size = min(size_dataset, 250)
        for j in range(size):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                for item in input_ids:
                    if item in where_untrained_set:
                        final_bad_items.append(item)
                        which_locations.append(j)
            pass
        pass

        # Re-check last 250
        left = max(size_dataset-250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                for item in input_ids:
                    if item in where_untrained_set:
                        final_bad_items.append(item)
                        which_locations.append(j)
            pass
        pass

        # If no bad tokens, possibly chat template itself has issues?
        if len(final_bad_items) == 0:
            # Recheck 2000 and last 2000 items
            size_dataset = len(train_dataset)
            size = min(size_dataset, 2000)
            for j in range(size):
                input_ids = train_dataset[j]
                if "input_ids" in input_ids:
                    input_ids = input_ids["input_ids"]
                    for item in input_ids:
                        if item in where_untrained_set:
                            final_bad_items.append(item)
                            which_locations.append(j)
                pass
            pass

            # Re-check last 2000
            left = max(size_dataset-2000, 0)
            for j in range(left, size_dataset):
                input_ids = train_dataset[j]
                if "input_ids" in input_ids:
                    input_ids = input_ids["input_ids"]
                    for item in input_ids:
                        if item in where_untrained_set:
                            final_bad_items.append(item)
                            which_locations.append(j)
                pass
            pass
            # Most likely false signal!
            if len(final_bad_items) == 0: return
        pass

        token_ids = list(set(final_bad_items))
        tokens = tokenizer.decode(token_ids)
        raise ValueError(
            f'Unsloth: Untrained tokens in rows [{list(set(which_locations))}] found.\n'\
            f"The token ids are [{token_ids}] and tokens are [{tokens}].\n"\
            f"The issue is the embed_tokens & lm_head not trainable, which will cause NaNs. "\
            'Restart then add `embed_tokens` & `lm_head` to '\
            '`FastLanguageModel.get_peft_model(target_modules = [..., "embed_tokens", "lm_head",]). `'\
            'Are you using the `base` model? Instead, use the `instruct` version to silence this warning.',
        )
    pass

    # Count all the possible bad tokens
    final_counts = np.zeros(max(len(tokenizer), embedding_matrix.shape[0]), dtype = np.int64)
    def mapping(examples):
        input_ids = examples["input_ids"]
        counter = np.fromiter(itertools.chain.from_iterable(input_ids), dtype = np.int32)
        np.add.at(final_counts, counter, 1)
    pass
    train_dataset.map(mapping, batched = True, desc = "Counting untrained tokens")

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head   = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Remove bad tokens
    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype = torch.float32, axis = 0)
    sum_lm_head   -= torch.sum(lm_head_matrix  [where_untrained], dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained)
    mean_lm_head   = (sum_lm_head   / n_trained)

    # Scale each to be equal to 1/max_frequency. Also set some to 0 if none seen
    scaling = final_counts[where_untrained] / max(final_counts.max(), 1)
    scaling = torch.tensor(scaling, device = mean_embedding.device).unsqueeze(1)
    mean_embedding = mean_embedding.repeat((n_untrained, 1,)) * scaling
    mean_lm_head   = mean_lm_head  .repeat((n_untrained, 1,)) * scaling
    where_null = scaling.ravel() == 0
    mean_embedding[where_null] = 0
    mean_lm_head  [where_null] = 0

    # Set them to the mean
    print(
        "Unsloth: Setting embed_tokens & lm_head untrained tokens to "\
        "mean(trained) to counteract NaNs during training."
    )
    embedding_matrix[where_untrained] = mean_embedding.to(embedding_matrix.dtype)
    lm_head_matrix  [where_untrained] = mean_lm_head  .to(lm_head_matrix  .dtype)

    # Clean up
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    pass
    return
pass


# Pad-token repair lives in the shared pad_token module (single source of truth
# for unsloth + unsloth-zoo). Re-exported here for backwards compatibility.
from .pad_token import (
    fix_pad_token,
    POSSIBLE_RESERVED_TOKENS,
    VISION_RESERVED_TOKENS,
)

@torch.inference_mode
def patch_tokenizer(model, tokenizer):
    """
        Set a sensible pad_token when missing (Phi3 -> <|placeholder...,
        Llama-3 -> <|reserved..., Llama-2 -> <unk>) and ensure it differs from
        eos_token so the loss does not ignore it.
        Fixes https://github.com/unslothai/unsloth/issues/5
    """
    # All Unsloth Zoo code licensed under LGPLv3

    # Guard against None tokenizer (e.g., some VLM processors without tokenizer)
    if tokenizer is None:
        return model, tokenizer

    original_tokenizer = tokenizer

    # Auto-apply chat template when a conversation is passed instead of a string
    if hasattr(tokenizer, "image_processor") and hasattr(tokenizer, "apply_chat_template"):
        patch_processor_call(tokenizer)

    if hasattr(tokenizer, "tokenizer"):
        inner = tokenizer.tokenizer
        if inner is None:
            # Processor exists but inner tokenizer is None - return as-is
            return model, original_tokenizer
        tokenizer = inner

    # Heal a bad/missing pad_token via the shared single source of truth: it picks
    # a reserved pad-like token already in the vocab (text-only models never reuse
    # a vision token), or adds one and raises if none exists, and stamps
    # model.config / generation_config pad_token_id when it changes.
    # Fixes https://github.com/unslothai/unsloth/issues/5 and #4104.
    model_config = getattr(model, "config", None)
    result = fix_pad_token(original_tokenizer, model = model, model_config = model_config)

    # No-op case: tokenizer already had a valid pad_token, but the model config
    # may still be missing pad_token_id - mirror it across.
    if not result["changed"] and model is not None and model_config is not None:
        if getattr(model_config, "pad_token_id", None) is None:
            model.config.update({"pad_token_id" : tokenizer.pad_token_id})
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.update(pad_token_id = tokenizer.pad_token_id)
    pass

    if model is not None:
        if getattr(model, "generation_config", None) is not None:
            if hasattr(model.config, "max_position_embeddings"):
                model.generation_config.update(max_length = model.config.max_position_embeddings)
    pass

    return model, original_tokenizer
pass


def _is_conversation_format(text):
    """Return True if text is conversation format (list of dicts with 'role')."""
    # All Unsloth Zoo code licensed under LGPLv3
    if not isinstance(text, list):
        return False
    if len(text) == 0:
        return False
    first = text[0]
    if isinstance(first, dict) and "role" in first:
        return True
    return False
pass


def patch_processor_call(processor):
    """
    Patch processor's __call__ to auto-apply the chat template when text is in
    conversation format. VLM processors (e.g. Qwen3VL) expect a string; passing
    a list of dicts otherwise raises
    `AttributeError: 'dict' object has no attribute 'replace'`.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if not hasattr(processor, "apply_chat_template"):
        return processor

    if hasattr(processor, "_unsloth_patched_call"):
        return processor

    original_call = processor.__class__.__call__

    def patched_call(self, images=None, text=None, videos=None, **kwargs):
        if text is not None and _is_conversation_format(text):
            add_generation_prompt = kwargs.pop("add_generation_prompt", True)
            text = self.apply_chat_template(
                text,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return original_call(self, images=images, text=text, videos=videos, **kwargs)

    # Patch via a dynamic subclass reusing the original class name so
    # save_pretrained writes the correct processor_class (fixes issue #4085).
    original_class = processor.__class__
    patched_class = type(
        original_class.__name__,
        (original_class,),
        {
            "__call__": patched_call,
            "__module__": original_class.__module__,
            "__qualname__": original_class.__qualname__,
        }
    )
    processor.__class__ = patched_class

    processor._unsloth_patched_call = True
    return processor
pass


# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
