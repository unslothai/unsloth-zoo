# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""sft_prepare_dataset's source is copied into the compiled trainer, so it must import its own helpers."""

import ast
import builtins
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

from datasets import Dataset, IterableDataset

import unsloth_zoo.dataset_utils as dataset_utils


def _source_tree():
    return ast.parse(textwrap.dedent(inspect.getsource(dataset_utils.sft_prepare_dataset)))


def test_every_dataset_utils_helper_it_calls_is_imported_inside():
    fn = _source_tree().body[0]
    bound = {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # Also imported by the compiled trainer module itself.
    provided = {"Dataset", "IterableDataset", "Any", "Callable", "Optional", "Union", "ConstantLengthDataset"}
    helpers = {
        name for name, value in vars(dataset_utils).items()
        if callable(value) and getattr(value, "__module__", None) == dataset_utils.__name__
    }
    missing = sorted((loaded - bound - provided - set(dir(builtins))) & helpers)
    assert missing == [], missing


class _Tokenizer:
    pad_token_id = 0
    bos_token = None
    eos_token = "</s>"
    chat_template = None

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) % 50 + 1 for c in t] for t in text]}
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_streaming_text_dataset_tokenizes_from_the_compiled_copy():
    namespace = {
        "Dataset": Dataset, "IterableDataset": IterableDataset,
        "Any": Any, "Callable": Callable, "Optional": Optional, "Union": Union,
        "pack_dataset": None,
    }
    exec(compile(_source_tree(), "<UnslothSFTTrainer>", "exec"), namespace)
    rows = [{"text": "hello world"}, {"text": "streaming"}]
    dataset = Dataset.from_list(rows).to_iterable_dataset()
    trainer = SimpleNamespace(model = None, completion_only_loss = None, data_collator = None)
    args = SimpleNamespace(
        max_length = 64, dataset_text_field = "text", remove_unused_columns = True,
        packing = False, packing_strategy = "bfd", dataset_kwargs = {},
    )
    out = namespace["sft_prepare_dataset"](trainer, dataset, _Tokenizer(), args, False, None, "train")
    got = [row["input_ids"] for row in out]
    assert got == [_Tokenizer()(r["text"])["input_ids"] for r in rows]
