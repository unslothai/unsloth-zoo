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

"""sft_prepare_dataset fingerprints its tokenizing map the same way in every process, so a rerun loads the
cached Arrow file, and any change to what the map produces still changes the fingerprint. CPU-only."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

TOKENIZER = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

_CHILD = textwrap.dedent(
    """
    import json, sys, warnings
    from types import SimpleNamespace
    warnings.filterwarnings("ignore")
    import datasets, transformers
    datasets.disable_progress_bars()
    import unsloth_zoo.dataset_utils as du
    du.trl_collator_applies_masks = lambda: True

    TOKENIZER = sys.argv[1]
    def conv(i, answer = "A"):
        return [{"role": "user", "content": f"Q {i}"}, {"role": "assistant", "content": f"{answer} {i}"}]

    def run(kind, rows_edit = False, max_length = 256, aol = False, add_token = False, template = None,
            text_field = "text"):
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
        if add_token: tok.add_tokens(["<r11_extra>"])
        if template is not None: tok.chat_template = template
        n = 24
        if kind == "text":
            rows = [{"instruction": f"I {i}", "input": "", "output": f"O {i}", "text": f"I {i} O {i}",
                     "body": f"I {i} O {i}"} for i in range(n)]
        elif kind == "messages":
            rows = [{"messages": conv(i), "source": "x", "id": i, "extra": "y"} for i in range(n)]
        else:
            rows = [{"prompt": conv(i)[:1], "completion": conv(i)[1:], "source": "x", "id": i} for i in range(n)]
        if rows_edit:
            if kind == "text": rows[3]["text"] += " changed"; rows[3]["body"] += " changed"
            elif kind == "messages": rows[3]["messages"] = conv(3, "B")
            else: rows[3]["completion"] = conv(3, "B")[1:]
        trainer = SimpleNamespace(chat_template = None, _unsloth_assistant_mask_fallback = True, data_collator = None,
                                  model = None, completion_only_loss = None)
        args = SimpleNamespace(max_length = max_length, dataset_text_field = text_field, remove_unused_columns = True,
                               assistant_only_loss = aol, dataset_num_proc = 1, packing_strategy = "bfd",
                               truncation_mode = "keep_start")
        out = du.sft_prepare_dataset(trainer, datasets.Dataset.from_list(rows), tok, args, False, None, "train")
        return out._fingerprint

    alt_template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\\n{% endfor %}"
    cases = {}
    for kind in ("text", "messages", "prompt_completion"):
        cases[kind] = run(kind)
        cases[kind + ":rows"] = run(kind, rows_edit = True)
        cases[kind + ":max_length"] = run(kind, max_length = 128)
        cases[kind + ":tokenizer"] = run(kind, add_token = True)
    cases["text:text_field"] = run("text", text_field = "body")
    for kind in ("messages", "prompt_completion"):
        cases[kind + ":assistant_only_loss"] = run(kind, aol = True)
        cases[kind + ":chat_template"] = run(kind, template = alt_template)
    print(json.dumps(cases))
    """
)


def _fingerprints(seeds):
    # One child per seed, all at once: one child is ~40 s, so five in a row outrun pytest-timeout on CI.
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, TOKENIZER],
            env = dict(os.environ, PYTHONHASHSEED = str(seed)),
            stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True,
        )
        for seed in seeds
    ]
    try:
        outputs = [child.communicate(timeout = 600) for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
    results = []
    for child, (stdout, stderr) in zip(children, outputs):
        if child.returncode != 0:
            if "OSError" in stderr or "ConnectionError" in stderr:
                pytest.skip(reason = "offline runner, tokenizer download failed")
            raise AssertionError(stderr[-4000:])
        results.append(json.loads(stdout.strip().splitlines()[-1]))
    return results


@pytest.fixture(scope = "module")
def runs():
    try:
        transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:
        pytest.skip(reason = f"offline runner, tokenizer download failed: {e}")
    # Several hash seeds: a set of 3 to 5 column names lists in a seed dependent order.
    return _fingerprints((1, 2, 3, 4, 5))


def test_identical_inputs_fingerprint_the_same_in_every_process(runs):
    first = runs[0]
    for other in runs[1:]:
        differing = sorted(k for k in first if first[k] != other[k])
        assert not differing, f"fingerprints differ between processes for {differing}"


@pytest.mark.parametrize("kind", ["text", "messages", "prompt_completion"])
def test_each_semantic_input_changes_the_fingerprint(runs, kind):
    cases = runs[0]
    base = cases[kind]
    variants = [k for k in cases if k.startswith(kind + ":")]
    assert len(variants) >= 3
    for variant in variants:
        assert cases[variant] != base, f"{variant} reused the {kind} fingerprint: a stale cache would be loaded"


def test_remove_columns_follow_dataset_column_order(monkeypatch):
    import unsloth_zoo.dataset_utils as du
    from types import SimpleNamespace

    try:
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:
        pytest.skip(reason = f"offline runner, tokenizer download failed: {e}")
    monkeypatch.setattr(du, "trl_collator_applies_masks", lambda: True)
    seen = []
    original = datasets.Dataset.map

    def recording_map(self, *args, **kwargs):
        if "remove_columns" in kwargs:
            seen.append((list(self.column_names), kwargs["remove_columns"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(datasets.Dataset, "map", recording_map)
    columns = ["zeta", "text", "alpha", "mid", "beta", "omega"]
    rows = [{c: f"{c} {i}" for c in columns} for i in range(4)]
    trainer = SimpleNamespace(chat_template = None, _unsloth_assistant_mask_fallback = True, data_collator = None,
                              model = None, completion_only_loss = None)
    args = SimpleNamespace(max_length = 64, dataset_text_field = "text", remove_unused_columns = True,
                           assistant_only_loss = False, dataset_num_proc = 1, packing_strategy = "bfd",
                           truncation_mode = "keep_start")
    du.sft_prepare_dataset(trainer, datasets.Dataset.from_list(rows), tok, args, False, None, "train")
    assert seen, "the tokenizing map was not reached"
    dataset_columns, removed = seen[0]
    assert list(removed) == dataset_columns == columns
