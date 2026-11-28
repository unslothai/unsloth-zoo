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

"""get_trl_metrics finds trainers that live only under trl/experimental (ORPO on TRL >= 0.29)."""

import importlib.util
import os
import sys
import textwrap

import pytest

pytest.importorskip("torch")

_LOGGING_UTILS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth_zoo", "logging_utils.py")

_ORPO_SOURCE = textwrap.dedent('''
    class ORPOTrainer:
        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            metrics = {}
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}rewards/chosen"] = 0
            metrics[f"{prefix}nll_loss"] = 0
            metrics[f"{prefix}log_odds_ratio"] = 0
            return metrics
''')
_DPO_SOURCE = 'class DPOTrainer:\n    def log(self, mode):\n        self._metrics[mode]["rewards/margins"] = 0\n'


def _load_logging_utils():
    spec = importlib.util.spec_from_file_location("_g150_logging_utils", _LOGGING_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        f.write(text)


def _fake_trl(root, orpo_in_trainer):
    # trl.trainer lists only the stable trainers; ORPO lives in trl/experimental/orpo, as on TRL >= 0.29.
    _write(os.path.join(root, "trl", "__init__.py"), "")
    names = ["dpo_trainer"] + (["orpo_trainer"] if orpo_in_trainer else [])
    _write(os.path.join(root, "trl", "trainer", "__init__.py"), "".join(f"{n} = None\n" for n in names))
    _write(os.path.join(root, "trl", "trainer", "dpo_trainer.py"), _DPO_SOURCE)
    if orpo_in_trainer:
        _write(os.path.join(root, "trl", "trainer", "orpo_trainer.py"), _ORPO_SOURCE)
    else:
        _write(os.path.join(root, "trl", "experimental", "__init__.py"), "")
        _write(os.path.join(root, "trl", "experimental", "orpo", "__init__.py"), "")
        _write(os.path.join(root, "trl", "experimental", "orpo", "orpo_trainer.py"), _ORPO_SOURCE)
        # A directory without a <name>_trainer.py is not a trainer.
        _write(os.path.join(root, "trl", "experimental", "utils", "__init__.py"), "")


def _drop_trl_modules():
    for name in [m for m in sys.modules if m == "trl" or m.startswith("trl.")]:
        del sys.modules[name]


@pytest.fixture
def fake_trl(tmp_path, monkeypatch):
    # The fake trl.* modules must not outlive the test: later tests import the installed TRL.
    saved = {m: sys.modules[m] for m in list(sys.modules) if m == "trl" or m.startswith("trl.")}

    def make(orpo_in_trainer):
        root = str(tmp_path / ("stable" if orpo_in_trainer else "experimental"))
        _fake_trl(root, orpo_in_trainer)
        _drop_trl_modules()
        monkeypatch.syspath_prepend(root)
        return root

    try:
        yield make
    finally:
        _drop_trl_modules()
        sys.modules.update(saved)


@pytest.mark.parametrize("orpo_in_trainer", [False, True], ids = ["trl_experimental", "trl_trainer"])
def test_orpo_metrics_found(fake_trl, orpo_in_trainer):
    fake_trl(orpo_in_trainer)
    lu = _load_logging_utils()
    metrics = lu.get_trl_metrics()
    assert "orpo_trainer" in metrics, sorted(metrics)
    assert metrics["orpo_trainer"] == ["rewards/chosen", "log_odds_ratio", "nll_loss"]
    assert metrics["dpo_trainer"] == ["rewards/margins"]
    assert "utils_trainer" not in metrics


def test_patch_rl_statistics_accepts_experimental_trainer(fake_trl, capsys):
    fake_trl(False)
    lu = _load_logging_utils()
    lu.PatchRLStatistics("orpo_trainer")
    assert "not yet implemented" not in capsys.readouterr().out
