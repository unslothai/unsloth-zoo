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

"""Compiled modules live outside transformers, so relative imports copied from a modeling file (MusicgenForConditionalGeneration.__init__) must be made absolute."""

import importlib
import inspect

import pytest

from unsloth_zoo.compiler import _absolute_relative_imports

LOCATION = "transformers.models.musicgen.modeling_musicgen"


def test_function_local_relative_imports_become_absolute():
    source = (
        "def __init__(self):\n"
        "    from ..auto.modeling_auto import AutoModelForTextEncoding\n"
        "    from . import configuration_musicgen\n"
        "    from .configuration_musicgen import MusicgenConfig\n"
        "    from transformers import AutoModel\n"
    )
    out = _absolute_relative_imports(source, LOCATION)
    assert "from transformers.models.auto.modeling_auto import AutoModelForTextEncoding" in out
    assert "    from transformers.models.musicgen import configuration_musicgen" in out
    assert "from transformers.models.musicgen.configuration_musicgen import MusicgenConfig" in out
    assert "from transformers import AutoModel" in out
    assert _absolute_relative_imports(out, LOCATION) == out  # idempotent
    assert "from ." not in out


def test_the_real_musicgen_init_resolves():
    transformers = pytest.importorskip("transformers")
    cls = getattr(transformers, "MusicgenForConditionalGeneration", None)
    if cls is None:
        pytest.skip("transformers has no MusicGen")
    out = _absolute_relative_imports(inspect.getsource(cls.__init__), LOCATION)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("from ") and " import " in line:
            module = line[len("from ") :].split(" import ")[0]
            assert not module.startswith(".")
            importlib.import_module(module)


def test_rnn_modules_are_not_compiled():
    """Dynamo refuses nn.LSTM (EnCodec's EncodecLSTM inside MusicGen), so such modules stay eager."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "unsloth_zoo" / "compiler.py").read_text(encoding = "utf-8")
    pattern = "(?:LSTM|GRU|RNN)" in source
    assert pattern and "+ rnn_modules" in source
    transformers = pytest.importorskip("transformers")
    lstm = getattr(transformers.models.encodec.modeling_encodec, "EncodecLSTM", None)
    if lstm is None:
        pytest.skip("transformers has no EnCodec")
    assert re.search(r"\bnn\.(?:LSTM|GRU|RNN)\(", inspect.getsource(lstm))
