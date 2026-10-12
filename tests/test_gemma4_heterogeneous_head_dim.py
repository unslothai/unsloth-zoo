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

"""transformers >= 5.15 raises on Gemma-4 config.head_dim (256/512 per layer); never getattr it bare."""
import inspect

import pytest

import unsloth_zoo.empty_model as empty_model
import unsloth_zoo.vllm_utils as vllm_utils


def _gemma4_text_config():
    transformers = pytest.importorskip("transformers")
    config_class = getattr(transformers, "Gemma4TextConfig", None)
    if config_class is None:
        pytest.skip("transformers has no Gemma-4")
    return config_class(head_dim = 256, global_head_dim = 512)


def test_heterogeneous_head_dim_reads_do_not_raise():
    config = _gemma4_text_config()
    assert empty_model._global_config_value(config, "head_dim", None) == 256
    assert vllm_utils._config_get(config, "head_dim") == 512


def test_no_bare_global_head_dim_read_in_the_vllm_path():
    for module, name in ((empty_model, "create_empty_causal_lm"), (vllm_utils, "load_vllm")):
        source = inspect.getsource(getattr(module, name))
        assert 'getattr(causal_config, "head_dim"' not in source, name
        assert 'getattr(_text_config, "head_dim"' not in source, name
