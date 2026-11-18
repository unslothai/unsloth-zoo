"""Cascade attention: off below vLLM 0.11.1 on pre-Hopper cards, vLLM's own default everywhere else.

vLLM 0.11.0 still pins the vllm_flash_attn commit just before the FA2 kv-split LSE fix
(vllm-project/flash-attention#87); 0.11.1 is the first release that carries it. vLLM >= 0.18 disables
cascade attention by default, so the zoo must never pass an explicit False.
"""

import ast
import pathlib

import pytest
from packaging.version import Version

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"


def _helper():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vllm_cascade_attn_needs_disable"
    )
    namespace = {"Version": Version}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace["_vllm_cascade_attn_needs_disable"]


@pytest.mark.parametrize(
    "vllm_version, major, expected",
    [
        ("0.10.2", 8, True),
        ("0.11.0", 8, True),  # pins flash-attention ee4d25bd, one commit before the fix
        ("0.11.1", 8, False),
        ("0.11.2", 8, False),
        ("0.18.0", 8, False),  # vLLM's own default (disabled) must stand
        ("0.10.2", 9, False),
        ("0.11.0", 10, False),
        ("0.11.0", 12, False),
    ],
)
def test_cascade_attention_is_only_ever_disabled(vllm_version, major, expected):
    assert _helper()(vllm_version, major) is expected


def test_load_vllm_never_forces_cascade_attention_on():
    """An explicit disable_cascade_attn = False would re-enable it on vLLM >= 0.18, where it is off by default."""
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
            for target in targets:
                name = None
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    name = target.slice.value
                elif isinstance(target, ast.Name):
                    name = target.id
                if name == "disable_cascade_attn":
                    assert not (isinstance(value, ast.Constant) and value.value is False), (
                        f"line {node.lineno} forces cascade attention on"
                    )
