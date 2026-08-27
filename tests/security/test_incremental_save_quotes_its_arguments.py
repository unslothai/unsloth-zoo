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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`incremental_save_pretrained` writes `repo_id` and `revision` into generated Python.

That source is handed to the `exec` in `merge_and_dequantize_lora`, so both values have
to arrive as string LITERALS. `revision` was interpolated bare, which made an ordinary
`revision = "main"` a name lookup and a revision spelled as an expression executable;
`repo_id` was wrapped in single quotes, which one apostrophe ends.
"""

import ast
import textwrap

import pytest

from unsloth_zoo.saving_utils import incremental_save_pretrained


# The two shapes `incremental_save_pretrained` rewrites, and nothing else.
_SAVE_PRETRAINED = textwrap.dedent(
    """
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok = True)
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            pass
        return None
    """
)


def _generated(repo_id, revision):
    return incremental_save_pretrained(
        _SAVE_PRETRAINED,
        low_disk_space_usage = True,
        use_temp_file = True,
        repo_id = repo_id,
        revision = revision,
    )


@pytest.mark.parametrize(
    "repo_id, revision",
    [
        ("me/model", "main"),
        # An apostrophe ends the single-quoted spelling the generator used to write.
        ("me/it's", "main"),
        ("me/model", None),
        # A revision that is Python rather than a name. It must land as a string.
        ("me/model", "__import__('os').getcwd()"),
        ('me/"quoted"', 'rev"quoted"'),
    ],
)
def test_the_generated_source_carries_both_values_as_literals(repo_id, revision):
    generated = _generated(repo_id, revision)
    tree = ast.parse(generated)

    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in ("repo_id", "revision"):
                found[keyword.arg] = keyword.value

    assert set(found) == {"repo_id", "revision"}, generated
    for name, expected in (("repo_id", repo_id), ("revision", revision)):
        node = found[name]
        assert isinstance(node, ast.Constant), f"{name} is not a literal:\n{generated}"
        assert node.value == expected


def test_the_generated_assignment_carries_the_repo_id_as_a_literal():
    # The `repo_id = ...` statement written into the temp-file branch is the second
    # site, and it was quoted the same fragile way.
    generated = _generated("me/it's", "main")
    tree = ast.parse(generated)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "repo_id" for t in node.targets)
    ]
    assert assignments, generated
    for node in assignments:
        assert isinstance(node.value, ast.Constant)
        assert node.value.value == "me/it's"
