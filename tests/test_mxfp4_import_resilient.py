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

"""The mxfp4 converter import must be resilient: convert_moe_packed_tensors_cpu is injected at
runtime by Unsloth's mxfp4 patch and is absent from stock transformers, so importing both in one
statement nulled BOTH names and wrongly tripped the `convert_moe_packed_tensors is None` guard in
the mxfp4 export path. convert_moe_packed_tensors must survive even when _cpu is missing.
"""
import subprocess
import sys
import textwrap


def test_convert_moe_packed_tensors_survives_missing_cpu_variant():
    # Fresh process WITHOUT importing unsloth first, so convert_moe_packed_tensors_cpu is not
    # injected: this is exactly the stock-transformers case the combined import broke.
    code = textwrap.dedent(
        """
        import transformers.integrations.mxfp4 as m
        from unsloth_zoo import saving_utils as sv
        base_in_stock = hasattr(m, "convert_moe_packed_tensors")
        # With the split import, saving_utils captures the base function whenever transformers
        # exposes it, regardless of whether the Unsloth-only _cpu variant is present.
        assert (sv.convert_moe_packed_tensors is not None) == base_in_stock, (
            "convert_moe_packed_tensors", sv.convert_moe_packed_tensors, "base_in_stock", base_in_stock,
        )
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "OK" in r.stdout
