#!/bin/bash

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

set -euo pipefail

SEQLENS=(1024)  
DTYPES=(bfloat16)
PERMUTE_X=(false true)
PERMUTE_Y=(false true)
AUTOTUNE=(false true)

for SEQLEN in "${SEQLENS[@]}"; do
    for DTYPE in "${DTYPES[@]}"; do
        for PX in "${PERMUTE_X[@]}"; do
            for PY in "${PERMUTE_Y[@]}"; do
                for AT in "${AUTOTUNE[@]}"; do

                    ARGS=()
                    [[ "$PX" == "true" ]] && ARGS+=("--permute_x")
                    [[ "$PY" == "true" ]] && ARGS+=("--permute_y")
                    [[ "$AT" == "true" ]] && ARGS+=("--autotune")

                    ARGS+=(--seqlen "$SEQLEN" --dtype "$DTYPE")

                    echo "Running with args: ${ARGS[*]}"
                    if ! python -m tests.test_qwen3_moe "${ARGS[@]}"; then
                        echo "❌ Test failed with args: --permute_x=$PX --permute_y=$PY --autotune=$AT --seqlen=$SEQLEN --dtype=$DTYPE" >&2
                    else
                        echo "✅ Test passed with args: --permute_x=$PX --permute_y=$PY --autotune=$AT --seqlen=$SEQLEN --dtype=$DTYPE"
                    fi

                done
            done
        done
    done
done
