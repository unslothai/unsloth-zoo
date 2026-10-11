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


"""The GRPO log-softmax helpers compiled their whole chunk loop. A list's length
cannot be symbolic, so Dynamo specialised on the chunk count torch.chunk returns,
ceil(T / ceil(T / chunks)), which moves with the token count T: every new count
recompiled a graph unrolling every chunk (598 s of a 639 s, 60 step GRPO run).
The chunks are now `chunks` row ranges [i*T//chunks, (i+1)*T//chunks), so the
traced loop length depends only on `chunks` and the graph count stays flat in T."""
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(
    """
    import json, math, sys
    import torch, torch._dynamo
    from torch._dynamo.utils import counters
    import unsloth_zoo.rl_replacements as R

    chunks, hidden, vocab = 32, 24, 64
    Ts = list(range(70, 430, 9))
    # 8 distinct torch.chunk counts; every T >= 2 * chunks.
    assert len({math.ceil(t / math.ceil(t / chunks)) for t in Ts}) == 8
    torch._dynamo.config.cache_size_limit = 64
    torch.manual_seed(0)
    lm_head = (torch.randn(vocab, hidden) * 0.1).requires_grad_(True)
    out = {}
    for name in ("hidden", "logits"):
        torch._dynamo.reset()
        counters.clear()
        worst = 0.0
        for T in Ts:
            h = torch.randn(1, T, hidden, requires_grad = True)
            idx = torch.randint(0, vocab, (1, T))
            if name == "hidden":
                got = R.chunked_hidden_states_selective_log_softmax(h, lm_head, idx, chunks = chunks)
            else:
                got = R.chunked_selective_log_softmax(h @ lm_head.t(), idx, chunks = chunks)
            got.sum().backward()
            ref = torch.log_softmax((h @ lm_head.t()).float(), -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            worst = max(worst, (got - ref).abs().max().item())
        out[name] = dict(graphs = counters["stats"]["unique_graphs"], err = worst)
    print("RESULT " + json.dumps(out))
    """
)


def test_graph_count_does_not_grow_with_the_chunk_count():
    env = dict(os.environ, PYTHONPATH = os.pathsep.join(
        [str(ROOT)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output = True,
                       text = True, timeout = 900, env = env)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, (r.stdout[-2000:], r.stderr[-3000:])
    import json
    got = json.loads(line[0][len("RESULT "):])
    for name, res in got.items():
        # main compiled a graph per distinct chunk count (8 here); now one,
        # independent of T (2 leaves slack for a respecialisation).
        assert res["graphs"] <= 2, (name, res)
        assert res["err"] < 1e-4, (name, res)

