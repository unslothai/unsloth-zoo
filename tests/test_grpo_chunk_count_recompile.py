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
Only the chunk body is compiled now, so the graph count stays flat in T."""
import inspect
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

    chunks, hidden, vocab = 8, 24, 64
    # 4 distinct chunk counts: 8, 5, 7, 6.
    Ts = [16, 9, 13, 17, 15, 10]
    assert len({math.ceil(t / math.ceil(t / chunks)) for t in Ts}) == 4
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
        # main compiled a graph per distinct chunk count (4 here, 6 graphs). The
        # chunk body needs one, plus a non-view variant (the 1-row re-split cats)
        # or a 1-row specialisation, independent of T.
        assert res["graphs"] <= 3, (name, res)
        assert res["err"] < 1e-4, (name, res)


def test_pasted_source_runs_outside_the_module():
    """The trainer generator pastes these functions' source into another module
    (inspect.getsource in unsloth/models/rl.py), so the compiled chunk helpers
    must be reachable from the pasted copy, not only from this module's globals."""
    import torch
    from unsloth_zoo import rl_replacements as R
    ns = {"torch": torch, "os": os, "math": math}
    for name in ("chunked_selective_log_softmax",
                 "chunked_hidden_states_selective_log_softmax"):
        f = getattr(R, name)
        f = getattr(f, "__wrapped__", f)
        exec(textwrap.dedent(inspect.getsource(f)), ns)
    torch.manual_seed(0)
    h = torch.randn(2, 5, 8)
    lm = torch.randn(32, 8)
    idx = torch.randint(0, 32, (2, 5))
    a = ns["chunked_hidden_states_selective_log_softmax"](h, lm, idx, 3)
    b = ns["chunked_selective_log_softmax"](h @ lm.t(), idx, 1.0, 3)
    ref = torch.log_softmax(h @ lm.t(), -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    assert (a - ref).abs().max().item() < 1e-4
    assert (b - ref).abs().max().item() < 1e-4
