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

"""execute_with_time_limit on the RL notebooks' path (unslothai/notebooks#122):
the decorated function looks safe, so the signal backend is used, and the
model-written strategy it calls swallows the TimeoutError. Each case runs in a
subprocess because the bug is a hang."""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "setitimer"), reason = "signal backend needs SIGALRM"
)

RL_ENVIRONMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth_zoo", "rl_environments.py"
)

HARNESS = textwrap.dedent('''
    import importlib.util, signal, sys, time
    # Loaded by path: importing the unsloth_zoo package pulls in torch.
    spec = importlib.util.spec_from_file_location("rl_environments", sys.argv[3])
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    create_locked_down_function, execute_with_time_limit = R.create_locked_down_function, R.execute_with_time_limit

    def _execute_strategy(strategy, board):
        steps = 0
        while True:  # the move never changes the board, so the game never ends
            strategy(board)
            steps += 1

    @execute_with_time_limit(0.5)
    def execute_strategy(strategy, board):
        return _execute_strategy(strategy, board)

    strategy = create_locked_down_function(sys.argv[1])
    for _ in range(int(sys.argv[2])):
        start = time.monotonic()
        try:
            execute_strategy(strategy, [[2, 4], [8, 16]])
            raise SystemExit("returned without timing out")
        except TimeoutError:
            pass
        elapsed = time.monotonic() - start
        assert signal.getsignal(signal.SIGALRM) is signal.SIG_DFL
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    time.sleep(0.2)  # a leftover alarm would kill the process here
    print("OK", round(elapsed, 2))
''')

SWALLOW_ONCE = """
def strategy(board):
    try:
        total = 0
        for i in range(20000):
            total += i * i
        return "W"
    except Exception:
        return "W"
"""

LOOP_CATCHES_CALL = """
def strategy(board):
    def score(b):
        return sum(sum(r) for r in b)
    best = 0
    while True:
        try:
            best = max(best, score(board))
        except Exception:
            pass
"""

NESTED_LOOP_CATCHES = """
def strategy(board):
    while True:
        try:
            while True:
                pass
        except Exception:
            pass
"""

STUCK_IN_HANDLER = """
def strategy(board):
    try:
        while True:
            pass
    except Exception:
        while True:
            pass
"""


def _run(strategy, repeats = 1):
    try:
        out = subprocess.run(
            [sys.executable, "-c", HARNESS, strategy, str(repeats), RL_ENVIRONMENTS],
            capture_output = True, text = True, timeout = 60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("execute_with_time_limit(0.5) still running after 60s")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1].startswith("OK")


def test_strategy_swallowing_timeout_does_not_hang():
    _run(SWALLOW_ONCE)


def test_strategy_looping_over_try_except_does_not_hang():
    _run(LOOP_CATCHES_CALL)


def test_strategy_catching_every_timeout_does_not_hang():
    _run(NESTED_LOOP_CATCHES)


def test_strategy_stuck_in_its_except_block_does_not_hang():
    _run(STUCK_IN_HANDLER)


def test_state_restored_after_repeated_swallowed_timeouts():
    _run(SWALLOW_ONCE, repeats = 5)


def test_cleanup_after_timeout_runs_to_completion():
    from unsloth_zoo.rl_environments import time_limit
    import time

    done = []
    with pytest.raises(TimeoutError):
        with time_limit(0.05):
            try:
                while True:
                    pass
            finally:
                end = time.monotonic() + 0.2  # several re-fire periods
                while time.monotonic() < end:
                    pass
                done.append(True)
    assert done == [True]
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_fast_body_unchanged():
    from unsloth_zoo.rl_environments import execute_with_time_limit, time_limit

    @execute_with_time_limit(5)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    with pytest.raises(TimeoutError):
        with time_limit(0.05):
            while True:
                pass
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
