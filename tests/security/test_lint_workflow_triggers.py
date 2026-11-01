"""Tests scripts/lint_workflow_triggers.py against GHSA-g7cv-rxg3-hmpx (TanStack)-class vectors: pull_request_target
(fork PRs run in base context) and shared PR/publish cache keys."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "lint_workflow_triggers.py"


def _run(workflows_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--workflows-dir", str(workflows_dir)],
        capture_output = True,
        text = True,
    )


def test_lint_passes_on_current_workflows():
    """The live `.github/workflows/` tree must pass the lint."""
    live = REPO_ROOT / ".github" / "workflows"
    proc = _run(live)
    assert (
        proc.returncode == 0
    ), f"live tree failed lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_lint_rejects_pull_request_target(tmp_path):
    """Synthetic PR_TARGET trigger must produce rc=1 with a named finding."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "bad.yml").write_text(
        "name: bad\n"
        "on:\n"
        "  pull_request_target:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo evil\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "BANNED trigger 'pull_request_target'" in proc.stderr
    assert "GHSA-g7cv-rxg3-hmpx" in proc.stderr


def test_lint_rejects_unjustified_workflow_run(tmp_path):
    """`workflow_run` requires an explicit allow-comment in the YAML."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "chained.yml").write_text(
        "name: chained\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: ['CI']\n"
        "    types: [completed]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo elevated\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "RESTRICTED trigger 'workflow_run'" in proc.stderr


def test_lint_allows_justified_workflow_run(tmp_path):
    """With the allow-comment, workflow_run is permitted."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "chained.yml").write_text(
        "# lint:workflow_triggers-allow-workflow_run -- justified by ticket #1234\n"
        "name: chained\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: ['CI']\n"
        "    types: [completed]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo elevated\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, f"justified workflow_run rejected:\n{proc.stderr}"


def test_lint_rejects_shared_cache_key_between_pr_and_publish(tmp_path):
    """A cache key shared by a PR-triggered workflow and the publish workflow is a cache-poisoning vector, so the lint must reject it."""
    wf = tmp_path / "wf"
    wf.mkdir()
    # PR-triggered: writes to a cache that the publish job will also restore.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: shared-cache-v1\n"
    )
    # Publish workflow with the IDENTICAL cache key -- the actual attack pattern.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: shared-cache-v1\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "cache-key" in proc.stderr.lower() or "cache key" in proc.stderr.lower()
    assert "shared-cache-v1" in proc.stderr


def _pr_workflow(text_key: str) -> str:
    return (
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        f"          key: {text_key}\n"
    )


def test_lint_rejects_a_publish_restore_keys_prefix_over_a_pr_namespace(tmp_path):
    """restore-keys takes the newest key that starts with the prefix, so a publish prefix can adopt a PR-written entry with no equal key; exact check misses it."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-v2-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: pip-v2-publish-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"prefix restore accepted:\n{proc.stdout}\n{proc.stderr}"
    assert "restore-keys" in proc.stderr
    assert "'pip-'" in proc.stderr


def test_a_partitioned_publish_prefix_is_accepted(tmp_path):
    """The fix must pass, or the rule above is just a ban on restore-keys."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-v2-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: pip-publish-only-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-publish-only-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, f"partitioned prefix rejected:\n{proc.stderr}"


def _publish_with_restore_keys(key: str, prefixes: str) -> str:
    return (
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        f"          key: {key}\n"
        "          restore-keys: |\n" + prefixes
    )


def test_a_publish_prefix_longer_than_the_pr_literal_head_is_caught(tmp_path):
    """Prefix check must run both ways: publish prefix pip-Linux- is longer than the PR literal head pip-, yet the runtime key pip-Linux-abc matches."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        _publish_with_restore_keys("pip-pub-${{ runner.os }}", "            pip-Linux-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"longer publish prefix accepted:\n{proc.stdout}\n{proc.stderr}"


def test_an_expression_led_pr_key_is_expanded_not_dropped(tmp_path):
    """Dropping these was a gate bypass: the PR side reduced to the empty string."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("${{ runner.os }}-shared-abc"))
    (wf / "wheel-smoke.yml").write_text(
        _publish_with_restore_keys("pub-${{ runner.os }}", "            Linux-shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"expression-led PR key silently dropped:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_composite_that_builds_its_key_in_shell_is_read(tmp_path):
    """pip-cache-restore builds its key or prefix in a shell run step, not in YAML key:, so the lint must read the shell to cover that action."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        '        prefix="pip-${name}-${{ runner.os }}-py${pyver}-"\n'
        '        echo "key=${prefix}${hash}" >> "$GITHUB_OUTPUT"\n'
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
    )
    (wf / "wheel-smoke.yml").write_text(
        _publish_with_restore_keys("pip-pub", "            pip-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the composite's shell-built pip- namespace was not seen:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_publish_only_composite_is_not_treated_as_a_pr_namespace(tmp_path):
    """Scanning every action made a publish-only cache reject its own prefix."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "release-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: release cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: release-only-${{ runner.os }}\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-${{ runner.os }}-abc"))
    (wf / "wheel-smoke.yml").write_text(
        "name: wheel-smoke\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/release-cache\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: release-only-pub\n"
        "          restore-keys: |\n"
        "            release-only-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a publish-only composite was treated as a PR namespace:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_restore_keys_prefix_after_a_blank_line_is_still_read(tmp_path):
    """A blank line in a restore-keys block scalar does not end the list (actions/cache skips empty entries), so entries after it must still be read."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "release-only-${{ runner.os }}",
            "            release-only-\n"
            "\n"
            "            shared-\n",
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a prefix after a blank line in the block was dropped, so the collision was "
        f"never compared:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_cache_key_in_a_local_reusable_workflow_is_seen(tmp_path):
    """A local reusable workflow reference names a file, not an action directory; probing only for action.yml missed its cache keys entirely."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: reuse-v1-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("reuse-v1-pub-${{ runner.os }}", "            reuse-v1-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the reusable workflow's reuse-v1- key was not seen, so the publish prefix "
        f"matched nothing:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "reuse-v1-" in proc.stderr


def test_a_composite_key_equal_to_a_publish_key_is_caught(tmp_path):
    """Composite-action keys must also feed the exact-equality check; an equal publish key with no restore-keys was otherwise accepted."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "shared-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: shared cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: wheels-shared-key\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/shared-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: wheels-shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a composite action's literal key equal to the publish key was accepted:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "wheels-shared-key" in proc.stderr


def test_a_shell_built_namespace_is_narrowed_by_the_inputs_callers_pass(tmp_path):
    """Narrow a shell-built namespace to the name: values callers pass (pip-${name}- becomes pip-mlx-), else pip-release- is falsely rejected."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pip-${name}-${{ runner.os }}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: mlx\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-release-pub-${{ runner.os }}", "            pip-release-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`pip-release-` cannot be written by a pull request whose only namespace is "
        f"`pip-mlx-`, so this must pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # And the narrowed namespace still has teeth: a publish prefix over `pip-mlx-` fails.
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-mlx-pub-${{ runner.os }}", "            pip-mlx-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a publish prefix over the PR's real `pip-mlx-` namespace must still be "
        f"rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-mlx-" in proc.stderr


def test_a_publish_composite_that_restores_a_pr_namespace_is_caught(tmp_path):
    """Publish-side restore-keys must be read from local composite actions too, not only the top-level workflow; an empty side must not report success."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "publish-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: publish cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: shared-publish-${{ runner.os }}\n"
        "        restore-keys: |\n"
        "          shared-\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/publish-cache\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish composite's `shared-` fallback was never collected, so the "
        f"collision was not compared:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_narrowing_keeps_the_broad_head_when_a_caller_is_dynamic(tmp_path):
    """Keep the broad head when any caller passes a dynamic input; narrowing to literal callers alone would falsely accept a pip-v2-shared- prefix."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pip-v2-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  literal:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: mlx\n"
        "  dynamic:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        cache_name: [shared, other]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: ${{ matrix.cache_name }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-v2-shared-pub-${{ runner.os }}", "            pip-v2-shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a dynamic call site's namespace was discarded by the narrowing, so a publish "
        f"prefix over it passed:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v2-" in proc.stderr


def test_a_folded_restore_keys_block_is_one_prefix_not_several(tmp_path):
    """A folded restore-keys block is one space-joined string at runtime, so each physical line is not its own fallback prefix."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: safe-only-${{ runner.os }}\n"
        "          restore-keys: >\n"
        "            safe-only-\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a folded block is one space-joined prefix and cannot reach the `shared-` "
        f"namespace, so this must pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # The literal form of the same block really does offer `shared-`, and must fail.
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "safe-only-${{ runner.os }}",
            "            safe-only-\n            shared-\n",
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a literal block DOES offer `shared-` as a separate fallback and must be "
        f"rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_quoted_restore_keys_field_is_read(tmp_path):
    """Quoted restore-keys keys are valid YAML and offer the same fallback, so the YAML parser must be used, not a lexical match on the bare token."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        '          "key": pub-${{ runner.os }}\n'
        '          "restore-keys": |\n'
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a quoted restore-keys field was not read:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_restore_keys_sequence_is_read(tmp_path):
    """`restore-keys: [a-, shared-]` is the sequence form, which the line reader never saw."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
          "          path: wheels\n"
        "          key: pub-${{ runner.os }}\n"
        "          restore-keys: [safe-, shared-]\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a sequence-form restore-keys was not read:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_flow_style_local_uses_is_followed(tmp_path):
    """Flow-style and quoted-key uses: entries are the same step mapping and must be followed; a lexical match on uses: missed them."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "shared-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: shared cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: flow-v1-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - {uses: ./.github/actions/shared-cache}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("flow-v1-pub-${{ runner.os }}", "            flow-v1-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a flow-style local `uses` was not followed, so the action's namespace was "
        f"invisible:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "flow-v1-" in proc.stderr


def test_a_caller_supplied_key_is_resolved_not_dismissed(tmp_path):
    """inputs.* keys are caller-supplied, not delegated like steps.* outputs, so they must be resolved from the callers' values, not dismissed."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        required: true\n"
        "        type: string\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
        "    with:\n"
        "      cache_key: shared-abc\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub-${{ runner.os }}", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a caller-supplied cache key was dismissed as delegation:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_key_delegated_to_a_step_output_is_still_accepted(tmp_path):
    """A steps.* output key is accepted as delegated when its namespace was already collected from the restoring action's shell, not reported undecidable."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "cache-save"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: cache save\n"
        "inputs:\n"
        "  key:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=own-v1-abc' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: ./.github/actions/cache-save\n"
        "        with:\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("unrelated-pub-${{ runner.os }}", "            unrelated-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a key delegated to a step output, against an unrelated publish namespace, "
        f"must pass:\n{proc.stdout}\n{proc.stderr}"
    )


def test_inputs_are_collected_through_a_wrapper_action(tmp_path):
    """Input call sites must also be collected inside composite wrappers; reading only top-level workflows hides the wrapper's call and drops its namespace."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    inner = root / "actions" / "pip-cache-restore"
    wrapper = root / "actions" / "setup-wrapper"
    wf.mkdir(parents = True)
    inner.mkdir(parents = True)
    wrapper.mkdir(parents = True)
    (inner / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pip-v3-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wrapper / "action.yml").write_text(
        "name: setup wrapper\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: ./.github/actions/pip-cache-restore\n"
        "      with:\n"
        "        name: wrapped\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  direct:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: direct\n"
        "  viawrapper:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/setup-wrapper\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "pip-v3-wrapped-pub-${{ runner.os }}", "            pip-v3-wrapped-\n"
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the wrapper's `name: wrapped` call site was not collected, so the narrowing "
        f"dropped that namespace:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v3-" in proc.stderr


def test_an_omission_before_the_first_literal_is_counted(tmp_path):
    """Omitted inputs must count even when a later call site passes a literal; otherwise call-site order hides the default namespace."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pipc"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pipc\n"
        "inputs:\n"
        "  name:\n"
        "    default: shared\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pipx-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    # The omitting call site comes FIRST, which is the ordering that used to be lost.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  defaulted:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pipc\n"
        "  named:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pipc\n"
        "        with:\n"
        "          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pipx-pub-${{ runner.os }}", "            pipx-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the first call site omits `name`, so the namespace is not fully resolved and "
        f"the broad `pipx-` head must still be defended:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pipx-" in proc.stderr


def test_a_prefix_that_opens_with_a_variable_is_recovered(tmp_path):
    """A prefix opening with a variable (${name}-pip-...) must still yield its literal part; the head pattern required an alphanumeric start and missed it."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "varfirst"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: varfirst\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "outputs:\n"
        "  key:\n"
        "    value: ${{ steps.probe.outputs.key }}\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"${name}-pip-${{ runner.os }}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: vf\n"
        "        uses: ./.github/actions/varfirst\n"
        "        with:\n"
        "          name: shared\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ steps.vf.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pip-pub-${{ runner.os }}", "            shared-pip-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the namespace is `shared-pip-` once `name` is substituted, so the publish "
        f"fallback over it must be rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-pip-" in proc.stderr


def test_an_input_backed_key_is_resolved_before_the_exact_comparison(tmp_path):
    """Input-backed keys must be resolved before the exact comparison; input resolution fed only the prefix pass, so equal keys compared an unexpanded expression."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        required: true\n"
        "        type: string\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-key"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
        "    with:\n"
        "      cache_key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish side's key resolves to `shared-key`, which the PR writes directly, "
        f"so this exact collision must be caught:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-key" in proc.stderr


def test_a_longer_fallback_over_a_complete_pr_key_is_accepted(tmp_path):
    """A longer fallback over a complete PR key (shared) is accepted: the reverse prefix check applies only to a head cut short by an expression."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-long-pub", "            shared-long-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`shared-long-` cannot restore a key that is exactly `shared`, so this must "
        f"pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # A head cut short by an expression can still meet the same fallback at runtime, so the truncated case must still be rejected.
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-long-${{ runner.os }}-abc"))
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the runtime key here really does start with `shared-long-`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def _lint_module():
    """Loads the lint script as a module so tests can call its predicates directly; the rest of the file runs it as a subprocess."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_lint_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_truncation_predicate_reads_the_key():
    """The rule above is only as good as this predicate, so it is tested directly."""
    lint = _lint_module()
    _is_truncated = lint._is_truncated
    _prefix_compatible = lint._prefix_compatible
    # runner.os expands before the head is taken, so a key with only that expression keeps a complete head; only
    # expressions this check cannot expand truncate heads.
    cases = [
        ("shared", False),
        ("shared-long-abc", False),
        ("pip-v2-${{ runner.os }}-abc", False),
        ("${{ runner.os }}-shared-abc", False),
        ("pip-${{ hashFiles('x') }}", True),
        ("pip-${{ matrix.flavour }}-abc", True),
    ]
    for key, expected in cases:
        assert _is_truncated(key) is expected, f"_is_truncated({key!r})"

    # A head cut short by an expression may be reached by a longer publish prefix; a complete key may not.
    assert _prefix_compatible("pip-v2-", "pip-v2-Linux-", True) is True
    assert _prefix_compatible("shared", "shared-long", False) is False
    # Either way, a prefix the head already starts with is always compatible.
    assert _prefix_compatible("shared-long-abc", "shared-", False) is True


def test_a_declared_input_default_is_part_of_the_namespace(tmp_path):
    """Actions applies a declared input default when a caller omits the input, so the default is part of the namespace and must be expanded."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "defaulted-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: defaulted cache\n"
        "inputs:\n"
        "  cache_key:\n"
        "    default: shared-key\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/defaulted-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the bare invocation writes the default namespace `shared-key`, which the "
        f"publish workflow uses exactly:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-key" in proc.stderr


def test_a_declared_default_does_not_settle_an_explicit_dynamic_value(tmp_path):
    """A declared default applies only to an omitted input; an explicit dynamic value overriding it stays unresolved, since its namespace is unknown."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "defaulted-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: defaulted cache\n"
        "inputs:\n"
        "  cache_key:\n"
        "    default: unrelated-default\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        cache_key: [a, b]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/defaulted-cache\n"
        "        with:\n"
        "          cache_key: ${{ matrix.cache_key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: shared-key\n"
        "          restore-keys: |\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the matrix caller overrode the default with a value the check cannot expand, "
        f"so the PR namespace is undecided and the publish prefix cannot be "
        f"cleared:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_shell_key_that_never_leaves_the_step_is_not_a_namespace(tmp_path):
    """A shell variable is a cache key only once written to $GITHUB_OUTPUT; unrelated key= or prefix= assignments must not be recorded as namespaces."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: |\n"
        '          key="shared-${RANDOM}"\n'
        '          echo hello > "/tmp/$key"\n'
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: shared-key-v1\n"
        "          restore-keys: |\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the pull request workflow caches nothing and the assignment never reaches "
        f"$GITHUB_OUTPUT, so `shared-` is not a namespace it can "
        f"write:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_input_embedded_in_a_key_is_expanded(tmp_path):
    """Inputs embedded in a longer key (prefix-${{ inputs.name }}) must be expanded, not just lone-expression keys, or exact collisions go unseen."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "embedded"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: embedded\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: prefix-${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/embedded\n"
        "        with:\n          name: shared\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-shared\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the composite writes `prefix-shared`, which the publish workflow restores by "
        f"that exact name:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "prefix-shared" in proc.stderr


def test_runner_os_is_expanded_before_the_exact_comparison(tmp_path):
    """runner.os is expanded to its known values before the exact comparison, so shared-${{ runner.os }} matches shared-Linux on a Linux runner."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ runner.os }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: shared-Linux\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"both jobs write `shared-Linux` on a Linux runner:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_an_unresolvable_pr_key_is_reported_against_an_exact_publish_key(tmp_path):
    """An unresolvable PR key must fail closed against exact publish keys too, not only in the restore-prefix pass; matrix values may equal them."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        tag: [a, b]\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ matrix.tag }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the matrix value is unknown and `shared-key` is one of the keys it could "
        f"produce:\n{proc.stdout}\n{proc.stderr}"
    )

    # A publish key the PR key's fixed head cannot lead to stays accepted, or every unresolved key in the tree would reject every publish key.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: wheels-publish-only\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a key headed `shared-` cannot become `wheels-publish-only` however its tail "
        f"resolves:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_targets_sharing_an_input_name_keep_their_own_namespaces(tmp_path):
    """Input values are scoped to the action that declares them, not merged by name across targets; otherwise a correct publish key is falsely rejected."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    caching = root / "actions" / "caching"
    unrelated = root / "actions" / "unrelated"
    wf.mkdir(parents = True)
    caching.mkdir(parents = True)
    unrelated.mkdir(parents = True)
    (caching / "action.yml").write_text(
        "name: caching\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: ${{ inputs.cache_key }}\n"
    )
    # Same input NAME, no cache anywhere in it.
    (unrelated / "action.yml").write_text(
        "name: unrelated\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.cache_key }}\n"
        "      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/caching\n"
        "        with:\n          cache_key: safe-key\n"
        "      - uses: ./.github/actions/unrelated\n"
        "        with:\n          cache_key: publish-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: publish-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`publish-key` only ever reaches the composite that caches nothing, so no PR "
        f"cache writes it:\n{proc.stdout}\n{proc.stderr}"
    )

    # The caching composite receiving it IS a collision, so the narrowing kept its teeth.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/caching\n"
        "        with:\n          cache_key: publish-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"now the caching composite is the one given `publish-key`:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_reusable_workflow_is_named_by_its_file_not_its_directory():
    """A local reusable workflow uses: reference names a file, while an action names a directory; using the parent dir for both loses the callers' values."""
    lint = _lint_module()
    # The full local reference that a uses: line writes; a basename cannot tell apart .github/actions/a/cache and
    # .github/actions/b/cache, so their call sites pooled.
    assert (
        lint._target_name(Path(".github/workflows/reuse.yml"))
        == ".github/workflows/reuse.yml"
    )
    assert (
        lint._target_name(Path(".github/actions/pip-cache/action.yml"))
        == ".github/actions/pip-cache"
    )
    assert (
        lint._target_name(Path(".github/actions/pip-cache/action.yaml"))
        == ".github/actions/pip-cache"
    )
    # An action is named by its directory and a reusable workflow by its file; the target name must keep both forms distinct.
    assert lint._target_name(Path(".github/actions/a/cache/action.yml")) != (
        lint._target_name(Path(".github/actions/b/cache/action.yml"))
    )


def test_an_unresolvable_publish_key_is_reported_too(tmp_path):
    """Unresolved publish keys must fail closed too, like PR keys: a matrix-supplied publish key may equal a PR literal, yet the lint skipped it."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pub-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pub cache\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n        path: wheels\n        key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        cache_key: [x, y]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pub-cache\n"
        "        with:\n          cache_key: ${{ matrix.cache_key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the matrix could produce `shared-key`, which the pull request writes:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_two_identically_spelled_unresolved_keys_collide(tmp_path):
    """Identically spelled unresolved keys on both sides (shared-${{ hashFiles('lock') }}) can collide at run time and must be compared, not skipped."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    body = (
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ hashFiles('lock') }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n" + body
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n" + body
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"identical keys resolve identically:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "identically" in proc.stderr


def test_a_publish_key_is_expanded_with_its_own_targets_inputs(tmp_path):
    """Publish keys must be expanded with their own target's inputs; publish_by_target was computed but never read, so the merged namespace was used."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    caching = root / "actions" / "pub-caching"
    unrelated = root / "actions" / "pub-unrelated"
    wf.mkdir(parents = True)
    caching.mkdir(parents = True)
    unrelated.mkdir(parents = True)
    (caching / "action.yml").write_text(
        "name: pub caching\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n        path: wheels\n        key: ${{ inputs.cache_key }}\n"
    )
    (unrelated / "action.yml").write_text(
        "name: pub unrelated\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.cache_key }}\n      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: safe-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pub-caching\n"
        "        with:\n          cache_key: publish-key\n"
        "      - uses: ./.github/actions/pub-unrelated\n"
        "        with:\n          cache_key: safe-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`safe-key` only ever reaches the action that caches nothing:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_composite_key_is_not_counted_a_second_time_without_its_inputs(tmp_path):
    """A composite key must not be recorded again without its target's inputs: the empty duplicate counts as unresolved and falsely fails an unrelated publish key."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "decided"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: decided\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/decided\n"
        "        with:\n          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-other\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"every caller passes `name: safe`, so the PR key can only be `prefix-safe` and "
        f"`prefix-other` is a different namespace:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_differently_spelled_unresolved_keys_are_paired(tmp_path):
    """Unresolved PR and publish keys spelled differently must still be paired: shared-${{ matrix.pr_part }} and shared-${{ matrix.pub_part }} can both be shared-x."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        pr_part: [a, b]\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ matrix.pr_part }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        pub_part: [x, y]\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ matrix.pub_part }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"both keys are headed `shared-` and neither tail is known, so they can be the "
        f"same entry:\n{proc.stdout}\n{proc.stderr}"
    )

    # Incompatible heads stay accepted, or every unresolved key in a tree would reject every other one.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        pub_part: [x, y]\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: wheels-only-${{ matrix.pub_part }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`shared-` and `wheels-only-` cannot become each other:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_delegated_key_whose_producer_was_not_read_stays_undecided(tmp_path):
    """A delegated key is settled only when its producer's namespace was recovered; _shell_built_key_prefixes reads just two spellings, so others stay undecided."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the producer's spelling was not recognised, so the key's namespace is "
        f"unknown and `shared-` cannot be cleared:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_literal_producer_output_is_read_as_a_key(tmp_path):
    """A fully literal producer output such as echo 'key=own-v1-abc' is an exact cache key; it joins the comparison, not just vouches for the step."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=own-v1-abc' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: unrelated-v1\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the key is known to be `own-v1-abc`, which is not `unrelated-v1`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )

    # And the literal really is compared, rather than merely vouching for the producer.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: own-v1-abc\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish workflow restores exactly the key the probe writes:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_publish_side_delegated_key_is_resolved_from_its_own_shell(tmp_path):
    """Shell-built heads must be collected from publish documents too, not only PR-reachable ones; a delegated publish key was dismissed unrecovered."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=shared-key' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish workflow's own shell says the key is `shared-key`, which the "
        f"pull request writes:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_actions_sharing_a_directory_name_keep_their_call_sites(tmp_path):
    """Actions are matched by full path, not last path component, so a/cache and b/cache keep their own call sites and values."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    first = root / "actions" / "a" / "cache"
    second = root / "actions" / "b" / "cache"
    wf.mkdir(parents = True)
    first.mkdir(parents = True)
    second.mkdir(parents = True)
    (first / "action.yml").write_text(
        "name: a cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (second / "action.yml").write_text(
        "name: b cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.name }}\n      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/a/cache\n"
        "        with:\n          name: safe\n"
        "      - uses: ./.github/actions/b/cache\n"
        "        with:\n          name: shared\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-shared\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`shared` only ever reaches b/cache, which caches nothing, so no PR cache "
        f"writes `prefix-shared`:\n{proc.stdout}\n{proc.stderr}"
    )


def test_one_readable_producer_does_not_vouch_for_an_unreadable_one(tmp_path):
    """Key resolution is per producing step, not per side: one readable echo must not certify another step's unrecovered printf-built key."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        # readable, and entirely unrelated to the key that is actually used
        "      - id: safe\n"
        "        run: echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
        # unreadable spelling, and this is the one whose value reaches the cache
        "      - id: make\n"
        "        run: printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the key in use comes from the step that could NOT be read, whatever the "
        f"other step spells correctly:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_producer_that_declares_its_key_inline_is_readable():
    """A step's inline key: is its output, and so is a local action's key:; such producers (frontend-dist-restore, actions/cache/restore) are readable."""
    lint = _lint_module()
    # Step identity is (document, job, step id) because step ids are unique only within a job; a bare id let a readable namesake answer for an unreadable one.
    here = ("wf.yml", "build")
    producers = {("wf.yml", "build", "probe"): {"key"}}
    assert lint._delegation_is_read(
        "${{ steps.probe.outputs.key }}", producers, here
    ) is True
    assert lint._delegation_is_read(
        "${{ steps.probe.outputs.key }}", {("wf.yml", "build", "probe"): set()}, here
    ) is False
    # Delegation is judged per output, not per step: a step may write several outputs, and recovering one says nothing about the others.
    assert lint._delegation_is_read(
        "${{ steps.probe.outputs.danger }}", producers, here
    ) is False
    # The SAME id in a different job is a different step and vouches for nothing.
    assert lint._delegation_is_read(
        "${{ steps.probe.outputs.key }}", {("wf.yml", "other", "probe"): {"key"}}, here
    ) is False
    assert lint._delegation_is_read(
        "${{ steps.probe.outputs.key }}", {("z.yml", "build", "probe"): {"key"}}, here
    ) is False
    # A step this check never saw is not evidence of anything.
    assert lint._delegation_is_read(
        "${{ steps.other.outputs.key }}", producers, here
    ) is False
    # Nor is a form that names no step at all.
    assert lint._delegation_is_read(
        "${{ needs.build.outputs.key }}", producers, here
    ) is False


def test_a_readable_namesake_in_another_job_vouches_for_nothing(tmp_path):
    """Step ids are unique only within a job, so producers are identified by (document, job, step id); a bare id lets a namesake vouch for another."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n"
        # unreadable producer, and the job that actually caches
        "  first:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
        # readable namesake in a different job, caching nothing of interest
        "  second:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the caching job's own `probe` could not be read, whatever the other job's "
        f"step of the same name spells:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_inline_key_input_does_not_certify_an_unrelated_output(tmp_path):
    """A with: key input certifies only an action that itself publishes that key; a local action's unrelated key input says nothing about its outputs.key."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "sneaky"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: sneaky\n"
        "inputs:\n  key:\n    description: unrelated\n"
        "outputs:\n"
        "  key:\n"
        "    description: the real cache key\n"
        "    value: ${{ steps.inner.outputs.key }}\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - id: inner\n"
        "      shell: bash\n"
        "      run: printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: maker\n"
        "        uses: ./.github/actions/sneaky\n"
        "        with:\n          key: safe-key\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.maker.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the action's published key comes from a command that could not be read; the "
        f"`key` input it happens to accept says nothing about it:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_top_level_publish_input_is_not_resolved_by_a_child_targets_value(tmp_path):
    """A top-level workflow's own inputs are chosen by the dispatcher and must not be expanded from an unrelated child action's literal value."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "unrelated"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: unrelated\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.cache_key }}\n      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        description: chosen by whoever dispatches\n"
        "        type: string\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/unrelated\n"
        "        with:\n          cache_key: safe-key\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the dispatch input can be given `shared-key`, and the unrelated action's "
        f"`safe-key` says nothing about it:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_action_used_from_a_checkout_subdirectory_is_reachable(tmp_path):
    """uses: references under a checkout subdirectory (./unsloth/.github/actions/x) name the same action; notebooks-ci.yml and version-compat-ci.yml use it."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "nested-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: nested cache\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: shared-inner\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        # checked out under `unsloth/`, so the reference carries that prefix
        "      - uses: ./unsloth/.github/actions/nested-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: shared-inner\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the prefixed reference names the same action, whose key the publish workflow "
        f"restores exactly:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_commented_out_output_does_not_certify_a_producer(tmp_path):
    """Commented-out shell lines such as echo 'key=safe-key' are not evidence about a step and must not mark an unreadable producer readable."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: |\n"
        "          # echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
        "          printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the only line that runs is the printf, which this check cannot read:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_an_unquoted_scalar_key_is_compared(tmp_path):
    """Unquoted scalars such as key: 123 parse as int in YAML but are cache keys to Actions, so non-str values must be compared too."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: 123\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: 123\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"both workflows use the same key:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_wrapper_forwarding_its_own_input_is_resolved(tmp_path):
    """A wrapper forwarding ${{ inputs.name }} to a nested composite takes its value from the wrapper's callers, so it resolves to their literals."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    inner = root / "actions" / "inner-cache"
    wrapper = root / "actions" / "wrapper"
    wf.mkdir(parents = True)
    inner.mkdir(parents = True)
    wrapper.mkdir(parents = True)
    (inner / "action.yml").write_text(
        "name: inner cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (wrapper / "action.yml").write_text(
        "name: wrapper\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: ./.github/actions/inner-cache\n"
        "      with:\n        name: ${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/wrapper\n"
        "        with:\n          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-other\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"every caller passes `name: safe`, so the only PR key is `prefix-safe`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )

    # And the resolution keeps its teeth: the value really reaching the cache collides.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-safe\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"`prefix-safe` is exactly what the wrapper produces:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_one_readable_output_does_not_certify_another_from_the_same_step(tmp_path):
    """Readability is per output, not per step: a readable key=safe-key must not certify an unrecovered danger= output from the same step."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: |\n"
        "          echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
        "          printf 'danger=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.danger }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the cache uses `danger`, which was written by the form this check cannot "
        f"read:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_unrelated_assignment_does_not_certify_the_output(tmp_path):
    """An unrelated assignment such as safe_key= in the step body must not certify an output written by an unreadable printf."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: |\n"
        '          safe_key="safe-${RANDOM}"\n'
        "          printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the only line writing an output is the printf:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_publish_jobs_sharing_a_raw_key_keep_their_own_scopes(tmp_path):
    """(path, key) is not a unique identity: publish jobs sharing a raw key need their own scopes, not the first job's readable producer."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n"
        "  first:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: echo 'key=unrelated-v1' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
        "  second:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the second job's producer could not be read, and the first job's readable "
        f"namesake says nothing about it:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_key_passed_to_a_non_cache_action_is_not_a_cache_namespace(tmp_path):
    """A `key` input passed to an unrelated action is not a cache namespace, or a publish workflow caching release-key is wrongly rejected."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "signer"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: signer\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: some/signing-action@v1\n"
        "      with:\n        key: release-key\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/signer\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: release-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the pull request path caches nothing; `key` there is a signing key:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_call_site_reached_through_a_checkout_prefix_is_matched(tmp_path):
    """Reachability strips the runtime prefix but the call-site comparison did not, so prefixed literal inputs stayed
    undecidable, rejecting an unrelated publish key."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "prefixed-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: prefixed cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./repo/.github/actions/prefixed-cache\n"
        "        with:\n          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-other\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the prefixed call site passes `name: safe`, so the only PR key is "
        f"`prefix-safe`:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_publish_restore_prefix_is_expanded_with_its_own_inputs(tmp_path):
    """A publish restore-keys prefix is expanded with its own inputs; reducing it to the bare prefix collided with every PR key under that head."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pub-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pub restore\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n        path: wheels\n"
        "        key: prefix-${{ inputs.name }}-exact\n"
        "        restore-keys: |\n"
        "          prefix-${{ inputs.name }}-\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: prefix-pr-exact\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pub-restore\n"
        "        with:\n          name: publish\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the runtime fallback is `prefix-publish-`, which cannot reach "
        f"`prefix-pr-exact`:\n{proc.stdout}\n{proc.stderr}"
    )
