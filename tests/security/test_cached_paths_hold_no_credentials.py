# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""No job may persist a directory a credential is configured to live in.

On 2026-09-21 the Rust project disclosed that `cargo miri` wrote the ENTIRE process
environment to a file under `target/`, that CI cached `target/`, and that GitHub lets
`pull_request` runs restore caches written on the default branch. A secret that had only
ever existed in a privileged run became readable by anyone who could open a pull request,
and none of it was visible in the workflow YAML, because the leak happened inside a tool
nobody had reason to distrust.

The class needs four things at once: a secret in the job's environment, something that
serialises it to disk, that path being persisted by a cache save or an artifact upload,
and the result being readable from a pull request. This repository has no instance of it.
The link that holds is the second one, and only the second one, which is the part a
future commit can remove without looking dangerous.

`gemma4-audio-probe.yml` is why this file exists. It cached `~/.cache/huggingface`, which
is huggingface_hub's DEFAULT credential home: `login()`, `hf auth login` and
`HfFolder.save_token` all write a token to `$HF_HOME/token`, and since 0.25 also
`stored_tokens`, so the token file lived inside the cached path by construction. Nothing
in that workflow logs in and it holds no secret, so no credential was ever written. The
protection was the absence of a token, not anything about the cache, and "nobody has
added a token yet" is not a property a test can rely on.

It now points `HF_HOME` at `${{ github.workspace }}/hf-cache` and caches that, which is
how every model cache in `unslothai/unsloth` is arranged. That is a real improvement
rather than a cosmetic one: the cached tree becomes a directory this workflow creates,
instead of a shared location in the runner's home that other tools also write to. But it
does not remove the shape, because `HF_HOME` still names a credential home and that home
is still inside the cached path. Only the guard below closes it.

Two spellings are therefore both checked, and the second is the one the old workflow had:

  1. the job sets a credential-home variable to a directory inside a persisted path;
  2. the job persists a path that IS a known default credential home, with no such
     variable set at all -- which reads as harmless and is not.

The sibling guard in `unslothai/unsloth`,
`tests/studio/test_cached_paths_hold_no_credentials.py`, covers case 1. It would have
passed this workflow, because there was no `HF_HOME` to find. Case 2 is here for that
reason.

Deliberately NOT asserted: that a job may not hold a secret and write a cache. Those
co-occur legitimately and often, and a rule that broad would fail on a tree with nothing
wrong with it, which teaches people to add exemptions. What is actually dangerous is the
narrow thing above: a tool told to keep its credentials somewhere that is about to be
uploaded.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ACTIONS = REPO / ".github" / "actions"

# Variables that tell a tool where to keep credentials. The value is the credential file
# it writes, used only in the failure message.
CREDENTIAL_HOMES = {
    "HF_HOME": "token, stored_tokens",
    "HUGGINGFACE_HUB_CACHE": "token",
    "TRANSFORMERS_CACHE": "token",
    "NPM_CONFIG_USERCONFIG": ".npmrc auth tokens",
    "CARGO_HOME": "credentials.toml",
    "DOCKER_CONFIG": "config.json auth entries",
    "AWS_SHARED_CREDENTIALS_FILE": "aws credentials",
    "GOOGLE_APPLICATION_CREDENTIALS": "service account json",
}

# Where those tools keep credentials when nothing overrides them. Caching one of these
# is the same hazard reached without setting any variable, so it reads as harmless.
DEFAULT_CREDENTIAL_HOMES = {
    "~/.cache/huggingface": "HF_HOME default; token, stored_tokens",
    "~/.huggingface": "legacy HF_HOME default; token",
    "~/.cargo": "CARGO_HOME default; credentials.toml",
    "~/.docker": "DOCKER_CONFIG default; config.json auth entries",
    "~/.npmrc": "npm auth tokens",
    "~/.aws": "aws credentials",
    "~/.config/gh": "gh CLI oauth token",
}

LOGIN_PATTERNS = (
    r"\bhf\s+auth\s+login\b",
    r"\bhuggingface-cli\s+login\b",
    r"\bhf\s+login\b",
    r"huggingface_hub[.\s]*\.?\s*login\s*\(",
    r"\bfrom\s+huggingface_hub\s+import\s+[^\n]*\blogin\b",
    r"\bHfFolder\b[^\n]*\bsave_token\b",
    r"\bsave_token\s*\(",
    r"add_to_git_credential\s*=\s*True",
    r"\bnpm\s+login\b",
    r"\bcargo\s+login\b",
    r"\bdocker\s+login\b",
    r"\bgcloud\s+auth\s+(?:application-default\s+)?login\b",
    r"\baws\s+configure\b",
)

_PERSIST = ("actions/cache/save", "actions/cache@", "actions/upload-artifact")


def _docs():
    for base, pattern in ((WORKFLOWS, "*.y*ml"), (ACTIONS, "action.y*ml")):
        if not base.is_dir():
            continue
        paths = sorted(base.glob(pattern)) if base is WORKFLOWS else sorted(base.rglob(pattern))
        for path in paths:
            try:
                doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
            except yaml.YAMLError:
                continue
            if isinstance(doc, dict):
                yield path, doc


def _jobs(doc):
    jobs = doc.get("jobs")
    if isinstance(jobs, dict):
        for jid, job in jobs.items():
            if isinstance(job, dict):
                yield jid, job
    runs = doc.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
        yield "runs", {"steps": runs["steps"]}


def _steps(job):
    steps = job.get("steps")
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _env_of(job, doc):
    env = {}
    for source in (doc.get("env"), job.get("env")):
        if isinstance(source, dict):
            env.update({str(k): str(v) for k, v in source.items()})
    return env


def _persisted_paths(job):
    out = []
    for step in _steps(job):
        uses = str(step.get("uses") or "").casefold()
        if not any(m.casefold() in uses for m in _PERSIST):
            continue
        with_ = step.get("with")
        if not isinstance(with_, dict) or with_.get("path") is None:
            continue
        for line in str(with_["path"]).splitlines():
            line = line.strip()
            if line and not line.startswith("!"):
                out.append(line)
    return out


def _normalise(path: str) -> str:
    """Strip expressions and separators so two path spellings can be compared."""
    path = re.sub(r"\$\{\{[^}]*\}\}", "", path)
    path = path.replace("\\", "/").strip().strip("'\"")
    path = re.sub(r"^\$(HOME|\{HOME\})/", "~/", path)
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def _inside(inner: str, outer: str) -> bool:
    inner, outer = _normalise(inner), _normalise(outer)
    inner, outer = inner.strip("/"), outer.strip("/")
    if not inner or not outer:
        return False
    return inner == outer or inner.startswith(outer + "/")


def _findings():
    """(label, what, detail) for every job persisting a credential home."""
    for path, doc in _docs():
        for jid, job in _jobs(doc):
            env = _env_of(job, doc)
            persisted = _persisted_paths(job)
            if not persisted:
                continue
            label = f"{path.name}:{jid}"
            for var, creds in CREDENTIAL_HOMES.items():
                if var not in env:
                    continue
                for p in persisted:
                    if _inside(env[var], p):
                        yield label, f"{var}={env[var]}", f"inside cached {p!r} ({creds})"
            for p in persisted:
                for default, creds in DEFAULT_CREDENTIAL_HOMES.items():
                    if _inside(default, p) or _inside(p, default):
                        yield label, f"cached path {p!r}", f"is a default credential home ({creds})"


def test_the_scan_finds_the_workflows_it_claims_to():
    """A scan that matched nothing would pass every check below on an empty set."""
    persisting = {
        f"{path.name}:{jid}"
        for path, doc in _docs()
        for jid, job in _jobs(doc)
        if _persisted_paths(job)
    }
    assert len(persisting) >= 3, (
        f"only found {len(persisting)} jobs that persist a path; the scan is wrong"
    )
    assert any(n.startswith("gemma4-audio-probe.yml") for n in persisting), (
        "gemma4-audio-probe.yml caches a model checkpoint but the scan missed it"
    )


def test_the_inside_predicate_reads_the_path():
    """The guard is only as good as this predicate, so the predicate is tested too."""
    cases = [
        ("hf-cache", "hf-cache", True),
        ("${{ github.workspace }}/hf-cache", "hf-cache", True),
        ("hf-cache/hub", "hf-cache", True),
        ("./hf-cache", "hf-cache", True),
        ("$HOME/.cache/huggingface", "~/.cache/huggingface", True),
        ("hf-cache-vision", "hf-cache", False),   # prefix, not a child
        ("other", "hf-cache", False),
        ("", "hf-cache", False),
    ]
    for inner, outer, expected in cases:
        assert _inside(inner, outer) is expected, f"_inside({inner!r}, {outer!r})"


def test_no_job_persists_a_default_credential_home():
    """The spelling that needs no variable set, and so reads as harmless.

    gemma4-audio-probe.yml cached `~/.cache/huggingface` until 2026-09-22. That is where
    huggingface_hub keeps its token unless HF_HOME says otherwise, so the token file was
    inside the cached path with nothing in the workflow hinting at it.
    """
    offenders = [
        f"{label}: {what} {detail}"
        for label, what, detail in _findings()
        if "default credential home" in detail
    ]
    assert not offenders, (
        "these jobs cache or upload a directory that is a tool's default credential "
        "home:\n  " + "\n  ".join(sorted(offenders)) + "\n\n"
        "Anything that logs in writes a token there, and the result is saved to a cache "
        "or uploaded. GitHub lets every pull request restore caches written on the "
        "default branch, so the token becomes readable by anyone who can open one.\n\n"
        "Point the tool at a directory this workflow owns and cache that instead, which "
        "is what gemma4-audio-probe.yml does with "
        "`HF_HOME: ${{ github.workspace }}/hf-cache` and `path: hf-cache`."
    )


@pytest.mark.parametrize(
    "label",
    sorted({label for label, _, _ in _findings()}) or ["<none>"],
)
def test_a_job_that_persists_a_credential_home_performs_no_login(label):
    if label == "<none>":
        pytest.skip("no job persists a credential home, which is the desired state")
    name, jid = label.split(":", 1)
    path = WORKFLOWS / name
    if not path.exists():
        candidates = list(ACTIONS.rglob(name))
        path = candidates[0] if candidates else path
    doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
    job = dict(_jobs(doc)).get(jid) or {}

    offenders = []
    for step in _steps(job):
        body = str(step.get("run") or "")
        for pattern in LOGIN_PATTERNS:
            if body and re.search(pattern, body, re.IGNORECASE):
                offenders.append(f"{step.get('name') or step.get('uses')}: /{pattern}/")

    detail = "; ".join(sorted({d for lb, _, d in _findings() if lb == label}))
    assert not offenders, (
        f"{label} persists a credential home ({detail}) and a step in it logs in:\n  "
        + "\n  ".join(offenders) + "\n\n"
        "A login writes a real token into that directory, and the directory is then "
        "cached or uploaded, where any pull request can restore it. Read the token from "
        "the environment instead of logging in, or move the credential home outside the "
        "persisted path."
    )
