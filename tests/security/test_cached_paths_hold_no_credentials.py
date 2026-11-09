# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""No job may persist a credential home, set via env var or at its default like ~/.cache/huggingface."""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ACTIONS = REPO / ".github" / "actions"

# Env vars telling a tool where to keep credentials; values name the file, shown only in failures.
CREDENTIAL_HOMES = {
    "HF_HOME": "token, stored_tokens",
    "HF_TOKEN_PATH": "the token file itself",
    "NPM_CONFIG_USERCONFIG": ".npmrc auth tokens",
    "CARGO_HOME": "credentials.toml",
    "DOCKER_CONFIG": "config.json auth entries",
    "AWS_SHARED_CREDENTIALS_FILE": "aws credentials",
    "GOOGLE_APPLICATION_CREDENTIALS": "service account json",
}

# HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE pick model caches, not credential homes (hub 1.32.0).

# Credential filenames per home; an empty tuple means the variable names the credential file itself.
CREDENTIAL_FILES = {
    "HF_HOME": ("token", "stored_tokens"),
    "HF_TOKEN_PATH": (),
    "NPM_CONFIG_USERCONFIG": (),
    "CARGO_HOME": ("credentials", "credentials.toml"),
    "DOCKER_CONFIG": ("config.json",),
    "AWS_SHARED_CREDENTIALS_FILE": (),
    "GOOGLE_APPLICATION_CREDENTIALS": (),
}


# Files each default home holds, so persisting one exactly (~/.cargo/credentials.toml) is caught.
DEFAULT_HOME_FILES = {
    "~/.cache/huggingface": ("token", "stored_tokens"),
    "~/.huggingface": ("token",),
    "~/.cargo": ("credentials", "credentials.toml"),
    "~/.docker": ("config.json",),
    "~/.npmrc": (),
    "~/.aws": ("credentials",),
    "~/.config/gh": ("hosts.yml",),
}

DEFAULT_CREDENTIAL_HOMES = {
    "~/.cache/huggingface": "HF_HOME default; token, stored_tokens",
    "~/.huggingface": "legacy HF_HOME default; token",
    "~/.cargo": "CARGO_HOME default; credentials.toml",
    "~/.docker": "DOCKER_CONFIG default; config.json auth entries",
    "~/.npmrc": "npm auth tokens",
    "~/.aws": "aws credentials",
    "~/.config/gh": "gh CLI oauth token",
}

# Vars that move a default home's credentials; HF_HOME moves all, HF_TOKEN_PATH only the token files.
DEFAULT_OWNERS = {
    "~/.cache/huggingface": ("HF_HOME", "HF_TOKEN_PATH"),
    "~/.huggingface": ("HF_HOME", "HF_TOKEN_PATH"),
    "~/.cargo": ("CARGO_HOME",),
    "~/.docker": ("DOCKER_CONFIG",),
    "~/.npmrc": ("NPM_CONFIG_USERCONFIG",),
    "~/.aws": ("AWS_SHARED_CREDENTIALS_FILE",),
    "~/.config/gh": (),
}

# Home each login writes; None means any credential home, used for Hugging Face logins.
_HF_HOMES = ("HF_HOME", "HF_TOKEN_PATH")

LOGIN_PATTERN_HOMES = {
    # hf auth login writes only the Hugging Face homes; unmapped logins fell back to any home and falsely paired with unrelated caches such as CARGO_HOME.
    r"\bhf\s+auth\s+login\b": _HF_HOMES,
    r"\bhuggingface-cli\s+login\b": _HF_HOMES,
    r"\bhf\s+login\b": _HF_HOMES,
    r"huggingface_hub[.\s]*\.?\s*login\s*\(": _HF_HOMES,
    r"\bfrom\s+huggingface_hub\s+import\s+[^\n]*\blogin\b": _HF_HOMES,
    r"\bHfFolder\b[^\n]*\bsave_token\b": _HF_HOMES,
    r"\bsave_token\s*\(": _HF_HOMES,
    r"add_to_git_credential\s*=\s*True": _HF_HOMES,
    r"\bnpm\s+login\b": ("NPM_CONFIG_USERCONFIG",),
    r"\bcargo\s+login\b": ("CARGO_HOME",),
    r"\bdocker\s+login\b": ("DOCKER_CONFIG",),
    r"\bgcloud\s+auth\s+(?:application-default\s+)?login\b": (
        "GOOGLE_APPLICATION_CREDENTIALS",
    ),
    r"\baws\s+configure\b": ("AWS_SHARED_CREDENTIALS_FILE",),
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

# Maps each action to the home variables it actually writes, since keying on the action alone falsely flagged
# setup-node; condition names a with: input the write needs (registry-url).
LOGIN_ACTIONS = {
    "docker/login-action": {
        "vars": ("DOCKER_CONFIG",),
        "why": "registry auth into $DOCKER_CONFIG/config.json",
    },
    "aws-actions/configure-aws-credentials": {
        "vars": ("AWS_SHARED_CREDENTIALS_FILE",),
        "why": "aws credentials",
    },
    "google-github-actions/auth": {
        "vars": ("GOOGLE_APPLICATION_CREDENTIALS",),
        "why": "an application default credentials file",
    },
    "actions/setup-node": {
        "vars": ("NPM_CONFIG_USERCONFIG",),
        "why": "an .npmrc auth token",
        "condition": "registry-url",
    },
}

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


def _step_envs(job):
    """One env dict per step, merged over the job env; job-level-only reads missed step-set homes."""
    out = []
    for step in _steps(job):
        source = step.get("env")
        if isinstance(source, dict):
            out.append({str(k): str(v) for k, v in source.items()})
    return out


def _expand(value: str, env: dict, inputs: dict | None = None) -> str:
    """Resolve env/inputs refs first: _normalise erases unresolved ones to empty, which _inside refuses."""
    # Iterates since one variable may name another (one pass let _normalise erase a nested expression); capped at 8
    # passes so mutual references end; leftovers are handled as unknown.
    for _ in range(8):
        before = value
        value = re.sub(
            r"\$\{\{\s*env\.([A-Za-z_]\w*)\s*\}\}",
            lambda m: env.get(m.group(1), ""),
            value,
        )
        if inputs:
            value = re.sub(
                r"\$\{\{\s*inputs\.([A-Za-z_][\w-]*)\s*\}\}",
                lambda m: str(inputs.get(m.group(1), "")),
                value,
            )
        if value == before:
            break
    return value

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




def _reusable_jobs(job, env = None, inputs = None):
    """A `uses:` job has no steps; the called workflow is scanned with the caller's `with:` values."""
    ref = str(job.get("uses") or "").strip().strip("'\"")
    if not ref.startswith("./"):
        return []
    target = REPO / ref[2:]
    if not target.is_file():
        return []
    try:
        doc = yaml.safe_load(target.read_text(encoding = "utf-8"))
    except yaml.YAMLError:
        return []
    if not isinstance(doc, dict):
        return []
    passed = {}
    on = doc.get(True) if True in doc else doc.get("on")
    call = on.get("workflow_call") if isinstance(on, dict) else None
    declared = call.get("inputs") if isinstance(call, dict) else None
    if isinstance(declared, dict):
        for name, spec in declared.items():
            if isinstance(spec, dict) and spec.get("default") is not None:
                passed[str(name)] = str(spec["default"])
    with_ = job.get("with")
    if isinstance(with_, dict):
        # Forwarded with: values resolve against the caller's env and inputs; copied verbatim, ${{ inputs.path }} went
        # self-referential and the login plus cache was wrongly accepted.
        passed.update({
            str(k): _expand(str(v), env or {}, inputs or {}) for k, v in with_.items()
        })
    out = []
    for _jid, inner in _jobs(doc):
        out.append((inner, _env_of(inner, doc), passed))
    return out


def _units(job, doc, env = None, inputs = None, depth = 0):
    """Each job of a called reusable workflow is its own runner, so their paths must not be pooled."""
    env = _env_of(job, doc) if env is None else env
    inputs = {} if inputs is None else inputs
    if depth > 8:
        return [(job, env, inputs)]
    inner = _reusable_jobs(job, env, inputs)
    if not inner:
        return [(job, env, inputs)]
    out = []
    for inner_job, inner_env, passed in inner:
        # Uses the called workflow's own env: GitHub does not pass a caller's env: into a reusable workflow (only with:
        # and secrets: cross), so merging them gave false reports; inputs still travel.
        out.extend(_units(inner_job, doc, inner_env, passed, depth + 1))
    return out



def _ref_candidates(uses: str):
    """Every source-tree path a `./` ref could name, incl. `./unsloth/...` after a subdirectory checkout."""
    ref = uses.strip()
    if ref.startswith("./"):
        ref = ref[2:]
    ref = ref.rstrip("/")
    bases = [REPO / ref]
    parts = ref.split("/")
    if ".github" in parts[1:]:
        bases.append(REPO / "/".join(parts[parts.index(".github"):]))
    for base in bases:
        for cand in (base, base / "action.yml", base / "action.yaml"):
            yield cand


def _flat_steps(job, inherited = None, inputs = None, stack = None):
    """Carries invoking env and `with:` into local composites; `stack` is the recursion path, not visited."""
    inherited = {} if inherited is None else inherited
    inputs = {} if inputs is None else inputs
    stack = () if stack is None else stack
    out = []
    # A job may delegate wholesale to a local reusable workflow instead of listing steps.

    for step in _steps(job):
        own = step.get("env")
        env = {**inherited, **({str(k): str(v) for k, v in own.items()}
                               if isinstance(own, dict) else {})}
        out.append((step, inherited, inputs))
        uses = str(step.get("uses") or "").strip().strip("'\"")
        if not uses.startswith("./"):
            continue
        for cand in _ref_candidates(uses):
            if not cand.is_file():
                continue
            if cand in stack:
                break                      # a cycle, not a sibling invocation
            try:
                doc = yaml.safe_load(cand.read_text(encoding = "utf-8"))
            except yaml.YAMLError:
                break
            if not isinstance(doc, dict):
                break
            runs = doc.get("runs")
            if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
                with_ = step.get("with")
                passed = {}
                # A declared input default is what Actions applies when the caller omits the input, so a composite invoked bare still names its real path.
                declared = doc.get("inputs")
                if isinstance(declared, dict):
                    for field, spec in declared.items():
                        if isinstance(spec, dict) and spec.get("default") is not None:
                            passed[str(field)] = str(spec["default"])
                if isinstance(with_, dict):
                    for field, value in with_.items():
                        # Resolved against the caller's env and inputs, so a forwarded ${{ inputs.path }} arrives as its
                        # value, not as the same expression one level down.
                        passed[str(field)] = _expand(str(value), env, inputs)
                out.extend(_flat_steps(
                    {"steps": runs["steps"]}, env, passed, stack + (cand,),
                ))
            break
    return out


def _local_action_steps(job):
    """The composite-provided steps only, for tests that assert flattening happened."""
    own = {id(s) for s in _steps(job)}
    return [step for step, _env, _in in _flat_steps(job) if id(step) not in own]


def _persisted_with_env(job, doc):
    """Persisted paths paired with the env of the saving step, since Actions resolves `path:` in that env."""
    out = []
    for unit, job_env, unit_inputs in _units(job, doc):
        out.extend(_persisted_in_unit(unit, job_env, unit_inputs))
    return out


def _persisted_in_unit(job, job_env, unit_inputs):
    """The paths one runner persists. See `_units` for why that boundary matters."""
    out = []
    # The job's own env seeds the walk; without it a root-level ${{ env.CACHE_DIR }} path expanded to empty and the composite looked like it persisted nothing.
    for step, inherited, inputs in _flat_steps(job, job_env, unit_inputs):
        uses = str(step.get("uses") or "").casefold()
        if not any(marker.casefold() in uses for marker in _PERSIST):
            continue
        with_ = step.get("with")
        if not isinstance(with_, dict) or with_.get("path") is None:
            continue
        own = step.get("env")
        env = {**job_env, **inherited, **({str(k): str(v) for k, v in own.items()}
                                         if isinstance(own, dict) else {})}
        for line in str(with_["path"]).splitlines():
            line = line.strip()
            if line and not line.startswith("!"):
                out.append((_expand(line, env, inputs), step))
    return out


def _login_offenders(doc, job):
    """Judged per step's own env; login actions (e.g. docker/login-action) count too, per LOGIN_ACTIONS."""
    offenders = []
    # Per runner, and a login is only ever paired with what its OWN runner persists.
    for unit, unit_env, unit_inputs in _units(job, doc):
        offenders.extend(_login_offenders_in_unit(unit, unit_env, unit_inputs))
    return offenders


def _login_offenders_in_unit(job, job_env, unit_inputs):
    persisted = [path for path, _s in _persisted_in_unit(job, job_env, unit_inputs)]
    if not persisted:
        return []
    offenders = []
    for step, inherited, inputs in _flat_steps(job, job_env, unit_inputs):
        body = str(step.get("run") or "")
        uses = str(step.get("uses") or "").strip().strip("'\"")
        action = uses.split("@")[0]
        # GitHub matches owner/repo case-insensitively, so Docker/login-action is the same action; a case-sensitive lookup let a capital letter skip the step.
        spec = LOGIN_ACTIONS.get(action.casefold())
        if spec is not None:
            with_ = step.get("with")
            needed = spec.get("condition")
            if needed and not (isinstance(with_, dict) and with_.get(needed) is not None):
                spec = None
        if not body and spec is None:
            continue
        own = step.get("env")
        env = {**job_env, **inherited, **({str(k): str(v) for k, v in own.items()}
                                         if isinstance(own, dict) else {})}
        homes = {v: _expand(str(env[v]), env, inputs) for v in CREDENTIAL_HOMES if v in env}
        # HF_TOKEN_PATH takes precedence over $HF_HOME/token when set, so naming HF_HOME for that token was a false failure that misdescribed where it lives.
        if "HF_TOKEN_PATH" in homes:
            homes.pop("HF_HOME", None)
        if not homes:
            continue
        for var, home in sorted(homes.items()):
            hit = next(
                (p for p in persisted if _inside(home, p, CREDENTIAL_FILES.get(var, ()))),
                None,
            )
            if hit is None:
                continue
            if spec is not None and var in spec["vars"]:
                offenders.append(
                    f"{step.get('name') or uses}: {var}={home} is inside cached "
                    f"{hit!r}, and this step uses {action} ({spec['why']})"
                )
                break
            matched = next(
                (
                    p
                    for p in LOGIN_PATTERNS
                    if body
                    and re.search(p, body, re.IGNORECASE)
                    and var in LOGIN_PATTERN_HOMES.get(p, (var,))
                ),
                None,
            )
            if matched is not None:
                offenders.append(
                    f"{step.get('name') or uses or 'run'}: {var}={home} is inside "
                    f"cached {hit!r}, and this step matches /{matched}/"
                )
                break
    return offenders

def _raw_persisted(job, doc = None):
    """Expanded persisted paths, including those declared inside local composites."""
    return [path for path, _step in _persisted_with_env(job, doc or {})]

def _normalise(path: str) -> str:
    """Strip expressions and separators so two path spellings can be compared."""
    path = re.sub(r"\$\{\{[^}]*\}\}", "", path)
    path = path.replace("\\", "/").strip().strip("'\"")
    path = re.sub(r"^\$(HOME|\{HOME\})/", "~/", path)
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")




def _glob_regex(pattern: str):
    """A glob compiled with `/` respected: `**` crosses separators, `*` does not."""
    out = []
    i = 0
    pattern = pattern.replace("\\", "/")
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            # **/ matches zero directories, so hf-cache/**/token matches hf-cache/token; the glob translation must not demand a separator there.
            out.append("(?:[^/]+/)*")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if ch == "[":
            # Bracket expressions compile as glob classes; escaping them matched only the literal text, so an upload that does include the token was permitted.
            close = pattern.find("]", i + 1)
            if close == -1:
                out.append(re.escape(ch))
                i += 1
                continue
            body = pattern[i + 1 : close]
            negate = body.startswith("!") or body.startswith("^")
            if negate:
                body = body[1:]
            out.append("[" + ("^" if negate else "") + body.replace("\\", "\\\\") + "]")
            i = close + 1
            continue
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _glob_captures(pattern: str, home: str, files = None) -> bool:
    """Tests the real glob: `hf-cache/**` can capture the token, `hf-cache/*.bin` cannot."""
    home = _normalise(home).rstrip("/")
    if files is None:
        # When files is None the filename is unknown, so fall back to plain containment; an empty tuple would silently switch the finding off.
        outer = _normalise(_deglob(pattern)).strip("/")
        return bool(outer) and (home == outer or home.startswith(outer + "/"))
    rx = _glob_regex(_normalise(pattern))
    # Candidates are the home itself, for a variable naming a credential file, plus each known filename beneath it, for one naming a directory.
    candidates = [home] + [home + "/" + f for f in files]
    return any(rx.match(c) for c in candidates)


def _deglob(path: str) -> str:
    """Fixed leading directory of a glob, used for containment; drops from the first wildcard segment on."""
    parts = []
    for segment in path.replace("\\", "/").split("/"):
        if any(ch in segment for ch in "*?["):
            break
        parts.append(segment)
    return "/".join(parts) if parts else path

def _inside(inner: str, outer: str, files = None) -> bool:
    """True if inner equals outer or lies below it; a glob suffix on outer is trimmed first."""
    # Identical expressions match before stripping: they name the same directory at run time, but _normalise erased both
    # to empty, which _inside refuses, so the login passed.
    if "${{" in inner and inner.strip() == outer.strip():
        return True
    if any(ch in outer for ch in "*?["):
        # Glob cases are decided by matching the credential's own path against the pattern, which is stricter than widening the pattern to its fixed prefix.
        return _glob_captures(outer, inner, files)
    inner, outer = _normalise(inner), _normalise(_deglob(outer))
    inner, outer = inner.strip("/"), outer.strip("/")
    if not inner or not outer:
        return False
    if inner == outer or inner.startswith(outer + "/"):
        return True
    # A persisted path may name the credential file itself, which containment misses; match it against the same known filenames a glob is tested against.
    return any(outer == inner.rstrip("/") + "/" + f for f in (files or ()))



def _default_home_hits(persisted: str, overridden: set) -> list:
    """Default credential homes a persisted path reaches; kept separate so tests can exercise the rule."""
    hits = []
    for default, creds in DEFAULT_CREDENTIAL_HOMES.items():
        # A default home only holds credentials when nothing overrides it: a job setting CARGO_HOME writes elsewhere, so
        # flagging persisted ~/.cargo was a false failure.
        if any(v in overridden for v in DEFAULT_OWNERS.get(default, ())):
            continue
        # Matched with the filenames that home holds, so persisting the credential file outright counts as well as persisting the directory.
        if _inside(default, persisted, DEFAULT_HOME_FILES.get(default, ())):
            hits.append((default, creds))
    return hits


def _findings():
    """Jobs persisting a credential home, resolved via `_units` so called reusable workflows are included."""
    for path, doc in _docs():
        for jid, job in _jobs(doc):
          for unit, unit_env, unit_inputs in _units(job, doc):
            job_env = unit_env
            raw = [p for p, _s in _persisted_in_unit(unit, unit_env, unit_inputs)]
            if not raw:
                continue
            label = f"{path.name}:{jid}"
            # Walks _flat_steps so an env declared inside a composite counts; _step_envs sees only the job's own steps,
            # so a composite's HF_HOME gave no label and the guard never got the job, though _login_offenders detects
            # it.
            scopes = [job_env]
            for step, inherited, _si in _flat_steps(unit, job_env, unit_inputs):
                own = step.get("env")
                scopes.append({
                    **inherited,
                    **({str(k): str(v) for k, v in own.items()}
                       if isinstance(own, dict) else {}),
                })
            for scope in scopes:
                env = {**job_env, **scope}
                persisted = [_expand(p, env, unit_inputs) for p in raw]
                present = [v for v in CREDENTIAL_HOMES if v in env]
                # HF_TOKEN_PATH overrides the location HF_HOME implies; the two are alternatives with a precedence, not two destinations.
                if "HF_TOKEN_PATH" in present:
                    present = [v for v in present if v != "HF_HOME"]
                for var in present:
                    creds = CREDENTIAL_HOMES[var]
                    home = _expand(env[var], env, unit_inputs)
                    for p in persisted:
                        if _inside(home, p, CREDENTIAL_FILES.get(var, ())):
                            yield label, f"{var}={env[var]}", f"inside cached {p!r} ({creds})"
            # Same hazard without setting any variable: only a credential home INSIDE the persisted path counts; caching
            # ~/.cache/huggingface/hub with the token as a sibling is the recommended setup; read the unit's own env,
            # since caller env does not cross a reusable workflow boundary.
            overridden = {v for v in CREDENTIAL_HOMES if v in job_env}
            for p in (_expand(p, job_env, unit_inputs) for p in raw):
                for _default, creds in _default_home_hits(p, overridden):
                    yield label, f"cached path {p!r}", f"is a default credential home ({creds})"


def test_the_scan_finds_the_workflows_it_claims_to():
    """A scan that matched nothing would pass every check below on an empty set."""
    persisting = {
        f"{path.name}:{jid}"
        for path, doc in _docs()
        for jid, job in _jobs(doc)
        if _persisted_with_env(job, doc)
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


def test_expanding_an_env_reference_finds_the_path_the_expression_names():
    """`path: ${{ env.X }}` must expand first: unexpanded it normalises to empty, which `_inside` refuses."""
    env = {"HF_HOME": "${{ github.workspace }}/hf-cache"}
    assert _expand("${{ env.HF_HOME }}", env) == "${{ github.workspace }}/hf-cache"
    assert _inside(_expand("${{ env.HF_HOME }}", env), "hf-cache") is True
    # An undefined name expands to empty, which is what Actions itself does.
    assert _expand("${{ env.NOT_SET }}", env) == ""
    # Nothing else is touched, so `github.workspace` is still normalised away later.
    assert _expand("hf-cache", env) == "hf-cache"


def test_a_subdirectory_of_a_credential_home_is_not_a_finding():
    """Caching ~/.cache/huggingface/hub is safe, not a hazard: the token sits beside hub, not inside it."""
    assert _inside("~/.cache/huggingface/hub", "~/.cache/huggingface") is True
    assert _inside("~/.cache/huggingface", "~/.cache/huggingface/hub") is False
    home, creds = "~/.cache/huggingface", "x"
    flagged = [p for p in ("~/.cache/huggingface/hub", "~/.cache", home) if _inside(home, p)]
    assert flagged == ["~/.cache", home], (
        "only a persisted path CONTAINING the credential home is a finding; the hub "
        f"subdirectory must not be, and the set flagged was {flagged}"
    )


def test_a_credential_home_set_on_a_step_is_seen():
    """Credential homes set in step-level `env:` must be read; job-level-only scanning missed them."""
    job = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"run": "python probe.py", "env": {"HF_HOME": "hf-cache"}},
    ]}
    scopes = _step_envs(job)
    assert {"HF_HOME": "hf-cache"} in scopes, f"step env not collected: {scopes}"
    assert _env_of(job, {}) == {}, "the job itself sets nothing, which is the point"
    assert any(
        _inside(scope["HF_HOME"], path)
        for scope in scopes if "HF_HOME" in scope
        for path in (p for p, _s in _persisted_with_env(job, {}))
    ), "the step-scoped credential home is inside the cached path and must be a finding"


def test_no_job_persists_a_default_credential_home():
    """Default credential home persisted with no variable set, e.g. ~/.cache/huggingface, holds the token."""
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

    # Checks each step with its own env, including the steps of local composites; a job-wide check was too broad (a
    # later uncached step) and too narrow (a login inside an action).
    offenders = _login_offenders(doc, job)

    detail = "; ".join(sorted({d for lb, _, d in _findings() if lb == label}))
    assert not offenders, (
        f"{label} persists a credential home ({detail}) and a step in it logs in:\n  "
        + "\n  ".join(offenders) + "\n\n"
        "A login writes a real token into that directory, and the directory is then "
        "cached or uploaded, where any pull request can restore it. Read the token from "
        "the environment instead of logging in, or move the credential home outside the "
        "persisted path."
    )


def test_model_cache_variables_are_not_treated_as_credential_homes():
    """HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE pick model caches, not credential homes (hub 1.32.0)."""
    for var in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        assert var not in CREDENTIAL_HOMES, (
            f"{var} names a model cache, not a credential home. The token is read from "
            f"HF_TOKEN_PATH (default $HF_HOME/token), computed independently of it, so "
            f"caching a directory {var} points at persists blobs and no token."
        )
    assert "HF_TOKEN_PATH" in CREDENTIAL_HOMES, (
        "HF_TOKEN_PATH names the token file directly and overrides HF_HOME, so it is the "
        "variable that actually has to be tracked"
    )

    hub = pytest.importorskip("huggingface_hub.constants")
    home = str(getattr(hub, "HF_HOME", ""))
    token = str(getattr(hub, "HF_TOKEN_PATH", ""))
    cache = str(getattr(hub, "HUGGINGFACE_HUB_CACHE", ""))
    assert token.startswith(home) and token.endswith("token"), (
        f"expected the token under HF_HOME; got HF_HOME={home!r} HF_TOKEN_PATH={token!r}"
    )
    assert not _inside(token, cache), (
        f"the token is supposed to sit OUTSIDE the hub cache, which is the reason "
        f"HUGGINGFACE_HUB_CACHE is not a credential home; got {token!r} inside {cache!r}"
    )


def test_a_login_is_judged_against_that_step_s_own_environment():
    """Judge each login by its own step's effective env, not every home against every login in the job."""
    safe = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"name": "warm the cache", "run": "python download.py", "env": {"HF_HOME": "hf-cache"}},
        {"name": "log in elsewhere", "run": "hf auth login --token x",
         "env": {"HF_HOME": "/tmp/scratch-home"}},
    ]}
    assert _login_offenders({}, safe) == [], (
        "the login step points HF_HOME at an uncached directory, so its token cannot "
        "enter the cache and it must not be a finding"
    )

    # And the same shape with the login step's own home inside the cache must fire.
    unsafe = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"name": "log in", "run": "hf auth login --token x", "env": {"HF_HOME": "hf-cache"}},
    ]}
    offenders = _login_offenders({}, unsafe)
    assert offenders, "a login whose own HF_HOME is inside the cached path must be a finding"
    assert "hf-cache" in offenders[0] and "log in" in offenders[0], offenders

    # A job-level home with a login anywhere in the job is still caught; the per-step check must not have lost that original rule.
    job_level = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
            {"name": "log in", "run": "huggingface-cli login"},
        ],
    }
    assert _login_offenders({}, job_level), (
        "a job-level credential home inside the cached path, with a login in any step, "
        "is the case this module was written for and must still fire"
    )


def test_a_login_inside_a_local_composite_action_is_seen(tmp_path, monkeypatch):
    """A local composite's steps run in the calling job, so its logins must be scanned with that job."""
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "hf-login"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: hf login\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - shell: bash\n"
        "      run: hf auth login --token \"$HF_TOKEN\"\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
            {"uses": "./.github/actions/hf-login"},
        ],
    }
    assert _local_action_steps(job), "the composite's steps were not flattened into the job"
    offenders = _login_offenders({}, job)
    assert offenders, (
        "the login happens inside the composite, and it writes into the cached HF_HOME "
        "just the same, so it must be a finding"
    )
    assert "hf auth login" in offenders[0] or "login" in offenders[0], offenders


def test_a_composite_invoked_with_a_credential_home_carries_that_env(tmp_path, monkeypatch):
    """The invoking step's env applies while a composite runs, so it is carried into its inner steps."""
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "hf-login"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: hf login\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - shell: bash\n"
        "      run: hf auth login --token \"$HF_TOKEN\"\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"uses": "./.github/actions/hf-login", "env": {"HF_HOME": "hf-cache"}},
    ]}
    offenders = _login_offenders({}, job)
    assert offenders, (
        "the credential home is set on the step that invokes the composite, and the "
        "composite's login writes into it, so this must be a finding"
    )


def test_persistence_inside_a_local_composite_is_seen(tmp_path, monkeypatch):
    """A local composite can hold `actions/cache/save`, so composites must be flattened for persistence."""
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "save-cache"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: save cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: hf-cache\n"
        "        key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/save-cache"},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert "hf-cache" in paths, f"the composite's cache save was not collected: {paths}"
    assert _login_offenders({}, job), (
        "the composite saves the job's credential home and the workflow logs in, which "
        "is the combination this module exists to refuse"
    )


def test_a_cached_path_is_expanded_with_its_own_step_s_environment():
    """Actions resolves `path:` against the saving step's own env, not another step's environment."""
    job = {"steps": [
        {"name": "save", "uses": "actions/cache/save@v4",
         "with": {"path": "${{ env.CACHE_DIR }}", "key": "k"},
         "env": {"CACHE_DIR": "creds"}},
        {"name": "log in", "run": "hf auth login --token x", "env": {"HF_HOME": "creds/hf"}},
    ]}
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["creds"], (
        f"the path had to resolve against the saving step's own CACHE_DIR; got {paths}"
    )
    assert _login_offenders({}, job), (
        "HF_HOME is creds/hf, inside the cached creds, and that step logs in"
    )


def test_every_invocation_of_a_composite_is_flattened(tmp_path, monkeypatch):
    """Each composite call has its own env, so none may be skipped; cycle checks use the recursion path."""
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "hf-login"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: hf login\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - shell: bash\n"
        "      run: hf auth login --token x\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"uses": "./.github/actions/hf-login", "env": {"HF_HOME": "/tmp/outside"}},
        {"uses": "./.github/actions/hf-login", "env": {"HF_HOME": "hf-cache"}},
    ]}
    offenders = _login_offenders({}, job)
    assert offenders, (
        "the SECOND invocation points HF_HOME inside the cached path, and suppressing it "
        "as already-visited is what hid the dangerous one behind the safe one"
    )
    # Two invocations of the one-step composite, so both inner steps are present.
    assert len(_local_action_steps(job)) == 2, _local_action_steps(job)


def test_a_composite_cached_path_given_by_input_is_resolved(tmp_path, monkeypatch):
    """A composite's cached path may be `${{ inputs.path }}`; resolve it from the caller's `with:`."""
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "save-in"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: save in\n"
        "inputs:\n"
        "  path:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: ${{ inputs.path }}\n"
        "        key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/save-in", "with": {"path": "hf-cache"}},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["hf-cache"], f"the input-backed path did not resolve; got {paths}"
    assert _login_offenders({}, job), (
        "the composite persists the job's credential home and the workflow logs in"
    )


def test_no_job_that_persists_anything_performs_a_login():
    """Sweeps every job for login offenders (`_login_offenders`), independent of label discovery."""
    offenders = []
    for path, doc in _docs():
        for jid, job in _jobs(doc):
            for item in _login_offenders(doc, job):
                offenders.append(f"{path.name}:{jid}: {item}")
    assert not offenders, (
        "these steps log in while their own credential home sits inside a path the job "
        "persists:\n  " + "\n  ".join(sorted(offenders)) + "\n\n"
        "A login writes a real token into that directory, and the directory is then "
        "saved to a cache or uploaded. GitHub lets every pull request restore caches "
        "written on the default branch, so the token becomes readable by anyone who can "
        "open one. Read the token from the environment instead of logging in, or point "
        "the credential home outside the persisted path."
    )


def test_a_login_performed_by_an_action_is_seen():
    """Logins done by actions count too: docker/login-action writes into $DOCKER_CONFIG/config.json."""
    job = {
        "env": {"DOCKER_CONFIG": "docker-cache"},
        "steps": [
            {"uses": "docker/login-action@v3", "with": {"username": "u", "password": "p"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "docker-cache", "key": "k"}},
        ],
    }
    offenders = _login_offenders({}, job)
    assert offenders, "an action-performed login into a cached credential home must fire"
    assert "docker/login-action" in offenders[0], offenders

    safe = {
        "env": {"DOCKER_CONFIG": "/tmp/docker"},
        "steps": [
            {"uses": "docker/login-action@v3"},
            {"uses": "actions/cache/save@v4", "with": {"path": "docker-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, safe) == [], "the credential home is outside the cache"


def test_a_login_action_only_counts_against_what_it_writes():
    """Login actions match only what they write: setup-node writes .npmrc only when registry-url is set."""
    unrelated = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "actions/setup-node@v4", "with": {"node-version": "20"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, unrelated) == [], (
        "setup-node does not write HF_HOME, so caching an HF_HOME directory beside it "
        "is not a finding"
    )

    no_registry = {
        "env": {"NPM_CONFIG_USERCONFIG": "npm-cache/.npmrc"},
        "steps": [
            {"uses": "actions/setup-node@v4", "with": {"node-version": "20"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "npm-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, no_registry) == [], (
        "without `registry-url` setup-node writes no token"
    )

    with_registry = {
        "env": {"NPM_CONFIG_USERCONFIG": "npm-cache/.npmrc"},
        "steps": [
            {"uses": "actions/setup-node@v4",
             "with": {"registry-url": "https://registry.npmjs.org"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "npm-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, with_registry), (
        "with `registry-url` it writes an .npmrc token into the cached config path"
    )


def test_an_input_forwarded_between_composites_is_resolved(tmp_path, monkeypatch):
    """Forwarded `${{ inputs.path }}` must resolve to the outer caller's value, not remain unexpanded."""
    import sys

    module = sys.modules[__name__]
    inner = tmp_path / ".github" / "actions" / "save-in"
    outer = tmp_path / ".github" / "actions" / "wrap"
    inner.mkdir(parents = True)
    outer.mkdir(parents = True)
    (inner / "action.yml").write_text(
        "name: save in\ninputs:\n  path:\n    required: true\nruns:\n"
        "  using: composite\n  steps:\n    - uses: actions/cache/save@v4\n"
        "      with:\n        path: ${{ inputs.path }}\n        key: k\n"
    )
    (outer / "action.yml").write_text(
        "name: wrap\ninputs:\n  path:\n    required: true\nruns:\n"
        "  using: composite\n  steps:\n    - uses: ./.github/actions/save-in\n"
        "      with:\n        path: ${{ inputs.path }}\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/wrap", "with": {"path": "hf-cache"}},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["hf-cache"], f"the forwarded input did not resolve; got {paths}"
    assert _login_offenders({}, job), "the nested composite saves the credential home"


def test_a_composite_input_default_is_applied(tmp_path, monkeypatch):
    """Declared input defaults apply even when `with:` is omitted; else inputs.path stays unresolved."""
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "save-default"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: save default\ninputs:\n  path:\n    default: hf-cache\nruns:\n"
        "  using: composite\n  steps:\n    - uses: actions/cache/save@v4\n"
        "      with:\n        path: ${{ inputs.path }}\n        key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/save-default"},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["hf-cache"], f"the declared default was not applied; got {paths}"
    assert _login_offenders({}, job), (
        "the composite persists the credential home via its default, and the job logs in"
    )


def test_a_glob_path_still_contains_its_directory():
    """`hf-cache/**` uploads the token beneath it, but `hf-cache/*.bin` does not and must not be flagged."""
    assert _deglob("hf-cache/**") == "hf-cache"
    assert _deglob("hf-cache/*.bin") == "hf-cache"
    assert _deglob("hf-cache") == "hf-cache"
    assert _inside("hf-cache", "hf-cache/**") is True
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/*.bin", hf) is False
    assert _inside("other", "hf-cache/**", hf) is False
    # A bare `*` matches any NAME, `token` among them, so it still counts.
    assert _inside("hf-cache", "hf-cache/*", hf) is True
    # A narrow pattern counts when it matches the file that home really holds: docker writes config.json, cargo writes credentials.toml.
    assert _inside("docker-cache", "docker-cache/*.json",
                   CREDENTIAL_FILES["DOCKER_CONFIG"]) is True
    assert _inside("cargo-home", "cargo-home/*.toml",
                   CREDENTIAL_FILES["CARGO_HOME"]) is True
    # A caller that names no variable gets the conservative answer, since the filename is unknown and could be anything.
    assert _inside("hf-cache", "hf-cache/*.bin") is True


def test_a_login_action_is_matched_case_insensitively():
    """GitHub treats owner/repo case-insensitively; a capital letter is not an escape."""
    job = {"env": {"DOCKER_CONFIG": "docker-cache"}, "steps": [
        {"uses": "Docker/Login-Action@v3"},
        {"uses": "actions/cache/save@v4", "with": {"path": "docker-cache", "key": "k"}},
    ]}
    assert _login_offenders({}, job), (
        "`Docker/Login-Action` runs the same credential-writing action, and the step has "
        "no `run:` body, so a case-sensitive lookup skipped it entirely"
    )


def test_a_cached_path_resolves_against_job_level_env():
    """The job's own env has to seed the walk, or a root-level call resolves to nothing."""
    job = {"env": {"CACHE_DIR": "hf-cache", "HF_HOME": "hf-cache"}, "steps": [
        {"name": "log in", "run": "hf auth login --token x"},
        {"uses": "actions/cache/save@v4",
         "with": {"path": "${{ env.CACHE_DIR }}", "key": "k"}},
    ]}
    assert [p for p, _s in _persisted_with_env(job, {})] == ["hf-cache"]
    assert _login_offenders({}, job)


def test_an_overridden_default_home_is_not_a_finding():
    """A default home is used only when nothing overrides it, e.g. CARGO_HOME moves credentials elsewhere."""
    for default, owners in DEFAULT_OWNERS.items():
        for owner in owners:
            assert owner in CREDENTIAL_HOMES, (
                f"{owner} overrides {default} but is not tracked as a credential home"
                )
    assert DEFAULT_OWNERS["~/.cargo"] == ("CARGO_HOME",)
    # Two variables reach the Hugging Face default, but HF_TOKEN_PATH alone moves the token and stored_tokens, so a job
    # setting only HF_TOKEN_PATH leaves nothing of interest in ~/.cache/huggingface.
    assert DEFAULT_OWNERS["~/.cache/huggingface"] == ("HF_HOME", "HF_TOKEN_PATH")


def test_a_reusable_workflow_job_is_flattened(tmp_path, monkeypatch):
    """Delegating jobs have no steps of their own, so the called reusable workflow must be flattened in."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared.yml").write_text(
        "name: shared\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.path }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"uses": "./.github/workflows/shared.yml", "with": {"path": "hf-cache"}}
    assert [p for p, _s in _persisted_with_env(caller, {})] == ["hf-cache"]
    assert _login_offenders({}, caller), (
        "the reusable job logs in and caches the same input-named directory"
    )


def test_a_reusable_workflow_resolves_the_inputs_it_was_handed(tmp_path, monkeypatch):
    """Forwarded `${{ inputs.path }}` means the caller's value; copying `with:` verbatim self-references."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.path }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    (wf / "outer.yml").write_text(
        "name: outer\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  forward:\n    uses: ./.github/workflows/inner.yml\n"
        "    with:\n      path: ${{ inputs.path }}\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"uses": "./.github/workflows/outer.yml", "with": {"path": "hf-cache"}}
    assert [p for p, _s in _persisted_with_env(caller, {})] == ["hf-cache"], (
        "the outer caller's concrete directory has to survive two hops of forwarding"
    )
    assert _login_offenders({}, caller), (
        "the innermost job logs in and caches the directory the outermost caller named"
    )


def test_a_shell_login_only_counts_against_what_that_command_writes():
    """`docker login` writes $DOCKER_CONFIG only, so it must not be paired with other credential homes."""
    safe = {
        "env": {"HF_HOME": "hf-cache", "DOCKER_CONFIG": "/tmp/docker"},
        "steps": [
            {"run": "echo y | docker login -u u --password-stdin"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, safe) == [], (
        "docker writes $DOCKER_CONFIG/config.json, which is outside the cached "
        f"directory: {_login_offenders({}, safe)}"
    )

    unsafe = dict(safe, steps = [
        {"run": "hf auth login --token x"},
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
    ])
    assert _login_offenders({}, unsafe), (
        "the Hugging Face login does write the cached HF_HOME, and still has to be caught"
    )

    docker_cached = {
        "env": {"DOCKER_CONFIG": "docker-cache"},
        "steps": [
            {"run": "echo y | docker login -u u --password-stdin"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "docker-cache", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, docker_cached), (
        "narrowing the pairing must not stop docker being caught against its own home"
    )


def test_a_hugging_face_login_is_not_judged_against_an_unrelated_home():
    """`hf auth login` writes only Hugging Face homes, never CARGO_HOME; HF patterns match only those."""
    safe = {
        "env": {"CARGO_HOME": "cargo-cache", "HF_HOME": "/tmp/hf"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "cargo-cache", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, safe) == [], (
        f"the token goes to /tmp/hf, nowhere near the cached Cargo home: "
        f"{_login_offenders({}, safe)}"
    )

    unsafe = {
        "env": {"CARGO_HOME": "cargo-cache", "HF_HOME": "cargo-cache/hf"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "cargo-cache", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, unsafe), (
        "with HF_HOME actually inside the cached path this is still a finding"
    )


def test_two_jobs_of_a_reusable_workflow_are_not_one_runner(tmp_path, monkeypatch):
    """Each job runs on its own machine, so the same path in two jobs is two directories; don't pool them."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "split.yml").write_text(
        "name: split\n"
        "on:\n  workflow_call:\n"
        "jobs:\n"
        # logs in, persists nothing
        "  login:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: hf-cache\n"
        "    steps:\n      - run: hf auth login --token x\n"
        # persists the same NAME, on a different runner, and never logs in
        "  save:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"uses": "./.github/workflows/split.yml"}
    assert _login_offenders({}, caller) == [], (
        f"the login and the save are on different runners: "
        f"{_login_offenders({}, caller)}"
    )

    # Both in ONE job is a real finding, so the split did not cost the check its teeth.
    (wf / "split.yml").write_text(
        "name: split\n"
        "on:\n  workflow_call:\n"
        "jobs:\n"
        "  both:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: hf-cache\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    assert _login_offenders({}, caller), (
        "one job logging in and caching its own credential home is still caught"
    )


def test_a_chain_of_environment_references_is_resolved():
    """Env values may name other env vars; expansion must iterate, as a single pass leaves an expression."""
    env = {"CACHE_ROOT": "hf-cache", "HF_HOME": "${{ env.CACHE_ROOT }}"}
    assert _expand("${{ env.HF_HOME }}", env) == "hf-cache"

    job = {
        "env": {"CACHE_ROOT": "hf-cache", "HF_HOME": "${{ env.CACHE_ROOT }}"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "${{ env.HF_HOME }}", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, job), (
        "the cached path and the credential home are the same directory, reached "
        "through two references"
    )


def test_a_reference_cycle_does_not_hang_the_expansion():
    """Expansion is bounded, so cyclic env refs terminate; a leftover expression counts as unknown."""
    env = {"A": "${{ env.B }}", "B": "${{ env.A }}"}
    out = _expand("${{ env.A }}", env)
    assert "${{" in out, f"an unresolvable cycle stays an expression, got {out!r}"


def test_a_restrictive_glob_does_not_capture_a_token():
    """`hf-cache/*.bin` is narrow enough to exclude the token; unknown names still count as capturable."""
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _glob_captures("hf-cache/**", "hf-cache", hf) is True
    assert _glob_captures("hf-cache/*", "hf-cache", hf) is True
    assert _glob_captures("hf-cache/*.bin", "hf-cache", hf) is False
    # docker writes config.json, so a `*.json` pattern does capture a credential.
    assert _glob_captures(
        "docker-cache/*.json", "docker-cache", CREDENTIAL_FILES["DOCKER_CONFIG"]
    ) is True
    # cargo writes credentials.toml, which a canned filename list had omitted.
    assert _glob_captures(
        "cargo-home/*.toml", "cargo-home", CREDENTIAL_FILES["CARGO_HOME"]
    ) is True
    # A file-valued home is matched as the path it names, whatever its basename; no list of likely filenames can cover those, so the real value is tested.
    assert _glob_captures("creds/*.ini", "creds/my-profile.ini", ()) is True
    assert _glob_captures("creds/*.ini", "elsewhere/my-profile.ini", ()) is False
    # A pattern reaching only into a SUBDIRECTORY cannot hold a token at the root.
    assert _glob_captures("hf-cache/models/**", "hf-cache", hf) is False

    safe = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "hf-cache/*.bin", "name": "w"},
            },
        ],
    }
    assert _login_offenders({}, safe) == [], (
        f"the upload cannot contain the token: {_login_offenders({}, safe)}"
    )


def test_hf_token_path_overrides_the_home_when_both_are_set():
    """HF_TOKEN_PATH replaces $HF_HOME/token when set, so the two are alternatives, not both written."""
    safe = {
        "env": {"HF_HOME": "hf-cache", "HF_TOKEN_PATH": "/tmp/token"},
        "steps": [
            {"run": "hf auth login --token x"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, safe) == [], (
        f"the token is written to /tmp/token: {_login_offenders({}, safe)}"
    )

    unsafe = {
        "env": {"HF_HOME": "/tmp/hf", "HF_TOKEN_PATH": "hf-cache/token"},
        "steps": [
            {"run": "hf auth login --token x"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, unsafe), (
        "HF_TOKEN_PATH inside the cache is the finding, and it is the variable that "
        "decides"
    )


def test_a_reusable_workflow_does_not_inherit_the_callers_env(tmp_path, monkeypatch):
    """`env:` stops at a called reusable workflow, `with:` crosses it; merging env misattributes homes."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"env": {"HF_HOME": "hf-cache"}, "uses": "./.github/workflows/inner.yml"}
    assert _login_offenders({}, caller) == [], (
        f"the caller's HF_HOME never reaches the called workflow: "
        f"{_login_offenders({}, caller)}"
    )

    # Passed as an INPUT, it does reach it, and that is still a finding.
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      home:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.home }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    passed = {
        "uses": "./.github/workflows/inner.yml",
        "with": {"home": "hf-cache"},
    }
    assert _login_offenders({}, passed), "an input does cross the boundary"


def test_the_discovery_scan_reaches_a_reusable_workflow_unit(tmp_path, monkeypatch):
    """Discovery must reach called reusable workflows; an HF_HOME set from an input is never checked."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.path }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    (wf / "caller.yml").write_text(
        "name: caller\n"
        "on:\n  push:\n"
        "jobs:\n  go:\n    uses: ./.github/workflows/inner.yml\n"
        "    with:\n      path: hf-cache\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS", wf)
    monkeypatch.setattr(module, "ACTIONS", tmp_path / ".github" / "actions")

    found = [f for f in _findings() if "caller.yml" in str(f[0])]
    assert found, (
        "the discovery scan has to reach the delegating job, or the guard below is "
        "never instantiated for it"
    )


def test_a_restrictive_glob_is_matched_against_the_real_credential_path():
    """Filenames per variable (CREDENTIAL_FILES), not a canned list; file homes take any basename."""
    cargo = CREDENTIAL_FILES["CARGO_HOME"]
    assert _inside("cargo-home", "cargo-home/*.toml", cargo) is True

    leaking = {
        "env": {"CARGO_HOME": "cargo-home"},
        "steps": [
            {"run": "cargo login $TOKEN"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "cargo-home/*.toml", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, leaking), (
        "cargo writes cargo-home/credentials.toml, which this pattern persists"
    )

    named = {
        "env": {"AWS_SHARED_CREDENTIALS_FILE": "creds/my-profile.ini"},
        "steps": [
            {"run": "aws configure set aws_access_key_id x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "creds/*.ini", "name": "c"},
            },
        ],
    }
    assert _login_offenders({}, named), (
        "the configured path matches the pattern, whatever its basename"
    )


def test_hf_token_path_alone_moves_the_token_out_of_the_default_home():
    """HF_HOME or HF_TOKEN_PATH each relocates the token; HF_TOKEN_PATH alone moves it out of the default."""
    assert DEFAULT_OWNERS["~/.cache/huggingface"] == ("HF_HOME", "HF_TOKEN_PATH")
    assert DEFAULT_OWNERS["~/.huggingface"] == ("HF_HOME", "HF_TOKEN_PATH")
    for default, owners in DEFAULT_OWNERS.items():
        for owner in owners:
            assert owner in CREDENTIAL_HOMES, f"{owner} for {default}"


def test_a_globstar_matches_zero_directories():
    """GitHub's `**` may match zero directories, so `hf-cache/**/token` also covers `hf-cache/token`."""
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/**/token", hf) is True
    assert _inside("hf-cache", "hf-cache/**/*", hf) is True
    assert _inside("hf-cache", "hf-cache/**", hf) is True
    # Depth beyond zero still matches.
    assert _glob_captures("hf-cache/**/token", "hf-cache", hf) is True
    # And the narrowing is not lost: a restrictive tail still excludes the token.
    assert _inside("hf-cache", "hf-cache/**/*.bin", hf) is False

    cargo = CREDENTIAL_FILES["CARGO_HOME"]
    leaking = {
        "env": {"CARGO_HOME": "cargo-home"},
        "steps": [
            {"run": "cargo login $TOKEN"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "cargo-home/**/credentials.toml", "name": "c"},
            },
        ],
    }
    assert _inside("cargo-home", "cargo-home/**/credentials.toml", cargo) is True
    assert _login_offenders({}, leaking), (
        "the upload includes cargo-home/credentials.toml at depth zero"
    )


def test_the_discovery_scan_sees_an_env_declared_inside_a_composite(tmp_path, monkeypatch):
    """Discovery must see env declared in a composite's own steps, or its login is never checked."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    act = tmp_path / ".github" / "actions" / "inner-login"
    wf.mkdir(parents = True)
    act.mkdir(parents = True)
    (act / "action.yml").write_text(
        "name: inner login\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: hf auth login --token x\n"
        "      shell: bash\n"
        "      env:\n        HF_HOME: hf-cache\n"
    )
    (wf / "caller.yml").write_text(
        "name: caller\n"
        "on:\n  push:\n"
        "jobs:\n  go:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/inner-login\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS", wf)
    monkeypatch.setattr(module, "ACTIONS", tmp_path / ".github" / "actions")

    found = [f for f in _findings() if "caller.yml" in str(f[0])]
    assert found, (
        "the env is declared inside the composite, and the scan has to reach it or the "
        "guard is never instantiated for this job"
    )


def test_a_character_class_is_compiled_not_escaped():
    """Character classes must compile as globs; escaping `[` made `hf-cache/[t]oken` match nothing."""
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/[t]oken", hf) is True
    assert _inside("hf-cache", "hf-cache/[a-z]oken", hf) is True
    # A negated class that excludes the token really does exclude it.
    assert _inside("hf-cache", "hf-cache/[!t]oken", hf) is False

    leaking = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "hf-cache/[t]oken", "name": "a"},
            },
        ],
    }
    assert _login_offenders({}, leaking), "the upload pattern includes the token"


def test_a_persisted_path_that_names_the_credential_file_is_caught():
    """A persisted path naming the credential file itself, e.g. hf-cache/token, captures it."""
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/token", hf) is True
    assert _inside("hf-cache", "hf-cache/stored_tokens", hf) is True
    assert _inside("hf-cache", "hf-cache/weights.bin", hf) is False

    leaking = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "hf-cache/token", "name": "a"},
            },
        ],
    }
    assert _login_offenders({}, leaking), "the token itself is the uploaded path"


def test_a_default_home_override_must_come_from_the_same_runner(tmp_path, monkeypatch):
    """Caller `env:` does not reach a called workflow, so it cannot exempt a default home there."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    (wf / "caller.yml").write_text(
        "name: caller\n"
        "on:\n  push:\n"
        "jobs:\n  go:\n"
        "    env:\n      CARGO_HOME: /tmp/cargo\n"
        "    uses: ./.github/workflows/inner.yml\n"
        "    with:\n      path: ~/.cargo\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS", wf)
    monkeypatch.setattr(module, "ACTIONS", tmp_path / ".github" / "actions")

    units = _units({"uses": "./.github/workflows/inner.yml", "with": {"path": "~/.cargo"},
                    "env": {"CARGO_HOME": "/tmp/cargo"}}, {})
    assert units, "the delegation resolves to at least one unit"
    for _unit, unit_env, _inputs in units:
        assert "CARGO_HOME" not in unit_env, (
            "the caller's variable must not appear in the called job's environment, or "
            "it will be read as overriding a default it cannot move"
        )


def test_a_composite_used_from_a_checkout_subdirectory_is_flattened(tmp_path, monkeypatch):
    """A `./unsloth/...` ref from a checkout subdirectory must resolve, or its steps are never flattened."""
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    act = tmp_path / ".github" / "actions" / "hidden-login"
    wf.mkdir(parents = True)
    act.mkdir(parents = True)
    (act / "action.yml").write_text(
        "name: hidden login\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: hf auth login --token x\n      shell: bash\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "./unsloth/.github/actions/hidden-login"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, job), (
        "the prefixed reference names the same composite, which logs in"
    )


def test_a_default_credential_file_persisted_exactly_is_caught():
    """Persisting an exact default credential file (~/.cargo/credentials.toml) must be caught as a match."""
    assert DEFAULT_HOME_FILES["~/.cargo"] == ("credentials", "credentials.toml")
    assert _inside(
        "~/.cargo", "~/.cargo/credentials.toml", DEFAULT_HOME_FILES["~/.cargo"]
    ) is True
    assert _inside(
        "~/.docker", "~/.docker/config.json", DEFAULT_HOME_FILES["~/.docker"]
    ) is True
    assert _inside(
        "~/.cache/huggingface",
        "~/.cache/huggingface/token",
        DEFAULT_HOME_FILES["~/.cache/huggingface"],
    ) is True
    # Something else under the same home is still not a credential.
    assert _inside(
        "~/.cargo", "~/.cargo/registry", DEFAULT_HOME_FILES["~/.cargo"]
    ) is False
    # The filenames are load-bearing: without them, persisting the credential file itself answers no, which was the default-home rule's original gap.
    assert _inside("~/.cargo", "~/.cargo/credentials.toml") is False
    for default in DEFAULT_CREDENTIAL_HOMES:
        assert default in DEFAULT_HOME_FILES, default
    # The rule must really consult the filenames: assert by calling it, not by scanning this file, which once counted
    # its own assertion text and passed however the rule behaved.
    assert _default_home_hits("~/.cargo/credentials.toml", set()), (
        "persisting the exact default credential file has to be a finding"
    )
    assert _default_home_hits("~/.docker/config.json", set())
    assert not _default_home_hits("~/.cargo/registry", set())
    # An override still exempts it, so the narrowing from earlier rounds is intact.
    assert not _default_home_hits("~/.cargo/credentials.toml", {"CARGO_HOME"})


def test_two_identical_unresolved_expressions_are_the_same_directory():
    """Identical unresolved expressions name the same directory; different ones stay unknown, not equal."""
    assert _inside("${{ matrix.path }}", "${{ matrix.path }}") is True
    # Two DIFFERENT unknowns stay unknown rather than being assumed equal.
    assert _inside("${{ matrix.path }}", "${{ matrix.other }}") is False

    job = {
        "env": {"HF_HOME": "${{ matrix.path }}"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "${{ matrix.path }}", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, job), (
        "the cached path is the credential home, by construction"
    )
