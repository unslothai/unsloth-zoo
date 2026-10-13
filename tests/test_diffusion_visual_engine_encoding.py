# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# unsloth#6489: the stub mirrors the visual server at the byte level (raw UTF-8 out, fopen of the stdin path).

from __future__ import annotations

import json
import locale
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unsloth_zoo.diffusion_studio import visual_engine as V

ENGINE = Path(V.__file__)
TEXTS = ["plain ascii", "Grüße → ok", "Ångström Á 😎", "日本語"]

STUB = r'''
import json, os, sys
out = sys.stdout.buffer
out.write(b"READY 262144 4096\n"); out.flush()
for raw in sys.stdin.buffer:
    raw = raw.rstrip(b"\r\n")
    if not raw: continue
    if raw == b"QUIT": break
    path = raw.decode("mbcs") if os.name == "nt" else raw
    try:
        with open(path, "rb") as f: req = json.loads(f.read().decode("utf-8"))
    except OSError:
        out.write(b"ERR cannot-open-request\n"); out.flush(); continue
    payload = json.dumps(req["messages"][-1]["content"], ensure_ascii=False).encode("utf-8")
    out.write(b"F 0 1 1 " + payload + b"\nC 0 " + payload + b"\nDONE\n"); out.flush()
'''

CHILD = r'''
import importlib.util, json, locale, sys
spec = importlib.util.spec_from_file_location("visual_engine", sys.argv[1])
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
srv = V.VisualServer(sys.argv[2], server_bin=sys.executable, req_path=sys.argv[3])
replies = []
for text in json.loads(sys.argv[4]):
    try: replies.append(V.generate_visual(srv, [{"role": "user", "content": text}]))
    except Exception as e: replies.append(f"{type(e).__name__}: {e}")
srv.close()
print(json.dumps({"encoding": locale.getpreferredencoding(False), "replies": replies}))
'''


def _stub(tmp_path):
    stub = tmp_path / "stub_server.py"
    stub.write_text(STUB, encoding = "utf-8")
    return stub


def test_replies_survive_a_non_utf8_locale(tmp_path):
    """Base decoded the server's UTF-8 with the locale codec: mojibake under cp1252, a crash under ascii."""
    env = dict(os.environ, PYTHONUTF8 = "0", PYTHONIOENCODING = "utf-8")
    if os.name != "nt":  # Windows already runs the ANSI code page
        env.update(LC_ALL = "C", LANG = "C", PYTHONCOERCECLOCALE = "0")
    run = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", CHILD, str(ENGINE), str(_stub(tmp_path)),
         str(tmp_path / "request.json"), json.dumps(TEXTS)],
        capture_output = True, encoding = "utf-8", env = env, timeout = 120,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["encoding"].lower().replace("-", "") not in ("utf8",), result  # else the test proves nothing
    assert result["replies"] == TEXTS


def test_request_path_keeps_the_locale_encoding(tmp_path):
    """stdin carries a path the server opens with narrow fopen: forcing UTF-8 there broke non-ASCII temp dirs."""
    encoding = locale.getpreferredencoding(False)
    for name in ("Jürgen", "Дмитрий", "日本語"):  # cp1252, cp1251, cp932 users
        try:
            name.encode(encoding)
            break
        except UnicodeEncodeError:
            continue
    else:
        pytest.skip(f"no non-ASCII directory name survives the {encoding} locale")
    req_dir = tmp_path / name
    req_dir.mkdir()
    srv = V.VisualServer(str(_stub(tmp_path)), server_bin = sys.executable, req_path = str(req_dir / "request.json"))
    try:
        assert V.generate_visual(srv, [{"role": "user", "content": "Grüße"}]) == "Grüße"
    finally:
        srv.close()
