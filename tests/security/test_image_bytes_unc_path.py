"""A dict image row whose path is bytes must hit the same network-share guard as a str path; PIL's
Image.open accepts bytes, so a UNC bytes path would open an SMB connection (NTLM leak) on
Windows; _WINDOWS is patched and Image.open replaced so nothing is opened."""
import os

import pytest

from unsloth_zoo import vision_utils


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    monkeypatch.delenv(vision_utils.UNSLOTH_ALLOW_PRIVATE_URL_FETCH_VAR, raising = False)
    opened = []
    monkeypatch.setattr(vision_utils.Image, "open", lambda fp, *a, **k: opened.append(fp) or (_ for _ in ()).throw(OSError("opened")))
    return opened


@pytest.mark.parametrize("path", [b"\\\\evil.example\\share\\x.png", b"//evil.example/share/x.png"])
def test_bytes_unc_path_in_dict_is_refused(windows, path):
    with pytest.raises(ValueError, match = "network share"):
        vision_utils._decode_image({"bytes": None, "path": path})
    assert windows == []


def test_str_unc_path_still_refused(windows):
    with pytest.raises(ValueError, match = "network share"):
        vision_utils._decode_image({"bytes": None, "path": "\\\\evil.example\\share\\x.png"})


def test_local_bytes_path_still_opens(windows):
    with pytest.raises(OSError, match = "opened"):
        vision_utils._decode_image({"bytes": None, "path": os.fsencode("C:\\data\\x.png")})
    assert windows == [os.fsencode("C:\\data\\x.png")]
