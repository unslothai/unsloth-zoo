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

"""Write a trained MTP head into a saved checkpoint, or beside a LoRA adapter.

transformers drops `mtp.*` on load, so the merge copies the stock head through from the
original shards. These helpers replace it with the trained one, touching only the bytes of
`mtp.*` tensors, so llama.cpp's converter and `reconcile_mtp_config` see a real head.
"""

__all__ = [
    "MTP_HEAD_WEIGHTS_NAME",
    "MTP_HEAD_INFO_NAME",
    "MTP_APPENDED_SHARD_NAME",
    "write_mtp_tensors",
    "save_mtp_head",
    "load_saved_mtp_head",
]

import hashlib
import json
import mmap
import os
import re
import shutil
import stat
import struct
import tempfile
from pathlib import Path

import torch

MTP_HEAD_WEIGHTS_NAME = "mtp_head.safetensors"
MTP_HEAD_INFO_NAME = "mtp_head.json"
# Starts with "model" and ends ".safetensors": llama.cpp lists parts by that prefix when no index exists.
MTP_APPENDED_SHARD_NAME = "model-mtp.safetensors"
_INDEX_NAME = "model.safetensors.index.json"
_MTP_CONFIG_KEY = "mtp_num_hidden_layers"

# Same spelling as `saving_utils._MTP_TENSOR_RE`; kept local so this module stays light to import.
_MTP_TENSOR_RE = re.compile(r"^(?:(?:model|language_model)\.)*mtp\.")
_MTP_LAYER_RE = re.compile(r"^layers\.(\d+)\.")

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _mtp_prefix(name):
    match = _MTP_TENSOR_RE.match(str(name))
    return None if match is None else match.group(0)


def _relative_head_state_dict(head_state_dict):
    """Head-relative names (`fc.weight`), accepting fully prefixed ones too."""
    names = list(head_state_dict.keys())
    if names and all(_mtp_prefix(name) is not None for name in names):
        return {name[len(_mtp_prefix(name)):]: tensor for name, tensor in head_state_dict.items()}
    return dict(head_state_dict)


def _num_mtp_layers(relative_names):
    indices = {int(m.group(1)) for m in (_MTP_LAYER_RE.match(n) for n in relative_names) if m}
    return max(indices) + 1 if indices else 0


def _read_header(path):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return header, 8 + length


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_signature(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def _checkpoint_layout(directory):
    """`{tensor_name: shard_filename}` plus whether an index backs it, or None if no safetensors."""
    index_path = directory / _INDEX_NAME
    if index_path.is_file():
        with open(index_path, "r", encoding = "utf-8") as f:
            weight_map = json.load(f).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Unsloth: {index_path} has no usable `weight_map`.")
        # Shards live in the checkpoint folder itself; never follow an index outside it.
        outside = sorted({f for f in weight_map.values() if not isinstance(f, str) or Path(f).name != f})
        if outside:
            raise ValueError(f"Unsloth: {index_path} names shards outside the folder: {outside[:3]}")
        return dict(weight_map), True
    single = directory / "model.safetensors"
    if single.is_file():
        header, _ = _read_header(single)
        return {name: single.name for name in header if name != "__metadata__"}, False
    return None, False


def _tensor_bytes(tensor, dtype):
    tensor = tensor.detach().to(device = "cpu", dtype = dtype).contiguous()
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    return tensor.view(torch.uint8).numpy().tobytes()


def _staged_copy(path):
    """A copy of `path` next to it, with the original's mode, for an atomic `os.replace`."""
    handle, staged = tempfile.mkstemp(prefix = f".{path.name}.", suffix = ".tmp", dir = path.parent)
    os.close(handle)
    try:
        shutil.copyfile(path, staged)
    except BaseException:
        try:
            os.unlink(staged)
        except OSError:
            pass
        raise
    try:
        os.chmod(staged, stat.S_IMODE(os.stat(path).st_mode))
    except OSError:
        pass
    return Path(staged)


def _patch_shard(path, replacements):
    """Overwrite the data bytes of `replacements` in a copy of `path`, then swap it in.

    Every other byte, header included, is untouched: shape and dtype are unchanged, so the
    data offsets are too.
    """
    header, data_start = _read_header(path)
    staged = _staged_copy(path)
    try:
        with open(staged, "r+b") as f:
            # The mapping is closed before the file, and both before os.replace (Windows).
            with mmap.mmap(f.fileno(), 0) as mm:
                for name, tensor in replacements.items():
                    entry = header[name]
                    begin, end = entry["data_offsets"]
                    payload = _tensor_bytes(tensor, _SAFETENSORS_DTYPES[entry["dtype"]])
                    if len(payload) != end - begin:
                        raise ValueError(
                            f"Unsloth: `{name}` would write {len(payload)} bytes into a "
                            f"{end - begin} byte slot in {path.name}."
                        )
                    mm[data_start + begin : data_start + end] = payload
                mm.flush()
            f.flush()
            os.fsync(f.fileno())
        os.replace(staged, path)
    except BaseException:
        try:
            os.unlink(staged)
        except OSError:
            pass
        raise


def _write_json_atomically(path, payload, mode_from = None):
    handle, staged = tempfile.mkstemp(prefix = f".{path.name}.", suffix = ".tmp", dir = path.parent)
    try:
        with os.fdopen(handle, "w", encoding = "utf-8") as f:
            json.dump(payload, f, indent = 2, ensure_ascii = False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        source = path if path.exists() else mode_from
        mode = 0o644
        if source is not None:
            try:
                mode = stat.S_IMODE(os.stat(source).st_mode)
            except OSError:
                pass
        os.chmod(staged, mode)
        os.replace(staged, path)
    except BaseException:
        try:
            os.unlink(staged)
        except OSError:
            pass
        raise


def _declare_mtp_layers(directory, num_layers):
    """Make config.json declare the head, in the container holding `num_hidden_layers`."""
    config_path = directory / "config.json"
    if not config_path.is_file() or num_layers <= 0:
        return False
    with open(config_path, "r", encoding = "utf-8") as f:
        config = json.load(f)
    container = config
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "num_hidden_layers" in text_config:
        container = text_config
    if container.get(_MTP_CONFIG_KEY) == num_layers:
        return False
    container[_MTP_CONFIG_KEY] = num_layers
    _write_json_atomically(config_path, config)
    return True


def _existing_mode(paths, default = 0o644):
    for path in paths:
        try:
            return stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            pass
    return default


def _append_mtp_shard(directory, layout, has_index, head, prefix, dtype):
    shard_path = directory / MTP_APPENDED_SHARD_NAME
    if shard_path.name in set(layout.values()):
        raise ValueError(f"Unsloth: {shard_path} already holds model tensors; refusing to overwrite it.")
    from safetensors.torch import save_file
    mode = _existing_mode(directory / f for f in sorted(set(layout.values())))

    tensors = {
        prefix + name: tensor.detach().to(device = "cpu", dtype = dtype).contiguous()
        for name, tensor in head.items()
    }
    handle, staged = tempfile.mkstemp(prefix = f".{shard_path.name}.", suffix = ".tmp", dir = directory)
    os.close(handle)
    try:
        save_file(tensors, staged, metadata = {"format": "pt"})
        os.chmod(staged, mode)
        os.replace(staged, shard_path)
    except BaseException:
        try:
            os.unlink(staged)
        except OSError:
            pass
        raise

    index_path = directory / _INDEX_NAME
    if has_index:
        with open(index_path, "r", encoding = "utf-8") as f:
            index = json.load(f)
    else:
        # A lone model.safetensors needs an index once a second part exists, or transformers
        # reads only the first file and the converter's part listing is the only thing left.
        index = {"metadata": {}, "weight_map": dict(layout)}
    index.setdefault("metadata", {})
    index["weight_map"].update({name: shard_path.name for name in tensors})
    total = 0
    for filename in set(index["weight_map"].values()):
        header, _ = _read_header(directory / filename)
        total += sum(
            entry["data_offsets"][1] - entry["data_offsets"][0]
            for key, entry in header.items() if key != "__metadata__"
        )
    index["metadata"]["total_size"] = total
    _write_json_atomically(index_path, index, mode_from = shard_path)
    return shard_path.name, list(tensors)


def write_mtp_tensors(save_directory, head_state_dict, prefix = None, strict = True, verify = False,
                      append_prefix = "mtp."):
    """Replace (or add) a checkpoint's `mtp.*` tensors with `head_state_dict`.

    `head_state_dict` uses head-relative names (`fc.weight`, `layers.0.mlp.up_proj.weight`);
    fully prefixed names are accepted too. Tensors already in the checkpoint keep their stored
    dtype and only their bytes change. A checkpoint without a head gets a new
    `model-mtp.safetensors` part, an index entry per tensor and `mtp_num_hidden_layers` in
    config.json. `prefix` picks the stored spelling (autodetected when None); `append_prefix` is
    the spelling for a head that is added. `verify` adds sha256 of every untouched shard before
    and after to the report.
    """
    directory = Path(save_directory)
    head = _relative_head_state_dict(head_state_dict)
    if not head:
        raise ValueError("Unsloth: the MTP head state dict is empty.")
    layout, has_index = _checkpoint_layout(directory)
    if layout is None:
        raise FileNotFoundError(f"Unsloth: no safetensors checkpoint in {directory} to write an MTP head into.")

    existing = {name: shard for name, shard in layout.items() if _mtp_prefix(name) is not None}
    shards = sorted(set(layout.values()))
    before = {s: (_sha256(directory / s) if verify else _file_signature(directory / s)) for s in shards}
    report = {
        "directory": str(directory),
        "mode": None,
        "prefix": None,
        "num_layers": _num_mtp_layers(head),
        "files_touched": [],
        "tensors_written": [],
        "untouched": {},
        "config_declared": False,
    }

    if existing:
        prefixes = sorted({_mtp_prefix(name) for name in existing})
        if prefix is None:
            if len(prefixes) != 1:
                raise ValueError(f"Unsloth: the checkpoint mixes MTP prefixes {prefixes}; pass `prefix`.")
            prefix = prefixes[0]
        stored = {name[len(prefix):]: shard for name, shard in existing.items() if name.startswith(prefix)}
        missing = sorted(set(stored) - set(head))
        extra = sorted(set(head) - set(stored))
        if strict and (missing or extra):
            raise ValueError(
                f"Unsloth: the trained MTP head does not match the checkpoint's `{prefix}*` tensors "
                f"(missing {missing[:5]}, unexpected {extra[:5]})."
            )
        by_shard = {}
        for name, tensor in head.items():
            if name in stored:
                by_shard.setdefault(stored[name], {})[prefix + name] = tensor
        for shard, replacements in by_shard.items():
            header, _ = _read_header(directory / shard)
            for full_name, tensor in replacements.items():
                if list(header[full_name]["shape"]) != list(tensor.shape):
                    raise ValueError(
                        f"Unsloth: `{full_name}` is {header[full_name]['shape']} in the checkpoint "
                        f"but {list(tensor.shape)} in the trained head."
                    )
        for shard, replacements in sorted(by_shard.items()):
            _patch_shard(directory / shard, replacements)
            report["files_touched"].append(shard)
            report["tensors_written"].extend(sorted(replacements))
        report["mode"] = "replaced"
    else:
        if prefix is None:
            prefix = append_prefix or "mtp."
        sample_shard = directory / shards[0]
        header, _ = _read_header(sample_shard)
        float_dtypes = [entry["dtype"] for key, entry in header.items()
                        if key != "__metadata__" and entry["dtype"] in ("BF16", "F16", "F32")]
        dtype = _SAFETENSORS_DTYPES[max(set(float_dtypes), key = float_dtypes.count)] if float_dtypes else torch.bfloat16
        shard_name, written = _append_mtp_shard(directory, layout, has_index, head, prefix, dtype)
        report["files_touched"].extend([shard_name, _INDEX_NAME])
        report["tensors_written"].extend(sorted(written))
        report["mode"] = "appended"
    report["prefix"] = prefix

    if _declare_mtp_layers(directory, report["num_layers"]):
        report["files_touched"].append("config.json")
        report["config_declared"] = True

    for shard in shards:
        if shard in report["files_touched"]:
            continue
        after = _sha256(directory / shard) if verify else _file_signature(directory / shard)
        if after != before[shard]:
            raise RuntimeError(f"Unsloth: {shard} changed while only the MTP head was being written.")
        report["untouched"][shard] = {"before": before[shard], "after": after}
    return report


def save_mtp_head(directory, head_state_dict, info = None):
    """Write the head beside a LoRA adapter so a later merge can stitch it in."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents = True, exist_ok = True)
    head = _relative_head_state_dict(head_state_dict)
    tensors = {name: tensor.detach().to("cpu").contiguous() for name, tensor in head.items()}
    weights_path = directory / MTP_HEAD_WEIGHTS_NAME
    mode = _existing_mode([weights_path, directory / "adapter_model.safetensors"])
    handle, staged = tempfile.mkstemp(prefix = f".{weights_path.name}.", suffix = ".tmp", dir = directory)
    os.close(handle)
    try:
        save_file(tensors, staged, metadata = {"format": "pt"})
        os.chmod(staged, mode)
        os.replace(staged, weights_path)
    except BaseException:
        try:
            os.unlink(staged)
        except OSError:
            pass
        raise

    record = {"format_version": 1, "num_layers": _num_mtp_layers(tensors), "prefix": "mtp."}
    for key, value in dict(info or {}).items():
        try:
            json.dumps(value)
        except TypeError:
            continue
        record[key] = value
    _write_json_atomically(directory / MTP_HEAD_INFO_NAME, record, mode_from = weights_path)
    return weights_path


def load_saved_mtp_head(directory):
    """`(state_dict, info)` written by `save_mtp_head`, or None when the folder has none."""
    directory = Path(directory)
    weights_path = directory / MTP_HEAD_WEIGHTS_NAME
    if not weights_path.is_file():
        return None
    from safetensors.torch import load_file

    info = {}
    info_path = directory / MTP_HEAD_INFO_NAME
    if info_path.is_file():
        with open(info_path, "r", encoding = "utf-8") as f:
            info = json.load(f)
    return load_file(weights_path), info
