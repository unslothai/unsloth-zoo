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

"""Lossless fixed-width codec for bf16 block swap fetches (fewer PCIe bytes, exact restore).

bf16 = sign | exponent (8) | mantissa (7). Weights use few exponents, so per element we store one byte of
sign + mantissa and a 4 bit index into a per-group table of the 15 most common exponents (code 15 = escape).
Escaped exponents go to a side list (position, exponent) patched after the main decode, so the restore is
bit exact for any 16 bit word. Encoding runs once on the host; the GPU decode is one memory-bound pass per
transfer chunk on a side stream, overlapped with the next chunk's copy.

Host blob per region of N words (N a multiple of 256):
    chunk j (CHUNK words each, last one shorter): sign+mantissa bytes [k], then packed codes [k / 2]
    aux (16 byte aligned): tables [groups * 16], escape positions (int32, int64 past 2**31 words, 8 byte
    aligned) [E], escape exponents (uint8) [E]
"""

import os
import sys
import threading
from contextlib import contextmanager
import torch

__all__ = [
    "GROUP",
    "CHUNK",
    "RegionCodec",
    "encode_region",
    "decode_region_cpu",
    "DeviceDecoder",
    "triton_decode_available",
    "self_check",
]

# Words per exponent table: 16 bytes per 64K words (0.01%), and tables follow each tensor's scale.
GROUP = 1 << 16
# Words per transfer chunk: 16 MiB raw, 12 MiB on the wire; a multiple of GROUP.
CHUNK = 1 << 23
_ESC = 15
# Keep a region raw unless the codec saves at least this much (fp16, fp32 and packed 4 bit words do not).
MAX_RATIO = 0.9
# And unless escapes stay rare (real bf16 weights: about 0.02%), which also bounds the card's aux buffer.
MAX_ESCAPE_FRACTION = 0.005
_LITTLE = sys.byteorder == "little"


def _align(n, a):
    return -(-int(n) // a) * a


class RegionCodec:
    """Layout of one encoded region (all offsets in bytes into the host blob)."""

    __slots__ = ("words", "chunk", "n_chunks", "groups", "aux_off", "tables_bytes", "esc_n", "pos_dtype",
                 "pos_off", "vals_off", "nbytes", "raw_nbytes")

    def __init__(self, words, chunk, esc_n):
        if int(words) <= 0 or int(words) % 256 or int(chunk) <= 0 or int(chunk) % GROUP or int(esc_n) < 0:
            raise ValueError(f"bad region geometry: words={words} chunk={chunk} escapes={esc_n}")
        self.words = int(words)
        self.chunk = int(chunk)
        self.n_chunks = -(-self.words // self.chunk)
        self.groups = -(-self.words // GROUP)
        self.aux_off = _align(self.words * 3 // 2, 16)
        self.tables_bytes = self.groups * 16
        self.esc_n = int(esc_n)
        self.pos_dtype = torch.int32 if self.words < 2**31 else torch.int64
        size = 4 if self.pos_dtype == torch.int32 else 8
        # 16 byte aligned, like every staging / aux base: the patch kernel compiles to one variant.
        self.pos_off = _align(self.aux_off + self.tables_bytes, 16)
        self.vals_off = _align(self.pos_off + self.esc_n * size, 16)
        self.nbytes = self.vals_off + self.esc_n
        self.raw_nbytes = 2 * self.words

    def chunk_span(self, j):
        """(first word, words, host byte offset) of chunk j."""
        s = j * self.chunk
        k = min(self.chunk, self.words - s)
        return s, k, s * 3 // 2

    @property
    def aux_nbytes(self):
        return self.nbytes - self.aux_off


def _planes(raw_u8):
    """bf16 words (little endian bytes) -> (sign + mantissa bytes, exponent bytes), all uint8."""
    w = raw_u8.view(-1, 2)
    lo, hi = w[:, 0], w[:, 1]
    sm = (hi & 0x80) | (lo & 0x7F)
    exp = ((hi & 0x7F) << 1) | (lo >> 7)
    return sm, exp


@contextmanager
def _integer_counts_ok():
    """bincount has no 'deterministic' CUDA kernel, but integer counts are exact whatever the order:
    lift torch.use_deterministic_algorithms(True) for the histogram only."""
    strict = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled() if strict else False
    if strict:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if strict:
            torch.use_deterministic_algorithms(True, warn_only = warn)


def _encode_chunk(raw_u8):
    """One chunk: (sm + packed codes bytes, tables [groups, 16], escape local positions, escape exponents)."""
    sm, exp = _planes(raw_u8)
    k = sm.numel()
    groups = -(-k // GROUP)
    gidx = torch.arange(k, device = raw_u8.device, dtype = torch.int32) >> 16
    key = (gidx * 256 + exp.to(torch.int32)).long()
    with _integer_counts_ok():
        counts = torch.bincount(key, minlength = groups * 256).view(groups, 256)
    top = counts.topk(_ESC, dim = 1).indices
    tables = torch.zeros(groups, 16, dtype = torch.uint8, device = raw_u8.device)
    tables[:, :_ESC] = top.to(torch.uint8)
    lut = torch.full((groups, 256), _ESC, dtype = torch.uint8, device = raw_u8.device)
    lut.scatter_(1, top, torch.arange(_ESC, dtype = torch.uint8, device = raw_u8.device).expand(groups, _ESC))
    code = lut.view(-1)[key]
    esc = code == _ESC
    pos = esc.nonzero().view(-1)
    vals = exp[pos]
    packed = code[0::2] | (code[1::2] << 4)
    return torch.cat((sm, packed)), tables, pos, vals


def max_aux_nbytes(raw_nbytes, max_escape_fraction = MAX_ESCAPE_FRACTION):
    """Largest aux block an accepted region of raw_nbytes can have (tables + escapes at the cap)."""
    words = int(raw_nbytes) // 2
    return _align(-(-words // GROUP) * 16, 8) + 16 + int(words * max_escape_fraction) * 9 + 16


def encode_region(raw_u8, chunk = CHUNK, max_ratio = MAX_RATIO, device = None, out = None,
                  max_escape_fraction = MAX_ESCAPE_FRACTION):
    """Encode a host region (1-D uint8, length a multiple of 512) of 16 bit words.

    Returns (parts, codec): parts are (blob offset, CPU uint8 tensor) to write at the region's base, or None
    when the encoding would not be at least 1 - max_ratio smaller or escapes pass max_escape_fraction. With
    `out` (a CPU uint8 view at least as long as the region) chunks are written there as they are made and
    parts holds only the aux pieces; after None, `out` holds partial bytes the caller overwrites.
    `device` runs the encode there."""
    if (not _LITTLE or raw_u8.dtype != torch.uint8 or raw_u8.dim() != 1 or raw_u8.numel() == 0
            or raw_u8.numel() % 512):
        return None
    if chunk <= 0 or chunk % GROUP:
        raise ValueError(f"chunk must be a positive multiple of {GROUP}, got {chunk}")
    if out is not None and (out.dtype != torch.uint8 or out.dim() != 1 or out.numel() < raw_u8.numel()):
        raise ValueError("out must be a 1-D uint8 view at least as long as the region")
    words = raw_u8.numel() // 2
    max_esc = int(words * max_escape_fraction)
    dev = torch.device(device) if device is not None else raw_u8.device
    parts, tables, pos_all, vals_all, esc_n = [], [], [], [], 0
    for s in range(0, words, chunk):
        k = min(chunk, words - s)
        src = raw_u8[2 * s:2 * (s + k)]
        if src.device != dev:
            src = src.to(dev)
        body, tab, pos, vals = _encode_chunk(src)
        esc_n += pos.numel()
        # Escapes past the budget: stop early, a raw copy is cheaper.
        if esc_n > max_esc or (s + k) * 3 // 2 + 5 * esc_n > max_ratio * 2 * words:
            return None
        if out is not None:
            out[s * 3 // 2:s * 3 // 2 + body.numel()].copy_(body)
        else:
            parts.append((s * 3 // 2, body.cpu()))
        tables.append(tab.cpu())
        pos_all.append((pos + s).cpu())
        vals_all.append(vals.cpu())
        del body, src
    codec = RegionCodec(words, chunk, esc_n)
    if codec.nbytes > max_ratio * codec.raw_nbytes or (out is not None and codec.nbytes > out.numel()):
        return None
    parts.append((codec.aux_off, torch.cat(tables).view(-1)))
    if esc_n:
        pos = torch.cat(pos_all).to(codec.pos_dtype)
        parts.append((codec.pos_off, pos.view(torch.uint8)))
        parts.append((codec.vals_off, torch.cat(vals_all)))
    return parts, codec


def write_parts(blob, parts):
    for off, t in parts:
        blob[off:off + t.numel()].copy_(t)


# Words per torch-decode slice: bounds its temporaries (about 20 bytes a word) on a card that is already full.
_TORCH_SLICE = 1 << 20


def _decode_chunk_torch(src, tables, first_word, k, out_u8):
    """Reference / fallback decode of one chunk into out_u8 (2k bytes), torch ops only (any device)."""
    for a in range(0, k, _TORCH_SLICE):
        n = min(_TORCH_SLICE, k - a)
        sm = src[a:a + n]
        packed = src[k + a // 2:k + (a + n) // 2]
        code = torch.stack((packed & 15, packed >> 4), dim = 1).view(-1)
        gidx = (torch.arange(n, device = src.device, dtype = torch.int64) + (first_word + a)) >> 16
        exp = tables.view(-1)[gidx * 16 + code.long()]
        o = out_u8[2 * a:2 * (a + n)].view(-1, 2)
        o[:, 0] = ((exp & 1) << 7) | (sm & 0x7F)
        o[:, 1] = (sm & 0x80) | (exp >> 1)


def _patch_torch(pos, vals, out_u8):
    if pos.numel() == 0:
        return
    o = out_u8.view(-1, 2)
    p = pos.long()
    o[p, 0] = (o[p, 0] & 0x7F) | ((vals & 1) << 7)
    o[p, 1] = (o[p, 1] & 0x80) | (vals >> 1)


def _aux_views(codec, aux):
    """(tables, positions, exponents) views of an aux buffer that holds blob[aux_off:]."""
    tables = aux[:codec.tables_bytes]
    base = codec.pos_off - codec.aux_off
    size = 4 if codec.pos_dtype == torch.int32 else 8
    pos = aux[base:base + codec.esc_n * size].view(codec.pos_dtype)
    v0 = codec.vals_off - codec.aux_off
    vals = aux[v0:v0 + codec.esc_n]
    return tables, pos, vals


def _check_buffers(codec, blob, out):
    if blob.dtype != torch.uint8 or blob.dim() != 1 or blob.numel() < codec.nbytes:
        raise ValueError(f"blob holds {blob.numel()} bytes, the region needs {codec.nbytes}")
    if out.dtype != torch.uint8 or out.dim() != 1 or out.numel() < codec.raw_nbytes:
        raise ValueError(f"output holds {out.numel()} bytes, the region needs {codec.raw_nbytes}")


def decode_region_cpu(codec, blob, out = None):
    """Decode a host blob back to the region's raw bytes (CPU, exact)."""
    if out is None:
        out = torch.empty(codec.raw_nbytes, dtype = torch.uint8)
    _check_buffers(codec, blob, out)
    tables, pos, vals = _aux_views(codec, blob[codec.aux_off:codec.nbytes])
    for j in range(codec.n_chunks):
        s, k, off = codec.chunk_span(j)
        _decode_chunk_torch(blob[off:off + k * 3 // 2], tables, s, k, out[2 * s:2 * (s + k)])
    _patch_torch(pos, vals, out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# GPU decode
# Module level, not inside a function: Triton 3.2 resolves `tl` and other names in a kernel body from the module's
# globals, not from an enclosing scope.
try:
    import triton
    import triton.language as tl
except Exception:
    triton = tl = None

_TRITON = None
if tl is not None:
    try:
        @triton.jit(do_not_specialize = ["k", "first_word", "n_blocks"])
        def _bswap_decode_kernel(src, tab, out, k, first_word, n_blocks, BLOCK: tl.constexpr):  # words per tile, fixed
            # Grid-stride over BLOCK-word tiles (the grid may be capped below n_blocks). Offsets inside a chunk stay
            # below CHUNK; the region-wide index (first_word + offs) is int64, since a region can pass 2**31 words.
            for b in range(tl.program_id(0), n_blocks, tl.num_programs(0)):
                offs = tl.arange(0, BLOCK).to(tl.int64) + b * BLOCK
                mask = offs < k
                sm = tl.load(src + offs, mask = mask, other = 0).to(tl.int32)
                packed = tl.load(src + k + (offs >> 1), mask = mask, other = 0).to(tl.int32)
                code = (packed >> ((offs & 1).to(tl.int32) * 4)) & 15
                g = (first_word + offs) >> 16
                e = tl.load(tab + g * 16 + code, mask = mask, other = 0).to(tl.int32)
                w = ((sm & 0x80) << 8) | (e << 7) | (sm & 0x7F)
                tl.store(out + offs, w.to(tl.int16), mask = mask)

        @triton.jit(do_not_specialize = ["m"])
        def _bswap_patch_kernel(pos, vals, out, m, BLOCK: tl.constexpr):  # escapes per program, fixed
            offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            mask = offs < m
            p = tl.load(pos + offs, mask = mask, other = 0).to(tl.int64)
            e = tl.load(vals + offs, mask = mask, other = 0).to(tl.int32)
            w = tl.load(out + p, mask = mask, other = 0).to(tl.int32)
            w = (w & 0x807F) | (e << 7)
            tl.store(out + p, w.to(tl.int16), mask = mask)

        @triton.jit(do_not_specialize = ["q_n", "first_word", "n_blocks"])
        def _bswap_decode4_kernel(sm32, nib16, tab, out64, q_n, first_word, n_blocks, BLOCK: tl.constexpr):  # 4-word lanes per tile, fixed
            # Four words per lane: one 32 bit load of sign + mantissa bytes, one 16 bit load of codes, one 64 bit
            # store. A tile never straddles a 64K-word table group (4 * BLOCK divides GROUP).
            for b in range(tl.program_id(0), n_blocks, tl.num_programs(0)):
                q = tl.arange(0, BLOCK).to(tl.int64) + b * BLOCK
                mask = q < q_n
                smw = tl.load(sm32 + q, mask = mask, other = 0)
                nw = tl.load(nib16 + q, mask = mask, other = 0).to(tl.int32)
                base = tab + ((first_word + q * 4) >> 16) * 16
                res = tl.zeros([BLOCK], dtype = tl.int64)
                for i in tl.static_range(4):
                    sb = (smw >> (8 * i)) & 0xFF
                    c = (nw >> (4 * i)) & 15
                    e = tl.load(base + c, mask = mask, other = 0).to(tl.int32)
                    w = ((sb & 0x80) << 8) | (e << 7) | (sb & 0x7F)
                    res = res | (w.to(tl.int64) << (16 * i))
                tl.store(out64 + q, res, mask = mask)

        _TRITON = (_bswap_decode_kernel, _bswap_patch_kernel, _bswap_decode4_kernel)
    except Exception:
        _TRITON = None


def _triton_kernels():
    return _TRITON


def triton_decode_available():
    return _triton_kernels() is not None


_BLOCK = 4096
_PATCH_BLOCK = 1024


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Decode stream priority ("high": its tiles are scheduled ahead of the training stream's) and a cap on decode
# CTAs (0 = one per tile), to trade decode latency against how many SMs it takes from the forward beside it.
_PRIORITY = os.environ.get("UNSLOTH_BLOCK_SWAP_DECODE_PRIORITY", "high").strip().lower()
_MAX_CTAS = _env_int("UNSLOTH_BLOCK_SWAP_DECODE_CTAS", 0)
# Staging chunks in flight: the copy engine runs up to this many chunks ahead of a decode that waits for SMs
# (a long attention or GEMM tile on the training stream), instead of idling.
_RING = max(2, _env_int("UNSLOTH_BLOCK_SWAP_DECODE_RING", 4))


# Four words per lane (wide loads and stores); 0 = one word per lane.
_VEC = _env_int("UNSLOTH_BLOCK_SWAP_DECODE_VEC", 1)
_BLOCK4 = 1024


def _decode_chunk(src, tables, first_word, k, out_u8, use_triton):
    if use_triton and _VEC:
        dec4 = _triton_kernels()[2]
        q_n = k // 4
        n_blocks = -(-q_n // _BLOCK4)
        grid = min(n_blocks, _MAX_CTAS) if _MAX_CTAS > 0 else n_blocks
        dec4[(grid,)](src[:k].view(torch.int32), src[k:k + k // 2].view(torch.int16), tables, out_u8.view(torch.int64),
                      q_n, first_word, n_blocks, BLOCK = _BLOCK4, num_warps = 4)
    elif use_triton:
        dec = _triton_kernels()[0]
        n_blocks = -(-k // _BLOCK)
        grid = min(n_blocks, _MAX_CTAS) if _MAX_CTAS > 0 else n_blocks
        dec[(grid,)](src, tables, out_u8.view(torch.int16), k, first_word, n_blocks, BLOCK = _BLOCK, num_warps = 4)
    else:
        _decode_chunk_torch(src, tables, first_word, k, out_u8)


def _patch(pos, vals, out_u8, use_triton):
    m = pos.numel()
    if m == 0:
        return
    if use_triton:
        patch = _triton_kernels()[1]
        patch[(-(-m // _PATCH_BLOCK),)](pos, vals, out_u8.view(torch.int16), m, BLOCK = _PATCH_BLOCK, num_warps = 4)
    else:
        _patch_torch(pos, vals, out_u8)


class DeviceDecoder:
    """Per card: a high priority decode stream, a ring of staging buffers and two aux buffers.

    Chunks are copied into the ring on the block's copy stream while earlier ones decode on the decode stream,
    so the copy engine only waits when the decode falls a whole ring behind. The ring and the aux buffers
    continue across fetches: the next block's copies start while this block's last chunks still decode."""

    def __init__(self, device, staging_bytes, aux_bytes, use_triton = True, ring = None):
        self.device = torch.device(device)
        self.use_triton = bool(use_triton) and triton_decode_available()
        prio = 0
        with torch.cuda.device(self.device):
            if _PRIORITY != "normal":
                try:
                    lo, hi = torch.cuda.Stream.priority_range()
                    prio = min(lo, hi)
                except Exception:
                    prio = 0
            self.stream = torch.cuda.Stream(device = self.device, priority = prio)
        ring = max(2, int(_RING if ring is None else ring))
        self.staging = [torch.empty(max(16, staging_bytes), dtype = torch.uint8, device = self.device) for _ in range(ring)]
        self.aux = [torch.empty(max(16, aux_bytes), dtype = torch.uint8, device = self.device) for _ in range(2)]
        # Staging and aux are written on a copy stream and read on the decode stream: freeing them (teardown) must
        # wait for both, so record each stream once.
        self._streams_seen = set()
        self._own(self.stream)
        self.copied = [torch.cuda.Event() for _ in range(ring)]
        self.freed = [torch.cuda.Event() for _ in range(ring)]
        self.aux_ready = [torch.cuda.Event() for _ in range(2)]
        self.aux_free = [torch.cuda.Event() for _ in range(2)]
        self._chunk = 0
        self.fetches = 0
        self.failed = None
        # One submitter at a time: the ring and aux events are re-recorded per fetch (a state_dict decode on
        # another thread must not interleave with a prefetch).
        self._lock = threading.Lock()

    def _own(self, stream):
        key = (stream.device, stream.stream_id)
        if key not in self._streams_seen:
            for t in self.staging + self.aux:
                t.record_stream(stream)
            self._streams_seen.add(key)

    def nbytes(self):
        return sum(t.numel() for t in self.staging) + sum(t.numel() for t in self.aux)

    def fetch(self, codec, blob, out_u8, copy_stream, done):
        """Enqueue host blob -> device raw region `out_u8`; `done` is recorded once the region is complete.
        The caller made `copy_stream` wait for the compute stream (the slot is free)."""
        if blob.numel() < codec.nbytes or out_u8.numel() < codec.raw_nbytes or out_u8.dtype != torch.uint8:
            raise ValueError("fetch buffers smaller than the region")
        if (codec.aux_nbytes > self.aux[0].numel()
                or min(codec.chunk, codec.words) * 3 // 2 > self.staging[0].numel()):
            raise ValueError("region larger than this decoder's staging / aux buffers")
        with self._lock:
            try:
                self._fetch(codec, blob, out_u8, copy_stream, done)
            except BaseException:
                # A partial submission skipped some release records: release every ring and aux slot after all
                # work queued so far, so a later fetch never reuses a buffer still in flight.
                ds = self.stream
                ds.wait_stream(copy_stream)
                for e in self.freed + self.aux_free:
                    e.record(ds)
                raise

    def _fetch(self, codec, blob, out_u8, copy_stream, done):
        ds = self.stream
        # The decode stream writes the slot: the allocator must not hand its memory out before that is done.
        out_u8.record_stream(ds)
        self._own(copy_stream)
        a = self.fetches & 1
        with torch.cuda.device(self.device):
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(self.aux_free[a])
                aux = self.aux[a][:codec.aux_nbytes]
                aux.copy_(blob[codec.aux_off:codec.nbytes], non_blocking = True)
                self.aux_ready[a].record(copy_stream)
            ds.wait_event(self.aux_ready[a])
            tables, pos, vals = _aux_views(codec, aux)
            for j in range(codec.n_chunks):
                s, k, off = codec.chunk_span(j)
                n = k * 3 // 2
                slot = self._chunk % len(self.staging)
                self._chunk += 1
                staged = self.staging[slot][:n]
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(self.freed[slot])
                    staged.copy_(blob[off:off + n], non_blocking = True)
                    self.copied[slot].record(copy_stream)
                ds.wait_event(self.copied[slot])
                with torch.cuda.stream(ds):
                    self._run(lambda: _decode_chunk(staged, tables, s, k, out_u8[2 * s:2 * (s + k)], self.use_triton),
                              lambda: _decode_chunk_torch(staged, tables, s, k, out_u8[2 * s:2 * (s + k)]))
                    self.freed[slot].record(ds)
            with torch.cuda.stream(ds):
                self._run(lambda: _patch(pos, vals, out_u8, self.use_triton), lambda: _patch_torch(pos, vals, out_u8))
                self.aux_free[a].record(ds)
                done.record(ds)
        self.fetches += 1

    def _run(self, fast, slow):
        if not self.use_triton:
            return slow()
        try:
            return fast()
        except Exception as e:
            if type(e).__module__.startswith("torch._dynamo") or isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            # A failed launch (JIT or driver): the torch decode writes the same bits.
            self.use_triton = False
            self.failed = repr(e)[:300]
            return slow()


def _self_check_data(words, seed = 0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(words, generator = g) * 0.02).to(torch.bfloat16)
    raw = x.view(torch.uint8).clone()
    w = raw.view(torch.int16)
    # Escapes: zeros, subnormals, inf / nan, large values, random words.
    idx = torch.randint(0, words, (words // 97,), generator = g)
    w[idx] = torch.randint(-32768, 32767, (idx.numel(),), generator = g, dtype = torch.int16)
    w[:8] = torch.tensor([0, -32768, 0x7F80, -128, 0x7FC0, 1, 0x3F80, 0x0080], dtype = torch.int16)
    return raw


def self_check(device, use_triton = True):
    """Round trip synthetic words (several chunks, a partial chunk, many escapes) through the device decode.
    Returns (ok, use_triton, reason). Eager only: call at install, never under capture or tracing."""
    try:
        chunk = 4 * GROUP
        words = 2 * chunk + 3 * GROUP + 256
        raw = _self_check_data(words)
        enc = encode_region(raw, chunk = chunk, max_ratio = 10.0, max_escape_fraction = 1.0)
        if enc is None:
            return False, False, "self-check data did not encode"
        parts, codec = enc
        blob = torch.zeros(codec.nbytes, dtype = torch.uint8)
        write_parts(blob, parts)
        if not torch.equal(decode_region_cpu(codec, blob), raw):
            return False, False, "CPU round trip mismatch"
        device = torch.device(device)
        for triton_on in ((True, False) if use_triton else (False,)):
            dec = DeviceDecoder(device, chunk * 3 // 2, codec.aux_nbytes, use_triton = triton_on)
            if triton_on and not dec.use_triton:
                continue
            out = torch.full((codec.raw_nbytes,), 0xAB, dtype = torch.uint8, device = device)
            done = torch.cuda.Event()
            cs = torch.cuda.Stream(device = device)
            cs.wait_stream(torch.cuda.current_stream(device))
            dec.fetch(codec, blob, out, cs, done)
            torch.cuda.current_stream(device).wait_event(done)
            got = out.cpu()
            if torch.equal(got, raw) and (not triton_on or dec.use_triton):
                return True, dec.use_triton, None
        return False, False, "device round trip mismatch"
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return False, False, "out of memory"
        return False, False, repr(e)[:300]
