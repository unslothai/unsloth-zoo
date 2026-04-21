# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Compile qualification, policy resolution, and runtime monkey patches for MLX.

This module is the canonical compile entrypoint for Unsloth's MLX training
stack. The file is large because it owns three different layers of behavior:

1. Qualification:
   Discover installed `mlx_vlm` architectures, scan them for known compile
   blocker patterns, and report whether each architecture has been verified.
2. Policy:
   Decide whether a particular training run should use `mx.compile`, fall back
   to eager mode, or raise when strict compile is requested.
3. Runtime patching:
   Install monkey patches for repeated blocker patterns so verified
   architectures can run compiled training without carrying architecture-local
   one-off shims throughout the trainer.

The qualification matrix is intentionally conservative. Architectures are only
marked compile-ready once explicitly verified. Everything else either runs eager
or stays in "pattern candidate" state until a verifier or real training sweep
confirms the patch set is functionally correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from itertools import accumulate
import importlib
import inspect
import mlx.core as mx
import numpy as np
import pkgutil
import re
from typing import Any, Callable, Iterable, Mapping


_PATCHES_INSTALLED = False
_PATCHED_ARCHES: set[str] = set()
_PATCHED_PATTERN_BUNDLES: set[str] = set()
_PATCH_BINDINGS: set[tuple[str, str, str, str]] = set()

# Architectures explicitly verified for mlx compile support.
# Training verification currently covers:
# - qwen2_5_vl: real end-to-end compiled training via train.py
# - qwen3_vl / qwen3_5 / qwen3_5_moe / gemma4 / paligemma / moondream3:
#   compiled synthetic forward+backward
# SmolVLM has processor/template support, but real mlx-vlm training still hits
# MLX primitive-less-array failures after a compiled call. Keep it patched but
# unqualified until a real dataset compile run passes.
_VERIFIED_TRAINING_ARCHES: set[str] = {
    "aya_vision",
    "deepseekocr",
    "deepseekocr_2",
    "gemma3",
    "gemma3n",
    "gemma4",
    "glm_ocr",
    "idefics2",
    "idefics3",
    "llama4",
    "llava",
    "llava_bunny",
    "llava_next",
    "mistral3",
    "mistral4",
    "mllama",
    "moondream3",
    "multi_modality",
    "paddleocr_vl",
    "paligemma",
    "phi4_siglip",
    "phi4mm",
    "phi3_v",
    "pixtral",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_vl_moe",
    "qwen3_vl",
}
_VERIFIED_GENERATION_ARCHES: set[str] = set()

_MODEL_REPO_TRAINING_COMPILE_BLOCKLIST: tuple[tuple[str, str], ...] = (
    (
        "smolvlm",
        "SmolVLM/Idefics3 real training currently leaves MLX primitive-less arrays after compiled execution",
    ),
)

_BACKEND_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "llm_config",
)

_BLOCKER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "data_to_python": (
        re.compile(r"\.item\s*\("),
        re.compile(r"\.tolist\s*\("),
        re.compile(r"np\.where\s*\("),
        re.compile(r"mx\.eval\s*\("),
    ),
    "ragged_python_control_flow": (
        re.compile(r"mx\.split\([^)]*tolist\s*\("),
        re.compile(r"\.append\s*\("),
        re.compile(r"for\s+\w+\s+in\s+enumerate\s*\("),
        re.compile(r"for\s+\w+\s+in\s+range\s*\("),
        re.compile(r"\bwhile\b"),
    ),
    "mutable_model_state": (
        re.compile(r"_position_ids"),
        re.compile(r"_rope_deltas"),
        re.compile(r"\.pop\s*\("),
    ),
}

_TRAIT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "qwen_like_image_merge": (
        re.compile(r"merge_input_ids_with_image_features"),
        re.compile(r"image_token_(?:id|index)"),
    ),
    "qwen2_vision_windowing": (
        re.compile(r"VisionRotaryEmbedding"),
        re.compile(r"get_window_index"),
        re.compile(r"rot_pos_emb"),
    ),
    "qwen3_deepstack_multimodal": (
        re.compile(r"_deepstack_process"),
        re.compile(r"visual_pos_masks"),
        re.compile(r"fast_pos_embed_interpolate"),
    ),
    "single_image_token_merge": (
        re.compile(r"_prepare_inputs_for_multimodal"),
        re.compile(r"merge_input_ids_with_image_features"),
    ),
    "mistral4_attention_backend": (
        re.compile(r"Mistral4Attention"),
        re.compile(r"qk_rope_head_dim"),
    ),
    "gemma3n_multiscale_fusion": (
        re.compile(r"MobileNetV5MultiScaleFusionAdapter"),
        re.compile(r"merge_multimodal_and_text"),
    ),
    "deepseek_ocr_multimodal": (
        re.compile(r"get_abs_pos_sam"),
        re.compile(r"MlpProjector"),
        re.compile(r"sam_config"),
    ),
    "masked_scatter_multimodal": (
        re.compile(r"def masked_scatter"),
        re.compile(r"special_image_mask"),
        re.compile(r"image_mask_expanded"),
    ),
    "padded_image_filtering": (
        re.compile(r"pixel_attention_mask"),
        re.compile(r"real_images_inds"),
        re.compile(r"patch_attention_mask"),
    ),
    "negative_image_placeholders": (
        re.compile(r"inputs\s*<\s*0"),
        re.compile(r"vision_embed_tokens"),
        re.compile(r"image_sizes"),
    ),
    "expanded_image_placeholders": (
        re.compile(r"add_image_token"),
        re.compile(r"num_image_tokens"),
        re.compile(r"image_token_index"),
    ),
    "phi4_multimodal_spans": (
        re.compile(r"pixel_attention_mask"),
        re.compile(r"spatial_shapes"),
        re.compile(r"audio_token_index|IMAGE_TOKEN_INDEX"),
    ),
    "cross_attention_kv": (
        re.compile(r"cross_attention_states"),
        re.compile(r"cross_attention_mask"),
    ),
    "mutable_position_state": (
        re.compile(r"_position_ids"),
        re.compile(r"_rope_deltas"),
    ),
}

_TRAINING_VERIFIER_HINTS: dict[str, str] = {
    "aya_vision": "verify_aya_vision",
    "deepseekocr": "verify_deepseekocr",
    "deepseekocr_2": "verify_deepseekocr2",
    "gemma3": "verify_gemma3",
    "gemma3n": "verify_gemma3n",
    "gemma4": "verify_gemma4",
    "glm_ocr": "verify_glm_ocr",
    "idefics2": "verify_idefics2",
    "idefics3": "verify_idefics3",
    "llama4": "verify_llama4",
    "llava": "verify_llava",
    "llava_bunny": "verify_llava_bunny",
    "llava_next": "verify_llava_next",
    "mistral3": "verify_mistral3",
    "mistral4": "verify_mistral4",
    "mllama": "verify_mllama",
    "moondream3": "verify_moondream3",
    "multi_modality": "verify_multi_modality",
    "paddleocr_vl": "verify_paddleocr_vl",
    "paligemma": "verify_paligemma",
    "phi4_siglip": "verify_phi4_siglip",
    "phi4mm": "verify_phi4mm",
    "phi3_v": "verify_phi3_v",
    "pixtral": "verify_pixtral",
    "qwen2_vl": "verify_qwen2_vl",
    "qwen2_5_vl": "verify_qwen2_5_vl",
    "qwen3_5": "verify_qwen3_5",
    "qwen3_5_moe": "verify_qwen3_5_moe",
    "qwen3_vl_moe": "verify_qwen3_vl_moe",
    "qwen3_vl": "verify_qwen3_vl",
    "smolvlm": "verify_smolvlm",
}


@dataclass(frozen=True)
class MLXVLMTraitReport:
    """Static scan result for one installed `mlx_vlm` architecture.

    Trait reports are architecture-level hints only. They are used to:
    - surface likely compile blocker categories for maintainers
    - identify shared patch bundles that can be applied generically
    - explain why an architecture is qualified, unqualified, or a candidate
    """

    arch: str
    blocker_categories: tuple[str, ...]
    pattern_traits: tuple[str, ...]
    backend_arches: tuple[str, ...]
    verification_state: str
    verifier_hint: str | None = None


@dataclass(frozen=True)
class CompilePatternBundle:
    """A reusable compile patch bundle keyed by architecture traits.

    The bundle registry is intentionally declarative. A bundle describes:
    - how to recognize a family via `matcher`
    - which semantic patch primitives it needs
    - which thin family adapter explains method/contract differences
    - which runtime patch primitives implement the monkey patches

    Installation happens through a generic patch-plan executor rather than
    through per-architecture installer dispatch. That keeps the public surface
    easy to trace even when a runtime primitive still needs specialized code
    under the hood.
    """

    name: str
    description: str
    matcher: Callable[[str, MLXVLMTraitReport], bool]
    primitive_names: tuple[str, ...] = ()
    adapter_name: str | None = None
    protocol_names: tuple[str, ...] = ()
    runtime_primitive_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class MLXVLMCompilePolicy:
    """User-facing compile policy after CLI/config normalization."""

    mode: str = "best_effort"
    arch_overrides: tuple[tuple[str, str], ...] = ()
    backend_overrides: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResolvedTrainingCompileDecision:
    """Final per-model compile decision used by the trainer."""

    arch: str
    enabled: bool
    policy_mode: str
    fallback_allowed: bool
    strict_requested: bool
    should_raise: bool
    reason: str
    qualification: "MLXVLMCompileQualification | None"
    backend_qualifications: tuple["MLXVLMCompileQualification", ...] = ()
    support_state: str = "unsupported"
    setting_recommendations: tuple["CompileSettingRecommendation", ...] = ()
    patch_mode: str = "patched"


@dataclass(frozen=True)
class MLXVLMCompileQualification:
    """Qualified compile status for one architecture after patches are installed."""

    arch: str
    training_compile: bool
    generation_compile: bool
    patched: bool
    blocker_categories: tuple[str, ...]
    reason: str
    verification_state: str = "unqualified"
    support_state: str = "unsupported"
    backend_arches: tuple[str, ...] = ()
    installed_patterns: tuple[str, ...] = ()
    trait_report: MLXVLMTraitReport | None = None


@dataclass(frozen=True)
class CompilePatchPrimitive:
    """Reusable compile-safe operation shared across multiple model families."""

    name: str
    description: str


@dataclass(frozen=True)
class CompilePatchAdapter:
    """Thin family-level contract description for shared patch primitives.

    Adapters should prefer semantic matching over explicit architecture lists.
    A future family should ideally pick up an existing adapter because it
    exposes the same source traits or naming convention, not because we added
    one more architecture string by hand.
    """

    name: str
    description: str
    arch_names: tuple[str, ...] = ()
    arch_prefixes: tuple[str, ...] = ()
    required_traits: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompilePatchPlan:
    """Resolved patch plan for one matched compile bundle.

    Patch plans are the maintainer-facing bridge between the declarative bundle
    registry and the actual monkey patches that get applied at runtime.
    """

    bundle_name: str
    adapter_name: str | None
    protocol_names: tuple[str, ...]
    semantic_primitives: tuple[str, ...]
    runtime_primitives: tuple[str, ...]


@dataclass(frozen=True)
class CompileSettingRecommendation:
    """Recommended or auto-applied training setting for compile stability."""

    setting: str
    recommended_value: object
    reason: str
    auto_applied: bool = False


@dataclass(frozen=True)
class CompileTraceReport:
    """Machine-readable explanation of compile support for one model or arch."""

    arch: str
    patch_mode: str
    backend_arches: tuple[str, ...]
    verification_state: str
    support_state: str
    blocker_categories: tuple[str, ...]
    pattern_traits: tuple[str, ...]
    matched_bundles: tuple[str, ...]
    installed_bundles: tuple[str, ...]
    patch_primitives: tuple[str, ...]
    runtime_primitives: tuple[str, ...]
    adapters: tuple[str, ...]
    qualification_reason: str
    decision_mode: str
    decision_enabled: bool
    decision_reason: str
    fallback_allowed: bool
    strict_requested: bool
    backend_qualification_arches: tuple[str, ...] = ()
    recommendations: tuple[CompileSettingRecommendation, ...] = ()
    direct_protocols: tuple[str, ...] = ()
    inferred_protocols: tuple[str, ...] = ()
    satisfied_protocols: tuple[str, ...] = ()
    missing_protocols: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompileProtocolRequirement:
    """Patchable runtime protocol used to discover generic MLX patch targets."""

    name: str
    description: str
    candidate_method_names: tuple[str, ...]
    module_keywords: tuple[str, ...] = ()
    source_tokens: tuple[str, ...] = ()
    required_traits: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompileProtocolTarget:
    """One discovered runtime target that may satisfy a compile patch protocol."""

    protocol_name: str
    arch: str
    module_name: str
    class_name: str
    method_name: str
    matched_traits: tuple[str, ...]
    missing_source_tokens: tuple[str, ...] = ()
    status: str = "protocol_matched"
    signature: str | None = None


@dataclass(frozen=True)
class CompilePatchabilityReport:
    """Maintainer-facing dry-run report for compile patch support on one arch."""

    arch: str
    support_state: str
    matched_bundles: tuple[str, ...]
    suggested_patch_plan: tuple[CompilePatchPlan, ...]
    discovered_targets: tuple[CompileProtocolTarget, ...]
    direct_protocols: tuple[str, ...]
    inferred_protocols: tuple[str, ...]
    satisfied_protocols: tuple[str, ...]
    missing_protocols: tuple[str, ...]
    blockers: tuple[str, ...]


def _iter_arch_modules() -> Iterable[tuple[str, Path]]:
    """Yield installed `mlx_vlm.models.<arch>` package roots.

    We inspect the installed package instead of maintaining a static registry so
    the qualification layer reflects the actual venv contents.
    """

    try:
        import mlx_vlm.models as vlm_models_pkg
    except ImportError:
        return ()

    roots = []
    for module_path in vlm_models_pkg.__path__:
        root = Path(module_path)
        if root.exists():
            roots.append(root)

    items = []
    seen = set()
    for root in roots:
        for _, modname, ispkg in pkgutil.iter_modules([str(root)]):
            if not ispkg or modname.startswith("__") or modname in seen:
                continue
            seen.add(modname)
            items.append((modname, root / modname))
    return tuple(sorted(items, key=lambda x: x[0]))


@lru_cache(maxsize=1)
def discover_architectures() -> tuple[str, ...]:
    """Return discovered `mlx_vlm` architecture names from the active env."""

    return tuple(name for name, _ in _iter_arch_modules())


@lru_cache(maxsize=None)
def _read_architecture_source(arch: str) -> str:
    """Read non-processor Python source for one installed architecture."""

    arch_dir = dict(_iter_arch_modules()).get(arch)
    if arch_dir is None or not arch_dir.exists():
        return ""

    chunks = []
    for path in arch_dir.rglob("*.py"):
        if path.name.startswith("processing") or "__pycache__" in path.parts:
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n\n".join(chunks)


def _scan_architecture_sources(arch: str) -> tuple[str, ...]:
    """Scan architecture source for coarse compile blocker categories."""

    source = _read_architecture_source(arch)
    if not source:
        return ()
    categories = set()
    for category, patterns in _BLOCKER_PATTERNS.items():
        if any(pattern.search(source) for pattern in patterns):
            categories.add(category)
    return tuple(sorted(categories))


def _scan_pattern_traits(arch: str) -> tuple[str, ...]:
    """Scan architecture source for reusable patch traits."""

    source = _read_architecture_source(arch)
    if not source:
        return ()
    traits = set()
    for trait, patterns in _TRAIT_PATTERNS.items():
        if any(pattern.search(source) for pattern in patterns):
            traits.add(trait)
    return tuple(sorted(traits))


def _verification_state(
    arch: str,
    patched: bool,
    has_pattern_candidates: bool,
) -> str:
    """Classify verification state for reporting and policy explanations."""

    if arch in _VERIFIED_TRAINING_ARCHES and arch in _VERIFIED_GENERATION_ARCHES:
        return "fully_verified"
    if arch in _VERIFIED_TRAINING_ARCHES:
        return "training_verified"
    if arch in _VERIFIED_GENERATION_ARCHES:
        return "generation_verified"
    if patched:
        return "patched_unverified"
    if has_pattern_candidates:
        return "pattern_candidate"
    return "unqualified"


def _support_state(
    *,
    training_ok: bool,
    generation_ok: bool,
    patched: bool,
    matched_patterns: tuple[str, ...] = (),
) -> str:
    """Return the user-facing support state for one architecture.

    `supported_inferred` is reserved for architectures where the shared patch
    system confidently matched reusable patterns, but the family has not yet
    been promoted to explicitly verified. We still keep runtime compile policy
    conservative unless verification exists.
    """

    if training_ok or generation_ok:
        return "supported_verified"
    if patched and matched_patterns:
        return "supported_inferred"
    if patched:
        return "patched_unverified"
    if matched_patterns:
        return "fallback_only"
    return "unsupported"


@lru_cache(maxsize=1)
def list_compile_patch_primitives() -> tuple[CompilePatchPrimitive, ...]:
    """Return the reusable compile-safe primitive registry."""

    return (
        CompilePatchPrimitive(
            name="safe_fused_sdpa_mask",
            description="Protect MLX fused SDPA from NaN gradients on fully masked additive float rows.",
        ),
        CompilePatchPrimitive(
            name="compile_safe_feature_merge",
            description="Merge placeholder-token multimodal features without NumPy or mutation-heavy scatter.",
        ),
        CompilePatchPrimitive(
            name="compile_safe_masked_scatter",
            description="Flatten masked-scatter style image/audio insertion into pure MLX array ops.",
        ),
        CompilePatchPrimitive(
            name="segmented_vision_attention",
            description="Split vision q/k/v by cumulative sequence lengths and run fused SDPA per chunk.",
        ),
        CompilePatchPrimitive(
            name="vision_metadata_normalization",
            description="Normalize tiny grid/shape/span metadata to immutable Python tuples before compiled array math.",
        ),
        CompilePatchPrimitive(
            name="explicit_position_plumbing",
            description="Replace mutable or host-side position state with explicit position-id handling.",
        ),
        CompilePatchPrimitive(
            name="cached_image_feature_plumbing",
            description="Support cached image feature injection through get_input_embeddings paths.",
        ),
        CompilePatchPrimitive(
            name="dtype_normalization",
            description="Cast pixels, masks, and multimodal features to stable dtypes for compiled training.",
        ),
        CompilePatchPrimitive(
            name="padded_image_filtering",
            description="Replace Python-side padded image filtering with compile-safe MLX operations.",
        ),
    )


@lru_cache(maxsize=1)
def list_protocol_requirements(
    protocol_name: str | None = None,
) -> tuple[CompileProtocolRequirement, ...]:
    """Return the protocol registry used for generic compile target discovery.

    These protocols are intentionally broader than architecture names. They
    describe the stable runtime contracts we look for when deciding whether an
    existing patch primitive can be reused by a new family.
    """

    protocols = (
        CompileProtocolRequirement(
            name="multimodal_merge",
            description="Model-level multimodal embedding merge path.",
            candidate_method_names=(
                "get_input_embeddings",
                "merge_input_ids_with_image_features",
                "_merge_input_ids_with_image_features",
                "_prepare_inputs_for_multimodal",
            ),
            source_tokens=("input_ids", "image", "embed"),
            required_traits=(
                "qwen_like_image_merge",
                "single_image_token_merge",
                "deepseek_ocr_multimodal",
                "negative_image_placeholders",
                "expanded_image_placeholders",
                "phi4_multimodal_spans",
            ),
        ),
        CompileProtocolRequirement(
            name="vision_attention",
            description="Vision attention implementation that can be chunked or mask-normalized.",
            candidate_method_names=("__call__",),
            module_keywords=("vision",),
            source_tokens=("qkv", "scale"),
            required_traits=(
                "qwen2_vision_windowing",
                "qwen3_deepstack_multimodal",
                "mistral4_attention_backend",
                "deepseek_ocr_multimodal",
            ),
        ),
        CompileProtocolRequirement(
            name="position_primitives",
            description="Vision or language position helper path requiring compile-safe metadata handling.",
            candidate_method_names=(
                "rot_pos_emb",
                "get_window_index",
                "fast_pos_embed_interpolate",
                "get_rope_index",
            ),
            source_tokens=("pos",),
            required_traits=(
                "qwen2_vision_windowing",
                "qwen3_deepstack_multimodal",
                "mutable_position_state",
            ),
        ),
        CompileProtocolRequirement(
            name="cached_image_features",
            description="Embedding path that accepts cached or precomputed visual features.",
            candidate_method_names=("get_input_embeddings", "_prepare_inputs_for_multimodal"),
            source_tokens=("cached", "image"),
            required_traits=(
                "qwen2_vision_windowing",
                "qwen3_deepstack_multimodal",
                "single_image_token_merge",
                "padded_image_filtering",
                "deepseek_ocr_multimodal",
                "negative_image_placeholders",
                "expanded_image_placeholders",
                "phi4_multimodal_spans",
            ),
        ),
        CompileProtocolRequirement(
            name="masked_scatter",
            description="Masked-scatter or placeholder insertion helper for multimodal features.",
            candidate_method_names=("masked_scatter", "merge_multimodal_and_text"),
            source_tokens=("mask",),
            required_traits=("masked_scatter_multimodal", "gemma3n_multiscale_fusion"),
        ),
        CompileProtocolRequirement(
            name="placeholder_expansion",
            description="Families that expand or replace negative/repeated placeholder spans.",
            candidate_method_names=(
                "_prepare_inputs_for_multimodal",
                "get_input_embeddings",
                "merge_input_ids_with_image_features",
            ),
            source_tokens=("token", "image"),
            required_traits=(
                "negative_image_placeholders",
                "expanded_image_placeholders",
                "phi4_multimodal_spans",
            ),
        ),
        CompileProtocolRequirement(
            name="padded_image_filtering",
            description="Python-side padded image filtering path that should become pure MLX ops.",
            candidate_method_names=("_prepare_inputs_for_multimodal", "get_input_embeddings"),
            source_tokens=("pixel_attention_mask", "real_images_inds"),
            required_traits=("padded_image_filtering",),
        ),
        CompileProtocolRequirement(
            name="deepstack_process",
            description="Visual deepstack side-channel that injects extra visual embeddings into the language path.",
            candidate_method_names=("_deepstack_process",),
            source_tokens=("visual", "embed"),
            required_traits=("qwen3_deepstack_multimodal",),
        ),
    )
    if protocol_name is None:
        return protocols
    return tuple(proto for proto in protocols if proto.name == protocol_name)


@lru_cache(maxsize=1)
def list_compile_patch_adapters() -> tuple[CompilePatchAdapter, ...]:
    """Return family-level adapters used by the compile patch registry."""

    return (
        CompilePatchAdapter(
            name="qwen_like_merge",
            description="Families that merge one placeholder token per image or video feature segment.",
            arch_names=("glm_ocr", "paddleocr_vl"),
            arch_prefixes=("qwen",),
            required_traits=("qwen_like_image_merge",),
        ),
        CompilePatchAdapter(
            name="qwen3_deepstack",
            description="Qwen3 multimodal families that return visual masks and deepstack embeddings.",
            arch_prefixes=("qwen3",),
            required_traits=("qwen3_deepstack_multimodal",),
        ),
        CompilePatchAdapter(
            name="single_image_token",
            description="Wrappers that merge image features through a single placeholder token path.",
            arch_prefixes=("aya", "llama", "llava", "mistral", "pixtral"),
            required_traits=("single_image_token_merge",),
        ),
        CompilePatchAdapter(
            name="idefics_family",
            description="Idefics-style padded-image families with shared multimodal filtering and embedding contracts.",
            arch_prefixes=("idefics", "smolvlm"),
            required_traits=("padded_image_filtering",),
        ),
        CompilePatchAdapter(
            name="phi_placeholder_family",
            description="Families that encode images or audio via placeholder spans in the text stream.",
            arch_names=("multi_modality",),
            arch_prefixes=("phi",),
            required_traits=(
                "negative_image_placeholders",
                "expanded_image_placeholders",
                "phi4_multimodal_spans",
            ),
        ),
        CompilePatchAdapter(
            name="gemma_vision_family",
            description="Gemma multimodal families with masked-scatter or multiscale fusion style image insertion.",
            arch_prefixes=("gemma",),
            required_traits=("masked_scatter_multimodal", "gemma3n_multiscale_fusion"),
        ),
        CompilePatchAdapter(
            name="ocr_projector_family",
            description="OCR-oriented vision/projector families with custom image sequence construction.",
            arch_names=("glm_ocr", "paddleocr_vl"),
            arch_prefixes=("deepseekocr",),
            required_traits=("deepseek_ocr_multimodal",),
        ),
        CompilePatchAdapter(
            name="mistral_attention_family",
            description="Mistral-family backends that need attention or multimodal wrapper compatibility patches.",
            arch_names=("mistral3", "mistral4", "pixtral", "mllama"),
            required_traits=("mistral4_attention_backend",),
        ),
    )


def normalize_mlx_patch_mode(value) -> str:
    """Normalize compile patch mode for loader, trainer, and test harnesses."""

    mode = str(value or "patched").strip().lower()
    if mode not in {"patched", "unpatched"}:
        raise ValueError(
            "Unsloth: patch_mode must be 'patched' or 'unpatched', "
            f"got {value!r}."
        )
    return mode


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _iter_backend_subconfigs(config) -> Iterable[tuple[str, object]]:
    """Yield likely nested backend configs from an MLX config object.

    The explicit key allowlist keeps the common paths readable and stable, but
    newer mlx-vlm families may introduce additional `*_config` children. We
    therefore also scan the config object's visible attributes / mapping keys
    for nested config-shaped values that advertise a `model_type`.
    """

    if config is None:
        return ()

    seen = set()
    items = []

    def add(name, value) -> None:
        if value is None or name in seen:
            return
        model_type = _config_get(value, "model_type")
        if not model_type:
            return
        seen.add(name)
        items.append((name, value))

    for key in _BACKEND_CONFIG_KEYS:
        add(key, _config_get(config, key))

    if isinstance(config, Mapping):
        for key, value in config.items():
            if isinstance(key, str) and key.endswith("_config"):
                add(key, value)
    else:
        for key in dir(config):
            if key.startswith("_") or not key.endswith("_config"):
                continue
            try:
                value = getattr(config, key)
            except Exception:
                continue
            add(key, value)

    return tuple(items)


def _extract_backend_arches(config, arch: str | None = None) -> tuple[str, ...]:
    """Extract nested backend model types from an MLX config object."""

    backend_arches = []
    seen = set()
    for key, subconfig in _iter_backend_subconfigs(config):
        model_type = _config_get(subconfig, "model_type")
        if not model_type or model_type == arch or model_type in seen:
            continue
        seen.add(model_type)
        backend_arches.append(model_type)
    return tuple(backend_arches)


def _get_model_config(model):
    return getattr(model, "_config", None) or getattr(model, "config", None)


def get_backend_architectures(model_or_arch) -> tuple[str, ...]:
    """Return nested backend architectures for a loaded MLX model instance."""

    if isinstance(model_or_arch, str):
        return ()

    arch = get_model_architecture(model_or_arch)
    config = _get_model_config(model_or_arch)
    return _extract_backend_arches(config, arch=arch)


def _build_trait_report(
    arch: str,
    backend_arches: tuple[str, ...] = (),
) -> MLXVLMTraitReport:
    """Build one trait report from static source scans and runtime patch state."""

    blockers = _scan_architecture_sources(arch)
    traits = _scan_pattern_traits(arch)
    patched = arch in _PATCHED_ARCHES
    return MLXVLMTraitReport(
        arch=arch,
        blocker_categories=blockers,
        pattern_traits=traits,
        backend_arches=backend_arches,
        verification_state=_verification_state(
            arch,
            patched=patched,
            has_pattern_candidates=bool(traits),
        ),
        verifier_hint=_TRAINING_VERIFIER_HINTS.get(arch),
    )


@lru_cache(maxsize=1)
def build_compile_trait_reports() -> dict[str, MLXVLMTraitReport]:
    """Build trait reports for every discovered architecture in the venv."""

    return {arch: _build_trait_report(arch) for arch in discover_architectures()}


def get_model_architecture(model) -> str | None:
    """Infer the top-level `mlx_vlm` architecture name from a loaded model."""

    module = getattr(model, "__module__", "")
    if ".models." not in module:
        return None
    return module.split(".models.", 1)[1].split(".", 1)[0]


def get_compile_trait_report(model_or_arch) -> MLXVLMTraitReport | None:
    """Return the trait report for an architecture or loaded model."""

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    if not arch:
        return None
    report = build_compile_trait_reports().get(arch)
    if report is None or isinstance(model_or_arch, str):
        return report

    backend_arches = get_backend_architectures(model_or_arch)
    if backend_arches == report.backend_arches:
        return report
    return _build_trait_report(arch, backend_arches=backend_arches)


def _iter_arch_python_module_names(arch: str) -> tuple[str, ...]:
    """Return importable Python module names for one installed architecture."""

    arch_dir = dict(_iter_arch_modules()).get(arch)
    if arch_dir is None or not arch_dir.exists():
        return ()

    module_names = []
    for path in sorted(arch_dir.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("processing"):
            continue
        rel = path.relative_to(arch_dir)
        stem = rel.with_suffix("")
        if stem.name == "__init__":
            module_name = f"mlx_vlm.models.{arch}"
        else:
            module_name = f"mlx_vlm.models.{arch}." + ".".join(stem.parts)
        module_names.append(module_name)
    return tuple(dict.fromkeys(module_names))


def _iter_arch_imported_modules(arch: str) -> tuple[tuple[str, object], ...]:
    """Import available architecture modules for protocol discovery."""

    imported = []
    for module_name in _iter_arch_python_module_names(arch):
        module = _try_import_module(module_name)
        if module is None:
            continue
        imported.append((module_name, module))
    return tuple(imported)


def _safe_getsource(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def _stringify_signature(callable_obj) -> str | None:
    try:
        return str(inspect.signature(callable_obj))
    except Exception:
        return None


def _discover_protocol_target_for_class(
    arch: str,
    module_name: str,
    cls,
    requirement: CompileProtocolRequirement,
    report: MLXVLMTraitReport,
) -> tuple[CompileProtocolTarget, ...]:
    """Return protocol targets discovered on one class."""

    if requirement.required_traits:
        matched_traits = tuple(
            trait for trait in requirement.required_traits if trait in report.pattern_traits
        )
        if not matched_traits:
            return ()
    else:
        matched_traits = ()

    lower_module_name = module_name.lower()
    if requirement.module_keywords and not any(
        keyword in lower_module_name for keyword in requirement.module_keywords
    ):
        return ()

    source = _safe_getsource(cls)
    targets = []
    for method_name in requirement.candidate_method_names:
        method = getattr(cls, method_name, None)
        if method is None or not callable(method):
            continue
        method_source = _safe_getsource(method)
        combined_source = source + "\n" + method_source
        missing_tokens = tuple(
            token for token in requirement.source_tokens if token not in combined_source
        )
        status = "protocol_matched" if not missing_tokens else "protocol_mismatch"
        targets.append(
            CompileProtocolTarget(
                protocol_name=requirement.name,
                arch=arch,
                module_name=module_name,
                class_name=getattr(cls, "__name__", repr(cls)),
                method_name=method_name,
                matched_traits=matched_traits,
                missing_source_tokens=missing_tokens,
                status=status,
                signature=_stringify_signature(method),
            )
        )
    return tuple(targets)


def _discover_protocol_target_for_module_functions(
    arch: str,
    module_name: str,
    module,
    requirement: CompileProtocolRequirement,
    report: MLXVLMTraitReport,
) -> tuple[CompileProtocolTarget, ...]:
    """Return protocol targets discovered on module-level helpers."""

    if requirement.required_traits:
        matched_traits = tuple(
            trait for trait in requirement.required_traits if trait in report.pattern_traits
        )
        if not matched_traits:
            return ()
    else:
        matched_traits = ()

    lower_module_name = module_name.lower()
    if requirement.module_keywords and not any(
        keyword in lower_module_name for keyword in requirement.module_keywords
    ):
        return ()

    module_source = _safe_getsource(module)
    targets = []
    for method_name in requirement.candidate_method_names:
        fn = getattr(module, method_name, None)
        if fn is None or not callable(fn) or inspect.isclass(fn):
            continue
        fn_source = _safe_getsource(fn)
        combined_source = module_source + "\n" + fn_source
        missing_tokens = tuple(
            token for token in requirement.source_tokens if token not in combined_source
        )
        status = "protocol_matched" if not missing_tokens else "protocol_mismatch"
        targets.append(
            CompileProtocolTarget(
                protocol_name=requirement.name,
                arch=arch,
                module_name=module_name,
                class_name="<module>",
                method_name=method_name,
                matched_traits=matched_traits,
                missing_source_tokens=missing_tokens,
                status=status,
                signature=_stringify_signature(fn),
            )
        )
    return tuple(targets)


def discover_compile_protocol_targets(
    model_or_arch,
) -> tuple[CompileProtocolTarget, ...]:
    """Discover patchable protocol targets for an architecture or loaded model.

    This is the main maintainer helper for new-family triage. It inspects the
    installed `mlx_vlm` modules for a family and reports which generic compile
    protocols match actual classes/methods, plus any missing source tokens that
    prevented a clean protocol match.
    """

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    if not arch:
        return ()
    report = get_compile_trait_report(model_or_arch if not isinstance(model_or_arch, str) else arch)
    if report is None:
        return ()

    targets = []
    for module_name, module in _iter_arch_imported_modules(arch):
        for requirement in list_protocol_requirements():
            targets.extend(
                _discover_protocol_target_for_module_functions(
                    arch,
                    module_name,
                    module,
                    requirement,
                    report,
                )
            )
        for _, cls in sorted(vars(module).items()):
            if not inspect.isclass(cls) or getattr(cls, "__module__", None) != module.__name__:
                continue
            for requirement in list_protocol_requirements():
                targets.extend(
                    _discover_protocol_target_for_class(
                        arch,
                        module_name,
                        cls,
                        requirement,
                        report,
                    )
                )
    unique = {}
    for target in targets:
        key = (
            target.protocol_name,
            target.module_name,
            target.class_name,
            target.method_name,
        )
        current = unique.get(key)
        if current is None or (current.status != "protocol_matched" and target.status == "protocol_matched"):
            unique[key] = target
    return tuple(sorted(unique.values(), key=lambda x: (x.protocol_name, x.module_name, x.class_name, x.method_name)))


def validate_compile_patchability(model_or_arch) -> CompilePatchabilityReport:
    """Validate whether a family is patchable by current generic compile protocols."""

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    if not arch:
        return CompilePatchabilityReport(
            arch="unknown",
            support_state="unsupported",
            matched_bundles=(),
            suggested_patch_plan=(),
            discovered_targets=(),
            direct_protocols=(),
            inferred_protocols=(),
            satisfied_protocols=(),
            missing_protocols=(),
            blockers=("could not determine mlx_vlm architecture",),
        )

    report = get_compile_trait_report(model_or_arch if not isinstance(model_or_arch, str) else arch)
    qualification = get_compile_qualification(model_or_arch if not isinstance(model_or_arch, str) else arch)
    bundles = _matching_pattern_bundles(arch, report)
    targets = discover_compile_protocol_targets(model_or_arch if not isinstance(model_or_arch, str) else arch)
    direct_protocols = tuple(
        sorted(
            {
                target.protocol_name
                for target in targets
                if target.status == "protocol_matched"
            }
        )
    )
    planned_protocols = tuple(
        sorted(
            {
                protocol_name
                for bundle in bundles
                for protocol_name in bundle.protocol_names
            }
        )
    )
    inferred_protocols = tuple(
        protocol_name
        for protocol_name in planned_protocols
        if protocol_name not in direct_protocols
    )
    matched_protocols = tuple(
        sorted(set(direct_protocols) | set(inferred_protocols))
    )
    required_protocols = tuple(
        sorted(planned_protocols)
    )
    missing_protocols = tuple(
        protocol_name
        for protocol_name in required_protocols
        if protocol_name not in matched_protocols
    )
    blockers = []
    if qualification is not None and qualification.blocker_categories:
        blockers.append("static blockers: " + ", ".join(qualification.blocker_categories))
    if missing_protocols:
        blockers.append("missing protocols: " + ", ".join(missing_protocols))
    mismatches = [
        target
        for target in targets
        if target.status != "protocol_matched"
    ]
    if mismatches:
        blockers.append(
            "protocol mismatches: "
            + ", ".join(
                f"{target.protocol_name}@{target.class_name}.{target.method_name}"
                for target in mismatches[:8]
            )
        )

    return CompilePatchabilityReport(
        arch=arch,
        support_state=qualification.support_state if qualification is not None else "unsupported",
        matched_bundles=tuple(bundle.name for bundle in bundles),
        suggested_patch_plan=tuple(_resolve_compile_patch_plan(bundle) for bundle in bundles),
        discovered_targets=targets,
        direct_protocols=direct_protocols,
        inferred_protocols=inferred_protocols,
        satisfied_protocols=matched_protocols,
        missing_protocols=missing_protocols,
        blockers=tuple(blockers),
    )


def suggest_compile_patch_plan(model_or_arch) -> tuple[CompilePatchPlan, ...]:
    """Return the dry-run bundle plan that would be used for a family."""

    return validate_compile_patchability(model_or_arch).suggested_patch_plan


def find_similar_compile_families(
    model_or_arch,
    *,
    limit: int = 5,
    verified_only: bool = False,
) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    """Return nearby families based on shared traits, bundles, and protocols.

    This helper is aimed at maintainers adding support for a new family. It
    answers "what existing architecture looks most like this one?" so the next
    step is usually to compare patchability and trace output against a known
    neighbor rather than reading every patch body from scratch.
    """

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    if not arch:
        return ()

    report = validate_compile_patchability(model_or_arch if not isinstance(model_or_arch, str) else arch)
    traits = set(get_compile_trait_report(model_or_arch if not isinstance(model_or_arch, str) else arch).pattern_traits)
    bundles = set(report.matched_bundles)
    protocols = set(report.satisfied_protocols)
    query_features = traits | bundles | protocols
    if not query_features:
        return ()

    qualifications = build_compile_qualifications()
    candidates = []
    for other_arch, other_report in build_compile_trait_reports().items():
        if other_arch == arch:
            continue
        qualification = qualifications.get(other_arch)
        if verified_only and (qualification is None or qualification.support_state != "supported_verified"):
            continue
        other_patchability = validate_compile_patchability(other_arch)
        other_features = (
            set(other_report.pattern_traits)
            | set(other_patchability.matched_bundles)
            | set(other_patchability.satisfied_protocols)
        )
        if not other_features:
            continue
        overlap = query_features & other_features
        if not overlap:
            continue
        score = len(overlap) / len(query_features | other_features)
        candidates.append(
            (
                other_arch,
                score,
                tuple(sorted(overlap)),
            )
        )
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return tuple(candidates[: max(0, limit)])


def _build_compile_qualification(
    arch: str,
    trait_report: MLXVLMTraitReport | None = None,
) -> MLXVLMCompileQualification:
    """Build a user-facing compile qualification for one architecture."""

    report = trait_report or build_compile_trait_reports().get(arch)
    if report is None:
        return MLXVLMCompileQualification(
            arch=arch,
            training_compile=False,
            generation_compile=False,
            patched=False,
            blocker_categories=(),
            reason="architecture not discovered",
        )

    training_ok = arch in _VERIFIED_TRAINING_ARCHES
    generation_ok = arch in _VERIFIED_GENERATION_ARCHES
    patched = arch in _PATCHED_ARCHES
    matched_patterns = _matching_pattern_bundle_names(arch, report)
    support_state = _support_state(
        training_ok=training_ok,
        generation_ok=generation_ok,
        patched=patched,
        matched_patterns=matched_patterns,
    )
    installed_patterns = tuple(
        name for name in matched_patterns if name in _PATCHED_PATTERN_BUNDLES
    )

    if training_ok or generation_ok:
        ready = []
        if training_ok:
            ready.append("training")
        if generation_ok:
            ready.append("generation")
        reason = "explicitly qualified for " + " and ".join(ready)
    elif patched:
        reason = "patched but not yet verified"
    elif report.blocker_categories:
        reason = "compile blockers detected: " + ", ".join(report.blocker_categories)
    elif matched_patterns:
        reason = "pattern candidate pending verification: " + ", ".join(matched_patterns)
    else:
        reason = "architecture not yet qualified"

    extras = []
    if report.backend_arches:
        extras.append("backends=" + ",".join(report.backend_arches))
    if installed_patterns:
        extras.append("patterns=" + ",".join(installed_patterns))
    if report.verifier_hint:
        extras.append("verifier=" + report.verifier_hint)
    if report.blocker_categories and (training_ok or generation_ok):
        extras.append("residual blockers=" + ",".join(report.blocker_categories))
    if extras:
        reason += " (" + "; ".join(extras) + ")"

    return MLXVLMCompileQualification(
        arch=arch,
        training_compile=training_ok,
        generation_compile=generation_ok,
        patched=patched,
        blocker_categories=report.blocker_categories,
        reason=reason,
        verification_state=report.verification_state,
        support_state=support_state,
        backend_arches=report.backend_arches,
        installed_patterns=installed_patterns,
        trait_report=report,
    )


@lru_cache(maxsize=1)
def build_compile_qualifications() -> dict[str, MLXVLMCompileQualification]:
    """Build compile qualifications for every discovered architecture."""

    return {
        arch: _build_compile_qualification(arch, build_compile_trait_reports().get(arch))
        for arch in discover_architectures()
    }


def get_compile_qualification(model_or_arch) -> MLXVLMCompileQualification | None:
    """Return the compile qualification for an architecture or loaded model."""

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    if not arch:
        return None
    if isinstance(model_or_arch, str):
        return build_compile_qualifications().get(arch)
    report = get_compile_trait_report(model_or_arch)
    return _build_compile_qualification(arch, report)


def get_backend_compile_qualifications(model_or_arch) -> tuple[MLXVLMCompileQualification, ...]:
    """Return compile qualifications for any nested backend architectures."""

    qualifications = []
    for backend_arch in get_backend_architectures(model_or_arch):
        qual = get_compile_qualification(backend_arch)
        if qual is not None:
            qualifications.append(qual)
    return tuple(qualifications)


def _model_repo_training_compile_block_reason(model_or_arch) -> str | None:
    """Return a repo-specific compile block reason when policy should stay eager.

    This is intentionally separate from architecture qualification. Sometimes an
    architecture is broadly patchable, but a known repo family still fails real
    training due to an unresolved runtime issue.
    """

    if isinstance(model_or_arch, str):
        return None
    repo = str(getattr(model_or_arch, "_hf_repo", "") or "").lower()
    if not repo:
        return None
    for needle, reason in _MODEL_REPO_TRAINING_COMPILE_BLOCKLIST:
        if needle in repo:
            return reason
    return None


def _normalize_policy_mode(mode: str | None) -> str:
    """Normalize user-facing compile mode aliases to the internal enum."""

    aliases = {
        None: "best_effort",
        "default": "best_effort",
        "best_effort": "best_effort",
        "best-effort": "best_effort",
        "strict": "strict",
        "eager": "eager",
        "off": "eager",
        "disable": "eager",
        "disabled": "eager",
        "none": "eager",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in {"best_effort", "strict", "eager"}:
        raise ValueError(f"Unsupported MLX VLM compile mode: {mode!r}")
    return normalized


def _normalize_override_items(
    overrides: Mapping[str, str] | Iterable[tuple[str, str]] | str | None,
) -> tuple[tuple[str, str], ...]:
    """Normalize architecture/backend override mappings from CLI or config."""

    if overrides is None:
        return ()
    if isinstance(overrides, str):
        items = []
        for item in overrides.split(","):
            item = item.strip()
            if not item:
                continue
            key, _, value = item.partition("=")
            if not _:
                raise ValueError(
                    "Compile overrides must use 'name=mode' entries."
                )
            items.append((key.strip(), value.strip()))
    elif isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = list(overrides)

    normalized = []
    for name, mode in items:
        normalized.append((str(name), _normalize_policy_mode(str(mode))))
    return tuple(normalized)


def build_compile_policy(
    policy: MLXVLMCompilePolicy | None = None,
    args=None,
) -> MLXVLMCompilePolicy:
    """Build the normalized compile policy from config objects or CLI args."""

    if policy is not None:
        return MLXVLMCompilePolicy(
            mode=_normalize_policy_mode(policy.mode),
            arch_overrides=_normalize_override_items(policy.arch_overrides),
            backend_overrides=_normalize_override_items(policy.backend_overrides),
        )

    compile_enabled = True if args is None else bool(getattr(args, "compile", True))
    mode = "eager" if not compile_enabled else getattr(args, "compile_mode", "best_effort")
    return MLXVLMCompilePolicy(
        mode=_normalize_policy_mode(mode),
        arch_overrides=_normalize_override_items(
            None if args is None else getattr(args, "compile_arch_overrides", None)
        ),
        backend_overrides=_normalize_override_items(
            None if args is None else getattr(args, "compile_backend_overrides", None)
        ),
    )


def _resolve_effective_policy_mode(
    arch: str,
    backend_arches: tuple[str, ...],
    policy: MLXVLMCompilePolicy,
) -> str:
    """Resolve the final policy mode after arch/backend overrides are applied."""

    arch_overrides = dict(policy.arch_overrides)
    backend_overrides = dict(policy.backend_overrides)
    if arch in arch_overrides:
        return arch_overrides[arch]

    backend_modes = [backend_overrides[name] for name in backend_arches if name in backend_overrides]
    if "eager" in backend_modes:
        return "eager"
    if "strict" in backend_modes:
        return "strict"
    return policy.mode


def _arch_matches_prefixes(arch: str, prefixes: tuple[str, ...]) -> bool:
    return bool(prefixes) and arch.startswith(prefixes)


def _adapter_matches(
    adapter: CompilePatchAdapter,
    arch: str,
    report: MLXVLMTraitReport,
) -> bool:
    """Return whether a declarative adapter applies to one architecture.

    Adapters intentionally match on multiple signals:
    - exact architecture names for true exceptions
    - architecture prefixes for stable family naming
    - required traits for semantic reuse across future variants
    """

    if arch in adapter.arch_names:
        return True
    if _arch_matches_prefixes(arch, adapter.arch_prefixes):
        return True
    if not adapter.required_traits:
        return False
    return any(trait in report.pattern_traits for trait in adapter.required_traits)


def _adapter_names_for_arch(arch: str, report: MLXVLMTraitReport | None = None) -> tuple[str, ...]:
    """Return matching adapter names for one architecture."""

    report = report or build_compile_trait_reports().get(arch)
    if report is None:
        return ()

    names = []
    for adapter in list_compile_patch_adapters():
        if _adapter_matches(adapter, arch, report):
            names.append(adapter.name)
    return tuple(dict.fromkeys(names))


def _recommend_compile_settings(
    model_or_arch,
    *,
    qualification: MLXVLMCompileQualification | None = None,
    decision: ResolvedTrainingCompileDecision | None = None,
    args=None,
) -> tuple[CompileSettingRecommendation, ...]:
    """Return compile-aware training setting recommendations.

    Recommendations are intentionally conservative. They target settings that
    materially affect compiled training stability or memory behavior.
    """

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    report = get_compile_trait_report(model_or_arch if arch else "")
    qualification = qualification or get_compile_qualification(model_or_arch if arch else "")
    recommendations: list[CompileSettingRecommendation] = []

    def add(setting: str, value, reason: str) -> None:
        recommendations.append(
            CompileSettingRecommendation(
                setting=setting,
                recommended_value=value,
                reason=reason,
            )
        )

    compile_requested = bool(getattr(args, "compile", True)) if args is not None else True
    if compile_requested and report is not None and report.pattern_traits:
        if getattr(args, "gradient_checkpointing", True) is False:
            add(
                "gradient_checkpointing",
                True,
                "Compiled multimodal training is substantially more memory-stable with gradient checkpointing enabled.",
            )
        if getattr(args, "use_cce", True) is False and any(
            trait in report.pattern_traits
            for trait in (
                "qwen_like_image_merge",
                "qwen3_deepstack_multimodal",
                "masked_scatter_multimodal",
                "padded_image_filtering",
                "phi4_multimodal_spans",
            )
        ):
            add(
                "use_cce",
                True,
                "CCE is the preferred compiled training loss path for these multimodal families and generally gives safer memory behavior.",
            )

    if decision is not None and not decision.enabled:
        add(
            "compile",
            False,
            f"Compile should stay disabled for this run ({decision.reason}).",
        )
    elif qualification is not None and qualification.support_state == "supported_inferred":
        add(
            "compile_mode",
            "best_effort",
            "This family matches reusable compile patterns but is not explicitly verified yet, so best-effort mode is safer than strict mode.",
        )

    return tuple(recommendations)


def resolve_training_compile(
    model_or_arch,
    policy: MLXVLMCompilePolicy | None = None,
    args=None,
) -> ResolvedTrainingCompileDecision:
    """Resolve whether a training run should use `mx.compile`.

    Resolution order:
    1. Normalize the global policy and any per-arch/backend overrides.
    2. Check for repo-specific blocklist reasons.
    3. Require the main architecture and every backend architecture to be
       training-qualified.
    4. Return either an enabled compile decision or an eager fallback / raise
       decision, depending on policy strictness.
    """

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    compiled_policy = build_compile_policy(policy=policy, args=args)
    patch_mode = normalize_mlx_patch_mode(
        getattr(
            args,
            "patch_mode",
            getattr(model_or_arch, "_unsloth_patch_mode", "patched"),
        )
    )

    def finalize(
        *,
        arch_name: str,
        enabled: bool,
        policy_mode: str,
        fallback_allowed: bool,
        strict_requested: bool,
        should_raise: bool,
        reason: str,
        qualification: MLXVLMCompileQualification | None,
        backend_qualifications: tuple[MLXVLMCompileQualification, ...] = (),
    ) -> ResolvedTrainingCompileDecision:
        support_state = qualification.support_state if qualification is not None else "unsupported"
        recommendations = _recommend_compile_settings(
            model_or_arch,
            qualification=qualification,
            decision=None,
            args=args,
        )
        decision = ResolvedTrainingCompileDecision(
            arch=arch_name,
            enabled=enabled,
            policy_mode=policy_mode,
            fallback_allowed=fallback_allowed,
            strict_requested=strict_requested,
            should_raise=should_raise,
            reason=reason,
            qualification=qualification,
            backend_qualifications=backend_qualifications,
            support_state=support_state,
            setting_recommendations=recommendations,
            patch_mode=patch_mode,
        )
        recommendations = _recommend_compile_settings(
            model_or_arch,
            qualification=qualification,
            decision=decision,
            args=args,
        )
        return ResolvedTrainingCompileDecision(
            arch=decision.arch,
            enabled=decision.enabled,
            policy_mode=decision.policy_mode,
            fallback_allowed=decision.fallback_allowed,
            strict_requested=decision.strict_requested,
            should_raise=decision.should_raise,
            reason=decision.reason,
            qualification=decision.qualification,
            backend_qualifications=decision.backend_qualifications,
            support_state=decision.support_state,
            setting_recommendations=recommendations,
            patch_mode=decision.patch_mode,
        )

    if not arch:
        return finalize(
            arch_name="unknown",
            enabled=False,
            policy_mode=compiled_policy.mode,
            fallback_allowed=compiled_policy.mode == "best_effort",
            strict_requested=compiled_policy.mode == "strict",
            should_raise=compiled_policy.mode == "strict",
            reason="could not determine mlx_vlm architecture",
            qualification=None,
        )

    if patch_mode == "unpatched":
        return finalize(
            arch_name=arch,
            enabled=False,
            policy_mode="eager",
            fallback_allowed=False,
            strict_requested=False,
            should_raise=False,
            reason="compile patches disabled by patch_mode=unpatched",
            qualification=get_compile_qualification(model_or_arch),
            backend_qualifications=get_backend_compile_qualifications(model_or_arch),
        )

    qualification = get_compile_qualification(model_or_arch)
    backend_qualifications = get_backend_compile_qualifications(model_or_arch)
    backend_arches = tuple(qual.arch for qual in backend_qualifications)
    policy_mode = _resolve_effective_policy_mode(arch, backend_arches, compiled_policy)
    strict_requested = policy_mode == "strict"
    fallback_allowed = policy_mode == "best_effort"

    if policy_mode == "eager":
        return finalize(
            arch_name=arch,
            enabled=False,
            policy_mode=policy_mode,
            fallback_allowed=False,
            strict_requested=False,
            should_raise=False,
            reason="compile disabled by policy",
            qualification=qualification,
            backend_qualifications=backend_qualifications,
        )

    model_block_reason = _model_repo_training_compile_block_reason(model_or_arch)
    if model_block_reason is not None:
        return finalize(
            arch_name=arch,
            enabled=False,
            policy_mode=policy_mode,
            fallback_allowed=fallback_allowed,
            strict_requested=strict_requested,
            should_raise=strict_requested,
            reason=model_block_reason,
            qualification=qualification,
            backend_qualifications=backend_qualifications,
        )

    backend_blockers = [
        qual.arch for qual in backend_qualifications if not qual.training_compile
    ]
    if qualification is None:
        reason = "architecture not discovered"
        return finalize(
            arch_name=arch,
            enabled=False,
            policy_mode=policy_mode,
            fallback_allowed=fallback_allowed,
            strict_requested=strict_requested,
            should_raise=strict_requested,
            reason=reason,
            qualification=qualification,
            backend_qualifications=backend_qualifications,
        )
    if not qualification.training_compile:
        reason = qualification.reason
        return finalize(
            arch_name=arch,
            enabled=False,
            policy_mode=policy_mode,
            fallback_allowed=fallback_allowed,
            strict_requested=strict_requested,
            should_raise=strict_requested,
            reason=reason,
            qualification=qualification,
            backend_qualifications=backend_qualifications,
        )
    if backend_blockers:
        reason = "backend compile not qualified: " + ", ".join(backend_blockers)
        return finalize(
            arch_name=arch,
            enabled=False,
            policy_mode=policy_mode,
            fallback_allowed=fallback_allowed,
            strict_requested=strict_requested,
            should_raise=strict_requested,
            reason=reason,
            qualification=qualification,
            backend_qualifications=backend_qualifications,
        )
    return finalize(
        arch_name=arch,
        enabled=True,
        policy_mode=policy_mode,
        fallback_allowed=fallback_allowed,
        strict_requested=strict_requested,
        should_raise=False,
        reason="compile enabled",
        qualification=qualification,
        backend_qualifications=backend_qualifications,
    )


def _invalidate_qualification_cache():
    """Clear cached trait/qualification reports after patch installation."""

    build_compile_trait_reports.cache_clear()
    build_compile_qualifications.cache_clear()


def _record_patch_binding(kind: str, cls, name: str) -> None:
    """Record one concrete runtime patch binding for maintainer tracing."""

    _PATCH_BINDINGS.add(
        (
            kind,
            getattr(cls, "__module__", "<unknown>"),
            getattr(cls, "__name__", repr(cls)),
            name,
        )
    )


def _patch_staticmethod(cls, name, fn):
    """Replace a class staticmethod only when it is not already patched."""

    current = getattr(cls, name, None)
    if current is fn:
        return
    setattr(cls, name, staticmethod(fn))
    _record_patch_binding("staticmethod", cls, name)


def _patch_method(cls, name, fn):
    """Replace a class method only when it is not already patched."""

    current = getattr(cls, name, None)
    if current is fn:
        return
    setattr(cls, name, fn)
    _record_patch_binding("method", cls, name)


def _try_import_module(module_name: str):
    """Best-effort import for optional `mlx_vlm` modules.

    Patch installers are intentionally permissive because the installed MLX
    stack may not include every architecture. Silent `None` keeps the patch
    registry idempotent across different environments.
    """

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _iter_trait_model_modules(
    trait: str,
    *,
    include_arches: Iterable[str] = (),
):
    """Yield model modules for architectures that share a compile trait.

    Most `mlx_vlm` architectures follow `mlx_vlm.models.<arch>.<arch>`. Using
    trait discovery here lets a future architecture inherit a shared patch
    automatically when it exposes the same source-level pattern and follows the
    standard module layout.
    """

    candidate_arches = set(include_arches)
    candidate_arches.update(
        arch
        for arch, report in build_compile_trait_reports().items()
        if trait in report.pattern_traits
    )
    for arch in sorted(candidate_arches):
        module = _try_import_module(f"mlx_vlm.models.{arch}.{arch}")
        if module is None:
            continue
        yield arch, module


def _install_safe_fused_sdpa_mask_patches():
    """Work around MLX SDPA NaN gradients on fully masked query rows.

    Patch `mx.fast.scaled_dot_product_attention` itself so every direct caller
    in mlx-vlm inherits the fix, including `ensure_fused_sdpa` and models that
    invoke the fast kernel directly.

    The upstream failure mode is:
    - additive float mask
    - one or more query rows entirely masked (all `-inf`)
    - finite forward output, non-finite backward gradients

    Preserve the intended semantics by:
    1. clearing fully masked rows in the additive mask before SDPA
    2. zeroing the corresponding output rows after SDPA
    """
    fast_ns = getattr(mx, "fast", None)
    current = getattr(fast_ns, "scaled_dot_product_attention", None) if fast_ns is not None else None
    if current is None or getattr(fast_ns, "_unsloth_safe_sdpa_mask_patch", False):
        return

    original_fast_sdpa = current

    @mx.compile
    def safe_scaled_dot_product_attention(q, k, v, scale=1.0, mask=None, **kwargs):
        row_all_masked = None
        if mask is not None and hasattr(mask, "dtype") and mx.issubdtype(mask.dtype, mx.floating):
            # MLX's fused SDPA backward is unstable when an additive float mask
            # contains one or more query rows that are entirely masked out with
            # negative infinity (every key position is -inf for that query
            # row). Forward remains
            # finite, but gradients can become NaN.
            #
            # Semantically those rows should contribute zero output because the
            # query is padding / otherwise invalid. Preserve that meaning by:
            # 1. clearing the all-masked rows before calling SDPA, avoiding the
            #    upstream backward bug
            # 2. explicitly zeroing those rows in the output afterward
            #
            # This only applies to additive float masks with true -inf rows.
            # Large finite negative biases (for example -1e9 or -1e30) are
            # valid finite masks with non-zero semantics and must pass through
            # unchanged.
            row_all_masked = mx.all(
                mx.logical_and(mx.logical_not(mx.isfinite(mask)), mask < 0),
                axis=-1,
            )
            mask = mx.where(
                mx.expand_dims(row_all_masked, axis=-1),
                mx.zeros_like(mask),
                mask,
            )

        out = original_fast_sdpa(q, k, v, scale=scale, mask=mask, **kwargs)
        if row_all_masked is not None:
            # Restore the intended zero-output semantics for fully masked query
            # rows after the SDPA call.
            out = mx.where(
                mx.expand_dims(row_all_masked, axis=-1),
                mx.zeros_like(out),
                out,
            )
        return out

    fast_ns.scaled_dot_product_attention = safe_scaled_dot_product_attention
    fast_ns._unsloth_safe_sdpa_mask_patch = True


def _merge_special_token_features(
    image_token_id,
    video_token_id,
    image_features,
    inputs_embeds,
    input_ids,
):
    import mlx.core as mx

    if image_features.ndim == 3 and image_features.shape[0] == 1:
        image_features = image_features[0]
    elif image_features.ndim > 2:
        image_features = image_features.reshape((-1, image_features.shape[-1]))

    special_mask = input_ids == image_token_id
    if video_token_id is not None:
        special_mask = mx.logical_or(special_mask, input_ids == video_token_id)

    return _merge_sequence_mask_features(special_mask, image_features, inputs_embeds)


def _merge_sequence_mask_features(
    sequence_mask,
    feature_rows,
    inputs_embeds,
):
    import mlx.core as mx

    if sequence_mask.dtype != mx.bool_:
        sequence_mask = sequence_mask.astype(mx.bool_)
    if feature_rows.ndim == 3 and feature_rows.shape[0] == 1:
        feature_rows = feature_rows[0]
    elif feature_rows.ndim > 2:
        feature_rows = feature_rows.reshape((-1, feature_rows.shape[-1]))

    hidden_dim = inputs_embeds.shape[-1]
    flat_mask = sequence_mask.reshape((-1,))
    flat_inputs = inputs_embeds.reshape((-1, hidden_dim))

    # Map each masked sequence position to the next feature row.
    feature_indices = mx.cumsum(flat_mask.astype(mx.int32)) - 1
    if feature_rows.shape[0] > 0:
        feature_indices = mx.minimum(mx.maximum(feature_indices, 0), feature_rows.shape[0] - 1)
        gathered = feature_rows[feature_indices]
    else:
        gathered = mx.zeros_like(flat_inputs)

    flat_out = mx.where(mx.expand_dims(flat_mask, axis=-1), gathered, flat_inputs)
    return flat_out.reshape(inputs_embeds.shape), mx.expand_dims(sequence_mask, axis=-1)


def _merge_special_token_features_only(
    image_token_id,
    video_token_id,
    image_features,
    inputs_embeds,
    input_ids,
):
    outputs, _ = _merge_special_token_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    )
    return outputs


def _merge_replacement_segments(inputs_embeds, replacements_by_batch):
    batch_size = inputs_embeds.shape[0]
    merged_batches = []
    max_len = 0

    for batch_idx in range(batch_size):
        cur_embeds = inputs_embeds[batch_idx]
        replacements = sorted(replacements_by_batch[batch_idx], key=lambda item: item[0])
        parts = []
        prev = 0
        seq_len = int(cur_embeds.shape[0])
        for start, end, features in replacements:
            start = int(start)
            end = int(end)
            if start > prev:
                parts.append(cur_embeds[prev:start])
            parts.append(features.astype(cur_embeds.dtype))
            prev = end
        if prev < seq_len:
            parts.append(cur_embeds[prev:])
        merged = mx.concatenate(parts, axis=0) if parts else cur_embeds
        merged_batches.append(merged)
        max_len = max(max_len, int(merged.shape[0]))

    if batch_size == 1:
        return merged_batches[0][None, :]

    hidden_dim = int(inputs_embeds.shape[-1])
    padded = mx.zeros((batch_size, max_len, hidden_dim), dtype=inputs_embeds.dtype)
    for idx, embeds in enumerate(merged_batches):
        padded[idx, : embeds.shape[0]] = embeds
    return padded


def _masked_scatter_no_numpy(final_embedding, image_mask_expanded, scaled_image_features):
    """Compile-safe replacement for `masked_scatter`-style multimodal merges.

    Upstream `mlx_vlm` variants often flatten to NumPy or use mutation-heavy
    scatter helpers. This version stays entirely in MLX array ops so it can run
    under `mx.compile` and backward.
    """

    import mlx.core as mx

    final_shape = final_embedding.shape
    hidden_dim = final_shape[-1]
    flat_mask = image_mask_expanded.reshape((-1,))
    flat_output = final_embedding.reshape((-1,))
    flat_features = scaled_image_features.reshape((-1,))

    feature_indices = mx.cumsum(flat_mask.astype(mx.int32)) - 1
    if flat_features.shape[0] > 0:
        feature_indices = mx.minimum(mx.maximum(feature_indices, 0), flat_features.shape[0] - 1)
        gathered = flat_features[feature_indices]
    else:
        gathered = mx.zeros_like(flat_output)

    flat_output = mx.where(flat_mask, gathered, flat_output)
    return flat_output.reshape(final_shape)


def _tolist_if_needed(values):
    """Convert MLX/NumPy metadata arrays to Python lists when available."""

    if hasattr(values, "tolist"):
        return values.tolist()
    return values


def _grid_to_tuple(grid_thw):
    """Normalize `(t, h, w)` vision grid metadata to immutable Python tuples."""

    if grid_thw is None:
        return ()
    grid_thw = _tolist_if_needed(grid_thw)
    return tuple(tuple(int(x) for x in item) for item in grid_thw)


def _size_pairs_to_tuple(values):
    """Normalize `(h, w)` metadata pairs to immutable Python tuples."""

    if values is None:
        return ()
    values = _tolist_if_needed(values)
    return tuple(tuple(int(x) for x in item) for item in values)


def _positions_to_tuple(positions):
    """Normalize 2-column position metadata to immutable Python tuples."""

    if positions is None:
        return ()
    positions = _tolist_if_needed(positions)
    return tuple((int(item[0]), int(item[1])) for item in positions)


def _positions_per_batch_to_tuple(positions):
    """Normalize batched 1D position metadata to immutable Python tuples."""

    if positions is None:
        return ()
    positions = _tolist_if_needed(positions)
    return tuple(
        tuple(int(pos) for pos in batch_positions)
        for batch_positions in positions
    )


def _spans_per_batch_to_tuple(spans):
    """Normalize batched `(start, end)` span metadata to Python tuples."""

    if spans is None:
        return ()
    spans = _tolist_if_needed(spans)
    return tuple(
        tuple((int(start), int(end)) for start, end in batch_spans)
        for batch_spans in spans
    )


def _lengths_to_tuple(lengths):
    """Normalize 1D length metadata to immutable Python tuples."""

    if lengths is None:
        return None
    lengths = _tolist_if_needed(lengths)
    return tuple(int(x) for x in lengths)


def _split_points(cu_seqlens):
    """Return Python split indices from cumulative sequence lengths."""

    if isinstance(cu_seqlens, (tuple, list)):
        return list(cu_seqlens[1:-1])
    return cu_seqlens[1:-1].tolist()


def _build_cu_seqlens(grid_thw):
    """Build cumulative sequence lengths from normalized `(t, h, w)` grids."""

    cu_seqlens = [0]
    total = 0
    for num_frames, height, width in grid_thw:
        frame_tokens = int(height) * int(width)
        for _ in range(int(num_frames)):
            total += frame_tokens
            cu_seqlens.append(total)
    return tuple(cu_seqlens)


def _attach_position_ids(features, position_ids):
    """Attach explicit position ids to an embeddings feature container."""

    if position_ids is not None:
        features.position_ids = position_ids
    return features


def _add_visual_embeds(hidden_states, visual_pos_masks, visual_embeds):
    """Compile-safe additive merge for sparse visual embeddings."""

    import mlx.core as mx

    flat_mask = visual_pos_masks.reshape((-1,))
    hidden_shape = hidden_states.shape
    flat_visual = visual_embeds.reshape((-1, hidden_shape[-1]))
    feature_indices = mx.cumsum(flat_mask.astype(mx.int32)) - 1
    if flat_visual.shape[0] > 0:
        feature_indices = mx.minimum(
            mx.maximum(feature_indices, 0), flat_visual.shape[0] - 1
        )
        gathered = flat_visual[feature_indices]
    else:
        gathered = mx.zeros((flat_mask.shape[0], hidden_shape[-1]), dtype=hidden_states.dtype)

    gathered = mx.where(mx.expand_dims(flat_mask, axis=-1), gathered, 0)
    return hidden_states + gathered.reshape(hidden_shape)


def _downsample_square_tokens(x, downsample_ratio):
    """Downsample square token grids without leaving MLX array ops."""

    bs, hw, input_dim = x.shape
    h = w = int(hw**0.5)
    pad = (downsample_ratio - h % downsample_ratio) % downsample_ratio

    x = x.reshape(bs, h, w, input_dim)
    if pad > 0:
        x = mx.pad(x, [(0, 0), (0, pad), (0, pad), (0, 0)], constant_values=0)

    hp, wp = x.shape[1], x.shape[2]
    x = x.reshape(
        bs,
        hp // downsample_ratio,
        downsample_ratio,
        wp // downsample_ratio,
        downsample_ratio,
        input_dim,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(bs, -1, input_dim * downsample_ratio * downsample_ratio)


def _install_qwen2_5_compile_patches():
    """Install compile-safe vision and embedding patches for Qwen2.5-VL."""

    try:
        module = importlib.import_module("mlx_vlm.models.qwen2_5_vl.qwen2_5_vl")
        vision_module = importlib.import_module("mlx_vlm.models.qwen2_5_vl.vision")
    except Exception:
        return

    InputEmbeddingsFeatures = module.InputEmbeddingsFeatures

    def patched_qwen2_rotary(self, seqlen):
        import mlx.core as mx

        seqlen = int(seqlen) if not hasattr(seqlen, "item") else int(seqlen.item())
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)

    def patched_qwen2_attention(self, x, cu_seqlens, rotary_pos_emb=None):
        import mlx.core as mx

        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        q = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        split_indices = _split_points(cu_seqlens)
        splits = [mx.split(tensor, split_indices, axis=2) for tensor in (q, k, v)]

        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(*splits):
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk, scale=self.scale, mask=None
                )
            )

        output = mx.concatenate(outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)

    def patched_qwen2_rot_pos_emb(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        pos_ids = []
        for t, h, w in grid_spec:
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3)).flatten()

            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3)).flatten()
            stacked = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked, (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_spec)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids]
        return rotary_pos_emb.reshape(pos_ids.shape[0], -1)

    def patched_qwen2_get_window_index(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_spec:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = np.arange(grid_t * llm_grid_h * llm_grid_w, dtype=np.int32).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = (window_size - llm_grid_h % window_size) % window_size
            pad_w = (window_size - llm_grid_w % window_size) % window_size
            num_windows_h = (llm_grid_h + pad_h) // window_size
            num_windows_w = (llm_grid_w + pad_w) // window_size

            index_padded = np.pad(
                index,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=-100,
            )
            index_padded = index_padded.reshape(
                grid_t, num_windows_h, window_size, num_windows_w, window_size
            ).transpose(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, window_size, window_size
            )

            seqlens = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)
            flat_index = index_padded.reshape(-1)
            index_new = flat_index[flat_index != -100]
            window_index.append(index_new + window_index_id)
            cu_window_seqlens.extend(
                (np.cumsum(seqlens) * self.spatial_merge_unit + cu_window_seqlens[-1]).tolist()
            )
            window_index_id += int(grid_t * llm_grid_h * llm_grid_w)

        deduped = []
        seen = set()
        for value in cu_window_seqlens:
            value = int(value)
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return mx.array(np.concatenate(window_index), dtype=mx.int32), tuple(deduped)

    def patched_qwen2_vision_call(self, hidden_states, grid_thw, output_hidden_states=None):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_spec)
        window_index, cu_window_seqlens = self.get_window_index(grid_spec)

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)

        cu_seqlens = _build_cu_seqlens(grid_spec)
        encoder_states = (hidden_states,) if output_hidden_states else None

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = (
                cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            )
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.merger(hidden_states)
        reverse_indices = mx.argsort(window_index, axis=0)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def patched_qwen2_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        explicit_position_ids = kwargs.get("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)
        hidden_states = (
            cached
            if cached is not None
            else self.vision_tower(pixel_values, grid_thw, output_hidden_states=False)
        )
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        position_ids = explicit_position_ids
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = None

        features = InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)
        return _attach_position_ids(features, position_ids)

    _patch_method(vision_module.VisionRotaryEmbedding, "__call__", patched_qwen2_rotary)
    _patch_method(vision_module.Attention, "__call__", patched_qwen2_attention)
    _patch_method(vision_module.VisionModel, "rot_pos_emb", patched_qwen2_rot_pos_emb)
    _patch_method(vision_module.VisionModel, "get_window_index", patched_qwen2_get_window_index)
    _patch_method(vision_module.VisionModel, "__call__", patched_qwen2_vision_call)
    _patch_method(module.Model, "get_input_embeddings", patched_qwen2_get_input_embeddings)
    _PATCHED_ARCHES.add("qwen2_5_vl")


def _install_qwen2_compile_patches():
    """Install compile-safe vision and embedding patches for Qwen2-VL."""

    try:
        module = importlib.import_module("mlx_vlm.models.qwen2_vl.qwen2_vl")
        vision_module = importlib.import_module("mlx_vlm.models.qwen2_vl.vision")
    except Exception:
        return

    InputEmbeddingsFeatures = module.InputEmbeddingsFeatures

    def patched_qwen2_rotary(self, seqlen):
        import mlx.core as mx

        seqlen = int(seqlen) if not hasattr(seqlen, "item") else int(seqlen.item())
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)

    def patched_qwen2_attention(self, x, cu_seqlens, rotary_pos_emb=None):
        import mlx.core as mx

        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        q = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        split_indices = _split_points(cu_seqlens)
        splits = [mx.split(tensor, split_indices, axis=2) for tensor in (q, k, v)]

        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(*splits):
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk, scale=self.scale, mask=None
                )
            )

        output = mx.concatenate(outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)

    def patched_qwen2_rot_pos_emb(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        pos_ids = []

        for t, h, w in grid_spec:
            h, w = int(h), int(w)
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3)).flatten()

            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3)).flatten()

            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked_pos_ids, (int(t), 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_spec)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb_full = rotary_pos_emb_full[pos_ids]
        return rotary_pos_emb_full.reshape(pos_ids.shape[0], -1)

    def patched_qwen2_vision_call(self, hidden_states, grid_thw, output_hidden_states=None):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_spec)
        cu_seqlens = _build_cu_seqlens(grid_spec)

        encoder_states = (hidden_states,) if output_hidden_states else None
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        return self.merger(hidden_states)

    def patched_qwen2_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        explicit_position_ids = kwargs.get("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)
        hidden_states = (
            cached
            if cached is not None
            else self.vision_tower(pixel_values, grid_thw, output_hidden_states=False)
        )
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        position_ids = explicit_position_ids
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = None

        features = InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)
        return _attach_position_ids(features, position_ids)

    _patch_method(vision_module.VisionRotaryEmbedding, "__call__", patched_qwen2_rotary)
    _patch_method(vision_module.Attention, "__call__", patched_qwen2_attention)
    _patch_method(vision_module.VisionModel, "rot_pos_emb", patched_qwen2_rot_pos_emb)
    _patch_method(vision_module.VisionModel, "__call__", patched_qwen2_vision_call)
    _patch_method(module.Model, "get_input_embeddings", patched_qwen2_get_input_embeddings)
    _PATCHED_ARCHES.add("qwen2_vl")


def _install_qwen3_family_compile_patches():
    """Install shared compile patches for the Qwen3 VL family."""

    try:
        module = importlib.import_module("mlx_vlm.models.qwen3_vl.qwen3_vl")
        vision_module = importlib.import_module("mlx_vlm.models.qwen3_vl.vision")
        qwen35_module = importlib.import_module("mlx_vlm.models.qwen3_5.qwen3_5")
    except Exception:
        return
    try:
        qwen3moe_module = importlib.import_module("mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe")
        qwen3moe_vision_module = importlib.import_module("mlx_vlm.models.qwen3_vl_moe.vision")
        qwen3moe_language_module = importlib.import_module("mlx_vlm.models.qwen3_vl_moe.language")
    except Exception:
        qwen3moe_module = None
        qwen3moe_vision_module = None
        qwen3moe_language_module = None

    def merge_qwen3(image_features, inputs_embeds, input_ids, image_token_index, video_token_index):
        import mlx.core as mx

        outputs, mask = _merge_special_token_features(
            image_token_index,
            video_token_index,
            image_features,
            inputs_embeds,
            input_ids,
        )
        mask = mask.astype(mx.bool_)
        mask = mx.broadcast_to(mask, inputs_embeds.shape)
        return outputs, mask

    def patched_qwen3_attention(self, x, cu_seqlens, rotary_pos_emb=None):
        import mlx.core as mx

        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        q = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        split_indices = _split_points(cu_seqlens)
        splits = [mx.split(tensor, split_indices, axis=2) for tensor in (q, k, v)]

        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(*splits):
            attn_outputs.append(
                vision_module.ensure_fused_sdpa(q_chunk, k_chunk, v_chunk, self.scale)
            )

        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)

    def patched_qwen3_rot_pos_emb(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        merge_size = self.spatial_merge_size
        max_hw = max(max(int(h), int(w)) for _, h, w in grid_spec)
        freq_table = self.rotary_pos_emb(max_hw)
        pos_ids = []

        for num_frames, height, width in grid_spec:
            merged_h = int(height) // merge_size
            merged_w = int(width) // merge_size
            block_rows = mx.arange(merged_h)
            block_cols = mx.arange(merged_w)
            intra_row = mx.arange(merge_size)
            intra_col = mx.arange(merge_size)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = mx.broadcast_to(
                row_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)
            col_idx = mx.broadcast_to(
                col_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)
            coords = mx.stack([row_idx, col_idx], axis=-1)
            if int(num_frames) > 1:
                coords = mx.tile(coords, (int(num_frames), 1))
            pos_ids.append(coords)

        pos_ids = mx.concatenate(pos_ids, axis=0)
        h_embeddings = freq_table[pos_ids[:, 0]]
        w_embeddings = freq_table[pos_ids[:, 1]]
        return mx.concatenate([h_embeddings, w_embeddings], axis=-1)

    def patched_qwen3_fast_pos_embed_interpolate(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_spec:
            h_idxs = mx.linspace(0, self.num_grid_per_side - 1, int(h))
            w_idxs = mx.linspace(0, self.num_grid_per_side - 1, int(w))

            h_floor = h_idxs.astype(mx.int32)
            w_floor = w_idxs.astype(mx.int32)
            h_ceil = mx.minimum(h_floor + 1, self.num_grid_per_side - 1)
            w_ceil = mx.minimum(w_floor + 1, self.num_grid_per_side - 1)

            dh = h_idxs - h_floor.astype(mx.float32)
            dw = w_idxs - w_floor.astype(mx.float32)
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[:, None] + w_floor[None, :]).flatten(),
                (base_h[:, None] + w_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = mx.array(idx_list, dtype=mx.int32)
        weight_tensor = mx.array(weight_list, dtype=self.pos_embed.weight.dtype)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        split_sizes = [int(h * w) for _, h, w in grid_spec]
        if len(split_sizes) > 1:
            split_indices = list(accumulate(split_sizes[:-1]))
            patch_pos_embeds_split = mx.split(patch_pos_embeds, split_indices, axis=0)
        else:
            patch_pos_embeds_split = [patch_pos_embeds]

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds_split, grid_spec):
            feature_dim = pos_embed.shape[-1]
            pos_embed = mx.tile(pos_embed, (int(t), 1))
            pos_embed = pos_embed.reshape(int(t), int(h), int(w), feature_dim)
            pos_embed = (
                pos_embed.reshape(
                    int(t),
                    int(h) // merge_size,
                    merge_size,
                    int(w) // merge_size,
                    merge_size,
                    feature_dim,
                )
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(-1, feature_dim)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return mx.concatenate(patch_pos_embeds_permute)

    def patched_qwen3_vision_call(self, hidden_states, grid_thw, **kwargs):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_spec)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_spec)

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        cu_seqlens = _build_cu_seqlens(grid_spec)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def patched_qwen3_deepstack(self, hidden_states, visual_pos_masks, visual_embeds):
        return _add_visual_embeds(hidden_states, visual_pos_masks, visual_embeds)

    def patched_qwen3_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        explicit_position_ids = kwargs.get("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return module.InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
            deepstack_image_embeds = None
        else:
            hidden_states, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw)

        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        position_ids = explicit_position_ids
        if position_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        features = module.InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=deepstack_image_embeds,
        )
        return _attach_position_ids(features, position_ids)

    def patched_qwen35_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        explicit_position_ids = kwargs.get("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return qwen35_module.InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        hidden_states = cached if cached is not None else self.vision_tower(pixel_values, grid_thw)[0]
        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        position_ids = explicit_position_ids
        if position_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        features = qwen35_module.InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)
        return _attach_position_ids(features, position_ids)

    module.masked_scatter = _masked_scatter_no_numpy
    _patch_staticmethod(module.Model, "merge_input_ids_with_image_features", merge_qwen3)
    _patch_staticmethod(qwen35_module.Model, "merge_input_ids_with_image_features", merge_qwen3)
    _patch_method(vision_module.Attention, "__call__", patched_qwen3_attention)
    _patch_method(vision_module.VisionModel, "rot_pos_emb", patched_qwen3_rot_pos_emb)
    _patch_method(vision_module.VisionModel, "fast_pos_embed_interpolate", patched_qwen3_fast_pos_embed_interpolate)
    _patch_method(vision_module.VisionModel, "__call__", patched_qwen3_vision_call)
    _patch_method(importlib.import_module("mlx_vlm.models.qwen3_vl.language").Qwen3VLModel, "_deepstack_process", patched_qwen3_deepstack)
    _patch_method(module.Model, "get_input_embeddings", patched_qwen3_get_input_embeddings)
    _patch_method(qwen35_module.Model, "get_input_embeddings", patched_qwen35_get_input_embeddings)
    if qwen3moe_module is not None:
        qwen3moe_module.masked_scatter = _masked_scatter_no_numpy
        _patch_staticmethod(qwen3moe_module.Model, "merge_input_ids_with_image_features", merge_qwen3)
        _patch_method(qwen3moe_vision_module.Attention, "__call__", patched_qwen3_attention)
        _patch_method(qwen3moe_vision_module.VisionModel, "rot_pos_emb", patched_qwen3_rot_pos_emb)
        _patch_method(qwen3moe_vision_module.VisionModel, "fast_pos_embed_interpolate", patched_qwen3_fast_pos_embed_interpolate)
        _patch_method(qwen3moe_vision_module.VisionModel, "__call__", patched_qwen3_vision_call)
        _patch_method(qwen3moe_language_module.Qwen3VLMoEModel, "_deepstack_process", patched_qwen3_deepstack)
        _patch_method(qwen3moe_module.Model, "get_input_embeddings", patched_qwen3_get_input_embeddings)
        _PATCHED_ARCHES.add("qwen3_vl_moe")
    _PATCHED_ARCHES.update({"qwen3_vl", "qwen3_5", "qwen3_5_moe"})


def _install_glm_ocr_compile_patches():
    """Install compile-safe GLM OCR vision and embedding patches."""

    try:
        module = importlib.import_module("mlx_vlm.models.glm_ocr.glm_ocr")
        vision_module = importlib.import_module("mlx_vlm.models.glm_ocr.vision")
    except Exception:
        return

    InputEmbeddingsFeatures = module.InputEmbeddingsFeatures

    def patched_glm_attention(self, hidden_states, cu_seqlens, position_embeddings):
        import mlx.core as mx

        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3, axis=0)
        q = self.q_norm(q.squeeze(0))
        k = self.k_norm(k.squeeze(0))
        v = v.squeeze(0)

        cos, sin = position_embeddings
        q, k = vision_module.apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(1, 0, 2)[None, ...]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        split_indices = _split_points(cu_seqlens)
        q_splits = mx.split(q, split_indices, axis=2)
        k_splits = mx.split(k, split_indices, axis=2)
        v_splits = mx.split(v, split_indices, axis=2)

        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q_splits, k_splits, v_splits):
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk, scale=self.scale, mask=None
                )
            )

        attn_output = mx.concatenate(outputs, axis=2)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(attn_output)

    def patched_glm_rot_pos_emb(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        pos_ids = []
        for t, h, w in grid_spec:
            hpos_ids = mx.expand_dims(mx.arange(int(h)), 1)
            hpos_ids = mx.repeat(hpos_ids, int(w), axis=1)
            hpos_ids = hpos_ids.reshape(
                int(h) // self.spatial_merge_size,
                self.spatial_merge_size,
                int(w) // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3)).flatten()

            wpos_ids = mx.expand_dims(mx.arange(int(w)), 0)
            wpos_ids = mx.repeat(wpos_ids, int(h), axis=0)
            wpos_ids = wpos_ids.reshape(
                int(h) // self.spatial_merge_size,
                self.spatial_merge_size,
                int(w) // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3)).flatten()

            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked_pos_ids, (int(t), 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_spec)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        emb = mx.concatenate((rotary_pos_emb, rotary_pos_emb), axis=-1)
        return (mx.cos(emb), mx.sin(emb)), pos_ids

    def patched_glm_vision_call(self, hidden_states, grid_thw, output_hidden_states=None):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings, _ = self.rot_pos_emb(grid_spec)
        cu_seqlens = _build_cu_seqlens(grid_spec)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.reshape(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = self.downsample(hidden_states).reshape(
            -1, self.config.out_hidden_size
        )
        return self.merger(hidden_states)

    def patched_glm_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        explicit_position_ids = kwargs.get("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)
        hidden_states = (
            cached
            if cached is not None
            else self.vision_tower(pixel_values, grid_thw, output_hidden_states=False)
        )

        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        position_ids = explicit_position_ids
        if position_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        features = InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)
        return _attach_position_ids(features, position_ids)

    _patch_method(vision_module.GlmOcrVisionAttention, "__call__", patched_glm_attention)
    _patch_method(vision_module.VisionModel, "rot_pos_emb", patched_glm_rot_pos_emb)
    _patch_method(vision_module.VisionModel, "__call__", patched_glm_vision_call)
    _patch_method(module.Model, "get_input_embeddings", patched_glm_get_input_embeddings)
    _PATCHED_ARCHES.add("glm_ocr")


def _install_paddleocr_vl_compile_patches():
    """Install compile-safe PaddleOCR-VL vision and embedding patches."""

    try:
        module = importlib.import_module("mlx_vlm.models.paddleocr_vl.paddleocr_vl")
        vision_module = importlib.import_module("mlx_vlm.models.paddleocr_vl.vision")
    except Exception:
        return

    InputEmbeddingsFeatures = module.InputEmbeddingsFeatures

    def patched_paddle_rotary(self, seqlen):
        import mlx.core as mx

        seqlen = int(seqlen) if not hasattr(seqlen, "item") else int(seqlen.item())
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)

    def patched_paddle_embeddings(self, hidden_states, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        batch_size, sequence_len, channel, patch_size, _ = hidden_states.shape
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.reshape(batch_size * sequence_len, channel, patch_size, patch_size)
        hidden_states = hidden_states.transpose(0, 2, 3, 1)
        patch_embeds = self.patch_embedding(hidden_states).astype(target_dtype)
        patch_embeds = patch_embeds.transpose(0, 3, 1, 2)
        embeddings = patch_embeds.flatten(-2).squeeze(-1).reshape(batch_size, sequence_len, -1)

        start = 0
        flat_embeddings = embeddings.squeeze(0)
        chunks = []
        for t, h, w in grid_spec:
            end = start + int(t) * int(h) * int(w)
            image_embeddings = flat_embeddings[start:end, :]
            position_embedding = self.interpolate_pos_encoding(int(h), int(w))
            chunks.append(image_embeddings + position_embedding)
            start = end
        return mx.concatenate(chunks, axis=0)

    def patched_paddle_projector(self, x, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        split_sizes = [int(t) * int(h) * int(w) for t, h, w in grid_spec]
        split_indices = list(accumulate(split_sizes[:-1])) if len(split_sizes) > 1 else []
        x_chunks = mx.split(x, split_indices, axis=0) if split_indices else [x]

        processed_features = []
        for x_chunk, (t, h, w) in zip(x_chunks, grid_spec):
            x_chunk = self.pre_norm(x_chunk)
            d = x_chunk.shape[-1]
            h_block = int(h) // self.spatial_merge_size
            w_block = int(w) // self.spatial_merge_size

            x_chunk = x_chunk.reshape(
                int(t),
                h_block,
                self.spatial_merge_size,
                w_block,
                self.spatial_merge_size,
                d,
            )
            x_chunk = x_chunk.transpose(0, 1, 3, 2, 4, 5)
            x_chunk = x_chunk.reshape(
                int(t) * h_block * w_block,
                self.spatial_merge_size * self.spatial_merge_size * d,
            )

            hidden_states = self.linear_1(x_chunk)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed_features.append(hidden_states)

        return mx.concatenate(processed_features, axis=0)

    def patched_paddle_attention(self, x, cu_seqlens, rotary_pos_emb=None):
        import mlx.core as mx

        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        q = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = vision_module.apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        split_indices = _split_points(cu_seqlens)
        q_splits = mx.split(q, split_indices, axis=2)
        k_splits = mx.split(k, split_indices, axis=2)
        v_splits = mx.split(v, split_indices, axis=2)

        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q_splits, k_splits, v_splits):
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk, scale=self.scale, mask=None
                )
            )

        output = mx.concatenate(outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.out_proj(output)

    def patched_paddle_rot_pos_emb(self, grid_thw):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        split_hids = []
        split_wids = []
        for t, h, w in grid_spec:
            image_pids = mx.arange(int(t) * int(h) * int(w)) % (int(h) * int(w))
            sample_hids = image_pids // int(w)
            sample_wids = image_pids % int(w)
            split_hids.append(sample_hids)
            split_wids.append(sample_wids)

        height_position_ids = mx.concatenate(split_hids, axis=0)
        width_position_ids = mx.concatenate(split_wids, axis=0)
        pos_ids = mx.stack([height_position_ids, width_position_ids], axis=-1)
        max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_spec)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb_full = rotary_pos_emb_full[pos_ids]
        return rotary_pos_emb_full.reshape(pos_ids.shape[0], -1)

    def patched_paddle_vision_call(self, hidden_states, grid_thw, output_hidden_states=None):
        import mlx.core as mx

        grid_spec = _grid_to_tuple(grid_thw)
        hidden_states = self.embeddings(hidden_states, grid_spec)
        rotary_pos_emb = self.rot_pos_emb(grid_spec)
        cu_seqlens = _build_cu_seqlens(grid_spec)

        encoder_states = (hidden_states,) if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.projector(hidden_states, grid_spec)
        return hidden_states

    def patched_paddle_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        explicit_position_ids = kwargs.get("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.visual.embeddings.patch_embedding.weight.dtype
        pixel_values = mx.array(pixel_values, dtype=dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)
        hidden_states = (
            cached
            if cached is not None
            else self.visual(pixel_values, grid_thw, output_hidden_states=False)
        )
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        self.language_model._position_ids = explicit_position_ids
        self.language_model._rope_deltas = None
        features = InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)
        return _attach_position_ids(features, explicit_position_ids)

    _patch_method(vision_module.VisionRotaryEmbedding, "__call__", patched_paddle_rotary)
    _patch_method(vision_module.PaddleOCRVisionEmbeddings, "__call__", patched_paddle_embeddings)
    _patch_method(vision_module.PaddleOCRProjector, "__call__", patched_paddle_projector)
    _patch_method(vision_module.Attention, "__call__", patched_paddle_attention)
    _patch_method(vision_module.VisionModel, "rot_pos_emb", patched_paddle_rot_pos_emb)
    _patch_method(vision_module.VisionModel, "__call__", patched_paddle_vision_call)
    _patch_method(module.Model, "get_input_embeddings", patched_paddle_get_input_embeddings)
    _PATCHED_ARCHES.add("paddleocr_vl")


def _install_qwen_like_image_merge_patches():
    """Patch qwen-style placeholder-token image merges with a shared helper.

    This covers the common pattern where a model replaces one placeholder token
    per image feature segment. We intentionally exclude the Qwen3 family here
    because it returns an additional broadcast mask and has its own multimodal
    deepstack plumbing.
    """

    for arch, module in _iter_trait_model_modules(
        "qwen_like_image_merge",
        include_arches=("qwen2_vl", "qwen2_5_vl", "glm_ocr", "paddleocr_vl"),
    ):
        if arch.startswith("qwen3"):
            continue
        model_cls = getattr(module, "Model", None)
        if (
            model_cls is None
            or getattr(model_cls, "merge_input_ids_with_image_features", None) is None
        ):
            continue
        _patch_staticmethod(
            model_cls,
            "merge_input_ids_with_image_features",
            _merge_special_token_features_only,
        )
        _PATCHED_ARCHES.add(arch)

    qwen3_module = _try_import_module("mlx_vlm.models.qwen3_vl.qwen3_vl")
    if qwen3_module is None:
        return

    def merge_qwen3(image_features, inputs_embeds, input_ids, image_token_index, video_token_index):
        outputs, mask = _merge_special_token_features(
            image_token_index,
            video_token_index,
            image_features,
            inputs_embeds,
            input_ids,
        )
        mask = mask.astype(bool)
        mask = importlib.import_module("mlx.core").broadcast_to(mask, inputs_embeds.shape)
        return outputs, mask

    qwen3_module.masked_scatter = _masked_scatter_no_numpy
    _patch_staticmethod(qwen3_module.Model, "merge_input_ids_with_image_features", merge_qwen3)
    _PATCHED_ARCHES.add("qwen3_vl")


def _install_qwen3_get_input_embeddings_patch():
    """Patch Qwen3-VL embeddings to expose compile-safe multimodal metadata."""

    module = _try_import_module("mlx_vlm.models.qwen3_vl.qwen3_vl")
    if module is None:
        return

    InputEmbeddingsFeatures = module.InputEmbeddingsFeatures

    def patched_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
            deepstack_image_embeds = None
        else:
            hidden_states, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw)

        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=deepstack_image_embeds,
        )

    _patch_method(module.Model, "get_input_embeddings", patched_get_input_embeddings)
    _PATCHED_ARCHES.add("qwen3_vl")


def _install_llama_pixtral_mistral_compile_patches():
    """Install compile-safe multimodal merge patches for llama-like families."""

    def merge_single_image_token_family(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        return _merge_special_token_features_only(
            image_token_index,
            None,
            image_features,
            inputs_embeds,
            input_ids,
        )

    for modname in ("mlx_vlm.models.pixtral.pixtral", "mlx_vlm.models.mistral3.mistral3"):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        _patch_staticmethod(
            module.Model,
            "merge_input_ids_with_image_features",
            merge_single_image_token_family,
        )
        _PATCHED_ARCHES.add(modname.split(".models.", 1)[1].split(".", 1)[0])

    try:
        llama4_module = importlib.import_module("mlx_vlm.models.llama4.llama4")
    except Exception:
        llama4_module = None

    def patched_single_image_prepare_inputs(self, image_features, inputs_embeds, input_ids):
        return _merge_special_token_features_only(
            self.config.image_token_index,
            None,
            image_features,
            inputs_embeds,
            input_ids,
        )

    if llama4_module is not None:
        _patch_method(
            llama4_module.Model,
            "_prepare_inputs_for_multimodal",
            patched_single_image_prepare_inputs,
        )
        _PATCHED_ARCHES.add("llama4")

    try:
        mllama_module = importlib.import_module("mlx_vlm.models.mllama.mllama")
        mllama_vision_module = importlib.import_module("mlx_vlm.models.mllama.vision")
    except Exception:
        mllama_module = None
        mllama_vision_module = None

    if mllama_module is not None and mllama_vision_module is not None:
        def patched_mllama_prepare_cross_attention_mask(
            self,
            cross_attention_mask,
            num_vision_tokens,
        ):
            batch_size, text_total_length, *_ = cross_attention_mask.shape
            target_dtype = self.language_model.model.embed_tokens.weight.dtype
            cross_attention_mask = mx.repeat(
                cross_attention_mask, num_vision_tokens, axis=3
            )
            cross_attention_mask = cross_attention_mask.reshape(
                batch_size, text_total_length, -1
            )
            cross_attention_mask = mx.expand_dims(
                cross_attention_mask.astype(target_dtype), 1
            )

            inverted_cross_attn_mask = 1.0 - cross_attention_mask
            fill_array = mx.full(
                inverted_cross_attn_mask.shape,
                -1e4 if target_dtype == mx.float16 else -1e9,
                dtype=target_dtype,
            )
            cross_attention_mask = mx.where(
                inverted_cross_attn_mask.astype(mx.bool_),
                fill_array,
                cross_attention_mask,
            )

            full_text_row_masked_out_mask = mx.any(
                cross_attention_mask != fill_array,
                axis=-1,
                keepdims=True,
            ).astype(target_dtype)
            cross_attention_mask = cross_attention_mask * full_text_row_masked_out_mask
            return cross_attention_mask, full_text_row_masked_out_mask

        def patched_mllama_vision_call(
            self,
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
        ):
            batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
                pixel_values.shape
            )
            aspect_ratio_ids = aspect_ratio_ids.reshape(
                batch_size * num_concurrent_media, -1
            )

            pixel_values = pixel_values.reshape(
                batch_size * num_concurrent_media * num_tiles,
                num_channels,
                height,
                width,
            )
            patch_embeds = self.patch_embedding(pixel_values.moveaxis(1, 3)).moveaxis(
                3, 1
            )
            hidden_state = patch_embeds.reshape(
                patch_embeds.shape[0], patch_embeds.shape[1], -1
            ).transpose(0, 2, 1)

            _, num_patches, dim = hidden_state.shape
            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media, num_tiles, -1, dim
            )
            hidden_state = self.pre_tile_positional_embedding(
                hidden_state, aspect_ratio_ids
            )

            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media * num_tiles, num_patches, dim
            )
            class_embedding = mx.broadcast_to(
                self.class_embedding,
                (batch_size * num_concurrent_media * num_tiles, 1, dim),
            ).astype(hidden_state.dtype)
            hidden_state = mx.concatenate([class_embedding, hidden_state], axis=1)
            num_patches += 1

            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media, num_tiles, num_patches, dim
            )
            hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)
            hidden_state = self.layernorm_pre(hidden_state)

            num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
            padding = [(0, 0), (0, 0), (0, num_padding_patches), (0, 0)]
            hidden_state = mx.pad(hidden_state, padding)
            slice_index = -num_padding_patches if num_padding_patches > 0 else None

            attention_mask = aspect_ratio_mask.reshape(
                batch_size * num_concurrent_media, -1
            )
            attention_mask = mllama_vision_module._prepare_aspect_ratio_attention_mask(
                aspect_ratio_mask=attention_mask,
                num_patches=self.num_patches,
                target_length=hidden_state.shape[2],
            ).astype(hidden_state.dtype)

            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media, -1, self.hidden_size
            )
            output = self.transformer(hidden_state, attention_mask=attention_mask)
            hidden_state = self.layernorm_post(output[0])

            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media,
                num_tiles,
                num_patches + num_padding_patches,
                self.hidden_size,
            )
            hidden_state = self.post_tile_positional_embedding(
                hidden_state, aspect_ratio_ids
            )
            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media,
                num_tiles * (num_patches + num_padding_patches),
                self.hidden_size,
            )
            global_output = self.global_transformer(
                hidden_state, attention_mask=attention_mask
            )
            hidden_state = global_output[0]

            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media,
                num_tiles,
                num_patches + num_padding_patches,
                dim,
            )
            hidden_state = hidden_state[:, :, :slice_index]
            hidden_state = hidden_state.reshape(
                batch_size, num_concurrent_media, num_tiles, num_patches, dim
            )

            intermediate_hidden_states = mx.stack(output[1], axis=-1)
            intermediate_hidden_states = intermediate_hidden_states[
                ..., self.intermediate_layers_indices
            ]
            intermediate_hidden_states = intermediate_hidden_states.reshape(
                batch_size * num_concurrent_media,
                num_tiles,
                num_patches + num_padding_patches,
                -1,
            )
            intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
            intermediate_hidden_states = intermediate_hidden_states.reshape(
                batch_size, num_concurrent_media, num_tiles, num_patches, -1
            )

            return mx.concatenate([hidden_state, intermediate_hidden_states], axis=-1)

        _patch_method(
            mllama_module.Model,
            "_prepare_cross_attention_mask",
            patched_mllama_prepare_cross_attention_mask,
        )
        _patch_method(
            mllama_vision_module.VisionModel,
            "__call__",
            patched_mllama_vision_call,
        )
        _PATCHED_ARCHES.add("mllama")

    for modname in ("mlx_vlm.models.llava_bunny.llava_bunny",):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        _patch_method(
            module.Model,
            "_prepare_inputs_for_multimodal",
            patched_single_image_prepare_inputs,
        )
        _PATCHED_ARCHES.add(modname.split(".models.", 1)[1].split(".", 1)[0])

    for modname in ("mlx_vlm.models.llava.llava", "mlx_vlm.models.aya_vision.aya_vision"):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue

        def patched_merge(self, image_features, inputs_embeds, input_ids):
            return _merge_special_token_features_only(
                self.config.image_token_index,
                None,
                image_features,
                inputs_embeds,
                input_ids,
            )

        _patch_method(module.Model, "_merge_input_ids_with_image_features", patched_merge)
        _PATCHED_ARCHES.add(modname.split(".models.", 1)[1].split(".", 1)[0])

    try:
        llava_next_module = importlib.import_module("mlx_vlm.models.llava_next.llava_next")
    except Exception:
        return

    def patched_llava_next_get_input_embeddings(
        self,
        input_ids=None,
        pixel_values=None,
        **kwargs,
    ):
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached.astype(inputs_embeds.dtype)
        else:
            pixel_values_hwc = pixel_values[0].transpose(0, 2, 3, 1)
            *_, hidden_states = self.vision_tower(
                pixel_values_hwc, output_hidden_states=True
            )
            image_features = hidden_states[self.vision_feature_layer]
            if self.vision_feature_select_strategy == "default":
                image_features = image_features[:, 1:]
            elif self.vision_feature_select_strategy != "full":
                raise ValueError(
                    "Unexpected feature selection strategy: "
                    f"{self.vision_feature_select_strategy}"
                )

            image_features = self.multi_modal_projector(image_features)
            if self.image_newline is not None:
                newline = mx.broadcast_to(
                    self.image_newline[None, None, :],
                    image_features.shape,
                )
                image_features = mx.concatenate([image_features, newline], axis=0)
            image_features = image_features.astype(inputs_embeds.dtype)

        outputs = _merge_special_token_features_only(
            self.config.image_token_index,
            None,
            image_features,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=outputs)

    def patched_llava_next_merge(self, image_features, inputs_embeds, input_ids):
        return _merge_special_token_features_only(
            self.config.image_token_index,
            None,
            image_features,
            inputs_embeds,
            input_ids,
        )

    _patch_method(
        llava_next_module.Model,
        "get_input_embeddings",
        patched_llava_next_get_input_embeddings,
    )
    _patch_method(
        llava_next_module.Model,
        "_merge_input_ids_with_image_features",
        patched_llava_next_merge,
    )
    _PATCHED_ARCHES.add("llava_next")


def _install_mistral4_compile_patches():
    """Install the Mistral4-specific attention backend compatibility patch."""

    try:
        module = importlib.import_module("mlx_vlm.models.mistral4.language")
    except Exception:
        return

    def patched_mistral4_attention(self, x, attn_scale, mask=None, cache=None):
        B, L, D = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)

        if self.qk_rope_head_dim > 0:
            q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        else:
            q_nope = q
            q_pe = None

        compressed_kv = self.kv_a_proj_with_mqa(x)
        if self.qk_rope_head_dim > 0:
            compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
            k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        else:
            compressed_kv = compressed_kv[..., : self.kv_lora_rank]
            k_pe = None

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if self.qk_rope_head_dim > 0:
            if cache is not None:
                q_pe = self.rope(q_pe, offset=cache.offset)
                k_pe = self.rope(k_pe, offset=cache.offset)
                k_pe = mx.broadcast_to(
                    k_pe, k_nope.shape[:-1] + (self.qk_rope_head_dim,)
                )
                keys = mx.concatenate([k_nope, k_pe], axis=-1)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                q_pe = self.rope(q_pe)
                k_pe = self.rope(k_pe)
                k_pe = mx.broadcast_to(
                    k_pe, k_nope.shape[:-1] + (self.qk_rope_head_dim,)
                )
                keys = mx.concatenate([k_nope, k_pe], axis=-1)
            queries = mx.concatenate([q_nope, q_pe], axis=-1)
        else:
            queries = q_nope
            keys = k_nope
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries * attn_scale

        output = module.scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    _patch_method(module.Mistral4Attention, "__call__", patched_mistral4_attention)
    _PATCHED_ARCHES.add("mistral4")


def _install_gemma3n_compile_patches():
    """Install Gemma3n multiscale fusion and merge compatibility patches."""

    try:
        module = importlib.import_module("mlx_vlm.models.gemma3n.gemma3n")
        vision_module = importlib.import_module("mlx_vlm.models.gemma3n.vision")
        language_module = importlib.import_module("mlx_vlm.models.gemma3n.language")
        kernels_module = importlib.import_module("mlx_vlm.models.kernels")
    except Exception:
        return

    def patched_merge_multimodal_and_text(
        inputs_embeds, features, special_modality_mask, modality="image"
    ):
        del modality
        features = features.astype(inputs_embeds.dtype)
        return _masked_scatter_no_numpy(inputs_embeds, special_modality_mask, features)

    def patched_msfa_call(self, inputs):
        import mlx.nn as nn

        inputs_nchw = [tensor.transpose(0, 3, 1, 2) for tensor in inputs]
        high_resolution = inputs_nchw[0].shape[-2:]
        resized_inputs = []

        for tensor in inputs_nchw:
            if any(dim < target for dim, target in zip(tensor.shape[-2:], high_resolution)):
                tensor = kernels_module._nearest_interpolate_mlx(
                    tensor, high_resolution[0], high_resolution[1]
                )
            resized_inputs.append(tensor)

        channel_cat_imgs = mx.concatenate(resized_inputs, axis=1)
        img = self.ffn(channel_cat_imgs.swapaxes(1, 3)).swapaxes(1, 3)

        if any(
            expected != actual
            for expected, actual in zip(high_resolution, self.output_resolution)
        ):
            if (
                high_resolution[0] % self.output_resolution[0] != 0
                or high_resolution[1] % self.output_resolution[1] != 0
            ):
                img = kernels_module._bicubic_interpolate_mlx(
                    img,
                    self.output_resolution[0],
                    self.output_resolution[1],
                )
            else:
                h_strides = high_resolution[0] // self.output_resolution[0]
                w_strides = high_resolution[1] // self.output_resolution[1]
                img = nn.AvgPool2d(
                    kernel_size=(h_strides, w_strides),
                    stride=(h_strides, w_strides),
                )(img.swapaxes(1, 3))
            img = self.norm(img) if self.noskip else img

        return img

    def _safe_branch_magnitude(x):
        # Gemma3n's AltUp branch rescaling squares hidden states to estimate an
        # RMS magnitude. In fp16, some finite branch activations exceed the
        # range where squaring remains finite, so `x ** 2` overflows even
        # though `x` itself is still valid. Compute the magnitude in float32
        # and cast back afterward to preserve the intended rescaling.
        return (mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) ** 0.5)

    def patched_gemma3model_call(
        self,
        inputs=None,
        inputs_embeds=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        min_scale = mx.array(mx.finfo(mx.float32).eps, dtype=mx.float32)
        per_layer_inputs = kwargs.pop("per_layer_inputs", None)
        n_to_process = kwargs.pop("n_to_process", None)

        if inputs_embeds is None:
            h = self.embed_tokens(inputs) * (self.hidden_size**0.5)
        else:
            h = inputs_embeds

        if per_layer_inputs is None and inputs is not None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)
        elif per_layer_inputs is not None:
            target_len = n_to_process if n_to_process is not None else h.shape[1]
            if target_len != h.shape[1]:
                target_len = h.shape[1]

            cache_offset = next(
                (
                    int(c.offset)
                    for c in (cache or [])
                    if c is not None and hasattr(c, "offset")
                ),
                0,
            )
            max_start = max(per_layer_inputs.shape[1] - target_len, 0)
            start = min(cache_offset, max_start)
            per_layer_inputs = per_layer_inputs[:, start : start + target_len]

        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            full_mask = language_module.create_attention_mask(
                h,
                cache[self.first_full_idx :],
            )
            sliding_window_mask = language_module.create_attention_mask(
                h,
                cache[self.first_sliding_idx :],
            )
        h0 = h

        target_magnitude = _safe_branch_magnitude(h0)

        h_list = [h0]
        h_list.extend([proj(h0) for proj in self.altup_projections])
        h = mx.stack(h_list, axis=0)
        mags = _safe_branch_magnitude(h[1:])
        scale = target_magnitude / mx.maximum(mags, min_scale)
        h = mx.concatenate([h[:1], h[1:] * scale.astype(h.dtype)], axis=0)

        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = self.config.layer_types[i] == "full_attention"

            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(
                h,
                local_mask,
                cache[self.layer_idx_to_cache_idx[i]],
                per_layer_input,
            )

        target_magnitude = _safe_branch_magnitude(h[0])
        head = h[:1]
        tail = [proj(h[i + 1]) for i, proj in enumerate(self.altup_unembed_projections)]
        tail = mx.stack(tail, axis=0)
        mags = _safe_branch_magnitude(tail)
        scale = target_magnitude / mx.maximum(mags, min_scale)
        h = mx.concatenate([head, tail * scale.astype(tail.dtype)], axis=0)

        h = mx.mean(h, axis=0)
        return self.norm(h)

    module.masked_scatter = _masked_scatter_no_numpy
    _patch_staticmethod(
        module.Model,
        "merge_multimodal_and_text",
        patched_merge_multimodal_and_text,
    )
    _patch_method(
        vision_module.MobileNetV5MultiScaleFusionAdapter,
        "__call__",
        patched_msfa_call,
    )
    _patch_method(
        language_module.Gemma3Model,
        "__call__",
        patched_gemma3model_call,
    )
    _PATCHED_ARCHES.add("gemma3n")


def _install_deepseek_ocr_compile_patches():
    """Install DeepSeek OCR compile patches for SAM/projector/image merging."""

    try:
        deepseekocr_module = importlib.import_module("mlx_vlm.models.deepseekocr.deepseekocr")
        sam_module = importlib.import_module("mlx_vlm.models.deepseekocr.sam")
        kernels_module = importlib.import_module("mlx_vlm.models.kernels")
    except Exception:
        deepseekocr_module = None

    if deepseekocr_module is not None:
        InputEmbeddingsFeatures = deepseekocr_module.InputEmbeddingsFeatures
        vision_module = importlib.import_module("mlx_vlm.models.deepseekocr.vision")

        def patched_get_abs_pos_sam(abs_pos, tgt_size):
            dtype = abs_pos.dtype
            src_size = abs_pos.shape[1]
            if src_size == tgt_size:
                return abs_pos
            old_pos_embed = abs_pos.transpose(0, 3, 1, 2).astype(mx.float32)
            new_pos_embed = kernels_module._bicubic_interpolate_mlx(
                old_pos_embed, tgt_size, tgt_size, antialias=True
            ).astype(dtype)
            return new_pos_embed.transpose(0, 2, 3, 1)

        def patched_deepseekocr_abs_pos(self, abs_pos, tgt_size):
            dtype = abs_pos.dtype
            abs_pos_new = mx.squeeze(abs_pos, axis=0)
            cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]
            src_size = int((abs_pos_new.shape[0] - 1) ** 0.5)
            tgt_tokens = max(int(tgt_size) - 1, 0)
            tgt_size_2d = int(tgt_tokens**0.5)

            if src_size == tgt_size_2d:
                return mx.expand_dims(abs_pos_new[: tgt_tokens + 1], axis=0)

            old_pos_embed = old_pos_embed.reshape(1, src_size, src_size, -1).transpose(
                0, 3, 1, 2
            )
            new_pos_embed = kernels_module._bicubic_interpolate_mlx(
                old_pos_embed.astype(mx.float32), tgt_size_2d, tgt_size_2d
            ).astype(dtype)
            new_pos_embed = new_pos_embed.transpose(0, 2, 3, 1).reshape(
                tgt_size_2d * tgt_size_2d, -1
            )
            vision_pos_embed = mx.concatenate([cls_token, new_pos_embed], axis=0)
            return mx.expand_dims(vision_pos_embed, axis=0)

        def patched_deepseekocr_projector(self, x):
            if self.config.projector_config.projector_type == "downsample_mlp_gelu":
                x = _downsample_square_tokens(
                    x, self.config.projector_config.downsample_ratio
                )

            if self.config.projector_config.projector_type == "linear":
                x = self.layers(x)
            else:
                for layer in self.layers:
                    x = layer(x)
            return x

        def patched_deepseekocr_init(self, config):
            import mlx.nn as nn

            nn.Module.__init__(self)
            self.config = config
            self.vision_model = deepseekocr_module.VisionModel(config.vision_config)
            sam_config = getattr(config, "sam_config", deepseekocr_module.SAMViTConfig())
            self.sam_model = deepseekocr_module.SAMEncoder(
                img_size=sam_config.image_size,
                patch_size=sam_config.patch_size,
                embed_dim=sam_config.width,
                depth=sam_config.layers,
                num_heads=sam_config.heads,
                window_size=sam_config.window_size,
                global_attn_indexes=sam_config.global_attn_indexes,
                final_out_chans=getattr(sam_config, "final_out_chans", 1024),
            )
            self.language_model = deepseekocr_module.LanguageModel(config.text_config)
            self.projector = deepseekocr_module.MlpProjector(config)

            self.tile_tag = config.tile_tag
            self.global_view_pos = config.global_view_pos
            embed_std = 1 / mx.sqrt(
                mx.array(config.projector_config.n_embed, dtype=mx.float32)
            )
            if self.tile_tag == "2D":
                self.image_newline = mx.array(
                    mx.random.normal((config.projector_config.n_embed,)) * embed_std
                )
                self.view_separator = mx.array(
                    mx.random.normal((config.projector_config.n_embed,)) * embed_std
                )
            else:
                raise ValueError(
                    f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
                )

        def patched_deepseekocr_get_input_embeddings(
            self,
            input_ids=None,
            pixel_values=None,
            images_spatial_crop=None,
            images_seq_mask=None,
            **kwargs,
        ):
            del kwargs
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            if pixel_values is None:
                return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

            all_patches = pixel_values[0]
            all_image_ori = pixel_values[1]
            seq_features = []
            patch_idx = 0

            for image_idx, crop_shape in enumerate(images_spatial_crop or ()):
                width_crop_num, height_crop_num = (int(crop_shape[0]), int(crop_shape[1]))
                has_crops = width_crop_num > 1 or height_crop_num > 1
                num_patches = width_crop_num * height_crop_num if has_crops else 0
                patches = (
                    all_patches[patch_idx : patch_idx + num_patches]
                    if num_patches > 0
                    else None
                )
                patch_idx += num_patches
                image_ori = all_image_ori[image_idx : image_idx + 1]

                if patches is not None and patches.shape[0] > 0:
                    local_features_1 = self.sam_model(patches.transpose(0, 2, 3, 1))
                    local_features_2 = self.vision_model(
                        patches.transpose(0, 2, 3, 1), patch_embeds=local_features_1
                    )
                    local_features = mx.concatenate(
                        (
                            local_features_2[:, 1:],
                            local_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori.transpose(0, 2, 3, 1))
                    global_features_2 = self.vision_model(
                        image_ori.transpose(0, 2, 3, 1), global_features_1
                    )
                    global_features = mx.concatenate(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    global_features = self.projector(global_features)[0]

                    hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    _, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)

                    global_features = global_features.reshape(h, w, n_dim)
                    global_features = mx.concatenate(
                        [
                            global_features,
                            mx.broadcast_to(self.image_newline[None, None, :], (h, 1, n_dim)),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim)

                    local_features = (
                        local_features.reshape(
                            height_crop_num, width_crop_num, h2, w2, n_dim2
                        )
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = mx.concatenate(
                        [
                            local_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :],
                                (height_crop_num * h2, 1, n_dim2),
                            ),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim2)
                    seq_features.append(
                        mx.concatenate(
                            [local_features, global_features, self.view_separator[None, :]],
                            axis=0,
                        )
                    )
                else:
                    global_features_1 = self.sam_model(image_ori.transpose(0, 2, 3, 1))
                    global_features_2 = self.vision_model(
                        image_ori.transpose(0, 2, 3, 1), global_features_1
                    )
                    global_features = mx.concatenate(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    global_features = self.projector(global_features)[0]
                    hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    global_features = global_features.reshape(h, w, n_dim)
                    global_features = mx.concatenate(
                        [
                            global_features,
                            mx.broadcast_to(self.image_newline[None, None, :], (h, 1, n_dim)),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim)
                    seq_features.append(
                        mx.concatenate([global_features, self.view_separator[None, :]], axis=0)
                    )

            if seq_features and images_seq_mask is not None:
                merged_features = mx.concatenate(seq_features, axis=0)
                input_embeds, _ = _merge_sequence_mask_features(
                    images_seq_mask,
                    merged_features,
                    input_embeds,
                )

            return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

        _patch_method(
            deepseekocr_module.Model,
            "get_input_embeddings",
            patched_deepseekocr_get_input_embeddings,
        )
        _patch_method(deepseekocr_module.Model, "__init__", patched_deepseekocr_init)
        _patch_method(deepseekocr_module.MlpProjector, "__call__", patched_deepseekocr_projector)
        _patch_method(vision_module.VisionEmbeddings, "_get_abs_pos", patched_deepseekocr_abs_pos)
        sam_module.get_abs_pos_sam = patched_get_abs_pos_sam
        _PATCHED_ARCHES.add("deepseekocr")

    try:
        deepseekocr2_module = importlib.import_module("mlx_vlm.models.deepseekocr_2.deepseekocr_2")
    except Exception:
        return

    InputEmbeddingsFeatures2 = deepseekocr2_module.InputEmbeddingsFeatures

    def patched_deepseekocr2_init(self, config):
        import mlx.nn as nn

        nn.Module.__init__(self)
        self.config = config
        self.vision_model = deepseekocr2_module.VisionModel(config.vision_config)
        sam_config = getattr(config, "sam_config", deepseekocr2_module.SAMViTConfig())
        self.sam_model = deepseekocr2_module.SAMEncoder(
            img_size=sam_config.image_size,
            patch_size=sam_config.patch_size,
            embed_dim=sam_config.width,
            depth=sam_config.layers,
            num_heads=sam_config.heads,
            window_size=sam_config.window_size,
            global_attn_indexes=sam_config.global_attn_indexes,
            final_out_chans=getattr(sam_config, "final_out_chans", 896),
        )
        self.language_model = deepseekocr2_module.LanguageModel(config.text_config)
        self.projector = deepseekocr2_module.MlpProjector(config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
        if self.tile_tag == "2D":
            self.view_separator = mx.zeros((config.projector_config.n_embed,))
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

    def patched_deepseekocr2_get_input_embeddings(
        self,
        input_ids=None,
        pixel_values=None,
        images_spatial_crop=None,
        images_seq_mask=None,
        **kwargs,
    ):
        del kwargs
        input_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return InputEmbeddingsFeatures2(inputs_embeds=input_embeds)

        if isinstance(pixel_values, (list, tuple)):
            patches, global_images = pixel_values
        else:
            patches = None
            global_images = pixel_values

        seq_features = []
        patch_offset = 0
        for idx in range(input_ids.shape[0]):
            all_features = []
            rows, cols = (1, 1)
            if images_spatial_crop is not None and idx < len(images_spatial_crop):
                rows, cols = (
                    int(images_spatial_crop[idx][0]),
                    int(images_spatial_crop[idx][1]),
                )
            num_patches = rows * cols if (rows > 1 or cols > 1) else 0
            sample_patches = None
            if patches is not None and num_patches > 0:
                sample_patches = patches[patch_offset : patch_offset + num_patches]
                patch_offset += num_patches

            if sample_patches is not None and sample_patches.shape[0] > 0:
                for patch_idx in range(sample_patches.shape[0]):
                    patch = sample_patches[patch_idx : patch_idx + 1]
                    patch_hwc = patch.transpose(0, 2, 3, 1)
                    sam_features = self.sam_model(patch_hwc)
                    vision_features = self.vision_model(patch_hwc, sam_features)
                    all_features.append(self.projector(vision_features)[0])

            global_image = global_images[idx : idx + 1]
            global_hwc = global_image.transpose(0, 2, 3, 1)
            sam_features = self.sam_model(global_hwc)
            global_features = self.vision_model(global_hwc, sam_features)
            all_features.append(self.projector(global_features)[0])
            all_features.append(self.view_separator[None, :])
            seq_features.append(mx.concatenate(all_features, axis=0))

        if seq_features and images_seq_mask is not None:
            merged_features = mx.concatenate(seq_features, axis=0)
            input_embeds, _ = _merge_sequence_mask_features(
                images_seq_mask,
                merged_features,
                input_embeds,
            )

        return InputEmbeddingsFeatures2(inputs_embeds=input_embeds)

    _patch_method(
        deepseekocr2_module.Model,
        "get_input_embeddings",
        patched_deepseekocr2_get_input_embeddings,
    )
    _patch_method(deepseekocr2_module.Model, "__init__", patched_deepseekocr2_init)
    _PATCHED_ARCHES.add("deepseekocr_2")


def _install_masked_scatter_multimodal_patches():
    """Patch `masked_scatter` implementations that rely on host-side mutation."""

    for arch, module in _iter_trait_model_modules(
        "masked_scatter_multimodal",
        include_arches=("gemma3", "idefics2", "idefics3"),
    ):
        if getattr(module, "masked_scatter", None) is None:
            continue
        setattr(module, "masked_scatter", _masked_scatter_no_numpy)
        _PATCHED_ARCHES.add(arch)


def _install_idefics_family_compile_patches():
    """Patch Idefics-family image filtering and multimodal merges for compile.

    The shared issue here is Python-side filtering of padded images and image
    placeholder merging. Architectures that expose the same source-level trait
    and standard model module layout can inherit this patch automatically.
    """

    def patched_prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        special_image_mask = input_ids == self.config.image_token_index
        outputs, _ = _merge_sequence_mask_features(
            special_image_mask,
            image_features,
            inputs_embeds,
        )
        return outputs

    def patched_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        import mlx.core as mx
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        batch_size, num_images, num_channels, height, width = pixel_values.shape
        flat_pixel_values = pixel_values.reshape(
            batch_size * num_images, num_channels, height, width
        )

        if pixel_attention_mask is None:
            pixel_attention_mask = mx.ones(
                (batch_size * num_images, height, width),
                dtype=mx.bool_,
            )
        else:
            pixel_attention_mask = pixel_attention_mask.reshape(
                batch_size * num_images, height, width
            ).astype(mx.bool_)

        patch_size = self.config.vision_config.patch_size
        patches_h = height // patch_size
        patches_w = width // patch_size
        reshaped = pixel_attention_mask[
            :, : patches_h * patch_size, : patches_w * patch_size
        ]
        reshaped = reshaped.reshape(
            batch_size * num_images, patches_h, patch_size, patches_w, patch_size
        )
        reshaped = reshaped.transpose(0, 1, 3, 2, 4)
        patch_attention_mask = reshaped.sum(axis=(-1, -2)) > 0

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            pooler_output, *_ = self.vision_model(
                flat_pixel_values.transpose(0, 2, 3, 1),
                patch_attention_mask=patch_attention_mask,
                output_hidden_states=True,
            )
            image_features = self.connector(pooler_output.astype(flat_pixel_values.dtype))

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    for arch, module in _iter_trait_model_modules(
        "padded_image_filtering",
        include_arches=("idefics2", "idefics3"),
    ):
        if arch == "smolvlm":
            continue
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            continue
        _patch_method(
            model_cls,
            "_prepare_inputs_for_multimodal",
            patched_prepare_inputs_for_multimodal,
        )
        if getattr(model_cls, "get_input_embeddings", None) is not None:
            _patch_method(model_cls, "get_input_embeddings", patched_get_input_embeddings)
        _PATCHED_ARCHES.add(arch)

    smol_module = _try_import_module("mlx_vlm.models.smolvlm.smolvlm")
    if smol_module is None:
        return

    _patch_method(
        smol_module.Model,
        "_prepare_inputs_for_multimodal",
        patched_prepare_inputs_for_multimodal,
    )
    _PATCHED_ARCHES.add("smolvlm")


def _install_negative_image_placeholder_patches():
    """Install Phi-3 vision patches for negative placeholder-token flows."""

    phi3_module = _try_import_module("mlx_vlm.models.phi3_v.phi3_v")
    phi3_vision_module = _try_import_module("mlx_vlm.models.phi3_v.vision")
    if phi3_module is None or phi3_vision_module is None:
        return

    def patched_phi3_get_input_embeddings(
        self,
        inputs,
        pixel_values=None,
        **kwargs,
    ):
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        image_sizes = kwargs.get("image_sizes", None)
        positions = kwargs.get("image_positions", None)
        positions = _positions_to_tuple(positions)

        if not positions:
            inputs_np = np.asarray(inputs)
            positions = _positions_to_tuple(np.argwhere(inputs_np < 0))

        inputs_embeds = self.model.embed_tokens(inputs)
        negative_mask = inputs < 0

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            outputs, _ = _merge_sequence_mask_features(
                negative_mask,
                cached,
                inputs_embeds,
            )
            return InputEmbeddingsFeatures(inputs_embeds=outputs)

        if pixel_values.dtype != inputs_embeds.dtype:
            pixel_values = pixel_values.astype(inputs_embeds.dtype)

        outputs = self.model.vision_embed_tokens(
            pixel_values,
            inputs_embeds,
            image_sizes,
            positions,
        )
        return InputEmbeddingsFeatures(inputs_embeds=outputs)

    def patched_phi3_vision_call(
        self,
        img_embeds,
        txt_embeds=None,
        img_sizes=None,
        positions=None,
        output_hidden_states=None,
    ):
        if output_hidden_states:
            return self.img_processor.vision_model(
                img_embeds, output_hidden_states=output_hidden_states
            )

        batch_size = img_embeds.shape[0]
        image_sizes = tuple(
            (height // 336, width // 336)
            for height, width in _size_pairs_to_tuple(img_sizes)
        )
        positions = _positions_to_tuple(positions)
        if not image_sizes or not positions:
            return txt_embeds

        img_features = self.img_processor.vision_model(
            img_embeds.reshape(-1, *img_embeds.shape[2:]).transpose(0, 2, 3, 1), True
        )[-1][-2][:, 1:]
        img_features = img_features.reshape(batch_size, -1, *img_features.shape[1:])
        channels = self.image_dim_out
        patch_hw = int(img_features.shape[2] ** 0.5)
        target_dtype = txt_embeds.dtype
        position_index = 0

        for batch_idx in range(batch_size):
            tiles_h, tiles_w = image_sizes[batch_idx]
            num_tiles = tiles_h * tiles_w

            def reshape_and_concatenate(img, shape, tile_shape):
                return mx.concatenate(
                    [
                        img.reshape(shape)
                        .transpose(0, 1, 3, 2, 4, 5)
                        .reshape(tile_shape),
                        mx.tile(self.sub_GN, (1, tile_shape[1], 1, 1)),
                    ],
                    axis=2,
                ).reshape(1, -1, 4 * channels)

            global_image = reshape_and_concatenate(
                img_features[batch_idx : batch_idx + 1, :1],
                (1, patch_hw // 2, 2, patch_hw // 2, 2, channels),
                (1, patch_hw // 2, patch_hw // 2, 4 * channels),
            )
            sub_image = reshape_and_concatenate(
                img_features[batch_idx, 1 : num_tiles + 1],
                (num_tiles, patch_hw // 2, 2, patch_hw // 2, 2, channels),
                (1, tiles_h * 12, tiles_w * 12, 4 * channels),
            )
            outputs = mx.concatenate([sub_image, self.glb_GN, global_image], axis=1)
            for layer in self.img_projection:
                outputs = layer(outputs)

            token_count = int((tiles_h * tiles_w + 1) * 144 + 1 + (tiles_h + 1) * 12)
            text_batch_idx, start_idx = positions[position_index]
            txt_embeds[text_batch_idx, start_idx : start_idx + token_count, :] = outputs[
                0
            ].astype(target_dtype)
            position_index += token_count

        return txt_embeds

    _patch_method(phi3_module.Model, "get_input_embeddings", patched_phi3_get_input_embeddings)
    _patch_method(phi3_vision_module.VisionModel, "__call__", patched_phi3_vision_call)
    _PATCHED_ARCHES.add("phi3_v")


def _install_expanded_image_placeholder_patches():
    """Install placeholder-expansion patches for repeated image-token families."""

    module = _try_import_module("mlx_vlm.models.multi_modality.multi_modality")
    vision_module = _try_import_module("mlx_vlm.models.multi_modality.vision")
    if module is None or vision_module is None:
        return

    def patched_fast_gelu(self, input):
        return (
            0.5
            * input
            * (
                1.0
                + mx.tanh(
                    0.7978845608028654 * (input + 0.044715 * (input**3))
                )
            )
        ).astype(input.dtype)

    def patched_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            if self.config.vision_config.cls == "HybridVisionTower":
                hidden_states = self.vision_model(
                    pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
                )
            else:
                hidden_states, _, _ = self.vision_model(
                    pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
                )
            image_features = self.aligner(hidden_states)

        outputs = _merge_special_token_features_only(
            self.config.image_token_index,
            None,
            image_features,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=outputs)

    _patch_method(vision_module.FastGELUActivation, "__call__", patched_fast_gelu)
    _patch_method(module.Model, "get_input_embeddings", patched_get_input_embeddings)
    _PATCHED_ARCHES.add("multi_modality")


def _install_phi4_multimodal_patches():
    """Install Phi-4 multimodal span and resize patches."""

    def patch_siglip_resize(module):
        interpolate_module = importlib.import_module("mlx_vlm.models.interpolate")

        def patched_resize_positional_embeddings(positional_embeddings, spatial_shapes, max_length):
            spatial_shapes = _size_pairs_to_tuple(spatial_shapes)
            embed_dim = positional_embeddings.shape[-1]
            source_dtype = positional_embeddings.dtype
            pos_emb = positional_embeddings.transpose(2, 0, 1)[None, :]
            outputs = []
            for height, width in spatial_shapes:
                resized = interpolate_module.resize_bilinear(
                    pos_emb,
                    (height, width),
                    align_corners=False,
                    antialias=True,
                )
                resized = resized.reshape(embed_dim, height * width).transpose(1, 0)
                if height * width < max_length:
                    padding = mx.broadcast_to(
                        resized[0:1],
                        (max_length - (height * width), embed_dim),
                    )
                    resized = mx.concatenate([resized, padding], axis=0)
                outputs.append(resized[:max_length].astype(source_dtype))
            return mx.stack(outputs)

        _patch_staticmethod(
            module.VisionEmbeddings,
            "resize_positional_embeddings",
            patched_resize_positional_embeddings,
        )

    try:
        phi4_siglip_module = importlib.import_module("mlx_vlm.models.phi4_siglip.phi4_siglip")
        phi4_siglip_vision_module = importlib.import_module("mlx_vlm.models.phi4_siglip.vision")
    except Exception:
        phi4_siglip_module = None
        phi4_siglip_vision_module = None

    if phi4_siglip_module is not None and phi4_siglip_vision_module is not None:
        patch_siglip_resize(phi4_siglip_vision_module)

        def patched_phi4_siglip_get_input_embeddings(self, inputs, pixel_values=None, **kwargs):
            from mlx_vlm.models.base import InputEmbeddingsFeatures

            spatial_shapes = _size_pairs_to_tuple(kwargs.get("spatial_shapes", None))
            pixel_valid_lengths = _lengths_to_tuple(kwargs.get("pixel_valid_lengths", None))
            inputs_embeds = self.language_model.model.embed_tokens(inputs)
            if pixel_values is None:
                return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

            encoder_outputs, _, _ = self.vision_tower(
                pixel_values,
                output_hidden_states=True,
                spatial_shapes=spatial_shapes,
            )
            hidden_states = encoder_outputs[self.config.mm_vision_select_layer]

            image_features_list = []
            if pixel_valid_lengths is None:
                pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
                if pixel_attention_mask is not None:
                    pixel_valid_lengths = _lengths_to_tuple(
                        np.asarray(pixel_attention_mask).astype(np.int32).sum(axis=1)
                    )
            if pixel_valid_lengths is not None:
                for img_idx, valid_len in enumerate(pixel_valid_lengths):
                    image_features_list.append(
                        self.mm_projector(hidden_states[img_idx, :valid_len, :])
                    )
            else:
                for img_idx in range(hidden_states.shape[0]):
                    image_features_list.append(self.mm_projector(hidden_states[img_idx]))

            merged_features = mx.concatenate(image_features_list, axis=0)
            outputs, _ = _merge_sequence_mask_features(
                inputs == -200,
                merged_features,
                inputs_embeds,
            )
            return InputEmbeddingsFeatures(inputs_embeds=outputs)

        _patch_method(
            phi4_siglip_module.Model,
            "get_input_embeddings",
            patched_phi4_siglip_get_input_embeddings,
        )
        _PATCHED_ARCHES.add("phi4_siglip")

    try:
        phi4mm_module = importlib.import_module("mlx_vlm.models.phi4mm.phi4mm")
        phi4mm_vision_module = importlib.import_module("mlx_vlm.models.phi4mm.vision")
    except Exception:
        return

    patch_siglip_resize(phi4mm_vision_module)

    def patched_phi4mm_vision_tower_call(
        self,
        pixel_values,
        pixel_attention_mask=None,
        spatial_shapes=None,
    ):
        _, _, hidden_states = self.vision_tower(
            pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            output_hidden_states=True,
        )
        selected = hidden_states[self.select_layer]
        if spatial_shapes is not None:
            features = []
            for idx, (height, width) in enumerate(_size_pairs_to_tuple(spatial_shapes)):
                num_valid = int(height) * int(width)
                features.append(selected[idx, :num_valid])
            return features
        return selected

    def patched_phi4mm_get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        input_audio_embeds = kwargs.get("input_audio_embeds", None)
        audio_embed_sizes = _lengths_to_tuple(kwargs.get("audio_embed_sizes", None))
        audio_attention_mask = kwargs.get("audio_attention_mask", None)
        spatial_shapes = _size_pairs_to_tuple(kwargs.get("spatial_shapes", None))
        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)

        has_images = pixel_values is not None
        has_audio = input_audio_embeds is not None and input_audio_embeds.size > 0
        if has_images or has_audio:
            self.set_modality(has_image=has_images, has_audio=has_audio)

        if not has_images and not has_audio:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        image_features = None
        if has_images:
            cached = kwargs.get("cached_image_features", None)
            if cached is not None:
                image_features = cached
            else:
                image_features = self.vision_tower(
                    pixel_values,
                    pixel_attention_mask,
                    spatial_shapes,
                )
                image_features = self.apply_mm_projector(image_features)

        audio_features = None
        if has_audio:
            encoded_audio, _ = self.audio_encoder(
                input_audio_embeds, audio_attention_mask
            )
            audio_features = self.audio_projection(encoded_audio, mode="speech")

        safe_input_ids = input_ids
        if has_images:
            safe_input_ids = mx.where(
                safe_input_ids == self.config.image_token_index,
                mx.array(0),
                safe_input_ids,
            )
        if has_audio:
            safe_input_ids = mx.where(
                safe_input_ids == self.config.audio_token_index,
                mx.array(0),
                safe_input_ids,
            )
        text_embeds = self.language_model.model.embed_tokens(safe_input_ids)

        outputs = text_embeds
        if has_images:
            if isinstance(image_features, list):
                image_features = mx.concatenate(image_features, axis=0)
            outputs, _ = _merge_sequence_mask_features(
                input_ids == self.config.image_token_index,
                image_features,
                outputs,
            )
        if has_audio:
            audio_chunks = []
            if audio_embed_sizes is not None:
                for idx, embed_size in enumerate(audio_embed_sizes):
                    audio_chunks.append(audio_features[idx, : int(embed_size)])
            else:
                audio_chunks.append(audio_features.reshape((-1, audio_features.shape[-1])))
            merged_audio = mx.concatenate(audio_chunks, axis=0)
            outputs, _ = _merge_sequence_mask_features(
                input_ids == self.config.audio_token_index,
                merged_audio,
                outputs,
            )
        return InputEmbeddingsFeatures(inputs_embeds=outputs)

    _patch_method(phi4mm_vision_module.VisionTower, "__call__", patched_phi4mm_vision_tower_call)
    _patch_method(phi4mm_module.Model, "get_input_embeddings", patched_phi4mm_get_input_embeddings)
    _PATCHED_ARCHES.add("phi4mm")


@lru_cache(maxsize=1)
def list_compile_pattern_bundles() -> tuple[CompilePatternBundle, ...]:
    """Return the shared compile patch registry.

    This is the main maintainer entrypoint for adding new compile patterns.
    Matchers should prefer source traits over hardcoded architecture lists when
    the same fix is semantically reusable.
    """

    return (
        CompilePatternBundle(
            name="qwen_like_image_merge",
            description="Shared image-token merge patches for qwen-like VLM families.",
            matcher=lambda arch, report: (
                arch in {"glm_ocr", "paddleocr_vl", "qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
                or (
                    (arch.startswith("qwen") or arch in {"glm_ocr", "paddleocr_vl"})
                    and "qwen_like_image_merge" in report.pattern_traits
                )
            ),
            primitive_names=("compile_safe_feature_merge", "compile_safe_masked_scatter"),
            adapter_name="qwen_like_merge",
            protocol_names=("multimodal_merge", "masked_scatter"),
            runtime_primitive_names=("qwen_like_image_merge_runtime",),
        ),
        CompilePatternBundle(
            name="qwen2_vision_windowing",
            description="Qwen2.5-VL vision/window/position patch set.",
            matcher=lambda arch, report: (
                arch == "qwen2_5_vl"
                or (arch.startswith("qwen2") and "qwen2_vision_windowing" in report.pattern_traits)
            ),
            primitive_names=(
                "segmented_vision_attention",
                "vision_metadata_normalization",
                "explicit_position_plumbing",
                "cached_image_feature_plumbing",
                "dtype_normalization",
            ),
            adapter_name="qwen_like_merge",
            protocol_names=("vision_attention", "position_primitives", "cached_image_features"),
            runtime_primitive_names=("qwen2_vision_windowing_runtime",),
        ),
        CompilePatternBundle(
            name="qwen3_family_multimodal",
            description="Qwen3 VL family merge, deepstack, and vision patch set.",
            matcher=lambda arch, report: (
                arch in {"qwen3_vl", "qwen3_5", "qwen3_5_moe"}
                or (arch.startswith("qwen3") and "qwen3_deepstack_multimodal" in report.pattern_traits)
            ),
            primitive_names=(
                "compile_safe_feature_merge",
                "compile_safe_masked_scatter",
                "segmented_vision_attention",
                "vision_metadata_normalization",
                "explicit_position_plumbing",
                "cached_image_feature_plumbing",
                "dtype_normalization",
            ),
            adapter_name="qwen3_deepstack",
            protocol_names=(
                "multimodal_merge",
                "masked_scatter",
                "vision_attention",
                "position_primitives",
                "cached_image_features",
                "deepstack_process",
            ),
            runtime_primitive_names=("qwen3_family_multimodal_runtime",),
        ),
        CompilePatternBundle(
            name="single_image_token_merge",
            description="Single-image-token merge patch set for llama/pixtral/mistral wrappers.",
            matcher=lambda arch, report: (
                arch in {"aya_vision", "llama4", "llava", "mistral3", "pixtral"}
                or (
                    arch.startswith(("aya", "llama", "llava", "mistral", "pixtral"))
                    and "single_image_token_merge" in report.pattern_traits
                )
            ),
            primitive_names=("compile_safe_feature_merge", "cached_image_feature_plumbing"),
            adapter_name="single_image_token",
            protocol_names=("multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("single_image_token_merge_runtime",),
        ),
        CompilePatternBundle(
            name="mistral4_attention_backend",
            description="Mistral4 backend attention fix for compile-safe zero-RoPE cases.",
            matcher=lambda arch, report: (
                arch == "mistral4"
                or (arch.startswith("mistral4") and "mistral4_attention_backend" in report.pattern_traits)
            ),
            primitive_names=("segmented_vision_attention", "dtype_normalization"),
            adapter_name="mistral_attention_family",
            protocol_names=("vision_attention",),
            runtime_primitive_names=("mistral4_attention_backend_runtime",),
        ),
        CompilePatternBundle(
            name="gemma3n_multiscale_fusion",
            description="Gemma3n multiscale fusion and merge patch set.",
            matcher=lambda arch, report: (
                arch == "gemma3n" or "gemma3n_multiscale_fusion" in report.pattern_traits
            ),
            primitive_names=(
                "compile_safe_masked_scatter",
                "vision_metadata_normalization",
                "dtype_normalization",
            ),
            adapter_name="gemma_vision_family",
            protocol_names=("masked_scatter", "multimodal_merge"),
            runtime_primitive_names=("gemma3n_multiscale_fusion_runtime",),
        ),
        CompilePatternBundle(
            name="deepseek_ocr_multimodal",
            description="DeepSeek OCR SAM/projector/input-embedding patch set.",
            matcher=lambda arch, report: (
                arch in {"deepseekocr", "deepseekocr_2"}
                or (
                    arch.startswith("deepseekocr")
                    and "deepseek_ocr_multimodal" in report.pattern_traits
                )
            ),
            primitive_names=(
                "compile_safe_feature_merge",
                "segmented_vision_attention",
                "vision_metadata_normalization",
                "cached_image_feature_plumbing",
                "dtype_normalization",
            ),
            adapter_name="ocr_projector_family",
            protocol_names=("multimodal_merge", "vision_attention", "cached_image_features"),
            runtime_primitive_names=("deepseek_ocr_multimodal_runtime",),
        ),
        CompilePatternBundle(
            name="masked_scatter_multimodal",
            description="Shared compile-safe flattened masked-scatter replacement.",
            matcher=lambda arch, report: (
                arch in {"gemma3", "idefics2", "idefics3"}
                or "masked_scatter_multimodal" in report.pattern_traits
            ),
            primitive_names=("compile_safe_masked_scatter",),
            protocol_names=("masked_scatter",),
            runtime_primitive_names=("masked_scatter_multimodal_runtime",),
        ),
        CompilePatternBundle(
            name="padded_image_filtering",
            description="Shared compile-safe replacement for Python-side padded image filtering.",
            matcher=lambda arch, report: (
                arch in {"idefics2", "idefics3", "smolvlm"}
                or "padded_image_filtering" in report.pattern_traits
            ),
            primitive_names=(
                "padded_image_filtering",
                "compile_safe_feature_merge",
                "cached_image_feature_plumbing",
            ),
            adapter_name="idefics_family",
            protocol_names=("padded_image_filtering", "multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("padded_image_filtering_runtime",),
        ),
        CompilePatternBundle(
            name="negative_image_placeholders",
            description="Compile-safe eager metadata path for negative image placeholder families.",
            matcher=lambda arch, report: (
                arch == "phi3_v" or "negative_image_placeholders" in report.pattern_traits
            ),
            primitive_names=(
                "vision_metadata_normalization",
                "compile_safe_feature_merge",
                "cached_image_feature_plumbing",
            ),
            adapter_name="phi_placeholder_family",
            protocol_names=("placeholder_expansion", "multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("negative_image_placeholders_runtime",),
        ),
        CompilePatternBundle(
            name="expanded_image_placeholders",
            description="Compile-safe eager placeholder expansion for repeated image-token families.",
            matcher=lambda arch, report: (
                arch == "multi_modality"
                or "expanded_image_placeholders" in report.pattern_traits
            ),
            primitive_names=("compile_safe_feature_merge", "cached_image_feature_plumbing"),
            adapter_name="phi_placeholder_family",
            protocol_names=("placeholder_expansion", "multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("expanded_image_placeholders_runtime",),
        ),
        CompilePatternBundle(
            name="phi4_multimodal_spans",
            description="Phi4 multimodal patch set for SigLIP2 vision and expanded image/audio spans.",
            matcher=lambda arch, report: (
                arch in {"phi4_siglip", "phi4mm"}
                or "phi4_multimodal_spans" in report.pattern_traits
            ),
            primitive_names=(
                "vision_metadata_normalization",
                "compile_safe_feature_merge",
                "cached_image_feature_plumbing",
                "dtype_normalization",
            ),
            adapter_name="phi_placeholder_family",
            protocol_names=("placeholder_expansion", "multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("phi4_multimodal_spans_runtime",),
        ),
        CompilePatternBundle(
            name="glm_ocr_vision_compile",
            description="GLM OCR vision/merge compile patch set.",
            matcher=lambda arch, report: arch == "glm_ocr",
            primitive_names=(
                "segmented_vision_attention",
                "vision_metadata_normalization",
                "explicit_position_plumbing",
                "cached_image_feature_plumbing",
                "dtype_normalization",
            ),
            adapter_name="ocr_projector_family",
            protocol_names=("vision_attention", "multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("glm_ocr_vision_compile_runtime",),
        ),
        CompilePatternBundle(
            name="paddleocr_vl_multimodal",
            description="PaddleOCR-VL vision/projector/input-embedding compile patch set.",
            matcher=lambda arch, report: arch == "paddleocr_vl",
            primitive_names=(
                "segmented_vision_attention",
                "vision_metadata_normalization",
                "explicit_position_plumbing",
                "cached_image_feature_plumbing",
                "dtype_normalization",
            ),
            adapter_name="ocr_projector_family",
            protocol_names=("vision_attention", "multimodal_merge", "cached_image_features"),
            runtime_primitive_names=("paddleocr_vl_multimodal_runtime",),
        ),
    )


def _matching_pattern_bundle_names(
    arch: str,
    report: MLXVLMTraitReport | None = None,
) -> tuple[str, ...]:
    """Return bundle names whose matchers apply to one architecture report."""

    report = report or build_compile_trait_reports().get(arch)
    if report is None:
        return ()
    return tuple(
        bundle.name
        for bundle in list_compile_pattern_bundles()
        if bundle.matcher(arch, report)
    )


def _matching_pattern_bundles(
    arch: str,
    report: MLXVLMTraitReport | None = None,
) -> tuple[CompilePatternBundle, ...]:
    """Return full bundle objects whose matchers apply to one architecture."""

    report = report or build_compile_trait_reports().get(arch)
    if report is None:
        return ()
    return tuple(
        bundle
        for bundle in list_compile_pattern_bundles()
        if bundle.matcher(arch, report)
    )


def _resolve_compile_patch_plan(bundle: CompilePatternBundle) -> CompilePatchPlan:
    """Lower one declarative bundle into the runtime plan we actually execute."""

    return CompilePatchPlan(
        bundle_name=bundle.name,
        adapter_name=bundle.adapter_name,
        protocol_names=bundle.protocol_names,
        semantic_primitives=bundle.primitive_names,
        runtime_primitives=bundle.runtime_primitive_names or bundle.primitive_names,
    )


def _runtime_patch_primitive_installers() -> dict[str, Callable[[], None]]:
    """Return the runtime primitive executor registry.

    These runtime primitives are intentionally small semantic units that can be
    composed by bundle plans. Some still wrap heavier helper bodies, but the
    installation path itself no longer dispatches directly on architecture-
    local installer names.
    """

    return {
        "qwen_like_image_merge_runtime": _install_qwen_like_image_merge_patches,
        "qwen2_vision_windowing_runtime": lambda: (
            _install_qwen2_compile_patches(),
            _install_qwen2_5_compile_patches(),
        ),
        "qwen3_family_multimodal_runtime": _install_qwen3_family_compile_patches,
        "single_image_token_merge_runtime": _install_llama_pixtral_mistral_compile_patches,
        "mistral4_attention_backend_runtime": _install_mistral4_compile_patches,
        "gemma3n_multiscale_fusion_runtime": _install_gemma3n_compile_patches,
        "deepseek_ocr_multimodal_runtime": _install_deepseek_ocr_compile_patches,
        "masked_scatter_multimodal_runtime": _install_masked_scatter_multimodal_patches,
        "padded_image_filtering_runtime": _install_idefics_family_compile_patches,
        "negative_image_placeholders_runtime": _install_negative_image_placeholder_patches,
        "expanded_image_placeholders_runtime": _install_expanded_image_placeholder_patches,
        "phi4_multimodal_spans_runtime": _install_phi4_multimodal_patches,
        "glm_ocr_vision_compile_runtime": _install_glm_ocr_compile_patches,
        "paddleocr_vl_multimodal_runtime": _install_paddleocr_vl_compile_patches,
    }


def _apply_compile_patch_plan(plan: CompilePatchPlan) -> None:
    """Apply one resolved runtime patch plan.

    The plan executor is intentionally generic so new model families usually
    only need registry updates:
    - add or extend a trait matcher
    - point a bundle at an adapter
    - reuse or add runtime primitives
    """

    executors = _runtime_patch_primitive_installers()
    for primitive_name in plan.runtime_primitives:
        executor = executors.get(primitive_name)
        if executor is None:
            raise KeyError(
                f"Unsloth: unknown MLX compile runtime primitive {primitive_name!r} "
                f"for bundle {plan.bundle_name!r}"
            )
        executor()


def trace_patch_bindings(model_or_arch) -> tuple[tuple[str, str, str, str], ...]:
    """Return concrete runtime patch bindings applied for one architecture.

    This is intentionally lower-level than `trace_compile_application(...)`.
    It answers "what class/method names were actually monkey-patched?" so a
    maintainer can quickly diff current bindings against upstream changes.
    """

    arch = model_or_arch if isinstance(model_or_arch, str) else get_model_architecture(model_or_arch)
    if not arch:
        return ()
    arch_token = f"mlx_vlm.models.{arch}"
    return tuple(
        sorted(
            binding
            for binding in _PATCH_BINDINGS
            if arch_token in binding[1]
        )
    )


def explain_compile_failure(model_or_arch, compile_error=None) -> str:
    """Summarize why compile support is missing or fragile for a family."""

    decision = resolve_training_compile(model_or_arch)
    patchability = validate_compile_patchability(model_or_arch)
    lines = [
        f"arch={patchability.arch}",
        f"decision={'enabled' if decision.enabled else 'disabled'}",
        f"decision_reason={decision.reason}",
    ]
    if compile_error is not None:
        lines.append(f"compile_error={compile_error}")
    if patchability.matched_bundles:
        lines.append("matched_bundles=" + ",".join(patchability.matched_bundles))
    if patchability.direct_protocols:
        lines.append("direct_protocols=" + ",".join(patchability.direct_protocols))
    if patchability.inferred_protocols:
        lines.append("inferred_protocols=" + ",".join(patchability.inferred_protocols))
    if patchability.satisfied_protocols:
        lines.append("satisfied_protocols=" + ",".join(patchability.satisfied_protocols))
    if patchability.missing_protocols:
        lines.append("missing_protocols=" + ",".join(patchability.missing_protocols))
    if patchability.blockers:
        lines.append("blockers=" + "; ".join(patchability.blockers))
    bindings = trace_patch_bindings(model_or_arch)
    if bindings:
        lines.append(
            "patch_bindings="
            + ", ".join(f"{class_name}.{method_name}" for _, _, class_name, method_name in bindings[:12])
        )
    similar = find_similar_compile_families(model_or_arch, verified_only=True)
    if similar:
        lines.append(
            "similar_verified="
            + ", ".join(f"{arch}:{score:.2f}" for arch, score, _ in similar[:3])
        )
    return "\n".join(lines)


def diff_patchability(model_or_arch_a, model_or_arch_b) -> dict[str, Any]:
    """Compare compile patchability between two families or loaded models."""

    left = validate_compile_patchability(model_or_arch_a)
    right = validate_compile_patchability(model_or_arch_b)
    return {
        "left_arch": left.arch,
        "right_arch": right.arch,
        "left_bundles": left.matched_bundles,
        "right_bundles": right.matched_bundles,
        "left_direct_protocols": left.direct_protocols,
        "right_direct_protocols": right.direct_protocols,
        "left_inferred_protocols": left.inferred_protocols,
        "right_inferred_protocols": right.inferred_protocols,
        "left_protocols": left.satisfied_protocols,
        "right_protocols": right.satisfied_protocols,
        "only_left_protocols": tuple(sorted(set(left.satisfied_protocols) - set(right.satisfied_protocols))),
        "only_right_protocols": tuple(sorted(set(right.satisfied_protocols) - set(left.satisfied_protocols))),
        "left_missing_protocols": left.missing_protocols,
        "right_missing_protocols": right.missing_protocols,
        "left_blockers": left.blockers,
        "right_blockers": right.blockers,
    }


def trace_compile_application(
    model_or_arch,
    *,
    policy: MLXVLMCompilePolicy | None = None,
    args=None,
) -> CompileTraceReport:
    """Return a machine-readable trace of compile support and patch selection."""

    qualification = get_compile_qualification(model_or_arch)
    report = get_compile_trait_report(model_or_arch)
    decision = resolve_training_compile(model_or_arch, policy=policy, args=args)
    arch = decision.arch
    patch_mode = decision.patch_mode
    backend_arches = tuple(
        report.backend_arches if report is not None else ()
    )
    bundles = _matching_pattern_bundles(arch, report)
    patchability = validate_compile_patchability(model_or_arch if not isinstance(model_or_arch, str) else arch)
    matched_bundle_names = tuple(bundle.name for bundle in bundles)
    installed_bundles = tuple(
        bundle.name for bundle in bundles if bundle.name in _PATCHED_PATTERN_BUNDLES
    ) if patch_mode == "patched" else ()
    primitive_names = []
    runtime_primitive_names = []
    adapter_names = list(_adapter_names_for_arch(arch, report))
    for bundle in bundles:
        primitive_names.extend(bundle.primitive_names)
        runtime_primitive_names.extend(bundle.runtime_primitive_names or bundle.primitive_names)
        if bundle.adapter_name is not None:
            adapter_names.append(bundle.adapter_name)

    if patch_mode == "patched" and "safe_fused_sdpa_mask" in _PATCHED_PATTERN_BUNDLES:
        primitive_names.append("safe_fused_sdpa_mask")
        runtime_primitive_names.append("safe_fused_sdpa_mask")

    return CompileTraceReport(
        arch=arch,
        patch_mode=patch_mode,
        backend_arches=backend_arches,
        verification_state=qualification.verification_state if qualification is not None else "unqualified",
        support_state=decision.support_state,
        blocker_categories=qualification.blocker_categories if qualification is not None else (),
        pattern_traits=report.pattern_traits if report is not None else (),
        matched_bundles=matched_bundle_names,
        installed_bundles=installed_bundles,
        patch_primitives=tuple(dict.fromkeys(primitive_names)),
        runtime_primitives=tuple(dict.fromkeys(runtime_primitive_names)),
        adapters=tuple(dict.fromkeys(adapter_names)),
        qualification_reason=qualification.reason if qualification is not None else "architecture not discovered",
        decision_mode=decision.policy_mode,
        decision_enabled=decision.enabled,
        decision_reason=decision.reason,
        fallback_allowed=decision.fallback_allowed,
        strict_requested=decision.strict_requested,
        backend_qualification_arches=tuple(qual.arch for qual in decision.backend_qualifications),
        recommendations=decision.setting_recommendations,
        direct_protocols=patchability.direct_protocols,
        inferred_protocols=patchability.inferred_protocols,
        satisfied_protocols=patchability.satisfied_protocols,
        missing_protocols=patchability.missing_protocols,
    )


def explain_compile_support(
    model_or_arch,
    *,
    policy: MLXVLMCompilePolicy | None = None,
    args=None,
) -> str:
    """Return a human-readable summary of compile support and patch selection."""

    trace = trace_compile_application(model_or_arch, policy=policy, args=args)
    lines = [
        f"arch={trace.arch}",
        f"patch_mode={trace.patch_mode}",
        f"support_state={trace.support_state}",
        f"verification_state={trace.verification_state}",
        f"decision={trace.decision_mode}:{'enabled' if trace.decision_enabled else 'disabled'}",
        f"decision_reason={trace.decision_reason}",
    ]
    if trace.backend_arches:
        lines.append("backend_arches=" + ",".join(trace.backend_arches))
    if trace.backend_qualification_arches:
        lines.append(
            "backend_qualifications=" + ",".join(trace.backend_qualification_arches)
        )
    if trace.pattern_traits:
        lines.append("traits=" + ",".join(trace.pattern_traits))
    if trace.matched_bundles:
        lines.append("matched_bundles=" + ",".join(trace.matched_bundles))
    if trace.patch_primitives:
        lines.append("patch_primitives=" + ",".join(trace.patch_primitives))
    if trace.runtime_primitives:
        lines.append("runtime_primitives=" + ",".join(trace.runtime_primitives))
    if trace.adapters:
        lines.append("adapters=" + ",".join(trace.adapters))
    if trace.satisfied_protocols:
        lines.append("protocols=" + ",".join(trace.satisfied_protocols))
    if trace.direct_protocols:
        lines.append("direct_protocols=" + ",".join(trace.direct_protocols))
    if trace.inferred_protocols:
        lines.append("inferred_protocols=" + ",".join(trace.inferred_protocols))
    if trace.missing_protocols:
        lines.append("missing_protocols=" + ",".join(trace.missing_protocols))
    if trace.blocker_categories:
        lines.append("blockers=" + ",".join(trace.blocker_categories))
    if trace.recommendations:
        lines.append(
            "recommendations="
            + "; ".join(
                f"{rec.setting}={rec.recommended_value} ({rec.reason})"
                for rec in trace.recommendations
            )
        )
    return "\n".join(lines)


def install_mlx_compile_patches():
    """Install every shared compile patch bundle once and rebuild reports."""

    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return build_compile_qualifications()

    _install_safe_fused_sdpa_mask_patches()
    _PATCHED_PATTERN_BUNDLES.add("safe_fused_sdpa_mask")

    for bundle in list_compile_pattern_bundles():
        _apply_compile_patch_plan(_resolve_compile_patch_plan(bundle))
        _PATCHED_PATTERN_BUNDLES.add(bundle.name)

    _PATCHES_INSTALLED = True
    _invalidate_qualification_cache()
    return build_compile_qualifications()


def summarize_compile_qualifications() -> dict[str, int]:
    """Return a small summary snapshot of current compile qualification state."""

    qualifications = build_compile_qualifications().values()
    out = {
        "architectures": 0,
        "training_compile_ready": 0,
        "generation_compile_ready": 0,
        "patched": 0,
        "pattern_candidates": 0,
        "bundles_installed": len(_PATCHED_PATTERN_BUNDLES),
        "supported_verified": 0,
        "supported_inferred": 0,
        "patched_unverified": 0,
        "fallback_only": 0,
        "unsupported": 0,
    }
    for qual in qualifications:
        out["architectures"] += 1
        out["training_compile_ready"] += int(qual.training_compile)
        out["generation_compile_ready"] += int(qual.generation_compile)
        out["patched"] += int(qual.patched)
        out["pattern_candidates"] += int(qual.verification_state == "pattern_candidate")
        out[qual.support_state] += 1
    return out
