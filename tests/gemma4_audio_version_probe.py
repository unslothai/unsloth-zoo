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

"""Does Gemma 4 audio training work on the installed mlx-vlm?

Re-measures the `gemma4` row of `_AUDIO_QUALIFIED_FAMILIES`, on real Apple
Silicon against a real checkpoint. That row cannot be reasoned about: the
0.4.4 it used to carry was correct when written and was invalidated by the
checkpoint being re-exported on the Hub. A version key cannot express a fact
about an artefact, so when the answer is in doubt, run this rather than argue.

It deliberately opens the gate for whatever mlx-vlm is installed, because the
question is what happens behind the gate, not what the gate says.

Loading goes through `FastMLXModel.from_pretrained`, not `mlx_vlm.load`,
because that is the path a user takes and zoo already repairs things upstream
gets wrong on the way -- `_ensure_audio_conv_sanitize` undoes mlx-vlm 0.6.4's
double transpose of the audio convs on pre-converted checkpoints. Probing raw
mlx-vlm would report failures zoo does not have.

Stage 0 does load raw mlx-vlm anyway, and never counts towards the verdict.
It is there to separate "upstream is broken here" from "zoo is broken here",
which is the difference between filing a bug and fixing one.

Each stage reports on its own, and a stage whose prerequisites failed is
skipped rather than errored, so a red cell names where that version breaks
without a single early failure hiding everything after it:

  0. mlx-vlm alone loads it       -- diagnostic only, never the verdict
  1. zoo loads it                 -- processor and weights, through zoo
  2. the model has an audio tower
  3. placeholders match the tower -- the invariant the gate protects: a clip's
                                     placeholder count must equal the positions
                                     the audio encoder emits, or the merge has
                                     nothing to put behind the surplus
  4. audio reaches the loss       -- distinct audio must give distinct losses,
                                     by more than this platform's own
                                     run-to-run spread, or the model is
                                     training on text alone and a drifting
                                     machine is supplying the difference

Run: python tests/gemma4_audio_version_probe.py [--model REPO]
Exit code 0 only if every stage that counts passed.
"""

import argparse
import json
import os
import sys
import traceback

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DURATIONS = (0.5, 1.0, 2.0, 3.7)
RATE = 16000

results = {}

# How far clear of the platform's own spread the two-tone gap has to land.
#
# Measured on macos-14, mlx-vlm 0.6.3, mlx 0.32.0, three separate runs of this
# probe against mlx-community/gemma-4-e2b-it-4bit:
#
#   run   signal (440 vs 1760)   noise (spread of three identical 440s)
#     1   0.812                  2.27
#     2   1.30                   0.827
#     3   0.719                  1.25
#
# The spread of three identical inputs exceeds the gap between two different
# tones. So on Apple Silicon this stage cannot separate connected audio from
# disconnected audio at any margin, and no choice of this constant rescues it;
# the honest report there is "could not measure", which is what Inconclusive
# is for. On Linux the same probe gives noise=0 and signal=0.327, and the
# stage decides normally.
#
# The margin only has to be large enough that jitter cannot masquerade as
# signal on a platform where the two are separable at all.
TWO_TONE_MARGIN = 4.0

TWO_TONE_REASONS = {
    "same_loss": "two different tones gave the same loss, so the audio is "
                 "not reaching the objective",
    "below_noise": "the two tones differ by no more than this platform's own "
                   "run-to-run noise, so this cannot say whether the audio "
                   "reached the objective",
}


def two_tone_verdict(repeats, other):
    """Did the second tone move the loss by more than the platform's jitter?

    `repeats` are losses for one tone run several times, so their spread is
    what this machine hands you for free with no audio involved; `other` is
    the second tone. Split out from the stage so it can be tested without a
    model, a checkpoint or a Mac -- which matters, because the branch that
    made this necessary only fires on hardware where the model is not
    reproducible.
    """
    noise = max(repeats) - min(repeats)
    # The mean, not the first sample: three forwards were paid for, and the
    # figure printed here is the one a reader will compare against `other`.
    base = sum(repeats) / len(repeats)
    signal = abs(base - other)
    # The repeats themselves, not just their spread. Jitter and a monotone
    # drift across sequential forwards produce the same spread and call for
    # opposite responses -- one is a platform fact to measure around, the
    # other is a bug in the forward path -- and this string is the only
    # artifact the macOS matrix leaves behind.
    detail = (f"440Hz={base:.4f} 1760Hz={other:.4f} "
              f"signal={signal:.3g} noise={noise:.3g} "
              f"repeats=[{', '.join(f'{r:.4f}' for r in repeats)}]")
    if signal < 1e-6:
        return "same_loss", detail
    if noise > 0.0 and signal <= TWO_TONE_MARGIN * noise:
        return "below_noise", detail
    return "pass", detail


class Inconclusive(Exception):
    """The stage could not decide, as distinct from deciding against.

    This matrix exists so that a red cell names the mlx-vlm version that
    breaks Gemma 4 audio. A stage that could not measure anything is not that,
    and reporting it as a failure would blame a version for a property of the
    machine it ran on. It still does not qualify the version -- nothing was
    demonstrated -- so it is not a pass either.
    """


def gating_stages(stages):
    """The stages whose result decides this version's exit code.

    Stage 0 and its subchecks are diagnostic and never gated. Neither does a
    stage that could not measure anything: this matrix exists to name the
    mlx-vlm version that breaks Gemma 4 audio, and on Apple Silicon stage 4
    cannot answer at all (see TWO_TONE_MARGIN). Failing the job over that
    would report every version as broken on the one platform the feature
    ships to. The measured verdict still appears in PROBE_RESULT, and stages
    1 to 3, which are what actually decide the version boundary, still gate.
    """
    return {k: v for k, v in stages.items()
            if not k.startswith("0_") and not v.get("inconclusive")}


def stage(name):
    def wrap(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k)
                results[name] = {"ok": True, "detail": detail}
                print(f"[PASS] {name}: {detail}", flush=True)
                return True
            except Inconclusive as exc:
                results[name] = {
                    "ok": False, "inconclusive": True, "error": str(exc),
                }
                print(f"[INCONCLUSIVE] {name}: {exc}", flush=True)
                return False
            except Exception as exc:
                results[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:],
                }
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
                print(traceback.format_exc()[-1500:], flush=True)
                return False
        return run
    return wrap


def tone(seconds, hertz):
    import numpy as np
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (0.5 * np.sin(2.0 * np.pi * hertz * t)).astype(np.float32)


def resolve(repo):
    """Download once, so every stage loads from one snapshot."""
    from mlx_vlm.utils import get_model_path
    path = get_model_path(repo)
    return path[0] if isinstance(path, tuple) else path   # 0.6.x returns a pair


@stage("0_mlx_vlm_alone_loads")
def load_upstream(path):
    """Diagnostic. Reported, never part of the verdict.

    Splits the processor from the weights because they fail for unrelated
    reasons, and reports both rather than stopping at the first.
    """
    from mlx_vlm.utils import load_model, load_processor
    notes = []
    for what, fn in (("processor", load_processor), ("weights", load_model)):
        try:
            fn(path)
            notes.append(f"{what} ok")
            results[f"0_mlx_vlm_alone_loads.{what}"] = {"ok": True}
        except Exception as exc:
            first = str(exc).strip().splitlines()[0][:120]
            notes.append(f"{what} {type(exc).__name__}: {first}")
            # A failed subcheck too, so the JSON says so without parsing prose.
            results[f"0_mlx_vlm_alone_loads.{what}"] = {
                "ok": False, "error": f"{type(exc).__name__}: {first}",
            }
    return "; ".join(notes)


@stage("1_zoo_loads_it")
def load_via_zoo(path):
    from unsloth_zoo.mlx.loader import FastMLXModel
    global _MODEL, _PROCESSOR
    _MODEL, _PROCESSOR = FastMLXModel.from_pretrained(
        model_name=str(path), max_seq_length=512,
    )
    return f"{type(_MODEL).__name__} + {type(_PROCESSOR).__name__}"


@stage("2_model_has_audio_tower")
def check_tower(_repo):
    model = _MODEL
    tower = getattr(model, "audio_tower", None)
    embed = getattr(model, "embed_audio", None)
    if tower is None or embed is None:
        raise AssertionError("no audio_tower/embed_audio on this checkpoint")
    return "audio_tower and embed_audio present"


@stage("3_placeholders_match_the_audio_tower")
def check_alignment(_repo):
    """The invariant the gate exists to protect.

    zoo derives the count from the installed extractor precisely because the
    framing arithmetic changed in 0.5.0; this checks the derivation still
    agrees with what the tower emits on this version.

    "What the tower emits" has to come from the tower. Deriving it from
    `_gemma4_audio_encoder_positions` instead would re-run the arithmetic
    `_gemma4_audio_placeholder_count` already ran, and assert a number against
    itself -- green on any release, including one whose subsampling changed
    underneath it, which is the whole failure this stage exists to catch. So
    the clip goes through `audio_tower` and the count comes off the mask it
    returns.
    """
    import mlx.core as mx
    import numpy as np

    from unsloth_zoo.mlx.utils import (
        _gemma4_audio_frame_mask, _gemma4_audio_placeholder_count,
    )

    processor = _PROCESSOR
    extractor = getattr(processor, "feature_extractor", None)
    if extractor is None:
        raise AssertionError("processor exposes no feature_extractor")
    tower = getattr(_MODEL, "audio_tower", None)
    if tower is None:
        raise AssertionError("no audio_tower to measure against")

    def tower_positions(wav):
        """Run the encoder and count the positions it reports as real."""
        features = extractor(
            [np.asarray(wav, dtype=np.float32)],
            sampling_rate=RATE,
            return_attention_mask=True,
        )
        mel = mx.array(np.asarray(features["input_features"], dtype=np.float32))
        raw_mask = features.get("input_features_mask")
        if raw_mask is None:
            valid = np.ones(mel.shape[:2], dtype=bool)
        else:
            valid = np.asarray(raw_mask).astype(bool)
        # The tower takes a PADDING mask, not a validity one: gemma4.py passes
        # `~input_features_mask` and gets a mask back in the same polarity. So
        # feed the inverse and count the False positions.
        encodings, out_mask = tower(mel, mx.array(~valid))
        mx.eval(encodings, out_mask)
        emitted = int((~np.asarray(out_mask)[0].astype(bool)).sum())
        return emitted, int(encodings.shape[1])

    rows = []
    for secs in DURATIONS:
        wav = tone(secs, 440.0)
        counted = _gemma4_audio_placeholder_count(processor, wav, RATE)
        # mlx-vlm's own count: reported, never asserted on, since it runs high.
        native = getattr(processor, "_compute_audio_num_tokens", None)
        try:
            native_count = None if native is None else int(native(wav, RATE))
        except Exception as exc:
            native_count = f"n/a ({type(exc).__name__})"
        frames = int(np.asarray(_gemma4_audio_frame_mask(extractor, wav, RATE)).sum())
        emitted, width = tower_positions(wav)
        if counted != emitted:
            raise AssertionError(
                f"{secs}s: zoo counts {counted} placeholders, tower emits "
                f"{emitted} valid of {width}")
        rows.append(f"{secs}s: frames={frames} placeholders={counted} "
                    f"tower={emitted}/{width}"
                    + (f" native={native_count}" if native_count is not None else ""))
    return "; ".join(rows)


@stage("4_audio_reaches_the_loss")
def check_loss(_repo):
    """Distinct audio must produce distinct losses, by more than the noise.

    A processor that accepts the argument and drops it, or a merge that lands
    features on the wrong positions, shows up here: the loss stops depending
    on what the audio actually was.

    "Distinct" only means something where the same input twice gives the same
    loss, and on Apple Silicon this model does not: two identical uncached
    forwards of Gemma 4 E2B differ, and a bisect puts the first differing
    decoder layer at 0, upstream of every KV-shared layer. Against that, a
    bare `!=` is satisfied by the noise whether or not the audio is connected,
    so it would pass on a checkpoint whose audio is entirely disconnected.

    So measure the noise first, by running one tone several times, and require
    the two-tone gap to clear it. On a platform where the model is
    reproducible the floor is 0 and this is the original test.
    """
    import numpy as np

    from unsloth_zoo.mlx.utils import (
        install_audio_merge_patch, make_vlm_baseline_loss_fn,
        remove_audio_merge_patch,
    )

    model, processor = _MODEL, _PROCESSOR
    audio_token_id = getattr(getattr(model, "config", None), "audio_token_id", None)
    if audio_token_id is None:
        raise AssertionError("no audio_token_id on the model config")

    held = install_audio_merge_patch(model, audio_token_id)
    try:
        from unsloth_zoo.mlx.utils import (
            _collate_vlm_batch, _finalize_vlm_batch,
        )

        def loss_at(hz):
            # The decoded-column shape datasets.Audio yields, the only one
            # collation accepts.
            clip = {"array": tone(1.0, hz), "sampling_rate": RATE}
            messages = [{"role": "user", "content": [
                {"type": "audio", "audio": clip},
                {"type": "text", "text": "Transcribe."}]},
                {"role": "assistant", "content": "ok"}]
            # Collate on the host, then finalize to MLX, as the trainer does.
            staged = _collate_vlm_batch(
                [{"messages": messages}], processor, 512, None)
            batch = _finalize_vlm_batch(staged)
            loss_fn = make_vlm_baseline_loss_fn(model, ignore_token_ids=[])
            out = loss_fn(model, batch)
            loss = float(out[0] if isinstance(out, tuple) else out)
            if not np.isfinite(loss):
                raise AssertionError(f"loss is not finite at {hz} Hz: {loss}")
            return loss

        # The same tone several times: whatever these disagree by is what the
        # platform hands you for free, with no audio involved. Three rather
        # than two because one pair is a single sample of a spread and can
        # land small by luck, which would put the floor back where it started.
        repeats = [loss_at(440.0) for _ in range(3)]
        verdict, detail = two_tone_verdict(repeats, loss_at(1760.0))
        if verdict == "below_noise":
            raise Inconclusive(f"{TWO_TONE_REASONS[verdict]} [{detail}]")
        if verdict != "pass":
            raise AssertionError(f"{TWO_TONE_REASONS[verdict]} [{detail}]")
        return detail
    finally:
        if held:
            remove_audio_merge_patch(model)


def main():
    # Set here rather than at import: the stage helpers below are worth
    # importing on their own from a test, and importing a module should not
    # rewrite the environment of whatever imported it. Both are read by
    # unsloth_zoo, which is first imported inside the stages that main() runs.
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    import mlx_vlm
    import transformers
    version = getattr(mlx_vlm, "__version__", "unknown")
    print(f"=== gemma4 audio probe: mlx-vlm {version}, "
          f"transformers {transformers.__version__}, {args.model} ===",
          flush=True)

    # Open the gate for whatever is installed: the question is what is behind it.
    from unsloth_zoo.mlx import utils as U
    U._AUDIO_QUALIFIED_FAMILIES = dict(
        U._AUDIO_QUALIFIED_FAMILIES,
        gemma4=U._AudioVersions(version, version),
    )
    U._AUDIO_MIN_TRANSFORMERS = {}

    # Stage 0 never gates. Later stages need zoo's load, so they skip rather
    # than error when it fails.
    path = resolve(args.model)
    load_upstream(path)
    ok_zoo = load_via_zoo(path)

    def skip(name):
        results[name] = {"ok": False, "skipped": True,
                         "error": "prerequisite stage failed"}
        print(f"[SKIP] {name}: prerequisite stage failed", flush=True)

    if ok_zoo:
        ok_tower = check_tower(path)
        check_alignment(path)
    else:
        skip("2_model_has_audio_tower")
        skip("3_placeholders_match_the_audio_tower")
        ok_tower = False

    if ok_tower:
        check_loss(path)
    else:
        skip("4_audio_reaches_the_loss")

    print("PROBE_RESULT " + json.dumps(
        {"mlx_vlm": version, "transformers": transformers.__version__,
         "model": args.model, "stages": results}), flush=True)
    sys.exit(0 if all(r["ok"] for r in gating_stages(results).values()) else 1)


if __name__ == "__main__":
    main()
