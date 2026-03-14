# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "PatchRLStatistics",
]

METRICS_MOVE_TO_END = [
    "nll",
    "aux",
    "beta",
    "alpha",
]

REMOVED_METRICS = [
    "num_tokens", # All extras - not necessary
    "mean_token_accuracy", # SFT extras
    "entropy",  # TRL >= 0.22.0
    "aux_loss", # TRL >= 0.23.0

    # GRPO extras
    "clip_ratio",
    'clip_ratio/low_mean',
    'clip_ratio/low_min',
    'clip_ratio/high_mean',
    'clip_ratio/high_max',
    'clip_ratio/region_mean',
    'frac_reward_zero_std',

    # Conditional metrics - only populated with vLLM importance sampling
    'sampling/sampling_logp_difference/mean',
    'sampling/sampling_logp_difference/max',
    'sampling/importance_sampling_ratio/min',
    'sampling/importance_sampling_ratio/mean',
    'sampling/importance_sampling_ratio/max',
]
REMOVED_METRICS = frozenset(REMOVED_METRICS)

import torch
try:
    from transformers.utils.notebook import (
        IntervalStrategy,
        NotebookTrainingTracker,
        NotebookProgressCallback,
    )
    HAS_NOTEBOOK = True
except:
    HAS_NOTEBOOK = False
pass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect
import os
import re
import functools


def NotebookProgressCallback_on_train_begin(Trainer_metrics):
    def _NotebookProgressCallback_on_train_begin(self, args, state, control, **kwargs):
        self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"
        self.training_loss = 0
        self.last_log = 0
        # Don't pre-create metric columns. Start with just the essentials;
        # columns are added dynamically by write_line as metrics actually appear.
        # This prevents empty "0 then blank" columns for conditional metrics
        # (kl when beta=0, sampling/* without importance sampling, etc.)
        column_names = [self.first_column] + ["Training Loss"]
        if args.eval_strategy != IntervalStrategy.NO:
            column_names.append("Validation Loss")
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)
    pass
    return _NotebookProgressCallback_on_train_begin
pass


def NotebookProgressCallback_on_log(Trainer_metrics):
    # Build an allowlist of known metrics (pre-extracted from TRL source).
    # Only these + dynamic rewards/* metrics pass through. This blocks
    # spurious keys injected at runtime (e.g. "train", "tools/*").
    set_Trainer_metrics = frozenset(Trainer_metrics)

    def _NotebookProgressCallback_on_log(self, args, state, control, logs = None, **kwargs):
        # Only for when there is no evaluation
        if args.eval_strategy == IntervalStrategy.NO and "loss" in logs:
            values = {}

            # 1) Pre-extracted metrics — only if actually present in logs
            for metric in Trainer_metrics:
                if metric in logs:
                    values[metric.replace("/", " / ")] = logs[metric]
            pass

            # 2) Dynamic per-reward-function metrics (rewards/*)
            #    These have user-defined names so can't be pre-extracted.
            #    Sort for stable column ordering across steps.
            dynamic_reward_keys = sorted(
                k for k in logs
                if k.startswith("rewards/") and k not in set_Trainer_metrics
            )
            for key in dynamic_reward_keys:
                display_key = key.replace("/", " / ")
                if display_key not in values:
                    values[display_key] = logs[key]
            pass

            # 3) Prepend Training Loss + Step (always first columns)
            values = {"Training Loss": logs["loss"], **values}
            values[self.first_column] = state.global_step
            self.training_tracker.write_line(values)
        pass
    pass
    return _NotebookProgressCallback_on_log
pass


def NotebookTrainingTracker_write_line(Trainer_metrics):
    def _NotebookTrainingTracker_write_line(self, values):
        """
        Write the values in the inner table.

        Args:
            values (`Dict[str, float]`): The values to display.
        """
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = self.inner_table[0]

            # Dynamically add new columns that appear in values
            # (e.g. per-reward-func metrics discovered at step > 1)
            for key in values:
                if key not in columns:
                    columns.append(key)
                    # Back-fill previous rows with empty string
                    for row in self.inner_table[1:]:
                        row.append("")
            pass

            self.inner_table[0] = columns
            first_column = columns[0]
            if len(self.inner_table) > 1:
                last_values = self.inner_table[-1]
                if last_values[0] != values[first_column]:
                    # write new line
                    self.inner_table.append([values[c] if c in values else "" for c in columns])
                else:
                    # update last line — preserve existing values for missing keys
                    new_values = values
                    for c in columns:
                        if c not in new_values:
                            new_values[c] = last_values[columns.index(c)]
                    self.inner_table[-1] = [new_values[c] for c in columns]
            else:
                # First data row (after header)
                self.inner_table.append([values[c] if c in values else "" for c in columns])
            pass
        pass
    pass
    return _NotebookTrainingTracker_write_line
pass


def _PatchRLStatistics(metrics, algorithm):
    if HAS_NOTEBOOK:
        if len(metrics) == 0: return
        from transformers.trainer import is_in_notebook
        if is_in_notebook():
            # Patch DPO notebook printing
            NotebookTrainingTracker.write_line = NotebookTrainingTracker_write_line(metrics)
            from transformers.trainer import DEFAULT_PROGRESS_CALLBACK
            DEFAULT_PROGRESS_CALLBACK.on_train_begin = NotebookProgressCallback_on_train_begin(metrics)
            DEFAULT_PROGRESS_CALLBACK.on_log         = NotebookProgressCallback_on_log(metrics)
        pass
    pass
pass


@functools.cache
def get_trl_metrics():
    # Gets metrics so we can output them in notebooks

    import trl.trainer
    trainers = dir(trl.trainer)
    trainers = [x for x in trainers if x.endswith("_trainer")]
    filepath = inspect.getfile(trl.trainer)
    filepath = os.path.split(filepath)[0]

    all_metrics = dict()
    for trainer in trainers:
        filename = os.path.join(filepath, f"{trainer}.py")
        if not os.path.exists(filename): continue
        with open(filename, "r", encoding = "utf-8") as file: file = file.read()

        # Get metrics['kl'] or stats['kl']
        metrics = re.findall(r"_?metrics\[[\"\']([^\"\']{1,})[\"\']\]", file)
        stats = re.findall(r"stats\[[\"\']([^\"\']{1,})[\"\']\]", file)
        metrics = metrics + stats

        # Get metrics[mode]['kl'] or stats[mode]['kl'] for new TRL
        metrics2 = re.findall(r"_?metrics\[mode\]\[[\"\']([^\"\']{1,})[\"\']\]", file)
        stats2 = re.findall(r"stats\[mode\]\[[\"\']([^\"\']{1,})[\"\']\]", file)
        metrics = metrics + metrics2 + stats2

        # Get optional f-strings (variable at start: f"{var}suffix")
        metrics_f = re.findall(r"_?metrics\[f[\"\']\{[^\}]{1,}\}([^\"\']{1,})[\"\']\]", file)
        stats_f = re.findall(r"stats\[f[\"\']\{[^\}]{1,}\}([^\"\']{1,})[\"\']\]", file)
        metrics_f = metrics_f + stats_f

        # Get optional f-strings for new TRL [mode] (variable at start)
        metrics_f2 = re.findall(r"_?metrics\[mode\]\[f[\"\']\{[^\}]{1,}\}([^\"\']{1,})[\"\']\]", file)
        stats_f2 = re.findall(r"stats\[mode\]\[f[\"\']\{[^\}]{1,}\}([^\"\']{1,})[\"\']\]", file)
        metrics_f = metrics_f + metrics_f2 + stats_f2

        # Get f-strings with literal prefix before variable: f"prefix/{var}/suffix"
        # Captures (prefix, suffix) pairs — these are dynamic columns we can't
        # pre-create, but we note them so the table can add them at runtime.
        metrics_f3 = re.findall(r"_?metrics(?:\[mode\])?\[f[\"\']([^\"\']*?)\{[^\}]{1,}\}([^\"\']*?)[\"\']\]", file)
        stats_f3 = re.findall(r"stats(?:\[mode\])?\[f[\"\']([^\"\']*?)\{[^\}]{1,}\}([^\"\']*?)[\"\']\]", file)
        # These are dynamic (contain runtime variable names) so we can't add them
        # as fixed columns. The write_line and on_log patches handle them dynamically.

        # Filter out prefixes if seen
        # metrics[f"{prefix}rewards/chosen"]
        left_prefix = 'prefix = "eval_" if train_eval == "eval" else ""' in file
        if left_prefix: metrics += metrics_f

        # Move all eval_ things to the end and reward to the front
        beginning = []
        middle = []
        end = []
        for x in metrics:
            lowered = x.lower()
            if "reward" in lowered:
                beginning.append(x)
            elif x.lower().startswith("eval"):
                end.append(x)
            else:
                # Check if we want to move to the end
                moved = False
                for move_end in METRICS_MOVE_TO_END:
                    if move_end in lowered:
                        end.append(x)
                        moved = True
                        break
                if not moved:
                    middle.append(x)
            pass
        pass
        metrics = beginning + middle + end

        metrics = [x for x in metrics if x not in REMOVED_METRICS]
        metrics = list(dict().fromkeys(metrics).keys())

        all_metrics[trainer] = metrics
    pass
    return all_metrics
pass


def PatchRLStatistics(algorithm = "grpo_trainer", other_metrics = []):
    # Get notebook statistics columns to show up
    all_metrics = get_trl_metrics()
    if algorithm not in all_metrics:
        print(
            f"Unsloth for {algorithm.upper()} is not yet implemented! Just ignore this function.\n"\
            f"We support: `{list(all_metrics.keys())}`"
        )
        return
    pass
    _PatchRLStatistics(all_metrics[algorithm] + other_metrics, algorithm)
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
