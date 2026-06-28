#!/usr/bin/env python3
"""
curriculum_trainer.py
=====================
Thin subclass of OPSDTrainer that enforces a **deterministic, pre-ordered data
schedule** AND records which curriculum stage each micro-batch came from so the
monitor callback can verify the schedule is respected. Curriculum is
DATA-ORDERING + INSTRUMENTATION ONLY — OPSD loss / generation / teacher logic is
inherited UNCHANGED (we always delegate to `super().training_step`).

Two overrides:
  1. `_get_train_sampler` -> SequentialSampler so the pre-built schedule (the
     dataset was physically reordered by train_opsd_curriculum.py) IS the order.
  2. `training_step`:
       - pop `stage_index` (REQUIRED) and `Answer` (optional, monitor-only),
       - `accelerator.gather` the stage indices so rank-0 sees the FULL global
         micro-batch (DDP shard-robust),
       - append to `self._stage_window` (accumulates across the
         gradient-accumulation micro-steps that make up ONE optimizer step),
       - **return `super().training_step(model, inputs, ...)` verbatim.**

The monitor callback drains `_stage_window` at on_step_end (one optimizer step),
computes modal/distinct/expected/respected, logs them, and resets the window.

DDP NOTE (the #1 debug point): with a SequentialSampler the HF/accelerate
dataloader shards the global order across ranks. If `stage_respected` ever drops
below 1.0 (esp. a persistent `distinct>1` on non-boundary steps), the shard is
interleaving stages — fix by switching to a rank-interleaved no-shuffle sampler.
Because we gather the ACTUAL stage_index column that travels with each example,
the monitor measures truth regardless of how the shard is laid out.
"""
from __future__ import annotations

from torch.utils.data import SequentialSampler

from opsd_trainer import OPSDTrainer  # from copied opsd_src/ (added to sys.path)

try:
    from accelerate.utils import gather_object
except Exception:  # pragma: no cover
    gather_object = None


def _extract_completion_texts(result):
    """Best-effort, read-only tap: find the list[str] of decoded completions in
    an OPSD generate() return tuple (it is the LAST element of
    generate_on_policy_outputs / _generate_on_policy_outputs_vllm). NEVER raises.
    """
    try:
        if isinstance(result, (tuple, list)):
            for item in reversed(result):
                if isinstance(item, list) and (
                    len(item) == 0 or isinstance(item[0], str)
                ):
                    return list(item)
    except Exception:
        pass
    return None



class CurriculumOPSDTrainer(OPSDTrainer):
    """OPSDTrainer + no-shuffle schedule order + per-step stage instrumentation."""

    STAGE_KEY = "stage_index"
    GOLD_KEY = "Answer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-optimizer-step window of (gathered) stage indices, drained by the
        # monitor callback at on_step_end. Also a buffer of gold answers + the
        # most recent surfaced completions for the (8B) reward proxy.
        self._stage_window: list[int] = []
        self._gold_window: list = []
        self._completion_window: list = []
        self._last_completions = None
        # reward proxy off by default; train_opsd_curriculum sets it from config.
        self.attach_gold = False

    # ---- order: schedule IS the order (no shuffle) ----
    def _get_train_sampler(self, *args, **kwargs):
        if self.train_dataset is None:
            return None
        return SequentialSampler(self.train_dataset)

    # ---- read-only completion taps (NO behavior change) ----
    # Both OPSD generate paths return completion_texts as the last tuple element;
    # we delegate verbatim and only STASH a copy for the monitor reward proxy.
    def generate_on_policy_outputs(self, *args, **kwargs):
        result = super().generate_on_policy_outputs(*args, **kwargs)
        if self.attach_gold:
            comps = _extract_completion_texts(result)
            if comps is not None:
                self._last_completions = comps
        return result

    def _generate_on_policy_outputs_vllm(self, *args, **kwargs):
        result = super()._generate_on_policy_outputs_vllm(*args, **kwargs)
        if self.attach_gold:
            comps = _extract_completion_texts(result)
            if comps is not None:
                self._last_completions = comps
        return result


    # ---- instrumentation + verbatim delegation ----
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Pop curriculum-only fields BEFORE the unchanged OPSD body sees them.
        stage_index = inputs.pop(self.STAGE_KEY, None)
        gold = inputs.pop(self.GOLD_KEY, None)

        if stage_index is not None:
            # Gather across DDP ranks so the window reflects the FULL global
            # micro-batch, not just this rank's shard.
            try:
                gathered = self.accelerator.gather(stage_index)
            except Exception:
                gathered = stage_index
            self._stage_window.extend(int(x) for x in gathered.detach().cpu().tolist())

        # OPSD loss/generation/teacher — UNCHANGED. The generate() taps above
        # populate self._last_completions during this call when attach_gold.
        self._last_completions = None
        out = super().training_step(model, inputs, num_items_in_batch)

        # ---- monitor-only reward proxy: gather completions + gold GLOBALLY so
        # they align with the (already global) stage window. No-op when smoke.
        if self.attach_gold:
            local_comps = self._last_completions or []
            local_gold = list(gold) if gold is not None else []
            if gather_object is not None:
                try:
                    g_comps = gather_object(local_comps)
                    g_gold = gather_object(local_gold)
                except Exception:
                    g_comps, g_gold = local_comps, local_gold
            else:
                g_comps, g_gold = local_comps, local_gold
            self._completion_window.extend(g_comps)
            self._gold_window.extend(g_gold)

        return out

