#!/usr/bin/env python3
"""
curriculum_monitor_manifest_once.py
==================================

Manifest-once curriculum monitor for boundary-mixed optimizer steps.

This file is intentionally separate from `curriculum_monitor.py`; the legacy
monitor keeps its single-stage-per-step abort guard.  Here, the source of truth
is an expected stage Counter per optimizer step, e.g. {"0": 21, "1": 11}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from transformers import TrainerCallback

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from grpo_train import reward_correctness
except Exception:  # pragma: no cover
    reward_correctness = None


def _counter_from_jsonable(obj) -> Counter:
    """Convert {"0": 21, "1": 11} or {0: 21} into Counter({0: 21, 1: 11})."""
    if obj is None:
        return Counter()
    return Counter({int(k): int(v) for k, v in dict(obj).items()})


class CurriculumManifestOnceMonitorCallback(TrainerCallback):
    def __init__(self, trainer, expected_stage_counters, attach_gold: bool = False):
        self.trainer = trainer
        self.expected_stage_counters = [_counter_from_jsonable(x) for x in expected_stage_counters]
        self.T = len(self.expected_stage_counters)
        self.attach_gold = attach_gold
        self.n_fail = 0

    def score_completions(self, stages, completions, gold):
        """Monitor-only reward proxy; mirrors the legacy callback behavior."""
        if not self.attach_gold or reward_correctness is None:
            return None
        n = min(len(stages), len(completions), len(gold))
        if n == 0:
            return None
        try:
            rewards = reward_correctness(list(completions[:n]), list(gold[:n]))
        except Exception:
            return None
        if not rewards:
            return None
        per: dict[int, list[float]] = {}
        for s, r in zip(stages[:n], rewards):
            per.setdefault(int(s), []).append(float(r))
        out = {
            "curriculum/rollout_acc": sum(rewards) / len(rewards),
            "curriculum/rollout_n": float(len(rewards)),
        }
        for s, vals in per.items():
            out[f"curriculum/rollout_acc_stage{s}"] = sum(vals) / len(vals)
        return out

    def on_step_end(self, args, state, control, **kwargs):
        win = getattr(self.trainer, "_stage_window", None)
        if not win:
            return control

        step_idx = int(state.global_step) - 1
        actual = Counter(int(x) for x in win)
        modal = actual.most_common(1)[0][0]
        distinct = len(actual)

        expected = None
        respected = None
        if 0 <= step_idx < self.T:
            expected = self.expected_stage_counters[step_idx]
            respected = 1.0 if actual == expected else 0.0
            if respected < 1.0:
                self.n_fail += 1
                print(
                    f"[MANIFEST-FAIL] step={step_idx} actual={dict(actual)} "
                    f"expected={dict(expected)} distinct={distinct} "
                    f"window_size={len(win)} (n_fail={self.n_fail})",
                    file=sys.stderr,
                    flush=True,
                )

        reward_log = self.score_completions(
            win,
            getattr(self.trainer, "_completion_window", []) or [],
            getattr(self.trainer, "_gold_window", []) or [],
        )

        if wandb is not None and getattr(wandb, "run", None) is not None \
                and self.trainer.accelerator.is_main_process:
            log = {
                "curriculum/stage_batch_modal": modal,
                "curriculum/stage_distinct": distinct,
                "curriculum/boundary_mixed_step": 1.0 if distinct > 1 else 0.0,
                "curriculum/stage_actual_counter_json": json.dumps(dict(sorted(actual.items()))),
            }
            if expected is not None:
                log["curriculum/stage_respected"] = respected
                log["curriculum/stage_expected"] = expected.most_common(1)[0][0]
                log["curriculum/stage_expected_counter_json"] = json.dumps(dict(sorted(expected.items())))
            if reward_log:
                log.update(reward_log)
            wandb.log(log)

        # In manifest-once mode, distinct>1 is allowed only if the exact Counter
        # matches.  Abort on mismatch to avoid wasting H200 hours — EXCEPT on the
        # final (tail) step where the HF Trainer may pad the partial batch with an
        # extra sample from the next SequentialSampler epoch; in that case we
        # warn but do not abort so trainer.save_model() can still run.
        is_tail_step = (step_idx == self.T - 1)
        if expected is not None and actual != expected:
            if is_tail_step:
                print(
                    f"[CURRICULUM-MANIFEST-WARN-TAIL] step={step_idx} actual={dict(actual)} "
                    f"expected={dict(expected)} — tail step, allowing (partial batch "
                    f"padding). n_fail={self.n_fail}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"[CURRICULUM-MANIFEST-ABORT] step={step_idx} actual={dict(actual)} "
                    f"expected={dict(expected)}. Aborting run."
                )

        self.trainer._stage_window = []
        if hasattr(self.trainer, "_gold_window"):
            self.trainer._gold_window = []
        if hasattr(self.trainer, "_completion_window"):
            self.trainer._completion_window = []
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer.accelerator.is_main_process:
            verdict = "PASS" if self.n_fail == 0 else f"FAIL ({self.n_fail} steps)"
            print(f"\n[CURRICULUM-MANIFEST-MONITOR] expected Counter gate: {verdict}\n",
                  file=sys.stderr, flush=True)
        return control