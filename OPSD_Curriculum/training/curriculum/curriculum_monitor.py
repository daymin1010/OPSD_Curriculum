#!/usr/bin/env python3
"""
curriculum_monitor.py
=====================
TrainerCallback that, at the end of EACH optimizer step, drains the trainer's
`_stage_window` (the gathered, global stage_index of every example consumed in
that step's micro-batches) and verifies it matches the curriculum schedule.

Logged to wandb each step:
  curriculum/stage_batch_modal : most-common stage in the just-finished step
  curriculum/stage_expected    : schedule's stage for this opt-step (source of
                                 truth = derive_stage_maps(meta, B_glob))
  curriculum/stage_respected   : 1.0 if modal == expected else 0.0
  curriculum/stage_distinct    : # of distinct stages in the step's window
                                 (should be 1 except across a stage boundary)

If modal != expected, prints a `[SMOKE-FAIL]` line to stderr. The smoke gate is
`stage_respected == 1.0` for ALL steps (ws=4). A persistent distinct>1 on
non-boundary steps == DDP shard interleaving (see curriculum_trainer.py note).

reward_proxy (option C): a scorer STUB is included and is a NO-OP when
attach_gold=False (smoke). When enabled (8B), wire `score_completions` to the
surfaced completions + drained gold (monitor-only; OPSD generation untouched).
"""
from __future__ import annotations

import sys
from collections import Counter

from transformers import TrainerCallback

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    # opsd_src/ on PYTHONPATH: boxed-answer extraction + math_verify (+ MCQ
    # string fallback). Returns a list of 1.0/0.0 per (completion, gold).
    from grpo_train import reward_correctness
except Exception:  # pragma: no cover
    reward_correctness = None



class CurriculumMonitorCallback(TrainerCallback):
    def __init__(self, trainer, stage_per_optstep, attach_gold: bool = False):
        self.trainer = trainer
        self.stage_per_optstep = list(stage_per_optstep)
        self.T = len(self.stage_per_optstep)
        self.attach_gold = attach_gold
        self.n_fail = 0

    # ---- reward proxy (option C): per-stage rollout accuracy ------------
    def score_completions(self, stages, completions, gold):
        """Monitor-only reward proxy. NO-OP for smoke (attach_gold=False).

        When enabled (8B), scores the OPSD-surfaced student completions against
        gold via opsd_src.reward_correctness (extract_boxed_answer + math_verify
        + MCQ string fallback) and aggregates accuracy overall and per stage for
        the just-finished optimizer step. Read-only: NEVER touches the OPSD loss
        or generation path. Returns a wandb-loggable dict or None.

        `stages`, `completions`, `gold` are the per-step GLOBAL windows; they
        accumulate micro-batch order identically, so index i aligns across all
        three.
        """
        if not self.attach_gold or reward_correctness is None:
            return None
        n = min(len(stages), len(completions), len(gold))
        if n == 0:
            return None
        comps = list(completions[:n])
        golds = list(gold[:n])
        try:
            rewards = reward_correctness(comps, golds)
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


    # ---- per optimizer step --------------------------------------------
    def on_step_end(self, args, state, control, **kwargs):
        win = getattr(self.trainer, "_stage_window", None)
        if not win:
            return control

        # HF increments global_step BEFORE on_step_end -> completed step is
        # 0-based index (global_step - 1).
        step_idx = int(state.global_step) - 1
        modal = Counter(win).most_common(1)[0][0]
        distinct = len(set(win))

        expected = None
        respected = None
        if 0 <= step_idx < self.T:
            expected = self.stage_per_optstep[step_idx]
            respected = 1.0 if modal == expected else 0.0
            if respected < 1.0:
                self.n_fail += 1
                print(
                    f"[SMOKE-FAIL] step={step_idx} modal={modal} "
                    f"expected={expected} distinct={distinct} "
                    f"window_size={len(win)} (n_fail={self.n_fail})",
                    file=sys.stderr, flush=True,
                )

        # reward proxy (no-op for smoke; per-stage rollout_acc when attach_gold)
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
            }
            if expected is not None:
                log["curriculum/stage_expected"] = expected
                log["curriculum/stage_respected"] = respected
            if reward_log:
                log.update(reward_log)
            # NOTE: no explicit step= — the OPSD/GOLD trainer advances wandb's
            # internal step during generation, so forcing step=global_step gets
            # silently dropped ("less than current step"). Commit at wandb's own
            # cadence so the curriculum curves actually render.
            wandb.log(log)

        # ABORT GUARD: each optimizer step's global window must be a SINGLE stage
        # (the schedule pads every opt-step to B_glob examples of one stage). A
        # distinct>1 window therefore means DDP sharding scrambled the schedule
        # ordering (the ws=4->2 sharding pattern is unverified). Raise immediately
        # so the run dies in the first step(s) instead of wasting GPU-hours.
        # NOTE: modal!=expected with distinct==1 (possible off-by-one) is logged
        # only (SMOKE-FAIL above), NOT aborted.
        if distinct > 1 and 0 <= step_idx < self.T:
            raise RuntimeError(
                f"[CURRICULUM-ABORT] step={step_idx} stage_distinct={distinct}>1 "
                f"window={dict(Counter(win))} expected={expected} — DDP sharding "
                f"scrambled the curriculum schedule. Aborting run."
            )

        # drain windows for the next optimizer step

        self.trainer._stage_window = []
        if hasattr(self.trainer, "_gold_window"):
            self.trainer._gold_window = []
        if hasattr(self.trainer, "_completion_window"):
            self.trainer._completion_window = []
        return control


    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer.accelerator.is_main_process:
            verdict = "PASS" if self.n_fail == 0 else f"FAIL ({self.n_fail} steps)"
            print(f"\n[CURRICULUM-MONITOR] stage_respected gate: {verdict}\n",
                  file=sys.stderr, flush=True)
        return control
