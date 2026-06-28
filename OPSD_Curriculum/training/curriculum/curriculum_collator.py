#!/usr/bin/env python3
"""
curriculum_collator.py
======================
Thin subclass of the upstream OPSD `SelfDistillationDataCollator`.

Curriculum is **data-ordering / instrumentation ONLY**. This collator changes
NOTHING about how student/teacher prompts are built or tokenized — it just calls
`super().__call__(features)` and *attaches* two passthrough fields so the trainer
can (a) verify the curriculum schedule is being respected and (b) optionally
score generations against gold answers (reward proxy, 8B only):

  * `batch["stage_index"]`  : LongTensor [batch] — the curriculum stage each
                              example was scheduled into (added as a dataset
                              column by train_opsd_curriculum.py). REQUIRED.
  * `batch["Answer"]`       : list[str|None] [batch] — gold answers, attached
                              ONLY when attach_gold=True (monitor-only reward
                              proxy; OFF for smoke per reward_proxy option C).

Both keys are popped by CurriculumOPSDTrainer.training_step BEFORE the inputs
reach the (unchanged) OPSD loss/generation body, so upstream logic is untouched.

Requires `training_args.remove_unused_columns = False` so these columns survive
into `features` (upstream opsd_src has no such flag → train script forces it).
"""
from __future__ import annotations

import torch

from data_collator import SelfDistillationDataCollator  # from opsd_src/ (on sys.path)


class CurriculumDataCollator(SelfDistillationDataCollator):
    """OPSD collator + curriculum passthrough (stage_index, optional gold Answer)."""

    STAGE_KEY = "stage_index"
    GOLD_KEY = "Answer"

    def __init__(self, *args, attach_gold: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.attach_gold = attach_gold

    def __call__(self, features):
        # 1) Upstream OPSD collation — fully unchanged.
        batch = super().__call__(features)

        # 2) stage_index passthrough (REQUIRED; hard-fail if missing so we never
        #    silently train an un-instrumented run that the gate can't verify).
        if self.STAGE_KEY not in features[0]:
            raise KeyError(
                f"CurriculumDataCollator: feature missing '{self.STAGE_KEY}'. "
                "Did train_opsd_curriculum.py add the stage_index column AND set "
                "training_args.remove_unused_columns=False?"
            )
        batch[self.STAGE_KEY] = torch.tensor(
            [int(f[self.STAGE_KEY]) for f in features], dtype=torch.long
        )

        # 3) optional gold passthrough (monitor-only reward proxy; smoke=OFF).
        if self.attach_gold:
            batch[self.GOLD_KEY] = [f.get(self.GOLD_KEY) for f in features]

        return batch
