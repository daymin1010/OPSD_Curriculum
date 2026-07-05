#!/usr/bin/env python3
"""
train_opsd_curriculum_manifest_once.py
========================
Curriculum entrypoint = OPSD self-distillation with a DETERMINISTIC, PRE-ORDERED
data schedule + per-step stage instrumentation. Mirrors opsd_src/opsd_train.py
exactly for everything except:

  (1) builds a curriculum schedule (curriculum_schedule.build_schedule) and
      physically reorders the OPSD train split into that order, attaching a
      `stage_index` column (derive_stage_maps -> stage_per_pos);
  (2) forces `training_args.remove_unused_columns = False` so stage_index (+ gold
      Answer when attach_gold) survive into the collator (upstream has no flag);
  (3) injects CurriculumDataCollator (super().__call__ + stage_index passthrough);
  (4) uses CurriculumOPSDTrainer (SequentialSampler => schedule IS the order;
      training_step pops stage_index, gathers across ranks, delegates to super);
  (5) sets max_steps = T (exactly one pass over the T*B_glob schedule);
  (6) adds CurriculumMonitorCallback (stage_respected gate, source-of-truth =
      derive_stage_maps(meta, B_glob)[1]).

OPSD loss / generation / teacher logic is INHERITED UNCHANGED. Run via
`accelerate launch` with PYTHONPATH including opsd_src/ AND this curriculum dir.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from transformers import AutoTokenizer, GenerationConfig
from trl import (
    LogCompletionsCallback,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

# opsd_src/ (on PYTHONPATH)
from opsd_train import CustomScriptArguments

# curriculum/ (this dir, on PYTHONPATH)
from curriculum_schedule import OUT_DIR
from curriculum_schedule_manifest_once import build_schedule_from_stage_manifest, save_manifest_schedule_meta
from curriculum_collator import CurriculumDataCollator
from curriculum_trainer import CurriculumOPSDTrainer
from curriculum_monitor_manifest_once import CurriculumManifestOnceMonitorCallback
from opsd_data import load_opsd_train

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CurriculumScriptArguments(CustomScriptArguments):
    """OPSD script args + curriculum knobs."""

    arm: str = field(default="manifest", metadata={"help": "Run label only (e.g. cond2_diff or cond3_ours_C2)."})
    stages_json: str = field(default=None, metadata={"help": "Path to the arm's stages JSON."})
    row_table: str = field(
        default=str(OUT_DIR / "join_setA_rows.parquet"),
        metadata={"help": "Phase-0 per-row join table (opsd_index, stage_index_*, in_setA)."},
    )
    curriculum_T: int = field(default=0, metadata={"help": "Deprecated for manifest_once; T is ceil(N/B_glob). If >0, assert equal."})
    curriculum_B_glob: int = field(default=32, metadata={"help": "Global effective batch (OPSD=32)."})
    curriculum_seed: int = field(default=42, metadata={"help": "Schedule shuffle seed."})
    attach_gold: bool = field(default=False, metadata={"help": "Attach gold Answer for monitor-only reward proxy (8B)."})
    within_stage_order: str = field(default="shuffle", metadata={"help": "shuffle or manifest."})
    tail_policy: str = field(default="partial", metadata={"help": "Only partial is supported."})
    curriculum_passes: int = field(default=1, metadata={"help": "Repeat full 0->last stage schedule this many times."})
    context_scaling: bool = field(default=False, metadata={"help": "Ramp on-policy generation budget (max_new_tokens) per stage from manifest context_per_stage. Generation length only; teacher/loss untouched."})
    stage_teacher_update: bool = field(default=False, metadata={"help": "Teacher-update at CURRICULUM STAGE BOUNDARIES only (φ←θ per stage, fixed within stage). Requires config use_ema_teacher=true (swap machinery); per-step EMA callback is removed. Mutually exclusive with plain EMA."})


def main():
    parser = TrlParser((CurriculumScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    assert script_args.stages_json, "stages_json is required"
    B_glob = int(script_args.curriculum_B_glob)

    is_main = os.environ.get("LOCAL_RANK", "0") == "0"

    # ---------------- manifest_once schedule (CPU, deterministic) ----------------
    schedule, stage_per_pos, expected_stage_counters, meta = build_schedule_from_stage_manifest(
        Path(script_args.stages_json),
        Path(script_args.row_table),
        B_glob=B_glob,
        seed=script_args.curriculum_seed,
        within_stage_order=script_args.within_stage_order,
        tail_policy=script_args.tail_policy,
        curriculum_passes=script_args.curriculum_passes,
    )
    T = int(meta["T"])
    if int(script_args.curriculum_T) > 0 and int(script_args.curriculum_T) != T:
        raise ValueError(f"curriculum_T={script_args.curriculum_T} was provided but manifest T={T}")
    assert len(schedule) == len(stage_per_pos), (len(schedule), len(stage_per_pos))
    assert len(expected_stage_counters) == T, (len(expected_stage_counters), T)
    if is_main:
        save_manifest_schedule_meta(meta, script_args.run_config or f"{script_args.arm}_run")
        print(f"[curriculum-manifest-once] arm={script_args.arm} spec={meta.get('spec')} "
              f"T={T} B_glob={B_glob} schedule_len={len(schedule)} "
              f"tail={meta['tail_size']} stages={meta['num_stages']} "
              f"order={script_args.within_stage_order} passes={script_args.curriculum_passes}", flush=True)

    # ---------------- model dtype (mirror opsd_train) ----------------
    if getattr(model_args, "torch_dtype", None) is not None:
        td = model_args.torch_dtype
        model_dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                       "float16": torch.float16, "fp16": torch.float16,
                       "float32": torch.float32, "fp32": torch.float32}.get(
            td.lower(), torch.bfloat16) if isinstance(td, str) else td
    elif getattr(model_args, "dtype", None) is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    qcfg = get_quantization_config(model_args)
    if qcfg is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = qcfg
    training_args.model_init_kwargs = model_kwargs
    training_args.presence_penalty = script_args.presence_penalty

    # ---------------- FORCE curriculum invariants ----------------
    training_args.remove_unused_columns = False   # stage_index/Answer must survive
    training_args.max_steps = T                   # exactly ceil(N/B_glob) manifest steps
    training_args.num_train_epochs = 1
    if hasattr(training_args, "dataloader_drop_last"):
        training_args.dataloader_drop_last = False

    # Nest output_dir under run_config on ALL ranks BEFORE building the trainer.
    # (resume-correctness: every rank must agree on the checkpoint dir passed to
    # trainer.train(); previously this only ran on the main rank inside wandb.)
    if script_args.run_config and not str(training_args.output_dir).endswith(script_args.run_config):
        training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- dataset: reorder + stage_index column ----------------
    ds = load_opsd_train()
    train_dataset = ds.select(schedule)
    train_dataset = train_dataset.add_column("stage_index", [int(x) for x in stage_per_pos])
    if is_main:
        print(f"[curriculum] reordered dataset -> {len(train_dataset)} rows; "
              f"columns={train_dataset.column_names}", flush=True)

    # ---------------- wandb (mirror opsd_train, main only) ----------------
    if is_main:
        num_processes = int(os.environ.get("WORLD_SIZE", 1))
        eff_bs = (training_args.per_device_train_batch_size
                  * training_args.gradient_accumulation_steps * num_processes)
        run_name = (script_args.run_config or f"opsd_curric_{script_args.arm}")
        run_name = f"{run_name}_T{T}_bs{eff_bs}"
        wandb.init(

            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=run_name,
            config={
                "arm": script_args.arm,
                "schedule_mode": "manifest_once",
                "manifest_spec": meta.get("spec"),
                "stages_json": str(script_args.stages_json),
                "curriculum_T": T,
                "B_glob": B_glob,
                "n_examples": meta.get("n_examples"),
                "tail_size": meta.get("tail_size"),
                "within_stage_order": script_args.within_stage_order,
                "tail_policy": script_args.tail_policy,
                "curriculum_passes": script_args.curriculum_passes,
                "effective_batch_size": eff_bs,
                "num_processes": num_processes,
                "model_name": model_args.model_name_or_path,
                "attach_gold": script_args.attach_gold,
                "remove_unused_columns": training_args.remove_unused_columns,
                "dataloader_drop_last": getattr(training_args, "dataloader_drop_last", None),
            },
        )

    # ---------------- collator + trainer ----------------
    collator = CurriculumDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        reason_first=script_args.reason_first,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        attach_gold=script_args.attach_gold,
    )

    trainer = CurriculumOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )
    # Enable the monitor-only reward proxy on the trainer (taps completions +
    # gathers gold). Off for smoke; OPSD loss/generation are never affected.
    trainer.attach_gold = bool(script_args.attach_gold)

    # per-stage context scaling (opt-in): ramp generation budget by the manifest's
    # context_per_stage. Generation length only — teacher/loss logic untouched.
    if getattr(script_args, "context_scaling", False):
        ctx_ps = meta.get("context_per_stage")
        if ctx_ps:
            trainer.context_per_stage = [int(x) for x in ctx_ps]
            if is_main:
                print(f"[curriculum] context_scaling ON: per-stage max_new_tokens="
                      f"{trainer.context_per_stage}", flush=True)
        elif is_main:
            print("[curriculum] context_scaling requested but manifest has no "
                  "context_per_stage; keeping global max_completion_length", flush=True)

    # stage-boundary teacher-update (opt-in): reuse OPSD teacher-swap machinery
    # (needs config use_ema_teacher=true) but refresh teacher only at stage
    # boundaries. Remove the per-step EMA callback so ONLY the boundary snapshot
    # updates the teacher.
    if getattr(script_args, "stage_teacher_update", False):
        if not getattr(trainer, "use_ema_teacher", False):
            raise ValueError("stage_teacher_update=true requires config use_ema_teacher=true (teacher-swap machinery).")
        trainer.stage_teacher_update = True
        try:
            from opsd_trainer import EMAUpdateCallback
            trainer.remove_callback(EMAUpdateCallback)
        except Exception as e:
            if is_main:
                print(f"[curriculum] EMA callback 제거 실패(무시): {e}", flush=True)
        if is_main:
            print("[curriculum] STAGE-BOUNDARY teacher-update ON "
                  "(φ←θ at stage boundaries; per-step EMA disabled)", flush=True)

    # manifest_once curriculum monitor (expected per-step stage Counter gate)
    trainer.add_callback(CurriculumManifestOnceMonitorCallback(
        trainer, expected_stage_counters, attach_gold=script_args.attach_gold
    ))

    if training_args.eval_strategy != "no":
        gen_cfg = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True, temperature=training_args.temperature,
        )
        trainer.add_callback(LogCompletionsCallback(trainer, gen_cfg, num_prompts=8))

    # Resume from the latest checkpoint if one exists (walltime backstop: an
    # 18h cap may not finish; requeue/resume picks up the right opt-step and the
    # deterministic schedule (curriculum_seed) reconstructs the identical order,
    # so generation is NOT re-run for already-completed steps).
    from transformers.trainer_utils import get_last_checkpoint
    resume_ckpt = None
    if os.path.isdir(training_args.output_dir):
        resume_ckpt = get_last_checkpoint(training_args.output_dir)
    if resume_ckpt and is_main:
        print(f"[curriculum] resuming from checkpoint: {resume_ckpt}", flush=True)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    main()
