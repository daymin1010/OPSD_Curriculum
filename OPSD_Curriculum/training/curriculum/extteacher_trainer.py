"""
extteacher_trainer.py — EXPERIMENTAL external-teacher variant (2026-07-12).
==========================================================================
별도(보통 더 큰) frozen teacher 모델로 cross-model distillation 하는 실험용 서브클래스.
⚠️ opsd_src/ 원본(OPSD_original과 동일)은 절대 수정하지 않는다 — 원본 self-distillation
   실험은 계속 기존 파이프라인(train_opsd_curriculum_manifest_once.py)으로 돌린다.

구성:
  - ExtTeacherCurriculumOPSDTrainer(CurriculumOPSDTrainer):
      * __init__: teacher_model_name_or_path 로 frozen teacher 로드 (rank마다 full 복제,
        DeepSpeed 밖. vocab 일치 검증 — Qwen3 계열은 전부 151936 공유라 4B/8B 학생 모두 OK)
      * compute_loss: opsd_src/opsd_trainer.py::compute_loss 를 그대로 복사하되
        티처 forward 만 external teacher 로 분기 (EXT-TEACHER 표시 블록)
  - 진입점은 train_opsd_curriculum_extteacher.py (이 디렉토리).

메모리 (bf16 full 복제 기준):
  teacher 8B≈16GB / 14B≈28GB / 32B≈64GB(→H200 전용).
  4B 학생+14B 티처: H100 80GB 가능(VLLM_UTIL≤0.3 권장). 8B 학생+14B 티처: H200 권장.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from transformers import AutoModelForCausalLM
from trl.trainer.utils import empty_cache

# curriculum/ (on PYTHONPATH)
from curriculum_trainer import CurriculumOPSDTrainer


class ExtTeacherCurriculumOPSDTrainer(CurriculumOPSDTrainer):
    """CurriculumOPSDTrainer + separate frozen (larger) teacher model."""

    def __init__(self, *args, teacher_model_name_or_path: str = None, **kwargs):
        if not teacher_model_name_or_path:
            raise ValueError(
                "ExtTeacherCurriculumOPSDTrainer requires teacher_model_name_or_path. "
                "For the original self-distillation pipeline use CurriculumOPSDTrainer."
            )
        if kwargs.get("fixed_teacher") or kwargs.get("use_ema_teacher"):
            raise ValueError(
                "teacher_model_name_or_path is mutually exclusive with fixed_teacher / use_ema_teacher "
                "(those reuse the student weights as teacher)."
            )
        super().__init__(*args, **kwargs)

        print(f"\n{'='*80}")
        print(f"EXTERNAL TEACHER MODE (experimental): loading frozen teacher '{teacher_model_name_or_path}'")
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # 토큰단위 KL/JSD가 성립하려면 vocab(support)이 학생과 동일해야 함
        student_cfg = self.accelerator.unwrap_model(self.model).config
        if teacher.config.vocab_size != student_cfg.vocab_size:
            raise ValueError(
                f"External teacher vocab_size={teacher.config.vocab_size} != student vocab_size="
                f"{student_cfg.vocab_size}. Teacher/student must share the tokenizer vocabulary."
            )
        teacher.eval()
        teacher.requires_grad_(False)
        teacher.config.use_cache = False
        # frozen forward 전용 → DeepSpeed 밖, rank 디바이스에 full 복제
        self.external_teacher = teacher.to(self.accelerator.device)
        print(f"Teacher on {self.accelerator.device}: vocab={teacher.config.vocab_size}, "
              f"hidden={teacher.config.hidden_size}, layers={teacher.config.num_hidden_layers} "
              f"(student hidden={student_cfg.hidden_size})")
        print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # opsd_src/opsd_trainer.py::compute_loss 를 그대로 복사.
    # 변경점은 'EXT-TEACHER' 표시 블록 하나뿐 (teacher forward를 external 모델로).
    # 원본이 바뀌면 이 복사본도 갱신할 것 (실험용 임시 변형).
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the self-distillation loss with memory-efficient log-prob extraction.

        Memory optimization: Extract only needed log-probs immediately and free large tensors.
        """
        # Get batch-level prompt lengths
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === STUDENT FORWARD - Extract log-probs immediately ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )

        # Extract only what we need and convert to log-probs immediately
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            # For reverse KL, we only need log-probs of sampled tokens
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs  # Free immediately!
        else:
            # For JSD, keep logits (temperature will be applied in generalized_jsd_loss)
            student_logits_for_loss = student_logits
            del student_logits

        # Free the full outputs (but keep reference for return_outputs if needed)
        if return_outputs:
            # Create a minimal output object to return (just the loss, no logits)
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        # === TEACHER FORWARD - Extract log-probs immediately ===
        # EXT-TEACHER: 원본은 (EMA/fixed/dynamic 모드로) 학생 모델을 티처로 재사용하지만,
        # 이 변형은 별도 frozen 모델을 티처로 사용한다. 특권 입력(teacher_input_ids =
        # gold+thinking 프롬프트)과 prompt_len 슬라이싱은 원본과 동일.
        with torch.no_grad():
            outputs_teacher = self.external_teacher(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs  # Free immediately!
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        # === COMPUTE LOSS with only small tensors ===
        if self.use_thinking_machines_loss:
            # Thinking Machines uses RL-style policy gradient:
            # Advantage = log π_teacher(x) - log π_student(x)
            # Loss = -E[Advantage * log π_student(x)]
            #
            # CRITICAL: advantage must be detached to prevent gradients flowing through it.
            # We want: ∇θ L = -E[A(x) * ∇θ log π_student(x)]
            # NOT: ∇θ L = -E[(T(x) - S(x)) * ∇θ S(x)] where both terms differentiate

            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            # Apply masking before computing loss
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            # Policy gradient loss: -advantage * log π_student
            # Negative because we minimize loss (gradient descent), but want to maximize reward
            loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            # Temperature is applied inside generalized_jsd_loss
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,  # Let the function handle temperature
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
            del student_logits_for_loss, teacher_logits_for_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss
