# OPSD Curriculum — Openthoughts 30K Labeling

`siyanzhao/Openthoughts_math_30k_opsd` (29,434 problems) 에 대해
GPT-4.1-mini 로 `(subject, level)` 라벨을 부착. 이후 subject×level unit
샘플링 → activation shift 추출의 기반 데이터.

---

## 1. 분류 기준

SYSTEM_PROMPT 는 `src/4.6_Task2/classifier/classify_full.py` 의 것과
**완전히 동일 (verbatim)**.

- Subject: 8 종 (Algebra, Counting & Probability, Geometry, Intermediate
  Algebra, Number Theory, Prealgebra, Precalculus, Other).
- Level: 1–8 (절대 난이도, source 무관).

기존 FastCuRL 40,315 문제 분류와의 일관성 확보가 목적이므로 프롬프트는
**절대 수정하지 않음**. 분포 보정/level binning 은 분석 단계에서.

---

## 2. ENV (필수)

```bash
export LAMI_OPENAI_API_KEY=sk-...     # 사용자 본인 키. .env 파일 안 만듦.
```

- 코드는 `os.environ["LAMI_OPENAI_API_KEY"]` **만** 읽음.
- 같은 노드의 다른 user 에게는 노출되지 않음 (process env).
- CSV / log / raw_response 어디에도 key 가 저장되지 않음.

---

## 3. 실행

GPU 불필요 → **login 노드 OK** (cluster 의 GPU partition 03/07 제약은
GPU job 에만 적용). 장시간 실행이므로 `tmux` 권장.

```bash
cd /scratch/lami2026/personal/jimin_2782
PY=envs/verl_new/bin/python

# (1) Smoke 200 — concurrency=20 검증
$PY src/OPSD_Curriculum/labeling/label_openthoughts_30k.py \
    --limit 200 --concurrency 20 \
    --output src/OPSD_Curriculum/labeling/outputs/smoke200_labels.csv

# (2) Full 29,434  (tmux 안에서)
tmux new -s opsd_label
$PY src/OPSD_Curriculum/labeling/label_openthoughts_30k.py \
    --concurrency 20 \
    --output src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels.csv
# detach: Ctrl-b, d   /   attach: tmux attach -t opsd_label
```

**Resume**: 중간 종료 후 같은 `--output` 으로 재실행하면
`*.partial.csv` 의 성공 row 를 스킵하고 미완료/실패만 재호출.

---

## 4. 출력 컬럼

`outputs/{smoke200,openthoughts_30k}_labels.csv`:

| Column | Source | Note |
|---|---|---|
| `row_index` | dataset enumeration | join key |
| `source` | HF `source` | olympiads / math / aops / amc_aime |
| `problem_text` | HF `problem` | 입력 보존 |
| `problem_char_len` | computed | confound check |
| `problem_qwen_tok_len` | Qwen3-8B tokenizer | activation 분석 시 길이 분포 |
| `r1_cot_token_count` | HF `generated_token_count` | 원본 R1 COT 길이 (난이도 proxy) |
| `solution_char_len` | HF `solution` | reference 답 길이 |
| `correct` | HF (전부 True) | 일관성 확인 |
| `answer` | HF `Answer` | join 용 |
| **`subject`** | GPT | 8 categories |
| **`level`** | GPT | 1–8 |
| `raw_response` | GPT | JSON 검증 |
| `error` | code | empty = ok |
| `finish_reason` | API | `length` 면 truncation |
| `prompt_tokens`, `completion_tokens` | API usage | 비용 추적 |
| `latency_s` | measured | concurrency tuning |
| `attempts` | retry 횟수 | 1=즉시성공 |
| `model` | `gpt-4.1-mini-2025-04-14` | 재현성 |
| `prompt_sha` | sha256(SYSTEM_PROMPT)[:12] | 프롬프트 버전 |

---

## 5. 예상 시간 / 비용

- 29,434 × concurrency=20, latency ≈ 1.5–2 s/req → **약 35–50 분**.
- GPT-4.1-mini 비용: prompt ~1.5 k tok × 29.4 k × $0.40/1M + 30 tok × 29.4 k × $1.60/1M ≈ **$18 정도**.

Smoke 결과 보고 concurrency 조정.

---

## 6. 분석 단계 (다음)

이 라벨 CSV 를 기반으로:
1. subject×level cross-tab + source 별 분포 → unit 정의 (sparse cell 병합).
2. unit 당 N개 균등 샘플링 → pilot ID list.
3. Qwen3-8B activation shift / pass-rate 측정 (이전 2666 pilot 과
   동일 파이프라인 재사용).
4. NAIT 3-phase 분석 + curriculum 설계.

---

## 7. 절대 수정 금지

- `src/4.6_Task2/classifier/classify_full.py` (옛 분류 코드, 참조 only).
- `outputs/*` 의 완성된 CSV (`*.partial.csv` 만 resume 으로 덮어쓰기 허용).
