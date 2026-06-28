# 🧭 OpenThoughts-30K Labeling — Hand-off

작성일: 2026-05-30
작업자: Jimin (jimin_2782)
선행 hand-off: `src/OPSD_Curriculum/analysis_qwen3_8b/HANDOFF_THINKING_MODE.md`

---

## 0. 한 줄 요약

**Siyanzhao/Openthoughts_math_30k_opsd 의 29,434개 문제 전부를 `gpt-4.1-mini` 로 `(subject, level)` 라벨링 완료.** 전수 통과 ($9, ~22분), ρ(level, r1_cot_token_count) = **0.61**.

산출물 핵심:
- `outputs/openthoughts_30k_labels_final.parquet` (29,434 rows, problem_id 포함)
- `outputs/pilot_universe_candidate.parquet` (3,000 rows, subject×level stratified)
- `outputs/REPORT_full.md`, `outputs/REPORT_pilot.md`

---

## 1. 운영 환경 (변경 없음)

- 공동 user `lami2026`, 작업물은 **`/scratch/lami2026/personal/jimin_2782/`** 내부에서만.
- GPU 작업은 `iREMB-C-03` (H200×2) 또는 `iREMB-C-07` (L40s×4) sbatch 전용.
- 이 라벨링은 OpenAI API 호출이라 GPU 불필요했음. compute 노드 외부에서 `tmux` 안에서 실행함 (정상).

API key: `LAMI_OPENAI_API_KEY` (export 후 tmux 안에서도 export 재확인 필요했음).

---

## 2. 결과 검증 (전부 PASS)

| 항목 | 결과 |
|---|---|
| rows | **29,434 / 29,434** |
| errors | **0** |
| finish_reason = "length" (truncation) | **0** |
| raw_response strict JSON (`{subject, level}` 두 키만) | **29,434 / 29,434** |
| row_index unique, range = 0..29433 | ✅ |
| attempts > 1 (retries) | **0** |
| subject in 8 allowed (after normalization) | 29,434 / 29,434 |
| level in [1,8] | 29,434 / 29,434 |
| latency p50 / p95 / p99 | 1.0s / 1.7s / 2.5s |
| **cost (실측)** | **≈ $9** (no-cache 추정 $20.24 → ~55% prompt caching 절감) |

### 2.1 신호 품질 (curriculum 가치)
- `ρ(level, r1_cot_token_count) = 0.61` (Spearman) — **강한 양의 상관**, level 이 실제 난이도를 잘 추적.
- `ρ(level, problem_qwen_tok_len) = 0.32` — 문제 길이도 따라가지만 r1_cot 보다 약함.

### 2.2 분포 (29,434 full, after normalization)

```
subject           total
Algebra           6823
Number Theory     5436
Geometry          5266
Counting & Prob   3859
Prealgebra        2903
Intermediate Alg  2289
Precalculus       2195
Other              663   (598 + 66 OOV remapped)
```

```
level   n
1     1343
2     4328
3     5914
4     6172
5     6110
6     3682
7     1819
8       66
```

### 2.3 sparse / dead cells
- subject × level = 8×8 = 64 cells.
- L8 은 전체에서 66개뿐 → 일부 subject 에서 사실상 비어있음 (Algebra L8 = 0, Precalculus L8 = 0, Prealgebra L8 = 0).
- Prealgebra 은 L4 부터 거의 사라짐 (L5+ 모두 0–17). 정의상 정상 (Prealgebra 는 쉬운 영역).
- 따라서 **실효 유효 cells ≈ 56** (zero cell 8개 제외).

### 2.4 OOV subject (lessons learned)
prompt 에서 strict 8-class enforce 했지만 model 이 6 종류, 66 rows 만들어냈음:
- Calculus 32, Logic 16, Physics 11, Linear Algebra 5, Trigonometry 1, Functional Equations 1.
- 모두 `Other` 로 mapping. 원본은 `subject_raw` 컬럼에 보존.
- 0.22% 라 무시 가능.

---

## 3. 산출물 (절대 경로 기준)

### 3.1 데이터
| 파일 | 설명 |
|---|---|
| `src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels.csv` | raw API output, 29,434 × 20 cols |
| `src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet` | **최종 정규화 본** (problem_id, subject 정규화 추가, 22 cols, 4.8 MB) |
| `src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet` | **pilot 3,000 stratified** (seed=42) |
| `src/OPSD_Curriculum/labeling/outputs/smoke200_labels.csv` | smoke run 보존본 |

### 3.2 코드
| 파일 | 설명 |
|---|---|
| `src/OPSD_Curriculum/labeling/label_openthoughts_30k.py` | API 라벨러 (concurrency=20, async, atomic save) |
| `src/OPSD_Curriculum/labeling/analyze_full.py` | 12-point sanity + 분포 분석 |
| `src/OPSD_Curriculum/labeling/postprocess.py` | subject 정규화 + problem_id 부여 + pilot 추출 |
| `src/OPSD_Curriculum/labeling/README.md` | 사용법 / prompt 명세 |

### 3.3 리포트
- `outputs/REPORT_full.md` — full sanity & distribution
- `outputs/REPORT_pilot.md` — pilot universe stratification 진단

### 3.4 메타 무결성
- `prompt_sha = 208fbdb6202f` (단일값) — 전 row 동일 prompt 사용 확인됨.
- `model = gpt-4.1-mini-2025-04-14` — 단일.

---

## 4. final parquet 스키마

| col | dtype | 비고 |
|---|---|---|
| `problem_id` | str (16 hex) | sha1(problem_text)[:16], **unique 29,434** |
| `row_index` | int | 0..29,433, original dataset order |
| `source` | str | dataset 안의 sub-source |
| `subject` | str | **8 canonical** (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus, Other) |
| `level` | int | 1..8 (8 = hardest) |
| `subject_raw` | str | model 의 원본 출력 (OOV 보존) |
| `problem_text` | str | original |
| `problem_char_len`, `problem_qwen_tok_len`, `r1_cot_token_count`, `solution_char_len`, `correct`, `answer` | passthrough |
| `raw_response`, `error`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `latency_s`, `attempts`, `model`, `prompt_sha` | API meta |

---

## 5. Pilot universe (3,000)

- Stratifier: `(subject, level)`, 64 cells (실효 ≈ 56).
- Sampler: `take = clip( round(sqrt(n_cell)*5), [5, 80] )`, 후 shuffle + slice 3,000.
- Seed: 42.
- ρ(level, r1_cot_token_count) in pilot: (REPORT_pilot.md 에 명시) — full 과 같은 0.6대 유지.

### 다음 사용처
이 pilot 3,000 universe 는 Qwen3-8B activation 추출 (thinking & non-thinking) 의 대상이 됨. 선행 hand-off 의 thinking-mode 재추출 task 와 직접 연결.

권장 워크플로:
1. (이 단계) ✅ 라벨링 완료
2. pass-rate 측정 (8-rollout, non-thinking) — pilot 3000 에 대해
3. activation shift 추출 — thinking + non-thinking 두 버전
4. NAIT 3-phase 분석

기존 2666 pilot universe 는 별도 dataset (1.5B sampled). **이번 3000 은 OpenThoughts 30K base, 더 큰 sample**.

---

## 6. 재현 명령

```bash
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
cd /scratch/lami2026/personal/jimin_2782

# 1. (이미 완료) 라벨링 자체
$PY src/OPSD_Curriculum/labeling/label_openthoughts_30k.py \
    --concurrency 20 \
    --output src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels.csv
# → 약 22분, $9, atomic save

# 2. sanity & 분포 분석
$PY src/OPSD_Curriculum/labeling/analyze_full.py
# → outputs/REPORT_full.md

# 3. 정규화 + problem_id + pilot 추출
$PY src/OPSD_Curriculum/labeling/postprocess.py
# → outputs/openthoughts_30k_labels_final.parquet
# → outputs/pilot_universe_candidate.parquet
# → outputs/REPORT_pilot.md
```

---

## 7. 절대 수정 금지

- `outputs/openthoughts_30k_labels.csv` (raw, ~$9 비용)
- `outputs/openthoughts_30k_labels_final.parquet`
- `outputs/pilot_universe_candidate.parquet`
- `outputs/smoke200_labels.csv`
- `label_openthoughts_30k.py` (prompt 변경 시 prompt_sha 가 바뀌어 재호출 비용 발생)

재실행이 필요하면 **새 output 경로**로 분기.

---

## 8. Quirks / 잊지 말 것

1. **prompt caching 효과 ~55%**: SYSTEM_PROMPT 고정 + 1024 chars 이상이면 OpenAI 자동 cache. concurrency 20 같은 워커도 같은 SYSTEM 토큰 prefix 를 공유함.
2. **OOV (0.22%) 무시 OK**, 하지만 prompt 수정 시 다시 늘어날 수 있음. `subject_raw` 컬럼이 항상 ground truth.
3. **CSV embedded newlines**: `wc -l` 결과 (75,738) ≠ row count (29,434). 항상 `pd.read_csv()` 또는 parquet 로 검증.
4. **MATH level convention**: 1=쉬움 ↔ 8=어려움. ρ(pass_rate, level) 은 항상 **음수** 가 정상.
5. tmux 에 env var 전파 주의: 새 tmux session 안에서는 `LAMI_OPENAI_API_KEY` 가 inherited 됐는지 항상 `echo "${LAMI_OPENAI_API_KEY:0:7}"` 로 확인.

---

## 9. 다음 세션의 첫 행동

1. 이 hand-off + 이전 `HANDOFF_THINKING_MODE.md` 읽기 (1–2분).
2. pilot universe 확인:
   ```bash
   $PY -c "
   import pandas as pd
   p = pd.read_parquet('src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet')
   print(p.shape, p.columns.tolist())
   print(p[['problem_id','subject','level','r1_cot_token_count']].head())
   "
   ```
3. 다음 작업 (pass-rate 측정 vs activation 추출 vs thinking mode 추출) 중 어떤 것부터 갈지 결정.
