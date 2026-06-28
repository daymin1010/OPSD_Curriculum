# STATUS — similarity_analysis.py 실행

## 진단 (2026-06-04)
- 기존 run (PID 1332330): State `R`, CPU 100% 1코어 bound, RSS 8.4G, 스왑 정상.
  → **멈춤 아님**. permutation test 가 single-thread 로 과도하게 무거웠던 것.
- 병목: `perm_pvalue()` 가 `N_PERM=1000` × grouping 다수 × 2 모드 반복, 매 perm 마다
  멤버 L2 정규화(N×36×12288)를 재계산.

## 적용한 수정 (option B)
1. `normalize_members()` 추가 → 멤버 정규화를 **perm 루프 밖에서 1회만** 계산,
   `within_between` / `perm_pvalue` 에 `DAn` 으로 주입 (가장 큰 가속).
2. `N_PERM` 1000 → **200** (p 해상도 1/201, screening 충분).
3. perm 진행 로그 `perm k/200 (ge=..)` flush 출력 추가.
4. 실행: `python -u` + `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=4` (공유 노드 배려, 코어 독점 금지).

## 재실행 커맨드
```bash
cd /scratch/lami2026/personal/jimin_2782
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
nohup envs/verl_new/bin/python -u \
  src/OPSD_Curriculum/reasoning_pivot/activation/analysis/similarity_analysis.py \
  --shifts-dir src/OPSD_Curriculum/reasoning_pivot/activation/outputs/pilot/shifts \
  --out-dir src/OPSD_Curriculum/reasoning_pivot/activation/analysis \
  --tag pilot > /tmp/cline/sim_analysis2.log 2>&1 &
```
- 로드: 1541 .pt files, kept 1541.
- 진행 모니터: `tail -f /tmp/cline/sim_analysis2.log`
- 속도: ~2.5s/perm (이전 대비 정상 진행, 가시적).

## 산출물 (완료 시)
- `analysis/REPORT_similarity_pilot.md`
- `analysis/sim_matrices_pilot.npz`
- `analysis/heatmap_{subject,level,unit}_*_pilot.png`

## ✅ 완료 (2026-06-04 12:08)
- run DONE: 14/14 perm 블록 종료, `REPORT_similarity_pilot.md` 생성 (PID 2674173 DONE).
- p값 정상 분포 확인 (FAITHFUL balanced LEVEL ge=80 → p≈0.40, 나머지 대부분 ge≈0 → p=0.005).
- 산출물 모두 생성: REPORT_similarity_pilot.md, sim_matrices_pilot.npz, heatmap 5종.
- **해석 마스터 문서 작성**: `analysis/MASTER_SIMILARITY_FINDINGS.md`
  (요약/방법론 수식/주석 코드/결과표/해석/한계/재현).

### 핵심 결과 (요약)
- raw ΔA: 공통축 지배(within·between ~0.9, gap 음수) → centering 필수.
- centered: subject/level/unit 모두 gap>0, perm p=0.005.
- LEVEL ordinality ρ≈0.85 (강한 단조 난이도 축).
- gen_len 균형 후 THINKING(think-span) 신호 생존, FAITHFUL LEVEL 은 소표본(N=106)에서 붕괴
  → think-span shift 가 길이 교란에 더 견고 (thinking-mode 가설 지지).

