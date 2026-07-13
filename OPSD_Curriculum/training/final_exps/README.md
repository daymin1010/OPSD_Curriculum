# final_exps — 최종 실험 네이밍 (2026-07-13~)

**이름 규칙**: `{method}[_키워드]`
- method: `base`(무학습) / `random`(완전셔플) / `cliff`(레벨-only 좁은밴드) / `main`(wide-level + discrete-late subject + k=2 재분배)
- 키워드: `L5`=레벨1–5만(하드꼬리 없어 k=1) · `d50`/`d70`=데이터 50/70% · `s43`=seed
- **run 접두어**: `fin4b_` / `fin8b_` (모델 구분; cliff는 베이스라인명이라 접두어에 안 씀)
- 실행: `STAGES=<이 폴더> RUN_PREFIX=fin4b ./train_mainphase.sh main_L5_d50` 식 (스크립트 env override)

## 구명 → 신명 매핑
| 신명 | 구명 | 비고 |
|---|---|---|
| main | benchsubj_k2 | wide-level·disc-late·k2 (28,771) |
| cliff | main_diff | 레벨-only, 좁은 밴드 (28,743; cond2_diff와 구조 동일 검증) |
| random | main_shuffle | 완전 랜덤 |
| main_d50 | benchsubj_n14k_k2 | 14,369 (T450) |
| main_d70 | benchsubj_n20k_k2 | 20,004 = **69.6%≈70% 근사** (T626) |
| main_L5_d50 | benchsubj_L5_n11k_k1 | L1–5 유니버스(23,418)의 50% = 11,701 (T366) |
| main_L5_d70 | benchsubj_L5_n16k_k1 | 〃 70% = 16,391 (T513) |
| main_L5 | benchsubj_L5_k1 | L1–5 전체 23,418 (T732) |
| (재평가) cliff_d25/d11/d06 | quarter/mini100/mini50 cond2_diff | 옛 8B ckpt val12 재채점 — cliff 저데이터 참조점 |

검증(2026-07-13): 4종 매니페스트 L5 상한·이산% disc-late 진행 확인 완료. cond3_ours_C2 계열은 구세대 subject라 재평가/비교 제외.
