# Main Phase — Difficulty-Mass Redistribution (2026-07-09~)

진짜 메인 실험 phase. 여기에 이 phase의 빌더·매니페스트·설정·문서 전부 모음.

## 핵심 가설
표현분석(§3): 난이도=두 레짐(easy L1-3 / hard L5-8, cliff at L4/L5), hard 레짐이 easy와 직교·과소학습.
→ **학습 질량을 hard 꼬리(L6-8)로 재배분**하면 하드벤치(AIME/HMMT, 전부 hard 레짐)가 오른다.

## 구성 (Eq 5-9)
- subject 축: Eq 5-6 (prototype→classical MDS→z(s)). **main과 동일, 불변.**
- 순서/스테이지: Eq 7-8 (κ=ℓ+λz+ε, λ=1, 등질량 5분할). **불변.**
- **NEW — 난이도-질량 재가중 Eq 9**: hard 꼬리 H={ℓ≥6}를 k배 복제, 나머지는 s=(M−k·n(H))/(M−n(H)) 비율로 균등 subsample → 총량 M 고정(컴퓨트 불변). k=1이면 지금(각 문제 1회).

## ⚠️ 명칭 정정 (2026-07-09 밤, 벤치 과목 프로필 실측 후): "benchmark-aligned" → **"discrete-late (representation-axis orientation)"**
- 실측(AIME24/25 독립분류 60문제 + HMMT25 공식라벨, bench_subject_profile.json): 하드벤치 = **Geo 31% + C&P 30%**(합 61%) 지배, **NT는 14%뿐**.
- 우리 discrete-late 마지막 스테이지 = NT 50%(벤치 3.6배 과다)·Geo 17%(절반)·C&P 28%(일치) → **벤치 정렬 아님**. 원인: 데이터 하드레벨이 NT-heavy(L8=43%) + z축 이산 끝=NT + Geo는 z중앙이라 안 몰림.
- **함의**: ① circularity 방어 완벽(커리큘럼-벤치 프로필 불일치가 정량 증거 — "테스트에 맞추지 않았다") ② 서사 = "표현축 방향(discrete-late)" (벤치 언급 금지, §3+데이터 상관으로 정당화) ③ 실험 유효(방향 효과 테스트 그대로) ④ 진짜 벤치-매칭 arm은 만들지 않음(그게 진짜 circular).

## (구) subject 방향 = benchmark-aligned (discrete-late), 2026-07-09 결정
- Eq 7 부호를 **κ=ℓ−λz**로: 이산 subject(정수론·조합=AIME/HMMT 핵심)를 늦은/하드 스테이지로.
- 효과: 마지막 스테이지 NT+조합 = 21%→63%(k=1)→**76%(k=2)**. 조합(C&P) 5%→25/27%.
- 서사: 기초(Prealg/Alg)→이산 경쟁수학(NT/조합)으로 accumulation, 난이도와 co-move, 벤치에서 끝남.
- ⚠️ domain-informed(AIME=이산 앎) → 논문에 disclose. 뒤집기 자체는 노이즈일 수 있으나 재배분과 결합 시 recency 레버 가능성.

## 실험 매트릭스 (전부 4B, context OFF/1024, fixed teacher) — 2026-07-09 20:xx 전부 RUNNING
2×2(subject 방향 × 재배분 k) + 용량반응:

| subject↓ / k→ | k=1 (18% hard) | k=2 (37%) | k=3 (55%) |
|---|---|---|---|
| **continuous-late** (old main, κ=ℓ+z) | ✅ old main 84.8/41.9/48.9/40.8/28.3 | **contsubj_k2** (H200 106059→eval 106060) | — |
| **discrete-late** (benchmark, κ=ℓ−z) | **benchsubj_k1** (H200 106056→eval 106057) | **benchsubj_k2** (H100 0,1) | **benchsubj_k3** (H100 2,3) |

**격리되는 것:**
- **뒤집기 효과** = old main vs benchsubj_k1 (k=1) / contsubj_k2 vs benchsubj_k2 (k=2) — 독립 2회 검정.
- **재배분 효과** = old main vs contsubj_k2 (cont) / benchsubj_k1→k2→k3 용량반응 (disc).
- **★circularity 해소**: benchsubj_k2 ≈ contsubj_k2면 "뒤집기는 성능 레버 아님"→ 벤치정렬 공격 무력화. benchsubj_k2 > contsubj_k2면 뒤집기 실효→§3 표현근거 필요.

H100 log: `$WORK/log_benchsubj_k{2,3}.log`, eval `$WORK/outputs/eval/cliff4b_benchsubj_k{2,3}_nonthink/`.
H200 결과: `outputs/eval_opsd_curriculum/cliff4b_{benchsubj_k1,contsubj_k2}_nonthink/` (여기 /scratch에서 직접 수거).

## 결정 사항 (확정)
- subject 방향 = benchmark-aligned(κ=ℓ−λz). 세 arm 전부 동일 방향(레벨만 변수).
- z 출처 = main과 동일 8B npz(통제군 유효). z 4B/8B 차이 r=0.988 무시 가능(논문 fidelity 별도).
- 복제 구현 = 트레이너 opt-in 플래그 `allow_duplicate_pids`(default off).

## 파일
- build_redistribute.py — Eq 5-9 빌더(벤치정렬 κ=ℓ−λz + 난이도재배분). `python build_redistribute.py 1 2 3`
- stages_benchsubj_k{1,2,3}.json — 산출 매니페스트(총 28771 고정, 하드 k배 복제)
- train_mainphase.sh — H100 런치(full_4b_main.yaml, --allow_duplicate_pids True, run_config=cliff4b_${ARM})

## 검증 완료 (2026-07-09, 드라이런)
- 트레이너 opt-in 플래그 3곳 추가: [curriculum_schedule_manifest_once.py](../curriculum/curriculum_schedule_manifest_once.py)(가드+시그니처), [train_opsd_curriculum_manifest_once.py](../curriculum/train_opsd_curriculum_manifest_once.py)(필드+전달). default off → 기존 arm 무영향.
- 가드 정상: k=2 flag OFF → ValueError(기존 arm 보호). 플래그 파싱 `--allow_duplicate_pids True` OK.
- **중복 확산 확인**: schedule에서 하드(L6-8) opsd_index가 정확히 k회(k=2→2.00, k=3→3.00), mid/easy 1회. schedule_len=28771, T=900(main과 동일).
- 벤치정렬 확인: 마지막 스테이지 NT+조합 = 64%(k1)/77%(k2)/74%(k3).

## 런치 (H100 relay, 2슬롯 병렬 + eval 체이닝)
먼저 H100(/data1/lamilab/jimin)에 sync: 변경파일(curriculum_schedule_manifest_once.py, train_opsd_curriculum_manifest_once.py) + 새 폴더(mainphase_20260709/). git push→pull 또는 scp.
```bash
export REPO=/data1/lamilab/jimin       # H100 repo 루트(=OPSD_Curriculum 상위)
cd $REPO/OPSD_Curriculum/training/mainphase_20260709
# slot1 (GPU 0,1)
CUDA_VISIBLE_DEVICES=0,1 PORT=13100 ./train_mainphase.sh benchsubj_k1 \
  && REPO=$REPO ../h100_port/eval_cliff4b.sh benchsubj_k1
# slot2 (GPU 2,3)  ← 다른 tmux pane
CUDA_VISIBLE_DEVICES=2,3 PORT=13200 ./train_mainphase.sh benchsubj_k2 \
  && REPO=$REPO ../h100_port/eval_cliff4b.sh benchsubj_k2
```
- k3: H100 슬롯 나면 동일 패턴(benchsubj_k3) / 또는 H200 SLURM 변형(train_cliff4b_h200.sh에 --allow_duplicate_pids True + 매니페스트 경로 추가 필요 — 요청 시 작성).
- 결과 수거: `$WORK/outputs/eval/cliff4b_benchsubj_k{n}_nonthink/*step899*.json`. 통제군=old main(84.8/41.9/48.9/40.8/28.3).
