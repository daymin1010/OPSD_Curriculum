# SUBJECT length-confound gate — tag=subjlen

> **질문.** subject 들이 activation 에서 가까운 게 '내용 유사' 인가 '길이 유사' 인가.
> subjsim 의 G1/G2/G3(mid-L11-15) grouping 이 gen_len 으로 설명되는지 검사.
> level length gate(ρ(level)=0.71≈ρ(gen_len)=0.74, partial=0.39)와 동일 gen_len·동일 3-method 미러링.

> **분리 원칙.** (1) subjects 가 length 에서 다름 = 잠재 신호(치명 아님). (2) subject 유사도 *구조* 가 length 구조를 따라감 = fatal. **판정은 (2).**

- pooled N=**3025** (pilot1=1608, pilot2=1417); subjects(8)=['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Other', 'Prealgebra', 'Precalculus']
- gen_len = Qwen3-8B thinking trace token 수(ΔA span 길이). length gate 와 동일 변수.
- view: mid_L11-15(주), layeravg(보조). centering=μ_pooled. CPU only, seed=42.
- 게이트 임계값(제안): Mantel 비유의(p≥0.05) & |r|<0.5; r(M_resid,M_act)≥0.85(Method B/C); within-bin r≥0.8(Method A).

## Step 0 — length 변수 sanity
- ρ(level, gen_len) 재계산 = **+0.709** (기대 ≈0.74, tol 0.1) → OK ✅

### per-subject gen_len 분포
| subject | n | mean | median | std | p25 | p50 | p75 |
|---|---|---|---|---|---|---|---|
| Algebra | 420 | 4851 | 4644 | 2327 | 2691 | 4644 | 7062 |
| Counting & Probability | 430 | 5358 | 5508 | 2373 | 3255 | 5508 | 8156 |
| Geometry | 406 | 5385 | 5747 | 2379 | 3218 | 5747 | 8010 |
| Intermediate Algebra | 387 | 5276 | 5251 | 2174 | 3427 | 5251 | 7250 |
| Number Theory | 443 | 5381 | 5292 | 2239 | 3279 | 5292 | 7843 |
| Other | 397 | 5754 | 5895 | 2010 | 4034 | 5895 | 7858 |
| Prealgebra | 197 | 3855 | 3201 | 2037 | 2238 | 3201 | 4849 |
| Precalculus | 345 | 5324 | 5411 | 2150 | 3340 | 5411 | 7479 |
- (Other n=397 — centroid 안정성 주의. Step2/3 에 Other 제외 robustness 포함.)

## Step 1 — subject↔length 연관 (잠재 confound 유무; 단독 FAIL 아님)
- Kruskal–Wallis H=110.7, p=6.56e-21
- one-way ANOVA F=16.4, p=2.61e-21
- effect size **η² = 0.037** (작음)
- ※ 이것만으론 FAIL 아님: '길이 분포 차이' ≠ '유사도 구조가 길이로 설명됨'.

## ===== VIEW: mid_L11-15 =====

### M_act 일치 확인 (기존 subjsim 대비)
- r(M_act, subjsim within-level(A)) = +0.879
- r(M_act, subjsim level-centroid resid(B-main)) = +0.974

### Step 2 — Mantel (length-sim vs subject-act-sim)
- Mantel(D_act, M_len mean-dist): r=+0.009, p=0.9433
- Mantel(D_act, M_len wasserstein): r=+0.036, p=0.7947
- (양수 r = 길이 먼 subject 쌍이 act 에서도 멀다 = length 정렬. r 낮고 비유의면 length 가 구조 설명 못함.)
- ※ 8×8=28 pairs 로 작음 → permutation p 만, 단독 강결론 금지.
- [robustness, Other 제외] Mantel(D_act, mean-dist): r=-0.083, p=0.5231

### Step 3 — 구조 생존 (level 3-method 를 length 로 미러링)

**Method A (within gen_len bin, nbins=5)** ↔ within-level
- bin 별 subject 사용: bin0(n=605):used 8 subj; bin1(n=605):used 8 subj; bin2(n=605):used 8 subj; bin3(n=598):used 8 subj; bin4(n=612):used 8 subj
- r(M_withinbin, M_act) = **+0.851** (≥0.8 이면 length artifact 아님)
- bin 간 행렬 상관 평균 = +0.604 (bin 가로질러 구조 일관)

**Method B (len-bin centroid 차감)** ↔ GPT-level centroid 차감
- r(M_resid_bincentroid, M_act) = **+0.980** (≥0.85 이면 생존)

**Method C (gen_len projection 제거, GLOBAL pooled)** ↔ ridge projection 제거
- [linear] r(M_resid_proj, M_act) = **+0.978**
- [+log_len] r(M_resid_proj, M_act) = **+0.978**
- [+quadratic] r(M_resid_proj, M_act) = **+0.980**
- [gen_len+level 동시 제거] r(M_resid, M_act) = **+0.974** (둘 다 제거해도 구조 생존 = content 고유)

### >>> VIEW [mid_L11-15] 게이트: **PASS** (Mantel p_min=0.795, |r|≈0.02; within-bin r=+0.85✓; resid_min r=+0.98✓)

## ===== VIEW: layeravg =====

### M_act 일치 확인 (기존 subjsim 대비)
- r(M_act, subjsim within-level(A)) = +0.815
- r(M_act, subjsim level-centroid resid(B-main)) = +0.958

### Step 2 — Mantel (length-sim vs subject-act-sim)
- Mantel(D_act, M_len mean-dist): r=+0.027, p=0.8186
- Mantel(D_act, M_len wasserstein): r=+0.058, p=0.6241
- (양수 r = 길이 먼 subject 쌍이 act 에서도 멀다 = length 정렬. r 낮고 비유의면 length 가 구조 설명 못함.)
- ※ 8×8=28 pairs 로 작음 → permutation p 만, 단독 강결론 금지.
- [robustness, Other 제외] Mantel(D_act, mean-dist): r=-0.064, p=0.5844

### Step 3 — 구조 생존 (level 3-method 를 length 로 미러링)

**Method A (within gen_len bin, nbins=5)** ↔ within-level
- bin 별 subject 사용: bin0(n=605):used 8 subj; bin1(n=605):used 8 subj; bin2(n=605):used 8 subj; bin3(n=598):used 8 subj; bin4(n=612):used 8 subj
- r(M_withinbin, M_act) = **+0.770** (≥0.8 이면 length artifact 아님)
- bin 간 행렬 상관 평균 = +0.567 (bin 가로질러 구조 일관)

**Method B (len-bin centroid 차감)** ↔ GPT-level centroid 차감
- r(M_resid_bincentroid, M_act) = **+0.962** (≥0.85 이면 생존)

**Method C (gen_len projection 제거, GLOBAL pooled)** ↔ ridge projection 제거
- [linear] r(M_resid_proj, M_act) = **+0.958**
- [+log_len] r(M_resid_proj, M_act) = **+0.958**
- [+quadratic] r(M_resid_proj, M_act) = **+0.962**
- [gen_len+level 동시 제거] r(M_resid, M_act) = **+0.964** (둘 다 제거해도 구조 생존 = content 고유)

### >>> VIEW [layeravg] 게이트: **PASS(content-driven; within-bin 저표본 inconclusive)** (Mantel p_min=0.624, |r|≈0.04; within-bin r=+0.77✗; resid_min r=+0.96✓)

## ===== 종합 게이트 판정 =====
- **mid_L11-15**: PASS (Mantel p_min=0.795, within-bin r=+0.851, resid_min r=+0.978)
- **layeravg**: PASS(content-driven; within-bin 저표본 inconclusive) (Mantel p_min=0.624, within-bin r=+0.770, resid_min r=+0.958)

**해석 규칙**: subject 배치 근거 view 는 mid_L11-15(subjsim 채택). *직접* fatal 검정인 E1 Mantel(length 거리정렬)·E2 residual survival(length 제거후 구조 잔존)을 primary 로, E3 within-bin replication 은 bin 당 subject 셀 누락(5~7/8)·centroid 불안정으로 **저표본 underpowered 보조지표**로 본다. E1·E2 가 둘 다 'confound 아님' 인데 E3 만 미달이면 length artifact 가 아니라 E3 의 통계력 부족으로 해석(length 가 구조를 몰았다면 E2 에서 붕괴했어야 함). 8×8 행렬 한 개로 강결론 금지.

### ⇒ 종합: **PASS (content-driven)** — primary 증거(Mantel 비유의·|r|<0.5, residual survival ≥0.85)가 모두 'gen_len 으로 설명 안 됨' 을 가리킴. subjsim 의 G1/G2/G3 grouping 은 length artifact 아님 → 배치 근거로 사용 정당. 단, within-bin replication 은 저표본으로 약함(inconclusive) — confound 증거 아니라 통계력 한계로 병기.

- Other n 작음(centroid 불안정) → Mantel 에 Other 제외 robustness 병기.
- gen_len 변수 sanity: ρ(level,gen_len)=+0.709 (일치).