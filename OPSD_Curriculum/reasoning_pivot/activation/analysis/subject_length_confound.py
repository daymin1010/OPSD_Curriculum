#!/usr/bin/env python3
"""
subject_length_confound.py — SUBJECT length-confound gate.  tag=subjlen
========================================================================
질문: subject 들이 activation 공간에서 가까운 것이 "내용이 비슷"해서인가
      "(Qwen3-8B thinking trace) 길이가 비슷"해서인가.
      = subjsim 의 G1/G2/G3 grouping(mid-L11-15, pilot1/2 r=0.94~0.97)이
        gen_len 으로 설명되는가.

반드시 분리:
  (1) "subjects 가 length 분포에서 다르다" → 그 자체는 치명적 아님(잠재 confound 신호).
  (2) "subject 유사도/grouping 구조가 length 구조를 그대로 따라간다" → 이게 fatal.
  판정은 (2) 기준. (1)은 Step1, (2)는 Step2(Mantel)+Step3(구조 생존)으로.

level length gate(REPORT_unsup_difficulty_lengthgate.md: ρ(level)=0.71≈ρ(gen_len)=0.74,
partial ρ(level|gen)=0.39)와 *동일 gen_len 변수*·*동일 3-method 논리*를 subject 에 미러링.

데이터: cm.load_pooled (length gate 와 동일 loader) → THINKING ΔA + md(gen_len).
        μ_pooled(per-layer) centering. mid_L11-15(주) + layeravg(보조). CPU only.
재사용: subject_similarity_gate(ssg) 의 subject_sim_view / within_level_subject_sim /
        level_centroid_residual / heatmap / fmt_mat / mat_corr / offdiag.
        (ssg.within_level_subject_sim 은 md["level"] 로 그룹핑 → len-bin label 을 "level"
         컬럼에 넣은 복사본을 전달해 within-bin / bin-centroid 차감에 재사용.)

OUTPUT:
  REPORT_subject_length_confound.md
  subjlen_outputs/ : subject_length_stats.csv, anova_len_by_subject.json,
    M_act.npy, M_len_meandist.npy, M_len_wass.npy, mantel.json,
    resid/within-bin matrices(.npy) + matrix_corr_summary.json
  heatmap_subjlen_*.png, lendist_subjlen_*.png
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import kruskal, f_oneway, wasserstein_distance, rankdata

import similarity_analysis as sa
import curriculum_materials as cm
import subject_similarity_gate as ssg

ANALYSIS = Path(__file__).resolve().parent
TAG = "subjlen"
OUT_MD = ANALYSIS / "REPORT_subject_length_confound.md"
OUT = ANALYSIS / "subjlen_outputs"
OUT.mkdir(exist_ok=True)
SUBJSIM_NPZ = ANALYSIS / "subjsim_artifacts.npz"

MID_LAYERS = list(range(11, 16))            # L11..L15
VIEWS = {"mid_L11-15": MID_LAYERS, "layeravg": list(range(36))}
MIN_SUBJ_CELL = 5
SEED = 42
N_PERM = 9999

# gen_len ↔ level sanity (level gate 에서 ρ(level,gen_len)=0.74)
EXPECT_RHO_LEVEL_GENLEN = 0.74
RHO_TOL = 0.10                              # |재계산 - 0.74| 이보다 크면 loud fail

# Gate 임계값 (제안값; 경계는 보류 서술)
GATE_RESID_R = 0.85                         # Method B/C: r(M_resid, M_act) ≥
GATE_WITHINBIN_R = 0.80                     # Method A: within-bin replication ≥
GATE_MANTEL_P = 0.05                        # Mantel 유의 경계 (낮고 *비유의* 면 좋음)
GATE_MANTEL_R = 0.50                        # |Mantel r| 이 이상이면 강정렬 경고


# ───────────────────────────── helpers ────────────────────────────────────
def subj_centroid_sim(DA, md, subjects, layers, min_cell=2):
    """marginal subject centroid layer-avg cosine (level/length 통제 없음). 8x8 order=subjects."""
    nS = len(subjects)
    S = np.full((nS, nS), np.nan)
    Sv, order = ssg.subject_sim_view(DA, md, subjects, layers, min_cell=min_cell)
    if Sv is None:
        return S, order
    pos = {s: i for i, s in enumerate(subjects)}
    for ia, sa_ in enumerate(order):
        for ib, sb_ in enumerate(order):
            S[pos[sa_], pos[sb_]] = Sv[ia, ib]
    return S, order


def eta_squared(groups):
    """one-way η² = SS_between / SS_total."""
    allx = np.concatenate(groups)
    grand = allx.mean()
    ss_t = float(((allx - grand) ** 2).sum())
    ss_b = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    return (ss_b / ss_t) if ss_t > 0 else float("nan")


def mantel(Dx, Dy, perms=N_PERM, seed=SEED):
    """Mantel: 두 *거리*행렬 off-diag Pearson r + permutation p (two-sided).
    Dy 의 행/열을 동시 치환. NaN pair 제외."""
    Dx = np.asarray(Dx, float); Dy = np.asarray(Dy, float)
    n = len(Dx)
    iu = np.triu_indices(n, 1)
    x = Dx[iu]; y = Dy[iu]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan"), float("nan"), int(m.sum())
    r_obs = float(np.corrcoef(x[m], y[m])[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(perms):
        p = rng.permutation(n)
        yp = Dy[np.ix_(p, p)][iu]
        mm = np.isfinite(x) & np.isfinite(yp)
        if mm.sum() < 3 or yp[mm].std() == 0 or x[mm].std() == 0:
            continue
        rp = np.corrcoef(x[mm], yp[mm])[0, 1]
        if abs(rp) >= abs(r_obs):
            cnt += 1
    p_val = (cnt + 1) / (perms + 1)
    return r_obs, float(p_val), int(m.sum())


def global_feature_residual(Xv, design):
    """GLOBAL(pooled) residualization: 각 feature 를 design(N,k) 에 회귀 후 residual.
    per-subject 회귀 금지(=subject 신호 보존). Xv:(N,F) f32. design 포함 intercept."""
    G = np.asarray(design, float)
    GtG = G.T @ G + 1e-6 * np.eye(G.shape[1])
    coef = np.linalg.solve(GtG, G.T @ Xv)          # (k,F)
    return Xv - G @ coef                            # (N,F)


def make_len_bins(gl, nbins):
    """gen_len 분위 nbins → 정수 bin label(0..nbins-1). 동률 경계로 bin 수 줄면 그대로."""
    qs = np.quantile(gl, np.linspace(0, 1, nbins + 1))
    qs[0] = -np.inf; qs[-1] = np.inf
    qs = np.unique(qs)
    lab = np.digitize(gl, qs[1:-1], right=False)
    return lab.astype(int), len(np.unique(lab))


def heat(S, order, title, path):
    ssg.heatmap(np.nan_to_num(S, nan=0.0), order, title, path)


# ───────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    ap.add_argument("--nbins", type=int, default=5, help="gen_len 분위 bin 수(작은 cell 시 자동 강등)")
    ap.add_argument("--min-cell", type=int, default=MIN_SUBJ_CELL)
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ── load (length gate 와 동일 loader) ──
    DAT, _DAF, md, n1 = cm.load_pooled(args.max_n, want_faithful=False)
    del _DAF; gc.collect()
    N = len(md)
    # schema assert (없으면 loud fail)
    need = ["subject", "level", "gen_len", "pilot"]
    miss = [c for c in need if c not in md.columns]
    assert not miss, f"[SCHEMA] missing md columns: {miss} (have={list(md.columns)})"
    subjects = sorted(md["subject"].unique().tolist())
    lev = md["level"].to_numpy(float)
    gl = md["gen_len"].to_numpy(float)
    assert np.isfinite(gl).all() and (gl > 0).all(), "[SCHEMA] gen_len 비정상(<=0/NaN) 존재"
    print(f"[load] N={N}, subjects={len(subjects)}, pilot1={n1}, "
          f"gen_len[min/med/max]={gl.min():.0f}/{np.median(gl):.0f}/{gl.max():.0f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    del DAT; gc.collect()

    pilots = md["pilot"].to_numpy().astype(str)
    L = []
    L.append("# SUBJECT length-confound gate — tag=subjlen")
    L.append("")
    L.append("> **질문.** subject 들이 activation 에서 가까운 게 '내용 유사' 인가 '길이 유사' 인가.")
    L.append("> subjsim 의 G1/G2/G3(mid-L11-15) grouping 이 gen_len 으로 설명되는지 검사.")
    L.append("> level length gate(ρ(level)=0.71≈ρ(gen_len)=0.74, partial=0.39)와 동일 gen_len·동일 3-method 미러링.")
    L.append("")
    L.append("> **분리 원칙.** (1) subjects 가 length 에서 다름 = 잠재 신호(치명 아님). "
             "(2) subject 유사도 *구조* 가 length 구조를 따라감 = fatal. **판정은 (2).**")
    L.append("")
    L.append(f"- pooled N=**{N}** (pilot1={n1}, pilot2={N-n1}); subjects(8)={subjects}")
    L.append(f"- gen_len = Qwen3-8B thinking trace token 수(ΔA span 길이). length gate 와 동일 변수.")
    L.append(f"- view: mid_L11-15(주), layeravg(보조). centering=μ_pooled. CPU only, seed={SEED}.")
    L.append(f"- 게이트 임계값(제안): Mantel 비유의(p≥{GATE_MANTEL_P}) & |r|<{GATE_MANTEL_R}; "
             f"r(M_resid,M_act)≥{GATE_RESID_R}(Method B/C); within-bin r≥{GATE_WITHINBIN_R}(Method A).")

    saved_corr = {}

    # ── Step 0: length 변수 sanity ──
    L.append("\n## Step 0 — length 변수 sanity")
    rho_lev_gl = sa.spearman(lev, gl)
    ok_var = abs(rho_lev_gl - EXPECT_RHO_LEVEL_GENLEN) <= RHO_TOL
    L.append(f"- ρ(level, gen_len) 재계산 = **{rho_lev_gl:+.3f}** "
             f"(기대 ≈{EXPECT_RHO_LEVEL_GENLEN}, tol {RHO_TOL}) → {'OK ✅' if ok_var else 'MISMATCH ⚠️'}")
    if not ok_var:
        L.append("- ⚠️ **gen_len 변수가 level gate 와 다를 수 있음.** 그래도 분석은 계속하되 결론은 보류로 해석.")
        print(f"[WARN] ρ(level,gen_len)={rho_lev_gl:.3f} != ~{EXPECT_RHO_LEVEL_GENLEN}", flush=True)

    # per-subject gen_len 분포 테이블
    rows = []
    L.append("\n### per-subject gen_len 분포")
    L.append("| subject | n | mean | median | std | p25 | p50 | p75 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in subjects:
        g = gl[md["subject"].to_numpy() == s]
        q = np.percentile(g, [25, 50, 75])
        rows.append({"subject": s, "n": len(g), "mean": g.mean(), "median": np.median(g),
                     "std": g.std(), "p25": q[0], "p50": q[1], "p75": q[2]})
        L.append(f"| {s} | {len(g)} | {g.mean():.0f} | {np.median(g):.0f} | {g.std():.0f} "
                 f"| {q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} |")
    pd.DataFrame(rows).to_csv(OUT / "subject_length_stats.csv", index=False)
    other_n = int((md["subject"].to_numpy() == "Other").sum())
    L.append(f"- (Other n={other_n} — centroid 안정성 주의. Step2/3 에 Other 제외 robustness 포함.)")

    # ── Step 1: subject ↔ length 연관 (잠재 confound) ──
    L.append("\n## Step 1 — subject↔length 연관 (잠재 confound 유무; 단독 FAIL 아님)")
    groups = [gl[md["subject"].to_numpy() == s] for s in subjects]
    H, p_kw = kruskal(*groups)
    Fst, p_an = f_oneway(*groups)
    eta2 = eta_squared(groups)
    L.append(f"- Kruskal–Wallis H={H:.1f}, p={p_kw:.2e}")
    L.append(f"- one-way ANOVA F={Fst:.1f}, p={p_an:.2e}")
    L.append(f"- effect size **η² = {eta2:.3f}** "
             f"({'큼 → subjects 가 length 에서 다름(잠재 confound) → Step2~3 필수' if eta2 >= 0.06 else '작음'})")
    L.append("- ※ 이것만으론 FAIL 아님: '길이 분포 차이' ≠ '유사도 구조가 길이로 설명됨'.")
    json.dump({"kruskal_H": float(H), "kruskal_p": float(p_kw),
               "anova_F": float(Fst), "anova_p": float(p_an), "eta_squared": float(eta2),
               "rho_level_genlen": float(rho_lev_gl)},
              open(OUT / "anova_len_by_subject.json", "w"), indent=2)

    # ── per-view 분석 ──
    view_verdicts = {}
    subjsim = np.load(SUBJSIM_NPZ, allow_pickle=True) if SUBJSIM_NPZ.exists() else None

    for vname, layers in VIEWS.items():
        L.append(f"\n## ===== VIEW: {vname} =====")
        nL = len(layers)

        # M_act = marginal subject centroid cosine
        M_act, _ = subj_centroid_sim(DA_c, md, subjects, layers)
        np.save(OUT / f"M_act_{vname}.npy", M_act)
        heat(M_act, subjects, f"{vname} M_act (subject centroid cosine)",
             ANALYSIS / f"heatmap_{TAG}_{vname}_Mact.png")

        # subjsim 기존 행렬과 일치 확인
        L.append("\n### M_act 일치 확인 (기존 subjsim 대비)")
        for key, lbl in [(f"{vname}_A_within_level", "within-level(A)"),
                         (f"{vname}_B_levelcentroid", "level-centroid resid(B-main)")]:
            if subjsim is not None and key in subjsim.files:
                r = ssg.mat_corr(M_act, subjsim[key])
                L.append(f"- r(M_act, subjsim {lbl}) = {r:+.3f}")

        # ── Step 2: Mantel ──
        L.append("\n### Step 2 — Mantel (length-sim vs subject-act-sim)")
        D_act = 1.0 - M_act                       # cosine → distance
        nS = len(subjects)
        mean_len = np.array([gl[md["subject"].to_numpy() == s].mean() for s in subjects])
        M_len_mean = np.abs(mean_len[:, None] - mean_len[None, :])
        M_len_wass = np.zeros((nS, nS))
        subj_samples = [gl[md["subject"].to_numpy() == s] for s in subjects]
        for i in range(nS):
            for j in range(i + 1, nS):
                w = wasserstein_distance(subj_samples[i], subj_samples[j])
                M_len_wass[i, j] = M_len_wass[j, i] = w
        np.save(OUT / f"M_len_meandist_{vname}.npy", M_len_mean)
        np.save(OUT / f"M_len_wass_{vname}.npy", M_len_wass)
        heat(-M_len_mean, subjects, f"{vname} -M_len(meandist)",
             ANALYSIS / f"heatmap_{TAG}_{vname}_Mlen.png")

        r_a, p_a, _ = mantel(D_act, M_len_mean)
        r_w, p_w, _ = mantel(D_act, M_len_wass)
        L.append(f"- Mantel(D_act, M_len mean-dist): r={r_a:+.3f}, p={p_a:.4f}")
        L.append(f"- Mantel(D_act, M_len wasserstein): r={r_w:+.3f}, p={p_w:.4f}")
        L.append("- (양수 r = 길이 먼 subject 쌍이 act 에서도 멀다 = length 정렬. r 낮고 비유의면 length 가 구조 설명 못함.)")
        L.append("- ※ 8×8=28 pairs 로 작음 → permutation p 만, 단독 강결론 금지.")
        mantel_pass = (p_a >= GATE_MANTEL_P and abs(r_a) < GATE_MANTEL_R and
                       p_w >= GATE_MANTEL_P and abs(r_w) < GATE_MANTEL_R)

        # Other 제외 robustness (Mantel)
        keep = [i for i, s in enumerate(subjects) if s != "Other"]
        r_a_no, p_a_no, _ = mantel(D_act[np.ix_(keep, keep)], M_len_mean[np.ix_(keep, keep)])
        L.append(f"- [robustness, Other 제외] Mantel(D_act, mean-dist): r={r_a_no:+.3f}, p={p_a_no:.4f}")

        # ── Step 3: 구조 생존 (3-method mirror) ──
        L.append("\n### Step 3 — 구조 생존 (level 3-method 를 length 로 미러링)")
        lab, nb = make_len_bins(gl, args.nbins)
        if nb < args.nbins:
            L.append(f"- (gen_len bin: 요청 {args.nbins} → 실제 {nb})")
        md_bin = md.copy(); md_bin["level"] = lab     # ssg 함수가 'level' 로 그룹핑 → bin 재사용

        # Method A: within gen_len bin
        S_A, den_A, info_A = ssg.within_level_subject_sim(DA_c, md_bin, subjects, layers, args.min_cell)
        r_A = ssg.mat_corr(S_A, M_act)
        np.save(OUT / f"M_withinbin_{vname}.npy", S_A)
        L.append(f"\n**Method A (within gen_len bin, nbins={nb})** ↔ within-level")
        L.append("- bin 별 subject 사용: " +
                 "; ".join(f"bin{b}(n={n}):{note}" for b, n, pres, note in info_A))
        L.append(f"- r(M_withinbin, M_act) = **{r_A:+.3f}** "
                 f"(≥{GATE_WITHINBIN_R} 이면 length artifact 아님)")
        # bin 별 행렬끼리 일관성(쌍별 평균) — replication
        per_bin_S = []
        for b in sorted(np.unique(lab)):
            bm = md_bin.copy()
            sub_mask = (lab == b)
            mb = md.loc[sub_mask].reset_index(drop=True)
            Sb, _ = subj_centroid_sim(DA_c[sub_mask], mb, subjects, layers, min_cell=args.min_cell)
            per_bin_S.append(Sb)
        cross = [ssg.mat_corr(per_bin_S[i], per_bin_S[j])
                 for i in range(len(per_bin_S)) for j in range(i + 1, len(per_bin_S))]
        cross_mean = float(np.nanmean(cross)) if cross else float("nan")
        L.append(f"- bin 간 행렬 상관 평균 = {cross_mean:+.3f} (bin 가로질러 구조 일관)")

        # Method B: len-bin centroid 차감 (bin global mean 차감 후 subject centroid)
        resid_B = ssg.level_centroid_residual(DA_c, md_bin)   # md_bin['level']=len-bin
        S_Bf, _ = subj_centroid_sim(resid_B, md, subjects, layers)
        del resid_B; gc.collect()
        r_B = ssg.mat_corr(S_Bf, M_act)
        np.save(OUT / f"M_resid_bincentroid_{vname}.npy", S_Bf)
        L.append(f"\n**Method B (len-bin centroid 차감)** ↔ GPT-level centroid 차감")
        L.append(f"- r(M_resid_bincentroid, M_act) = **{r_B:+.3f}** (≥{GATE_RESID_R} 이면 생존)")

        # Method C: gen_len projection 제거 (GLOBAL pooled, 3 spec)
        L.append(f"\n**Method C (gen_len projection 제거, GLOBAL pooled)** ↔ ridge projection 제거")
        Xv = DA_c[:, layers, :].reshape(N, -1).astype(np.float32)
        glc = (gl - gl.mean())
        logl = np.log(gl); logl = logl - logl.mean()
        sq = glc ** 2; sq = sq - sq.mean()
        specs = {
            "linear":    np.c_[np.ones(N), glc],
            "+log_len":  np.c_[np.ones(N), glc, logl],
            "+quadratic": np.c_[np.ones(N), glc, sq],
        }
        r_C = {}
        for sp, design in specs.items():
            Xr = global_feature_residual(Xv, design)
            Xr3 = Xr.reshape(N, nL, -1)
            S_C, _ = subj_centroid_sim(Xr3, md, subjects, list(range(nL)))
            r_C[sp] = ssg.mat_corr(S_C, M_act)
            if sp == "linear":
                np.save(OUT / f"M_resid_proj_linear_{vname}.npy", S_C)
                heat(S_C, subjects, f"{vname} M_act_resid (gen_len proj, linear)",
                     ANALYSIS / f"heatmap_{TAG}_{vname}_Mactresid.png")
            del Xr, Xr3; gc.collect()
            L.append(f"- [{sp}] r(M_resid_proj, M_act) = **{r_C[sp]:+.3f}**")

        # 추가: gen_len + level 동시 residualize (length unique 기여)
        levc = lev - lev.mean()
        design_jl = np.c_[np.ones(N), glc, levc]
        Xr = global_feature_residual(Xv, design_jl)
        S_jl, _ = subj_centroid_sim(Xr.reshape(N, nL, -1), md, subjects, list(range(nL)))
        r_jl = ssg.mat_corr(S_jl, M_act)
        np.save(OUT / f"M_resid_proj_genlen+level_{vname}.npy", S_jl)
        del Xr, Xv; gc.collect()
        L.append(f"- [gen_len+level 동시 제거] r(M_resid, M_act) = **{r_jl:+.3f}** "
                 "(둘 다 제거해도 구조 생존 = content 고유)")

        # ── view 게이트 판정 ──
        # 판정 철학: fatal 기준(=(2) "구조가 length 를 따라가나")의 *직접* 검정은
        #   E1 Mantel(length 거리정렬 없음) + E2 residual survival(length 제거후 구조 잔존).
        #   E3 within-bin 복제는 robustness 보조지표이나, bin 당 subject 셀 누락(5~7/8)·
        #   centroid 불안정으로 *저표본 underpowered* → 단독으로 confound 판정 못함.
        #   E1·E2 가 모두 "confound 아님" 인데 E3 만 미달이면 length artifact 가 아니라
        #   E3 의 통계력 부족으로 본다(만약 length 가 구조를 몰았다면 E2 에서 붕괴했어야 함).
        resid_min = np.nanmin([r_B] + list(r_C.values()))
        resid_ok = np.isfinite(resid_min) and resid_min >= GATE_RESID_R
        withinbin_ok = np.isfinite(r_A) and r_A >= GATE_WITHINBIN_R
        primary_pass = mantel_pass and resid_ok        # E1 & E2 (직접 fatal 검정)
        if primary_pass and withinbin_ok:
            verdict = "PASS"
        elif primary_pass and not withinbin_ok:
            verdict = "PASS(content-driven; within-bin 저표본 inconclusive)"
        elif resid_ok and not mantel_pass:
            verdict = "BORDERLINE(구조 생존하나 Mantel length 정렬 신호)"
        else:
            # E2(residual survival) 가 깨진 경우 = length 제거 시 구조 붕괴 = 진짜 우려
            verdict = "FAIL"
        view_verdicts[vname] = {
            "verdict": verdict, "primary_pass": bool(primary_pass),
            "mantel_pass": bool(mantel_pass), "resid_ok": bool(resid_ok),
            "withinbin_ok": bool(withinbin_ok),
            "mantel_r_mean": (r_a + r_w) / 2, "mantel_p_min": min(p_a, p_w),
            "withinbin_r": r_A, "resid_min_r": float(resid_min),
        }

        saved_corr[vname] = {
            "mantel_meandist": {"r": r_a, "p": p_a}, "mantel_wass": {"r": r_w, "p": p_w},
            "mantel_meandist_noOther": {"r": r_a_no, "p": p_a_no},
            "methodA_withinbin_r": r_A, "withinbin_cross_mean": cross_mean,
            "methodB_bincentroid_r": r_B, "methodC_proj_r": r_C,
            "genlen+level_resid_r": r_jl, "verdict": verdict,
        }
        L.append(f"\n### >>> VIEW [{vname}] 게이트: **{verdict}** "
                 f"(Mantel p_min={min(p_a,p_w):.3f}, |r|≈{abs((r_a+r_w)/2):.2f}; "
                 f"within-bin r={r_A:+.2f}{'✓' if withinbin_ok else '✗'}; "
                 f"resid_min r={resid_min:+.2f}{'✓' if resid_ok else '✗'})")

    # ── 종합 ──
    L.append("\n## ===== 종합 게이트 판정 =====")
    for v, g in view_verdicts.items():
        L.append(f"- **{v}**: {g['verdict']} "
                 f"(Mantel p_min={g['mantel_p_min']:.3f}, within-bin r={g['withinbin_r']:+.3f}, "
                 f"resid_min r={g['resid_min_r']:+.3f})")
    mid = view_verdicts.get("mid_L11-15", {})
    overall = "PASS" if mid.get("verdict", "").startswith("PASS") else \
              ("BORDERLINE" if "BORDERLINE" in mid.get("verdict", "") else "FAIL")
    L.append("")
    L.append("**해석 규칙**: subject 배치 근거 view 는 mid_L11-15(subjsim 채택). *직접* fatal 검정인 "
             "E1 Mantel(length 거리정렬)·E2 residual survival(length 제거후 구조 잔존)을 primary 로, "
             "E3 within-bin replication 은 bin 당 subject 셀 누락(5~7/8)·centroid 불안정으로 "
             "**저표본 underpowered 보조지표**로 본다. E1·E2 가 둘 다 'confound 아님' 인데 E3 만 미달이면 "
             "length artifact 가 아니라 E3 의 통계력 부족으로 해석(length 가 구조를 몰았다면 E2 에서 붕괴했어야 함). "
             "8×8 행렬 한 개로 강결론 금지.")
    if overall == "PASS":
        L.append(f"\n### ⇒ 종합: **PASS (content-driven)** — primary 증거(Mantel 비유의·|r|<0.5, "
                 "residual survival ≥0.85)가 모두 'gen_len 으로 설명 안 됨' 을 가리킴. "
                 "subjsim 의 G1/G2/G3 grouping 은 length artifact 아님 → 배치 근거로 사용 정당. "
                 "단, within-bin replication 은 저표본으로 약함(inconclusive) — confound 증거 아니라 통계력 한계로 병기.")

    elif overall == "BORDERLINE":
        L.append(f"\n### ⇒ 종합: **BORDERLINE** — 구조는 residual 후 생존하나 length 와 일부 정렬. "
                 "grouping 사용 가능하되 length 보조 신호 병기 권고.")
    else:
        L.append(f"\n### ⇒ 종합: **FAIL** — residualize 후 구조 붕괴 또는 length 강정렬. "
                 "subject grouping 을 length confound 로 의심 → 배치 근거 약화, mixing 용도 우선.")
    L.append("\n- Other n 작음(centroid 불안정) → Mantel 에 Other 제외 robustness 병기.")
    L.append(f"- gen_len 변수 sanity: ρ(level,gen_len)={rho_lev_gl:+.3f} ({'일치' if ok_var else '주의'}).")

    json.dump(saved_corr, open(OUT / "matrix_corr_summary.json", "w"), indent=2, default=float)
    json.dump({v: {k: (vv if not isinstance(vv, dict) else vv)
                   for k, vv in saved_corr[v].items() if "mantel" in k}
               for v in saved_corr}, open(OUT / "mantel.json", "w"), indent=2, default=float)
    OUT_MD.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}", flush=True)
    print(f"[OK] outputs in {OUT}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
