#!/usr/bin/env python3
"""
levsubj_length_confound_pooled.py — LENGTH-confound 3-method gate (POOLED N=3025).
==================================================================================
목적: REPORT_level_subject_similarity_pooled_N3025 의 SUBJECT/LEVEL centroid-cosine
      유사도가 "내용(content)" 때문인지 "gen_len(길이)" 때문인지를, 서로 독립적인
      3가지 각도로 검사한다. **subject·level 둘 다**에 적용.

세 방법 (각각 무엇을 통제하나 — 보고서에 그대로 인용):
  ① Mantel        : 그룹 활성화 *거리행렬*(1-cos) vs gen_len *거리행렬*(평균차/Wasserstein)
                    의 off-diag Pearson r + permutation p. [거리행렬 상관]
                    낮고 비유의면 = length 정렬 아님.
  ② residual      : 각 feature 를 gen_len(±log/±quad, +level)에 GLOBAL 회귀한 *잔차* 에서
    survival        그룹 centroid-cosine 행렬을 다시 만들어 원본 M_act 과 상관.
                    높게 생존하면 = length 제거해도 구조 유지(content). [회귀잔차]
  ③ gen_len-      : gen_len 5분위로 그룹 표본을 매칭한 *부분표본* 에서 separability gap
    balanced        (within-between) 재계산. gap 이 유지되면 = length artifact 아님.
                    [분위 매칭]

데이터/방법: pilot1+pilot2 pooled THINKING ΔA (canonical finite N=3025), 검증된
  similarity_analysis.py primitive 재사용(load_pilot/centroids/sim_matrix/
  within_between/perm_pvalue/genlen_balanced_indices). per-layer μ_pooled centering.
  뷰: layeravg(36) 주 + mid_L11-15 보조. CPU only. 기존 산출물 미변경(신규 파일명).

residual 은 메모리 안전을 위해 full residual array 를 만들지 않고 centroid-level 로
  조정한다: C_resid[g] = C_X[g] - (mean_G[g] @ coef), coef = (G'G)^-1 G'X.

OUTPUT (analysis/):
  LENGTHCONF_pooled3025.txt
  lengthconf_pooled_outputs.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import similarity_analysis as sa

BASE = Path("/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/"
            "reasoning_pivot/activation")
PILOT1 = BASE / "outputs/pilot/shifts"
PILOT2 = BASE / "outputs/pilot2/shifts"
OUT = BASE / "analysis"

SEED = 42
N_PERM_MANTEL = 9999
MID_LAYERS = list(range(11, 16))            # L11..L15
VIEWS = {"layeravg": list(range(36)), "mid_L11-15": MID_LAYERS}

# 게이트 임계값(제안; 경계는 보류 서술)
GATE_MANTEL_P = 0.05      # Mantel 비유의 경계
GATE_MANTEL_R = 0.50      # |Mantel r| 강정렬 경고
GATE_RESID_R = 0.85       # residual survival r(M_resid, M_act) ≥


# ───────────────────────────── helpers ────────────────────────────────────
def offdiag(M):
    return M[np.triu_indices(len(M), 1)]


def mat_corr(A, B):
    a, b = offdiag(np.asarray(A, float)), offdiag(np.asarray(B, float))
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def mantel(Dx, Dy, perms=N_PERM_MANTEL, seed=SEED):
    """두 거리행렬 off-diag Pearson r + permutation p (two-sided). Dy 행/열 동시 치환."""
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
        if abs(np.corrcoef(x[mm], yp[mm])[0, 1]) >= abs(r_obs):
            cnt += 1
    return r_obs, float((cnt + 1) / (perms + 1)), int(m.sum())


def centroid_view(DA_c, idxg, order, layers):
    """그룹 centroid (layers 슬라이스) + sim matrix."""
    sub = np.ascontiguousarray(DA_c[:, layers, :])
    cents = sa.centroids(sub, idxg)
    S = sa.sim_matrix(cents, order)
    return cents, S


def resid_centroid_sim(DA_c, idxg, order, layers, design):
    """residual survival: C_resid[g] = C_X[g] - (mean_G[g] @ coef).
    coef = (G'G)^-1 G'X, full residual array 미생성(메모리 안전)."""
    G = np.asarray(design, float)                       # (N,k)
    sub = np.ascontiguousarray(DA_c[:, layers, :]).astype(np.float32)  # (N,Lv,D)
    N, Lv, D = sub.shape
    GtG = G.T @ G + 1e-6 * np.eye(G.shape[1])
    GtX = np.tensordot(G, sub, axes=(0, 0))             # (k,Lv,D)
    coef = np.linalg.solve(GtG, GtX.reshape(G.shape[1], -1)).reshape(G.shape[1], Lv, D)
    cents_r = {}
    for g in order:
        idx = idxg[g]
        Cx = sub[idx].mean(axis=0)                      # (Lv,D)
        meanG = G[idx].mean(axis=0)                     # (k,)
        cents_r[g] = Cx - np.tensordot(meanG, coef, axes=(0, 0))
    return sa.sim_matrix(cents_r, order)


def len_distance_matrices(gl, group_samples, order):
    """그룹별 gen_len mean-abs-diff + Wasserstein 거리행렬."""
    nG = len(order)
    means = np.array([group_samples[g].mean() for g in order])
    M_mean = np.abs(means[:, None] - means[None, :])
    M_wass = np.zeros((nG, nG))
    for i in range(nG):
        for j in range(i + 1, nG):
            w = wasserstein_distance(group_samples[order[i]], group_samples[order[j]])
            M_wass[i, j] = M_wass[j, i] = w
    return M_mean, M_wass


# ───────────────────────────── main ───────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)
    t0 = time.time()

    print("[load] pilot1 ...", flush=True)
    _, DAT1, md1 = sa.load_pilot(PILOT1, args.max_n)
    print("[load] pilot2 ...", flush=True)
    _, DAT2, md2 = sa.load_pilot(PILOT2, args.max_n)
    DAT = np.concatenate([DAT1, DAT2], axis=0)
    md = pd.concat([md1, md2], ignore_index=True)
    N = len(md)
    n1 = len(md1)
    print(f"[pooled] N={N} (pilot1 {n1} + pilot2 {len(md2)}) ({time.time()-t0:.0f}s)", flush=True)
    assert N == len(DAT)

    mu = DAT.astype(np.float32).mean(axis=0, keepdims=True)
    DA_c = DAT.astype(np.float32) - mu
    del DAT, DAT1, DAT2

    gl = md["gen_len"].to_numpy(float)
    lev = md["level"].to_numpy(float)
    assert np.isfinite(gl).all() and (gl > 0).all(), "gen_len 비정상"
    rho_lev_gl = sa.spearman(lev, gl)

    # gen_len 회귀 design 성분(공통)
    glc = gl - gl.mean()
    logl = np.log(gl); logl = logl - logl.mean()
    sq = glc ** 2; sq = sq - sq.mean()
    levc = lev - lev.mean()

    results = {"N": N, "pilot1": n1, "pilot2": int(N - n1),
               "rho_level_genlen": rho_lev_gl, "views": {}}

    L = []
    L.append("## Length-confound robustness — 3-method 일치 (POOLED N=3025)")
    L.append("")
    L.append("> 본 절은 위 SUBJECT/LEVEL centroid-cosine 유사도가 gen_len(Qwen3-8B thinking")
    L.append("> trace 길이)으로 설명되는 가짜 신호인지, 내용(content) 신호인지를 서로 독립적인")
    L.append("> 3각도로 검사한다. **subject·level 둘 다** 적용. (handoff §0의 'length 보류'")
    L.append("> 입장을 약화하지 않는 보조 robustness — 활성화 난이도 축은 정당 신호로 사용.)")
    L.append("")
    L.append("**세 방법이 각각 통제하는 것 (중복 아님):**")
    L.append("- **① Mantel** = 그룹 활성화 *거리행렬*(1−cos) vs gen_len *거리행렬*(평균차/Wasserstein) "
             "off-diag Pearson r + perm p. 낮고 비유의 → length 정렬 아님. [거리행렬 상관]")
    L.append("- **② residual survival** = 각 feature 를 gen_len(±log/±quad, +level)에 GLOBAL 회귀한 "
             "*잔차* 에서 centroid-cosine 재계산 → 원본과 상관. 높게 생존 → content. [회귀잔차]")
    L.append("- **③ gen_len-balanced** = gen_len 5분위로 그룹 매칭한 *부분표본* 에서 gap 재현. "
             "유지되면 length artifact 아님. [분위 매칭]")
    L.append("")
    L.append(f"- pooled N=**{N}** (pilot1={n1}, pilot2={N-n1}); μ_pooled(per-layer) centering; "
             f"seed={SEED}; CPU only.")
    L.append(f"- ρ(level, gen_len) = **{rho_lev_gl:+.3f}** (level gate 와 동일 변수 sanity).")
    L.append(f"- 게이트(제안): Mantel 비유의(p≥{GATE_MANTEL_P}) & |r|<{GATE_MANTEL_R}; "
             f"residual survival r≥{GATE_RESID_R}. 8×8/8그룹 행렬 한 개로 강결론 금지.")

    groupings = [("subject", "SUBJECT"), ("level", "LEVEL")]

    # ── ① Mantel + ② residual survival (per view) ──
    for col, nm in groupings:
        vc = md[col].value_counts()
        if col == "level":
            order = sorted(vc.index.tolist(), key=lambda x: int(x))
        else:
            order = sorted(vc.index.tolist())
        idxg = {g: md.index[md[col] == g].to_numpy() for g in order}
        group_samples = {g: gl[idxg[g]] for g in order}
        M_mean, M_wass = len_distance_matrices(gl, group_samples, order)

        results["views"].setdefault(col, {})
        L.append(f"\n### ===== {nm} =====")
        L.append(f"groups({len(order)}) = {order}")

        for vname, layers in VIEWS.items():
            _, S_act = centroid_view(DA_c, idxg, order, layers)
            D_act = 1.0 - S_act

            r_a, p_a, _ = mantel(D_act, M_mean)
            r_w, p_w, _ = mantel(D_act, M_wass)
            mantel_pass = (p_a >= GATE_MANTEL_P and abs(r_a) < GATE_MANTEL_R and
                           p_w >= GATE_MANTEL_P and abs(r_w) < GATE_MANTEL_R)

            # residual specs
            specs = {
                "linear": np.c_[np.ones(N), glc],
                "+log_len": np.c_[np.ones(N), glc, logl],
                "+quadratic": np.c_[np.ones(N), glc, sq],
            }
            if col == "subject":
                specs["gen_len+level"] = np.c_[np.ones(N), glc, levc]
            r_resid = {}
            for sp, design in specs.items():
                S_r = resid_centroid_sim(DA_c, idxg, order, layers, design)
                r_resid[sp] = mat_corr(S_r, S_act)
            resid_min = float(np.nanmin(list(r_resid.values())))
            resid_ok = np.isfinite(resid_min) and resid_min >= GATE_RESID_R

            verdict = ("PASS(content-driven)" if (mantel_pass and resid_ok)
                       else ("BORDERLINE" if resid_ok else "FAIL"))

            results["views"][col][vname] = {
                "mantel_meandist": {"r": r_a, "p": p_a},
                "mantel_wass": {"r": r_w, "p": p_w},
                "mantel_pass": bool(mantel_pass),
                "residual_r": r_resid, "residual_min": resid_min,
                "resid_ok": bool(resid_ok), "verdict": verdict,
            }

            L.append(f"\n**[{vname}]**")
            L.append(f"- ① Mantel(D_act, gen_len mean-dist): r={r_a:+.3f}, p={p_a:.4f}")
            L.append(f"- ① Mantel(D_act, gen_len Wasserstein): r={r_w:+.3f}, p={p_w:.4f} "
                     f"→ {'비유의·약상관 ✅' if mantel_pass else '주의 ⚠️'}")
            L.append("- ② residual survival r(M_resid, M_act): " +
                     ", ".join(f"{sp}={rr:+.3f}" for sp, rr in r_resid.items()) +
                     f"  → min={resid_min:+.3f} {'(생존 ✅)' if resid_ok else '(붕괴 ⚠️)'}")
            L.append(f"- 뷰 판정: **{verdict}**")

    # ── ③ gen_len-balanced (layeravg, both groupings) ──
    L.append("\n### ===== ③ gen_len-balanced subsample (layeravg) =====")
    results["balanced"] = {}
    for col, nm in groupings:
        vc = md[col].value_counts()
        if col == "level":
            order = sorted(vc.index.tolist(), key=lambda x: int(x))
        else:
            order = sorted(vc.index.tolist())
        idxg = {g: md.index[md[col] == g].to_numpy() for g in order}

        # full gap (layeravg, centered)
        cents = sa.centroids(DA_c, idxg)
        _, _, gap_full, _ = sa.within_between(DA_c, idxg, cents, order)

        bidx = sa.genlen_balanced_indices(md, col, order)
        if bidx is None or len(bidx) < 2 * len(order):
            L.append(f"\n**{nm}**: gen_len-balanced subsample 사용 불가")
            results["balanced"][col] = {"gap_full": gap_full, "balanced": None}
            continue
        md_b = md.loc[bidx].reset_index(drop=True)
        DA_b = DA_c[bidx]
        idxg_b = {g: md_b.index[md_b[col] == g].to_numpy() for g in order
                  if (md_b[col] == g).sum() >= 2}
        ord_b = [g for g in order if g in idxg_b]
        cents_b = sa.centroids(DA_b, idxg_b)
        _, _, gap_bal, _ = sa.within_between(DA_b, idxg_b, cents_b, ord_b)
        labels_b = md_b[col].to_numpy()
        keep_b = np.isin(labels_b, ord_b)
        p_bal = sa.perm_pvalue(DA_b[keep_b], labels_b[keep_b], ord_b, gap_bal)

        retain = gap_bal / gap_full if gap_full != 0 else float("nan")
        results["balanced"][col] = {
            "gap_full": gap_full, "gap_balanced": gap_bal, "p_balanced": p_bal,
            "N_balanced": int(len(bidx)), "retain_frac": retain,
        }
        L.append(f"\n**{nm}** (balanced N={len(bidx)}, groups={len(ord_b)})")
        L.append(f"- full gap = {gap_full:+.4f} → balanced gap = {gap_bal:+.4f} "
                 f"(유지율 {retain:.0%}); perm p(balanced) = {p_bal:.4f} (N_PERM={sa.N_PERM})")

    # ── 종합 ──
    L.append("\n### ===== 종합 (3-method 일치) =====")
    for col, nm in groupings:
        mid = results["views"][col].get("mid_L11-15", {})
        lay = results["views"][col].get("layeravg", {})
        bal = results["balanced"].get(col, {})
        L.append(f"- **{nm}**: Mantel "
                 f"layeravg r≈{lay.get('mantel_meandist',{}).get('r',float('nan')):+.2f}"
                 f"(p={lay.get('mantel_meandist',{}).get('p',float('nan')):.3f}); "
                 f"residual_min layeravg={lay.get('residual_min',float('nan')):+.2f}, "
                 f"mid={mid.get('residual_min',float('nan')):+.2f}; "
                 f"balanced 유지율="
                 f"{(bal.get('retain_frac') if bal.get('retain_frac') is not None else float('nan')):.0%}"
                 if bal.get('retain_frac') is not None else
                 f"- **{nm}**: balanced 불가")
    L.append("")
    L.append("**해석.** 세 각도(거리행렬 상관·회귀잔차·분위매칭)가 모두 content-driven 으로 "
             "수렴하면 'length 가 유사도 구조를 설명하지 못함'을 강하게 지지한다. Mantel 은 "
             "그룹 수가 적어(8그룹) permutation p 만 보고, residual survival 과 balanced gap 을 "
             "주증거로 본다.")

    # write
    txt = OUT / "LENGTHCONF_pooled3025.txt"
    txt.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    json.dump(results, open(OUT / "lengthconf_pooled_outputs.json", "w"),
              indent=2, default=float)
    print("\n".join(str(x) for x in L))
    print(f"\n[OK] wrote {txt} and lengthconf_pooled_outputs.json "
          f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
