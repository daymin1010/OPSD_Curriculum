#!/usr/bin/env python3
"""
subject_similarity_gate.py — SUBJECT 유사도 구조 게이트 (level 통제).  tag=subjsim
=================================================================================
목적 (커리큘럼 선결 게이트):
  curriculum 의 난이도 축은 GPT level stage 로 확정. subject 는 "모델 내부 표현상
  유사한 subject 를 같은/인접 stage 에 배치"(novelty)하려는 것. 이 스크립트는 그
  배치의 *선결 게이트* — "subject 간 유사도 구조가 LEVEL 오염을 빼고도 비자명하고
  안정적인가" 를 측정한다. subject 라벨=GPT, subject 간 *관계(유사도)*=activation
  으로 도출하므로 circularity 약함.

데이터: pooled (pilot1+pilot2) THINKING ΔA, pooled-mean(μ_pooled, per-layer) centering.
        (pooled_analysis.load_pooled 재사용; canonical finite N=3025.)
재사용: similarity_analysis(sa) 의 centroids / sim_matrix / layeravg_cos / spearman 등.
CPU only. 기존 산출물 미변경, 신규 태그 subjsim.

────────────────────────────────────────────────────────────────────────────────
작업 1 — Subject 유사도 (level 통제), 3 방식의 일관성으로 게이트 판정
  (A) within-level   : 각 level 안에서만 subject centroid 간 cosine → level 가로질러
                       가중평균(weight = level 안 두 subject 표본수의 min). level 고정
                       이므로 level 오염 0. 표본 적은 subject 셀(n<MIN_SUBJ_CELL)은 제외.
  (B-main) GPT-level centroid 차감 : 각 sample 에서 자기 level 그룹평균(per-layer)을
                       빼서 level 주효과 제거한 잔차에서 subject centroid cosine.
  (B-aux) ridge_level projection 제거 : currmat_artifacts.npz 의 difficulty_score_all
                       (3025, pooled load 순서 일치)로, centered ΔA 의 각 차원에서 그
                       스칼라 난이도점수와 상관된 1D 성분을 회귀 제거(= 난이도 축 projection
                       제거). refit/circularity 없음(GPT-level 영향 *제거* 용도).
  → 세 방식이 같은 subject 구조를 일관되게 주면(행렬 상관 높음) 게이트 결론이 단단해진다.

작업 2 — Supervised 대조 (검증용, 배치 정의 아님)
  mid-layer L11–15 특징 → PCA→LDA (pilot1 train / pilot2 test). subject confusion matrix
  + LDA 공간 subject centroid 거리. "어느 subject 가 헷갈리나(=내부 유사)" 가 작업1
  unsupervised 행렬과 맞는지 대조 + macro-F1(정보 존재 sanity).

작업 3 — 안정성 + 의미 + 게이트 판정
  pilot1 vs pilot2 각각 within-level subject 행렬 → 행렬 상관(안정성). 직관 sanity.
  hierarchical clustering + dendrogram. 게이트 판정.

레이어 뷰: layeravg(36) 와 mid-L11–15 를 *동등 후보* 로 작업1·3 전부 수행. 게이트 판정을
  두 view 각각 독립적으로 내리고, mid 에서라도 subject 구조가 안정적이면 mid 를 배치
  근거로 채택(subject 신호가 mid 에 집중 → layeravg 는 희석될 수 있음).

작업 4 (PASS 시) — 배치 재료(확정 X): subject 그룹 + 그룹 간 거리, level stage × subject
  그룹 결합 골격.

OUTPUT: REPORT_subject_similarity_gate.md, subjsim_artifacts.npz, heatmap/dendro PNG.
"""
from __future__ import annotations
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import similarity_analysis as sa
import pooled_analysis as pa

ANALYSIS = Path(__file__).resolve().parent
TAG = "subjsim"
OUT_MD = ANALYSIS / "REPORT_subject_similarity_gate.md"
OUT_NPZ = ANALYSIS / f"{TAG}_artifacts.npz"
CURRMAT_NPZ = ANALYSIS / "currmat_artifacts.npz"

LAYERS = 36
MID_LAYERS = list(range(11, 16))          # L11..L15 (subject-channel window)
VIEWS = {"layeravg": list(range(LAYERS)), "mid_L11-15": MID_LAYERS}

MIN_SUBJ_CELL = 5     # within-level: subject 가 해당 level 에서 가질 최소 표본수
SEED = 42

# 게이트 임계값 (리포트에 명시; 경계 사례는 보류로 서술)
GATE_STRUCT_STD = 0.05    # off-diag cosine 표준편차가 이 이상이면 "비자명 구조"
GATE_CONSIST_R = 0.60     # 3 방식 행렬 간 평균 상관
GATE_STABLE_R = 0.60      # pilot1 vs pilot2 행렬 상관


# ───────────────────────────── helpers ────────────────────────────────────
def offdiag(M):
    return M[np.triu_indices(len(M), 1)]


def mat_corr(A, B):
    """두 대칭행렬 상삼각 off-diagonal 의 Pearson 상관."""
    a, b = offdiag(np.asarray(A, float)), offdiag(np.asarray(B, float))
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def subject_sim_view(DA, md, subjects, layers, min_cell=2):
    """DA:(N,L,D). 주어진 layers 슬라이스에서 subject centroid layer-avg cosine 행렬.
    subject 별 표본 < min_cell 이면 그 subject 는 제외(반환 order 로 알림)."""
    vc = md["subject"].value_counts()
    order = [s for s in subjects if vc.get(s, 0) >= min_cell]
    if len(order) < 2:
        return None, order
    sub = np.ascontiguousarray(DA[:, layers, :])
    idxg = {s: md.index[md["subject"] == s].to_numpy() for s in order}
    cents = sa.centroids(sub, idxg)
    S = sa.sim_matrix(cents, order)
    return S, order


def within_level_subject_sim(DA, md, subjects, layers, min_cell=MIN_SUBJ_CELL):
    """작업1-(A): 각 level 안에서 subject centroid cosine → level 가로질러 가중평균.
    weight(a,b,level) = min(n_a^level, n_b^level) (둘 다 >= min_cell 일 때만).
    Returns: S (8x8, subjects order), W (사용된 가중치 합), per_level_info(list)."""
    nS = len(subjects)
    sidx = {s: i for i, s in enumerate(subjects)}
    num = np.zeros((nS, nS), float)
    den = np.zeros((nS, nS), float)
    levels = sorted(md["level"].unique().tolist())
    info = []
    sub_all = np.ascontiguousarray(DA[:, layers, :])
    for lv in levels:
        lvmask = (md["level"] == lv).to_numpy()
        md_lv = md.loc[lvmask].reset_index(drop=True)
        DA_lv = sub_all[lvmask]
        vc = md_lv["subject"].value_counts()
        present = [s for s in subjects if vc.get(s, 0) >= min_cell]
        if len(present) < 2:
            info.append((lv, len(md_lv), present, "skip(<2 subj w/ enough n)"))
            continue
        idxg = {s: md_lv.index[md_lv["subject"] == s].to_numpy() for s in present}
        cents = sa.centroids(DA_lv, idxg)
        S_lv = sa.sim_matrix(cents, present)
        for ia in range(len(present)):
            for ib in range(ia + 1, len(present)):
                sa_, sb_ = present[ia], present[ib]
                w = float(min(len(idxg[sa_]), len(idxg[sb_])))
                i, j = sidx[sa_], sidx[sb_]
                num[i, j] += w * S_lv[ia, ib]; num[j, i] = num[i, j]
                den[i, j] += w; den[j, i] = den[i, j]
        info.append((lv, len(md_lv), present, f"used {len(present)} subj"))
    S = np.eye(nS)
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(den > 0, num / den, np.nan)
    for i in range(nS):
        for j in range(nS):
            if i != j:
                S[i, j] = off[i, j]
    return S, den, info


def level_centroid_residual(DA_c, md):
    """(B-main) 각 sample 에서 자기 level 그룹평균(per-layer)을 뺀 잔차. DA_c:(N,L,D) f32."""
    resid = DA_c.copy()
    for lv in md["level"].unique():
        idx = md.index[md["level"] == lv].to_numpy()
        if len(idx) == 0:
            continue
        resid[idx] -= DA_c[idx].mean(axis=0, keepdims=True)
    return resid


def ridge_projection_residual(DA_c, score):
    """(B-aux) centered ΔA 의 각 (layer,dim) 에서 난이도 스칼라 score 와 상관된 1D 성분
    회귀 제거: resid_d = X_d - b_d * s_c,  b_d = <X_d, s_c> / <s_c, s_c>."""
    s = np.asarray(score, float)
    s_c = s - s.mean()
    denom = float((s_c * s_c).sum())
    if denom <= 0:
        return DA_c.copy()
    # b[l,d] = sum_n X[n,l,d]*s_c[n] / denom
    b = np.tensordot(s_c, DA_c, axes=(0, 0)) / denom      # (L,D)
    resid = DA_c - s_c[:, None, None] * b[None, :, :]
    return resid


def heatmap(S, order, title, path):
    n = len(order)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2, 1.0 * n + 2))
    im = ax.imshow(S, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(o) for o in order], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([str(o) for o in order], fontsize=7)
    for i in range(n):
        for j in range(n):
            v = S[i, j]
            ax.text(j, i, ("-" if not np.isfinite(v) else f"{v:.2f}"),
                    ha="center", va="center", fontsize=6, color="black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def dendro(S, order, title, path):
    """S: 대칭 cosine sim. dist=1-cos, average linkage. flat 2/3 cluster cut 반환."""
    D = 1.0 - np.asarray(S, float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.nan_to_num(D, nan=float(np.nanmax(D[np.isfinite(D)])) if np.isfinite(D).any() else 1.0)
    D = np.clip(D, 0.0, 2.0)
    Z = linkage(squareform(D, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(order) + 2), 5))
    dendrogram(Z, labels=[str(o) for o in order], ax=ax, leaf_rotation=45,
               leaf_font_size=9, color_threshold=0.7 * D.max())
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("distance (1 - centroid cosine)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    cuts = {}
    for k in (2, 3):
        fam = fcluster(Z, t=k, criterion="maxclust")
        groups = {}
        for lab, f in zip(order, fam):
            groups.setdefault(int(f), []).append(str(lab))
        cuts[k] = groups
    return Z, cuts, D


def fmt_mat(S, order):
    w = max(8, max(len(str(o)) for o in order) + 1)
    head = " " * w + "".join(f"{str(o)[:7]:>8}" for o in order)
    rows = [head]
    for i, o in enumerate(order):
        cells = "".join((f"{'-':>8}" if not np.isfinite(S[i, j]) else f"{S[i, j]:>8.3f}")
                        for j in range(len(order)))
        rows.append(f"{str(o):>{w}}" + cells)
    return "\n".join(rows)


# ───────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    ap.add_argument("--min-cell", type=int, default=MIN_SUBJ_CELL,
                    help="within-level subject 최소 표본수")
    ap.add_argument("--pca-comps", type=int, default=150, help="LDA 전 PCA 차원")
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ── load pooled THINKING (centered by μ_pooled), drop FAITHFUL ──
    DAF, DAT, md, ninfo = pa.load_pooled(args.max_n)
    del DAF; gc.collect()
    N = len(md)
    subjects = sorted(md["subject"].unique().tolist())
    print(f"[load] N={N}, subjects={subjects} ({time.time()-t0:.0f}s)", flush=True)

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    del DAT; gc.collect()

    # ── ridge_level 점수 정렬 (B-aux) ──
    score = None
    if CURRMAT_NPZ.exists():
        z = np.load(CURRMAT_NPZ, allow_pickle=True)
        s = z["difficulty_score_all"]
        if len(s) == N:
            score = np.asarray(s, float)
            print(f"[ridge] difficulty_score_all 정렬 OK (len={N})", flush=True)
        else:
            print(f"[ridge] 길이 불일치 (npz={len(s)} vs N={N}) → B-aux 생략", flush=True)
    else:
        print("[ridge] currmat_artifacts.npz 없음 → B-aux 생략", flush=True)

    saved = {"subjects": np.array(subjects)}
    L = []
    L.append("# SUBJECT 유사도 구조 게이트 (level 통제) — tag=subjsim")
    L.append("")
    L.append("> **목적.** 난이도축=GPT level stage 확정. subject 는 내부표현 유사 subject 를")
    L.append("> 같은/인접 stage 에 배치(novelty)하려는 것. 이 리포트는 그 *선결 게이트* —")
    L.append("> \"subject 유사도 구조가 LEVEL 오염을 빼고도 비자명하고 안정적인가\".")
    L.append("> subject 라벨=GPT, subject 간 관계(유사도)=activation → circularity 약함.")
    L.append("")
    L.append("**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. "
             "metric = group-centroid layer-averaged cosine (sa.* 재사용). CPU only.")
    L.append("")
    L.append("## Population (canonical N)")
    L.append(f"- raw .pt = **{ninfo['n_loaded']}**, non-finite drop = **{ninfo['n_nonfinite']}**, "
             f"**finite N = {ninfo['n_final']}** ('3000' 은 별칭).")
    pc = md["_pilot"].value_counts()
    L.append(f"- provenance: " + ", ".join(f"{k}={int(v)}" for k, v in pc.items()))
    L.append(f"- subjects (8-canonical): {subjects}")
    L.append(f"- 레이어 뷰: layeravg(36) 와 mid-L11–15 를 **동등 후보** 로 각각 판정.")
    L.append(f"- within-level 최소 subject 셀 = {args.min_cell}. (표본 적은 셀 제외/명시.)")
    L.append("")
    L.append("**게이트 임계값** (경계 사례는 보류 서술): "
             f"구조 std(off-diag)≥{GATE_STRUCT_STD}; 3방식 평균 행렬상관≥{GATE_CONSIST_R}; "
             f"pilot1-2 행렬상관≥{GATE_STABLE_R}.")
    L.append("")

    # ── (B) 잔차 배열 준비 ──
    resid_main = level_centroid_residual(DA_c, md)
    print(f"[resid] level-centroid 차감 완료 ({time.time()-t0:.0f}s)", flush=True)
    resid_aux = ridge_projection_residual(DA_c, score) if score is not None else None
    if resid_aux is not None:
        print(f"[resid] ridge projection 제거 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ── 작업 1 + 3 (view 별) ──
    view_gate = {}
    for vname, layers in VIEWS.items():
        L.append(f"\n## ===== VIEW: {vname} (layers={layers if len(layers)<=6 else f'{layers[0]}..{layers[-1]} ({len(layers)})'}) =====")

        # (A) within-level
        S_A, den_A, info_A = within_level_subject_sim(DA_c, md, subjects, layers, args.min_cell)
        # (B-main) level-centroid 차감 후 subject sim (전 sample)
        S_Bm, ord_Bm = subject_sim_view(resid_main, md, subjects, layers)
        # (B-aux)
        if resid_aux is not None:
            S_Ba, ord_Ba = subject_sim_view(resid_aux, md, subjects, layers)
        else:
            S_Ba = None

        saved[f"{vname}_A_within_level"] = S_A
        saved[f"{vname}_B_levelcentroid"] = S_Bm
        if S_Ba is not None:
            saved[f"{vname}_B_ridgeproj"] = S_Ba

        L.append("\n### 작업1 — subject 유사도 행렬 (level 통제, 3 방식)")
        L.append("\n**(A) within-level (level 안 subject centroid cosine, level 가중평균)**")
        L.append("```\n" + fmt_mat(S_A, subjects) + "\n```")
        L.append("per-level 사용 현황: " +
                 "; ".join(f"L{lv}(n={n}):{note}" for lv, n, pres, note in info_A))
        L.append("\n**(B-main) GPT-level centroid 차감 후 subject cosine**")
        L.append("```\n" + fmt_mat(S_Bm, ord_Bm) + "\n```")
        if S_Ba is not None:
            L.append("\n**(B-aux) ridge_level projection 제거 후 subject cosine**")
            L.append("```\n" + fmt_mat(S_Ba, ord_Ba) + "\n```")
        else:
            L.append("\n**(B-aux) ridge projection 제거: 생략 (점수 정렬 불가/smoke).**")

        # heatmaps
        heatmap(S_A, subjects, f"{vname} (A) within-level subject cosine", ANALYSIS / f"heatmap_{TAG}_{vname}_A.png")
        heatmap(S_Bm, ord_Bm, f"{vname} (B-main) level-centroid resid", ANALYSIS / f"heatmap_{TAG}_{vname}_Bmain.png")
        if S_Ba is not None:
            heatmap(S_Ba, ord_Ba, f"{vname} (B-aux) ridge-proj resid", ANALYSIS / f"heatmap_{TAG}_{vname}_Baux.png")

        # 3 방식 일관성 (공통 subject order = subjects, 단 행렬은 동일 order)
        mats = {"A": S_A, "Bmain": S_Bm}
        if S_Ba is not None:
            mats["Baux"] = S_Ba
        keys = list(mats.keys())
        consist = []
        L.append("\n### 작업1 — 3 방식 행렬 일관성 (off-diag Pearson r)")
        for ii in range(len(keys)):
            for jj in range(ii + 1, len(keys)):
                r = mat_corr(mats[keys[ii]], mats[keys[jj]])
                consist.append(r)
                L.append(f"- r({keys[ii]}, {keys[jj]}) = {r:+.3f}")
        consist_mean = float(np.nanmean(consist)) if consist else float("nan")
        L.append(f"- **평균 일관성 r = {consist_mean:+.3f}**")

        # 구조 비자명성 (A 행렬 off-diag 통계)
        od = offdiag(S_A); od = od[np.isfinite(od)]
        struct_std = float(od.std()) if od.size else float("nan")
        L.append(f"\n### 작업3 — 구조 비자명성: off-diag(A) mean={od.mean():+.3f}, "
                 f"std={struct_std:.3f}, min={od.min():+.3f}, max={od.max():+.3f}")

        # 안정성 (pilot1 vs pilot2, within-level A)
        S_p = {}
        for pilot in ("pilot1", "pilot2"):
            pmask = (md["_pilot"] == pilot).to_numpy()
            md_p = md.loc[pmask].reset_index(drop=True)
            DA_p = DA_c[pmask]
            Sp, _, _ = within_level_subject_sim(DA_p, md_p, subjects, layers, args.min_cell)
            S_p[pilot] = Sp
            saved[f"{vname}_A_{pilot}"] = Sp
        stab_r = mat_corr(S_p["pilot1"], S_p["pilot2"])
        L.append(f"### 작업3 — 안정성: pilot1 vs pilot2 within-level(A) 행렬 r = **{stab_r:+.3f}**")

        # 직관 sanity
        L.append("\n### 작업3 — 직관 sanity (A 행렬)")
        def cos(a, b):
            if a in subjects and b in subjects:
                v = S_A[subjects.index(a), subjects.index(b)]
                return v
            return float("nan")
        pairs_chk = [("Algebra", "Intermediate Algebra"), ("Algebra", "Prealgebra"),
                     ("Algebra", "Precalculus"), ("Number Theory", "Counting & Probability"),
                     ("Geometry", "Number Theory")]
        for a, b in pairs_chk:
            L.append(f"- cos({a}, {b}) = {cos(a,b):+.3f}")
        # 각 subject 의 최근접 subject
        L.append("- 각 subject 최근접(A):")
        for i, s in enumerate(subjects):
            row = S_A[i].copy(); row[i] = -np.inf
            j = int(np.nanargmax(np.where(np.isfinite(row), row, -np.inf)))
            L.append(f"    {s} → {subjects[j]} ({S_A[i, j]:+.3f})")

        # dendrogram (A)
        _, cuts, _ = dendro(S_A, subjects, f"{vname} (A) subject clustering",
                            ANALYSIS / f"dendro_{TAG}_{vname}_A.png")
        L.append("\n### 작업3 — hierarchical clustering (A, average-linkage)")
        for k, groups in cuts.items():
            L.append(f"- {k}-cluster cut: " +
                     " | ".join(f"G{g}={{{', '.join(m)}}}" for g, m in sorted(groups.items())))
        saved[f"{vname}_cut3"] = np.array(str(cuts.get(3)))

        # view 별 게이트 판정
        struct_ok = np.isfinite(struct_std) and struct_std >= GATE_STRUCT_STD
        consist_ok = np.isfinite(consist_mean) and consist_mean >= GATE_CONSIST_R
        stable_ok = np.isfinite(stab_r) and stab_r >= GATE_STABLE_R
        verdict = "PASS" if (struct_ok and stable_ok and consist_ok) else "FAIL"
        if (struct_ok and stable_ok) and not consist_ok:
            verdict = "BORDERLINE(구조·안정 OK, 방식 일관성 약)"
        view_gate[vname] = {"struct_std": struct_std, "consist_mean": consist_mean,
                            "stab_r": stab_r, "verdict": verdict,
                            "cuts": cuts, "S_A": S_A}
        L.append(f"\n### >>> VIEW [{vname}] 게이트: **{verdict}**  "
                 f"(struct_std={struct_std:.3f}{'✓' if struct_ok else '✗'}, "
                 f"consist_r={consist_mean:+.3f}{'✓' if consist_ok else '✗'}, "
                 f"stable_r={stab_r:+.3f}{'✓' if stable_ok else '✗'})")

    del resid_main
    if resid_aux is not None:
        del resid_aux
    gc.collect()

    # ── 작업 2 — supervised LDA 대조 (mid-L11–15) ──
    L.append("\n## ===== 작업2 — Supervised 대조 (검증용, mid-L11–15 PCA→LDA) =====")
    lda_block = run_lda(DA_c, md, subjects, MID_LAYERS, args.pca_comps, L, saved)

    # LDA confusion vs unsupervised(A, mid) 대조
    if lda_block is not None and "mid_L11-15" in view_gate:
        conf_sym = lda_block["conf_sym"]; lda_order = lda_block["order"]
        # A(mid) 를 같은 order 로 정렬
        S_A_mid = view_gate["mid_L11-15"]["S_A"]
        idx = [subjects.index(s) for s in lda_order if s in subjects]
        if len(idx) == len(lda_order):
            S_A_aligned = S_A_mid[np.ix_(idx, idx)]
            r_conf = mat_corr(conf_sym, S_A_aligned)
            L.append(f"\n### 작업2 — LDA confusion(대칭) vs unsup A(mid) 행렬 r = **{r_conf:+.3f}** "
                     "(양수 = 헷갈리는 subject 쌍이 unsup 에서도 고유사 → 대조 일치)")
            saved["lda_conf_vs_A_mid_r"] = np.array([r_conf])

    # ── 종합 게이트 + 작업4 ──
    L.append("\n## ===== 종합 게이트 판정 =====")
    for vname, g in view_gate.items():
        L.append(f"- **{vname}**: {g['verdict']} "
                 f"(struct_std={g['struct_std']:.3f}, consist_r={g['consist_mean']:+.3f}, "
                 f"stable_r={g['stab_r']:+.3f})")
    any_pass = any(g["verdict"].startswith("PASS") for g in view_gate.values())
    mid_pass = view_gate.get("mid_L11-15", {}).get("verdict", "").startswith("PASS")
    L.append("")
    L.append("**판정 규칙**: subject 신호는 mid-layer 에 집중 → layeravg 는 희석될 수 있으므로 "
             "두 view 중 *어느 하나라도* (특히 mid) PASS 면 그 view 를 배치 근거로 채택. "
             "둘 다 FAIL 이면 subject 는 mixing(다양성) 용도로만 권고.")
    if any_pass:
        basis = "mid_L11-15" if mid_pass else next(v for v, g in view_gate.items() if g["verdict"].startswith("PASS"))
        L.append(f"\n### ⇒ 종합: **PASS** (배치 근거 view = `{basis}`). 작업4 배치 재료 생성.")
        g = view_gate[basis]
        L.append(f"\n## ===== 작업4 — 배치 재료 (확정 X, view={basis}) =====")
        cuts = g["cuts"]; S_A = g["S_A"]
        groups3 = cuts.get(3, cuts.get(2))
        L.append(f"- subject 그룹 후보 (3-cluster cut @ {basis}):")
        for gid, members in sorted(groups3.items()):
            L.append(f"    그룹 {gid}: {{{', '.join(members)}}}")
        # 그룹 간 평균 cosine (인접성)
        gl = {gid: [subjects.index(m) for m in members] for gid, members in groups3.items()}
        gids = sorted(gl)
        L.append("- 그룹 간 평균 cosine (높을수록 인접 → 인접 stage 후보):")
        for ii in range(len(gids)):
            for jj in range(ii + 1, len(gids)):
                vals = [S_A[a, b] for a in gl[gids[ii]] for b in gl[gids[jj]] if np.isfinite(S_A[a, b])]
                mv = float(np.mean(vals)) if vals else float("nan")
                L.append(f"    G{gids[ii]}↔G{gids[jj]}: {mv:+.3f}")
        L.append("\n- **결합 골격(확정 X)**: 난이도축=GPT level stage(고정) × subject 그룹(위 cluster). "
                 "각 level stage 내부에서 유사 subject 그룹을 같은/인접 배치, 그룹 간 거리로 인접성 결정. "
                 "stage 경계·schedule·혼합비는 이후 세션에서 점진 확정.")
    else:
        L.append("\n### ⇒ 종합: **FAIL** — level 통제 후 subject 구조가 약/불안정. "
                 "subject 는 배치축이 아니라 **mixing(다양성) 용도**로만 사용 권고. 작업4 생략.")

    np.savez(OUT_NPZ, **saved)
    OUT_MD.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}", flush=True)
    print(f"[OK] wrote {OUT_NPZ}  (total {time.time()-t0:.0f}s)", flush=True)


def run_lda(DA_c, md, subjects, layers, pca_comps, L, saved):
    """mid-layer PCA→LDA, pilot1 train / pilot2 test. confusion + centroid dist."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import confusion_matrix, f1_score
    except Exception as e:
        L.append(f"- [LDA] sklearn import 실패: {e}"); return None

    tr = (md["_pilot"] == "pilot1").to_numpy()
    te = (md["_pilot"] == "pilot2").to_numpy()
    if tr.sum() < 50 or te.sum() < 50:
        L.append("- [LDA] train/test 표본 부족 → 생략 (smoke?)"); return None

    X = DA_c[:, layers, :].reshape(len(md), -1).astype(np.float32)
    Xtr, Xte = X[tr], X[te]
    ytr = md.loc[tr, "subject"].to_numpy()
    yte = md.loc[te, "subject"].to_numpy()

    nc = int(min(pca_comps, Xtr.shape[0] - 1, Xtr.shape[1]))
    pca = PCA(n_components=nc, svd_solver="randomized", random_state=SEED).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Ztr, ytr)
    pred = lda.predict(Zte)

    order = sorted(set(ytr.tolist()) & set(yte.tolist()))
    macro_f1 = f1_score(yte, pred, labels=order, average="macro", zero_division=0)
    cm = confusion_matrix(yte, pred, labels=order).astype(float)
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    conf_sym = (cm_norm + cm_norm.T) / 2.0
    np.fill_diagonal(conf_sym, 0.0)

    # LDA 공간 subject centroid 거리 (test)
    Zte_l = lda.transform(Zte)
    cents = np.stack([Zte_l[yte == s].mean(axis=0) for s in order])
    nO = len(order)
    Dc = np.zeros((nO, nO))
    for i in range(nO):
        for j in range(nO):
            Dc[i, j] = float(np.linalg.norm(cents[i] - cents[j]))

    L.append(f"- PCA({nc}) → LDA; pilot1 train(n={tr.sum()}) / pilot2 test(n={te.sum()}); "
             f"**macro-F1 = {macro_f1:.3f}** (chance≈{1/len(order):.3f}).")
    L.append("- subject confusion matrix (row-normalized, test):")
    L.append("```\n" + fmt_mat(cm_norm, order) + "\n```")
    L.append("- LDA 공간 subject centroid 거리 (작을수록 유사):")
    L.append("```\n" + fmt_mat(Dc, order) + "\n```")
    saved["lda_order"] = np.array(order)
    saved["lda_confusion_norm"] = cm_norm
    saved["lda_centroid_dist"] = Dc
    saved["lda_macro_f1"] = np.array([macro_f1])
    return {"order": order, "conf_sym": conf_sym, "cdist": Dc, "f1": macro_f1}


if __name__ == "__main__":
    main()
