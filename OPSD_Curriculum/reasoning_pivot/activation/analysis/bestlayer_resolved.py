#!/usr/bin/env python3
"""
bestlayer_resolved.py — best-layer 윈도우별 SUBJECT/LEVEL/UNIT 정밀 *공정* 비교.
================================================================================
배경. 직전 세션(level_unit_resolved.py)에서 §B/§C 가 'level 우세 윈도우 L16–32'
한 군데서만 subject 까지 평가해 subject 한테 불리했음. 36레이어 layer-scan 으로
LEVEL/SUBJECT 신호가 *서로 다른 레이어 대역* 에 분리됨이 드러났고, 그 핸디캡을
제거한 공정 비교가 필요하다.

이 스크립트는 **새 통계 0줄**. `subject_layer_resolved`(=slr) 의 헬퍼와,
`level_unit_resolved`(=lur) 의 대칭 미러 헬퍼를 import 해서 호출만 한다.
또한 직전 산출물(`REPORT_levelunit_controlled_*.md`, `levunit_artifacts.npz`)을
**절대 수정/덮어쓰지 않는다.** 출력은 별도 파일.

윈도우 상수 (직전 보고서 §A.1 Top-5 union 으로 데이터에서 도출됨):
  W_SUBJ = [9, 10, 11, 12, 14]               # subject 우세 (Fisher∪probe-F1 Top-5)
  W_LEV  = [20, 25, 26, 27, 29, 30, 31]       # level   우세 (Fisher∪probe-F1 Top-5)
  W_ALL  = list(range(36))                    # 전체 36 레이어 (기준선)
  W_SUBJ ∩ W_LEV = ∅

수행 (각 (축, 윈도우) 마다):
  §S SUBJECT : W_SUBJ, W_ALL  (블록=level, 라벨=subject)
  §L LEVEL   : W_LEV,  W_ALL  (블록=subject, 라벨=level)
  §U UNIT    : W_SUBJ, W_LEV, W_ALL
각 (축, 윈도우)별 측정:
  1) layer-averaged L2-norm sample-pairwise cosine 행렬 (slr.window_cos_matrix)
  2) within-block same-label vs diff-label 분포 (lur.within_block_pair_dists)
     → mean cos, ratio, Cohen's d(slr.cohens_d), Mann–Whitney p(asymptotic)
  3) block-permutation ×n_perm (lur.block_perm)  — SUBJECT/LEVEL 축에서만
  4) LDA confusion: PCA(150)→LDA, pilot1 train→pilot2 test, macro-F1, row-norm cm
     (lur.lda_confusion_col) — SUBJECT/LEVEL 축에서만
  5) UNIT 축: same-unit vs diff-unit 전체쌍 cosine + Cohen's d + silhouette
     (unit/level/subject) + level-residual 후 subject/unit silhouette
     + 윈도우 평균 η²(level/subject/interaction)  (slr.eta2_partition)
마지막에 cross-window 비교표 한 장 출력.

OUTPUT:
  REPORT_bestlayer_resolved_<YYYY-MM-DD>.md
  bestlayer_artifacts.npz
  bestlayer_pairhist_<axis>_<win>.png (옵션, 기본 ON)
  bestlayer_unithist_<win>.png (옵션, UNIT)
"""
from __future__ import annotations
import argparse
import gc
import time
from datetime import date
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

import similarity_analysis as sa
import pooled_analysis as pa
import subject_similarity_gate as ssg
import subject_layer_resolved as slr
import level_unit_resolved as lur

ANALYSIS = Path(__file__).resolve().parent
TAG = "bestlayer"
LAYERS = slr.LAYERS
SEED = slr.SEED

# 윈도우 상수 (data-derived; 보고서 §A.1 Top-5 union)
W_SUBJ = [9, 10, 11, 12, 14]
W_LEV = [20, 25, 26, 27, 29, 30, 31]
W_ALL = list(range(LAYERS))

WIN_NAMES = {"Wsubj": W_SUBJ, "Wlev": W_LEV, "Wall": W_ALL}


def fmt_layers(layers):
    if len(layers) <= 8:
        return str(layers)
    return f"[L{min(layers)}..L{max(layers)} n={len(layers)}]"


def safe_mwu(a, b):
    try:
        if a.size == 0 or b.size == 0:
            return float("nan")
        u, p = mannwhitneyu(a, b, alternative="greater", method="asymptotic")
        return float(p)
    except Exception as e:
        print(f"    [mwu] 실패: {e}", flush=True)
        return float("nan")


def fmt_top_confusion(order, cm_norm, k=5):
    """confusion (row-norm) 대각 제외 대칭화 후 top-k 혼동쌍 텍스트."""
    cms = (cm_norm + cm_norm.T) / 2.0
    np.fill_diagonal(cms, 0.0)
    n = len(order)
    iu = np.triu_indices(n, 1)
    pairs = sorted(zip(cms[iu], [(order[a], order[b]) for a, b in zip(*iu)]),
                   reverse=True)[:k]
    return "; ".join(f"{a}↔{b}={v:.3f}" for v, (a, b) in pairs)


# ───────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    ap.add_argument("--n-perm", type=int, default=1000, help="block-permutation 횟수")
    ap.add_argument("--pca-comps", type=int, default=150, help="LDA PCA 차원")
    ap.add_argument("--no-png", action="store_true", help="히스토그램 PNG 생략")
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)
    t0 = time.time()

    DAF, DAT, md, ninfo = pa.load_pooled(args.max_n)
    del DAF; gc.collect()
    N = len(md)
    subjects = sorted(md["subject"].unique().tolist())
    levels = sorted(md["level"].unique().tolist())
    subj_arr = md["subject"].to_numpy()
    lev_arr = md["level"].to_numpy()
    unit_arr = (md["subject"].astype(str) + "|" + md["level"].astype(str)).to_numpy()
    n_units = len(np.unique(unit_arr))
    print(f"[load] N={N} subjects={subjects} levels={levels} units={n_units} "
          f"({time.time()-t0:.0f}s)", flush=True)

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    del DAT; gc.collect()

    print("[norm] per-layer L2-normalize members ...", flush=True)
    DAn = sa.normalize_members(DA_c)

    saved = {
        "subjects": np.array(subjects),
        "levels": np.array(levels),
        "n_units": np.array([n_units]),
        "W_SUBJ": np.array(W_SUBJ),
        "W_LEV": np.array(W_LEV),
        "W_ALL": np.array(W_ALL),
    }

    L = []
    today = date.today().isoformat()
    L.append(f"# best-layer 윈도우별 SUBJECT/LEVEL/UNIT 정밀 공정 비교 — {today}  (tag={TAG})")
    L.append("")
    L.append("> **목적.** 직전 보고서(`REPORT_levelunit_controlled_2026-06-22.md`)는 §B/§C 를")
    L.append("> `levwin_L16-32`(=level 우세 구간) 한 윈도우로만 SUBJECT/LEVEL 을 같이 평가해서")
    L.append("> subject 한테 불리했다. 본 보고서는 각 축마다 *그 축에 유리한 윈도우*(데이터에서")
    L.append("> 도출된 Fisher∪probe-F1 Top-5 union) 를 쓰는 *공정 비교* 를 별도 파일로 기록한다.")
    L.append("> 새 통계 로직은 없음 — `subject_layer_resolved`/`level_unit_resolved` 의 헬퍼만 호출.")
    L.append("")
    L.append(f"**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. "
             f"finite **N={ninfo['n_final']}** (raw {ninfo['n_loaded']}, non-finite drop "
             f"{ninfo['n_nonfinite']}). units={n_units}. CPU only. seed={SEED}.")
    pc = md["_pilot"].value_counts()
    L.append(f"- provenance: " + ", ".join(f"{k}={int(v)}" for k, v in pc.items()))
    L.append("")
    L.append("**윈도우 상수 (직전 `levunit_artifacts.npz` 의 36-vec Fisher/probe-F1 에서 도출)**")
    L.append(f"- `W_SUBJ` = {W_SUBJ}  (SUBJECT Fisher∪probe-F1 Top-5 union, n={len(W_SUBJ)})")
    L.append(f"- `W_LEV ` = {W_LEV}  (LEVEL Fisher∪probe-F1 Top-5 union, n={len(W_LEV)})")
    L.append(f"- `W_ALL ` = list(range({LAYERS}))  (전 레이어 기준선)")
    L.append(f"- `W_SUBJ ∩ W_LEV` = ∅  (두 신호는 서로 다른 레이어 대역에 분리)")
    L.append("")
    L.append("**측정 매트릭스**")
    L.append("- §S SUBJECT 정밀검증 : 윈도우 = W_SUBJ, W_ALL  (블록=level, 라벨=subject)")
    L.append("- §L LEVEL   정밀검증 : 윈도우 = W_LEV,  W_ALL  (블록=subject, 라벨=level)")
    L.append("- §U UNIT    정밀검증 : 윈도우 = W_SUBJ, W_LEV, W_ALL  (silhouette + η² + cohesion)")
    L.append("")

    # ─────────────────────────── §S SUBJECT ────────────────────────────────
    print("[S] SUBJECT  blocks=level  label=subject", flush=True)
    L.append("## §S. SUBJECT 정밀검증 (block=level, label=subject)")
    L.append("")
    L.append("metric = per-layer L2-norm 후 layer-averaged sample-pairwise cosine. "
             "level 블록 안에서 same-subject vs diff-subject 비교(→ level 오염 0). "
             "block-permutation 은 level 고정·subject 라벨만 셔플.")

    subject_windows = [("Wsubj", W_SUBJ), ("Wall", W_ALL)]
    for wname, layers in subject_windows:
        print(f"  [S/{wname}] cos matrix ({fmt_layers(layers)}) ...", flush=True)
        C = slr.window_cos_matrix(DAn, layers)
        same, diff, per_block = lur.within_block_pair_dists(C, md, "level", "subject")
        d = slr.cohens_d(same, diff)
        p_mwu = safe_mwu(same, diff)
        print(f"  [S/{wname}] block_perm n_perm={args.n_perm} ...", flush=True)
        obs, p_perm, nblk = lur.block_perm(C, md, "level", "subject", args.n_perm)
        print(f"  [S/{wname}] LDA confusion ...", flush=True)
        res, err = lur.lda_confusion_col(DA_c, md, layers, "subject", args.pca_comps)

        ratio = (same.mean() / diff.mean()) if (diff.size and diff.mean() != 0) else float("nan")
        key = f"S_{wname}"
        saved[f"{key}_same_mean"] = np.array([float(same.mean()) if same.size else np.nan])
        saved[f"{key}_diff_mean"] = np.array([float(diff.mean()) if diff.size else np.nan])
        saved[f"{key}_ratio"] = np.array([ratio])
        saved[f"{key}_cohend"] = np.array([d])
        saved[f"{key}_mwu_p"] = np.array([p_mwu])
        saved[f"{key}_perm_stat"] = np.array([obs])
        saved[f"{key}_perm_p"] = np.array([p_perm])
        saved[f"{key}_layers"] = np.array(layers)

        L.append("")
        L.append(f"### S/{wname}  layers={fmt_layers(layers)}")
        L.append(f"- within-level **same-subject** mean cos = {same.mean():+.4f} "
                 f"(n_pairs={same.size}); **diff-subject** mean cos = {diff.mean():+.4f} "
                 f"(n_pairs={diff.size}); ratio = **{ratio:.3f}x**")
        L.append(f"- **Cohen's d (same−diff) = {d:+.3f}**; "
                 f"Mann–Whitney p(same>diff) = {p_mwu:.2e}")
        L.append(f"- **block-permutation**(level 고정, subject 셔플 ×{args.n_perm}): "
                 f"stat(mean_same−mean_diff)={obs:+.4f}, **p={p_perm:.4f}** (blocks={nblk})")
        L.append("- per-level (same_mean / diff_mean / n_same / n_diff):")
        for bk, n, sm, dm, ns, nd in per_block:
            L.append(f"    level={bk} (n={n}): same={sm:+.4f} diff={dm:+.4f}  ({ns}/{nd} pairs)")
        if res is None:
            L.append(f"- LDA: 생략 ({err})")
            saved[f"{key}_lda_f1"] = np.array([np.nan])
        else:
            L.append(f"- LDA (PCA({args.pca_comps})→LDA, pilot1→pilot2): "
                     f"**macro-F1={res['f1']:.3f}** (chance≈{1/len(res['order']):.3f})")
            L.append("  confusion (row-normalized, test):")
            L.append("```\n" + ssg.fmt_mat(res['cm_norm'], [str(o) for o in res['order']]) + "\n```")
            L.append("  top 혼동쌍(=내부 유사): " +
                     fmt_top_confusion(res['order'], res['cm_norm']))
            saved[f"{key}_lda_f1"] = np.array([float(res["f1"])])
            saved[f"{key}_lda_cm"] = res["cm_norm"]
            saved[f"{key}_lda_order"] = np.array(res["order"])

        if (not args.no_png) and same.size and diff.size:
            lur.plot_pair_hist(same, diff,
                               f"S/{wname}: within-level same vs diff subject",
                               "within-level pairwise cosine",
                               ANALYSIS / f"{TAG}_pairhist_subject_{wname}.png")
        del C; gc.collect()

    # ─────────────────────────── §L LEVEL ──────────────────────────────────
    print("[L] LEVEL  blocks=subject  label=level", flush=True)
    L.append("")
    L.append("## §L. LEVEL 정밀검증 (block=subject, label=level)")
    L.append("")
    L.append("metric 동일. subject 블록 안에서 same-level vs diff-level 비교(→ subject 오염 0). "
             "block-permutation 은 subject 고정·level 라벨만 셔플.")

    level_windows = [("Wlev", W_LEV), ("Wall", W_ALL)]
    for wname, layers in level_windows:
        print(f"  [L/{wname}] cos matrix ({fmt_layers(layers)}) ...", flush=True)
        C = slr.window_cos_matrix(DAn, layers)
        same, diff, per_block = lur.within_block_pair_dists(C, md, "subject", "level")
        d = slr.cohens_d(same, diff)
        p_mwu = safe_mwu(same, diff)
        print(f"  [L/{wname}] block_perm n_perm={args.n_perm} ...", flush=True)
        obs, p_perm, nblk = lur.block_perm(C, md, "subject", "level", args.n_perm)
        print(f"  [L/{wname}] LDA confusion ...", flush=True)
        res, err = lur.lda_confusion_col(DA_c, md, layers, "level", args.pca_comps)

        ratio = (same.mean() / diff.mean()) if (diff.size and diff.mean() != 0) else float("nan")
        key = f"L_{wname}"
        saved[f"{key}_same_mean"] = np.array([float(same.mean()) if same.size else np.nan])
        saved[f"{key}_diff_mean"] = np.array([float(diff.mean()) if diff.size else np.nan])
        saved[f"{key}_ratio"] = np.array([ratio])
        saved[f"{key}_cohend"] = np.array([d])
        saved[f"{key}_mwu_p"] = np.array([p_mwu])
        saved[f"{key}_perm_stat"] = np.array([obs])
        saved[f"{key}_perm_p"] = np.array([p_perm])
        saved[f"{key}_layers"] = np.array(layers)

        L.append("")
        L.append(f"### L/{wname}  layers={fmt_layers(layers)}")
        L.append(f"- within-subject **same-level** mean cos = {same.mean():+.4f} "
                 f"(n_pairs={same.size}); **diff-level** mean cos = {diff.mean():+.4f} "
                 f"(n_pairs={diff.size}); ratio = **{ratio:.3f}x**")
        L.append(f"- **Cohen's d (same−diff) = {d:+.3f}**; "
                 f"Mann–Whitney p(same>diff) = {p_mwu:.2e}")
        L.append(f"- **block-permutation**(subject 고정, level 셔플 ×{args.n_perm}): "
                 f"stat(mean_same−mean_diff)={obs:+.4f}, **p={p_perm:.4f}** (blocks={nblk})")
        L.append("- per-subject (same_mean / diff_mean / n_same / n_diff):")
        for bk, n, sm, dm, ns, nd in per_block:
            L.append(f"    {bk} (n={n}): same={sm:+.4f} diff={dm:+.4f}  ({ns}/{nd} pairs)")
        if res is None:
            L.append(f"- LDA: 생략 ({err})")
            saved[f"{key}_lda_f1"] = np.array([np.nan])
        else:
            L.append(f"- LDA (PCA({args.pca_comps})→LDA, pilot1→pilot2): "
                     f"**macro-F1={res['f1']:.3f}** (chance≈{1/len(res['order']):.3f})")
            L.append("  confusion (row-normalized, test):")
            L.append("```\n" + ssg.fmt_mat(res['cm_norm'], [str(o) for o in res['order']]) + "\n```")
            L.append("  top 혼동쌍(=내부 유사): " +
                     fmt_top_confusion(res['order'], res['cm_norm']))
            saved[f"{key}_lda_f1"] = np.array([float(res["f1"])])
            saved[f"{key}_lda_cm"] = res["cm_norm"]
            saved[f"{key}_lda_order"] = np.array(res["order"])

        if (not args.no_png) and same.size and diff.size:
            lur.plot_pair_hist(same, diff,
                               f"L/{wname}: within-subject same vs diff level",
                               "within-subject pairwise cosine",
                               ANALYSIS / f"{TAG}_pairhist_level_{wname}.png")
        del C; gc.collect()

    # ─────────────────────────── §U UNIT ───────────────────────────────────
    print("[U] UNIT  same-unit vs diff-unit + silhouette + η²", flush=True)
    L.append("")
    L.append("## §U. UNIT(subject×level) 정밀검증")
    L.append("")
    L.append("metric: same-unit vs diff-unit *전체쌍* cosine (블록 없음). silhouette 는 "
             "동일 (1−cos) precomputed distance 로 unit/level/subject 3 라벨 모두 계산. "
             "level-residualize(ssg.level_centroid_residual) 후 subject/unit silhouette "
             "재계산. η² 는 전 레이어 1번 계산 후 윈도우 평균.")

    # η² 36-vec 1번
    print("  [U] η² partition (36 layers) ...", flush=True)
    eta_level = np.full(LAYERS, np.nan)
    eta_subj = np.full(LAYERS, np.nan)
    eta_inter = np.full(LAYERS, np.nan)
    for l in range(LAYERS):
        e = slr.eta2_partition(DA_c[:, l, :], lev_arr, subj_arr)
        eta_level[l] = e["level"]; eta_subj[l] = e["subject"]; eta_inter[l] = e["interaction"]
        if (l + 1) % 9 == 0:
            print(f"    eta layer {l+1}/{LAYERS} ({time.time()-t0:.0f}s)", flush=True)
    saved["eta_level"] = eta_level
    saved["eta_subject"] = eta_subj
    saved["eta_interaction"] = eta_inter

    # level-residual 1번 (윈도우 독립)
    print("  [U] level_centroid_residual + normalize ...", flush=True)
    resid = ssg.level_centroid_residual(DA_c, md)
    DAn_resid = sa.normalize_members(resid)
    del resid; gc.collect()

    unit_windows = [("Wsubj", W_SUBJ), ("Wlev", W_LEV), ("Wall", W_ALL)]
    for wname, layers in unit_windows:
        print(f"  [U/{wname}] cos matrix ({fmt_layers(layers)}) ...", flush=True)
        C = slr.window_cos_matrix(DAn, layers)
        su, du = lur.unit_pair_dists(C, md)
        du_mean = float(du.mean()) if du.size else float("nan")
        su_mean = float(su.mean()) if su.size else float("nan")
        ratio_u = (su_mean / du_mean) if (du.size and du_mean != 0) else float("nan")
        d_u = slr.cohens_d(su, du)
        sil_unit = slr.subject_silhouette(C, unit_arr)
        sil_level = slr.subject_silhouette(C, lev_arr)
        sil_subj = slr.subject_silhouette(C, subj_arr)
        del C; gc.collect()

        # level-residual silhouette (같은 윈도우에서 C_res 만들고 사용 후 del)
        print(f"  [U/{wname}] residual silhouette ...", flush=True)
        C_res = slr.window_cos_matrix(DAn_resid, layers)
        sil_subj_res = slr.subject_silhouette(C_res, subj_arr)
        sil_unit_res = slr.subject_silhouette(C_res, unit_arr)
        del C_res; gc.collect()

        eta_l_win = float(np.nanmean(eta_level[layers]))
        eta_s_win = float(np.nanmean(eta_subj[layers]))
        eta_i_win = float(np.nanmean(eta_inter[layers]))

        key = f"U_{wname}"
        saved[f"{key}_unit_same_mean"] = np.array([su_mean])
        saved[f"{key}_unit_diff_mean"] = np.array([du_mean])
        saved[f"{key}_unit_ratio"] = np.array([ratio_u])
        saved[f"{key}_unit_cohend"] = np.array([d_u])
        saved[f"{key}_sil_unit"] = np.array([sil_unit])
        saved[f"{key}_sil_level"] = np.array([sil_level])
        saved[f"{key}_sil_subject"] = np.array([sil_subj])
        saved[f"{key}_sil_subject_residlevel"] = np.array([sil_subj_res])
        saved[f"{key}_sil_unit_residlevel"] = np.array([sil_unit_res])
        saved[f"{key}_eta_level_mean"] = np.array([eta_l_win])
        saved[f"{key}_eta_subject_mean"] = np.array([eta_s_win])
        saved[f"{key}_eta_interaction_mean"] = np.array([eta_i_win])
        saved[f"{key}_layers"] = np.array(layers)

        L.append("")
        L.append(f"### U/{wname}  layers={fmt_layers(layers)}")
        L.append(f"- same-unit cos = {su_mean:+.4f} (n={su.size}) / "
                 f"diff-unit cos = {du_mean:+.4f} (n={du.size}); "
                 f"ratio = **{ratio_u:.3f}x**; Cohen's d = {d_u:+.3f}")
        L.append(f"- silhouette: unit={sil_unit:+.4f}, level={sil_level:+.4f}, "
                 f"subject={sil_subj:+.4f}")
        L.append(f"- (level-residual) subject silhouette {sil_subj:+.4f} → {sil_subj_res:+.4f}; "
                 f"unit silhouette {sil_unit:+.4f} → {sil_unit_res:+.4f}")
        L.append(f"- 윈도우 평균 η²: level={eta_l_win:.3f}, subject={eta_s_win:.3f}, "
                 f"interaction={eta_i_win:.3f}")

        if (not args.no_png) and su.size and du.size:
            lur.plot_pair_hist(su, du, f"U/{wname}: same vs diff UNIT",
                               "pairwise cosine",
                               ANALYSIS / f"{TAG}_unithist_{wname}.png")

    del DAn_resid; gc.collect()

    # ─────────────────────── §X cross-window 표 ────────────────────────────
    L.append("")
    L.append("## §X. cross-window 비교표 (요약)")
    L.append("")
    L.append("**SUBJECT / LEVEL** (supervised + permutation)")
    L.append("")
    L.append("| axis | window | n_layers | same / diff cos | ratio | Cohen's d | MWU p | perm p | LDA macro-F1 |")
    L.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    def row_SL(axis, key, layers):
        sm = float(saved[f"{key}_same_mean"][0])
        dm = float(saved[f"{key}_diff_mean"][0])
        r = float(saved[f"{key}_ratio"][0])
        d = float(saved[f"{key}_cohend"][0])
        p_mwu = float(saved[f"{key}_mwu_p"][0])
        p_perm = float(saved[f"{key}_perm_p"][0])
        f1 = float(saved[f"{key}_lda_f1"][0])
        return (f"| {axis} | {key.split('_',1)[1]} | {len(layers)} | "
                f"{sm:+.4f} / {dm:+.4f} | {r:.3f}x | {d:+.3f} | "
                f"{p_mwu:.2e} | {p_perm:.4f} | "
                f"{('nan' if not np.isfinite(f1) else f'{f1:.3f}')} |")
    L.append(row_SL("SUBJECT", "S_Wsubj", W_SUBJ))
    L.append(row_SL("SUBJECT", "S_Wall",  W_ALL))
    L.append(row_SL("LEVEL",   "L_Wlev",  W_LEV))
    L.append(row_SL("LEVEL",   "L_Wall",  W_ALL))

    L.append("")
    L.append("**UNIT** (cohesion + silhouette + η²)")
    L.append("")
    L.append("| window | n_layers | same / diff cos | ratio | d | sil(unit) | sil(level) | sil(subj) | sil(subj|residL) | sil(unit|residL) | η²_lev | η²_subj | η²_inter |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    def row_U(wname, layers):
        key = f"U_{wname}"
        sm = float(saved[f"{key}_unit_same_mean"][0])
        dm = float(saved[f"{key}_unit_diff_mean"][0])
        r = float(saved[f"{key}_unit_ratio"][0])
        d = float(saved[f"{key}_unit_cohend"][0])
        su = float(saved[f"{key}_sil_unit"][0])
        sl = float(saved[f"{key}_sil_level"][0])
        sj = float(saved[f"{key}_sil_subject"][0])
        sjr = float(saved[f"{key}_sil_subject_residlevel"][0])
        sur = float(saved[f"{key}_sil_unit_residlevel"][0])
        el = float(saved[f"{key}_eta_level_mean"][0])
        es = float(saved[f"{key}_eta_subject_mean"][0])
        ei = float(saved[f"{key}_eta_interaction_mean"][0])
        return (f"| {wname} | {len(layers)} | {sm:+.4f} / {dm:+.4f} | {r:.3f}x | {d:+.3f} | "
                f"{su:+.4f} | {sl:+.4f} | {sj:+.4f} | {sjr:+.4f} | {sur:+.4f} | "
                f"{el:.3f} | {es:.3f} | {ei:.3f} |")
    L.append(row_U("Wsubj", W_SUBJ))
    L.append(row_U("Wlev",  W_LEV))
    L.append(row_U("Wall",  W_ALL))

    L.append("")
    L.append("## §Y. 결론 (요약 가이드 — 수치 해석은 표 참조)")
    L.append("- §S(SUBJECT): `W_SUBJ` 에서의 효과크기/perm-p/LDA-F1 가 `W_ALL` 보다 *유리* 한지 비교.")
    L.append("- §L(LEVEL): `W_LEV` 에서의 효과크기/perm-p/LDA-F1 가 `W_ALL` 보다 *유리* 한지 비교.")
    L.append("- §U(UNIT): 세 윈도우에서 same-unit/diff-unit ratio, silhouette(unit), η²_level vs η²_subject 비교.")
    L.append("- level-residual 후 subject/unit silhouette 가 어떻게 변하는지로 'level 제거 시 잔여 구조'를 확인.")
    L.append("")
    L.append("> 주의: η² 분해는 불균형 설계의 근사 (interaction 음수면 0 취급). "
             "단일 행렬 결과 하나로 강결론 금지. permutation p · 효과크기 · pilot1/pilot2 일반화(LDA) 를 함께 본다.")
    L.append("")
    L.append("**산출물.**")
    L.append(f"- `REPORT_bestlayer_resolved_{today}.md` (이 파일)")
    L.append(f"- `bestlayer_artifacts.npz` (모든 raw 수치)")
    L.append(f"- `bestlayer_pairhist_{{subject,level}}_{{Wsubj,Wlev,Wall}}.png`, "
             f"`bestlayer_unithist_{{Wsubj,Wlev,Wall}}.png` (옵션)")

    out_md = ANALYSIS / f"REPORT_bestlayer_resolved_{today}.md"
    np.savez(ANALYSIS / f"{TAG}_artifacts.npz", **saved)
    out_md.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"[OK] wrote {out_md}", flush=True)
    print(f"[OK] wrote {ANALYSIS / f'{TAG}_artifacts.npz'}  "
          f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
