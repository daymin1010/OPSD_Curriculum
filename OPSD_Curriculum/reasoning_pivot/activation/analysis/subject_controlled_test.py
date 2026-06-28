#!/usr/bin/env python3
"""
subject_controlled_test.py — SUBJECT 신호를 LEVEL 통제 하에 단독 검정.
================================================================================
배경(직전 세션, unit=subject×level 유사도 결론):
  - subject block within/across ≈ 0.99x(raw)/1.02x(resid) → 응집 ≈0
  - level  block within/across  ≈ 1.46x(raw)/1.84x(resid)
  - silhouette 둘 다 음수(subject -0.160 < level -0.095)
  → "level 신호 절댓값이 더 커서 subject 신호가 묻힌다"는 가설.
이 스크립트는 **level을 고정/제거하면 subject 신호가 살아나는가**를 검정한다.

방법(모두 CPU, numpy):
  1) **level 고정 within- vs across-subject**: 같은 level 블록 안에서만
     same-subject vs diff-subject 의 *sample-pairwise* cosine 분포 비교
     (level 오염 = 0). level별 + 전체 pooled. Cohen's d, Mann-Whitney.
  2) **순열검정(permutation)**: level 블록 고정, subject 라벨만 블록 내 셔플
     → subject 효과(=level-가중 mean_same - mean_diff)의 p값/효과크기.
     대칭 대조군으로 (subject 블록 고정, level 셔플) 도 같이 산출 → 두 축 직접 비교.
  3) (선택) level 회귀잔차(self-level centroid 차감) 후 subject silhouette 재계산
     + within-level subject centroid cosine 행렬(ssg 재사용).

데이터/재사용: pooled(pilot1+pilot2) THINKING ΔA, finite N=3025,
  μ_pooled(per-layer) centering. pooled_analysis.load_pooled,
  similarity_analysis.normalize_members, subject_similarity_gate
  (within_level_subject_sim / level_centroid_residual / fmt_mat) 재사용.
기존 산출물/파일 미변경. opsd_src 미변경.

OUTPUT: REPORT_subject_controlled_<date>.md, subjctrl_artifacts.npz
"""
from __future__ import annotations
import argparse
import gc
import time
from datetime import date
from pathlib import Path

import numpy as np

from scipy.stats import mannwhitneyu

import similarity_analysis as sa
import pooled_analysis as pa
import subject_similarity_gate as ssg

ANALYSIS = Path(__file__).resolve().parent
TAG = "subjctrl"
LAYERS = 36
MID_LAYERS = list(range(11, 16))     # 직전 세션 subject 윈도우(참고값)
SEED = 42


# ───────────────────────────── cosine helpers ──────────────────────────────
def window_cos_matrix(DAn, layers):
    """layer-averaged pairwise cosine 행렬 (N×N). DAn = per-layer L2-normalized."""
    sub = np.ascontiguousarray(DAn[:, layers, :]).astype(np.float32)
    N = sub.shape[0]
    C = np.zeros((N, N), dtype=np.float32)
    for li in range(sub.shape[1]):
        Al = sub[:, li, :]
        C += Al @ Al.T
    C /= sub.shape[1]
    return C


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


def within_block_pair_dists(C, md, block_col, test_col):
    """각 `block_col` 값(통제축) 안에서 same-`test_col` vs diff-`test_col` 의
    *sample-pairwise* cosine 수집. Returns (same, diff, per_block_stats)."""
    same_all, diff_all = [], []
    per_block = []
    blocks = sorted(md[block_col].unique().tolist())
    tarr = md[test_col].to_numpy()
    for b in blocks:
        idx = md.index[md[block_col] == b].to_numpy()
        if len(idx) < 4:
            per_block.append((b, len(idx), np.nan, np.nan, 0, 0))
            continue
        tb = tarr[idx]
        Csub = C[np.ix_(idx, idx)]
        n = len(idx)
        iu = np.triu_indices(n, 1)
        eq = (tb[:, None] == tb[None, :])[iu]
        vals = Csub[iu]
        sv = vals[eq]; dv = vals[~eq]
        if sv.size:
            same_all.append(sv)
        if dv.size:
            diff_all.append(dv)
        per_block.append((b, n, float(sv.mean()) if sv.size else np.nan,
                          float(dv.mean()) if dv.size else np.nan,
                          int(sv.size), int(dv.size)))
    same = np.concatenate(same_all) if same_all else np.array([])
    diff = np.concatenate(diff_all) if diff_all else np.array([])
    return same, diff, per_block


def block_permutation(C, md, block_col, test_col, n_perm, seed=SEED):
    """`block_col` 고정, `test_col` 라벨만 블록 내 셔플. 통계량 =
    block-가중 평균 (mean_same - mean_diff). p = P(stat_perm >= stat_obs)."""
    rng = np.random.default_rng(seed)
    blocks_v = sorted(md[block_col].unique().tolist())
    tarr = md[test_col].to_numpy()
    blocks = []
    for b in blocks_v:
        idx = md.index[md[block_col] == b].to_numpy()
        if len(idx) < 4:
            continue
        n = len(idx)
        Csub = C[np.ix_(idx, idx)].astype(np.float32)
        iu = np.triu_indices(n, 1)
        lab = tarr[idx].copy()
        # 라벨 다양성 없는 블록은 same/diff 구분 불가 → 스킵
        if len(np.unique(lab)) < 2:
            continue
        blocks.append({"lab": lab, "iu": iu, "vals": Csub[iu],
                       "npairs": len(iu[0])})
    if not blocks:
        return float("nan"), float("nan"), 0

    def stat(label_lists):
        num = 0.0; den = 0.0
        for bk, lab in zip(blocks, label_lists):
            eq = (lab[:, None] == lab[None, :])[bk["iu"]]
            sv = bk["vals"][eq]; dv = bk["vals"][~eq]
            if sv.size and dv.size:
                w = bk["npairs"]
                num += w * (sv.mean() - dv.mean()); den += w
        return num / den if den > 0 else float("nan")

    obs = stat([bk["lab"] for bk in blocks])
    ge = 0
    log_every = max(1, n_perm // 10)
    for it in range(n_perm):
        perm = [rng.permutation(bk["lab"]) for bk in blocks]
        s = stat(perm)
        if np.isfinite(s) and s >= obs:
            ge += 1
        if (it + 1) % log_every == 0:
            print(f"    [perm {block_col}->shuffle {test_col}] "
                  f"{it+1}/{n_perm} (ge={ge})", flush=True)
    p = (ge + 1) / (n_perm + 1)
    return float(obs), float(p), len(blocks)


def subject_silhouette(C, subj):
    from sklearn.metrics import silhouette_score
    D = (1.0 - C).astype(np.float64)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    try:
        return float(silhouette_score(D, subj, metric="precomputed"))
    except Exception as e:
        print(f"    [silhouette] 실패: {e}", flush=True)
        return float("nan")


# ───────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot당 .pt 제한")
    ap.add_argument("--n-perm", type=int, default=2000, help="permutation 횟수")
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)
    t0 = time.time()

    DAF, DAT, md, ninfo = pa.load_pooled(args.max_n)
    del DAF; gc.collect()
    N = len(md)
    subjects = sorted(md["subject"].unique().tolist())
    levels = sorted(md["level"].unique().tolist())
    print(f"[load] N={N} subjects={subjects} levels={levels} "
          f"({time.time()-t0:.0f}s)", flush=True)

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    del DAT; gc.collect()

    # per-layer normalize (raw) + level-residualize 후 normalize (resid)
    print("[norm] per-layer L2-normalize (raw + level-residual) ...", flush=True)
    DAn = sa.normalize_members(DA_c)
    resid = ssg.level_centroid_residual(DA_c, md)
    DAn_res = sa.normalize_members(resid)
    del resid; gc.collect()

    VIEWS = {"layeravg": list(range(LAYERS)),
             f"mid_L{MID_LAYERS[0]}-{MID_LAYERS[-1]}": MID_LAYERS}

    saved = {"subjects": np.array(subjects), "levels": np.array(levels)}
    L = []
    today = date.today().isoformat()
    L.append(f"# SUBJECT 단독 검정 (LEVEL 통제) — {today}  (tag={TAG})")
    L.append("")
    L.append("> **가설.** 직전 세션에서 unit(subject×level) 구조는 LEVEL이 지배하고 SUBJECT는")
    L.append("> 거의 무신호(within/across≈1.0, silhouette 음수)였다. \"level 신호의 절댓값이")
    L.append("> 더 커서 subject 신호가 묻힌다\"면, level을 고정/제거하면 subject 신호가")
    L.append("> 살아나야 한다. 이를 (1) level-고정 pairwise 비교, (2) 순열검정, (3) level-잔차")
    L.append("> 후 silhouette 로 검정한다.")
    L.append("")
    L.append(f"**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. "
             f"finite **N={ninfo['n_final']}** (raw {ninfo['n_loaded']}, non-finite drop "
             f"{ninfo['n_nonfinite']}). CPU only. seed={SEED}, n_perm={args.n_perm}.")
    pc = md["_pilot"].value_counts()
    L.append(f"- provenance: " + ", ".join(f"{k}={int(v)}" for k, v in pc.items()))
    L.append(f"- metric: per-layer L2-normalized → layer-averaged sample-pairwise cosine. "
             f"views = {list(VIEWS.keys())}. (mid = 직전 세션 subject 윈도우 참고값)")
    L.append("")

    summary = []   # (view, space, subj_ratio, subj_d, subj_p, lev_ratio, lev_d, lev_p)

    for vname, layers in VIEWS.items():
        for space, DA_use in (("raw", DAn), ("resid_on_level", DAn_res)):
            print(f"[run] view={vname} space={space} ...", flush=True)
            C = window_cos_matrix(DA_use, layers)

            # (1) level 고정 → subject 검정
            s_same, s_diff, per_lv = within_block_pair_dists(C, md, "level", "subject")
            s_d = cohens_d(s_same, s_diff)
            try:
                _, s_mwu = mannwhitneyu(s_same, s_diff, alternative="greater",
                                        method="asymptotic")
            except Exception:
                s_mwu = float("nan")
            s_obs, s_p, s_nb = block_permutation(C, md, "level", "subject", args.n_perm)
            s_ratio = (s_same.mean() / s_diff.mean()) if (s_diff.size and s_diff.mean() != 0) else float("nan")

            # (2) 대칭 대조: subject 고정 → level 검정
            l_same, l_diff, per_sj = within_block_pair_dists(C, md, "subject", "level")
            l_d = cohens_d(l_same, l_diff)
            try:
                _, l_mwu = mannwhitneyu(l_same, l_diff, alternative="greater",
                                        method="asymptotic")
            except Exception:
                l_mwu = float("nan")
            l_obs, l_p, l_nb = block_permutation(C, md, "subject", "level", args.n_perm)
            l_ratio = (l_same.mean() / l_diff.mean()) if (l_diff.size and l_diff.mean() != 0) else float("nan")

            pfx = f"{vname}__{space}"
            saved[f"{pfx}_subj_same"] = np.array([s_same.mean() if s_same.size else np.nan])
            saved[f"{pfx}_subj_diff"] = np.array([s_diff.mean() if s_diff.size else np.nan])
            saved[f"{pfx}_subj_d"] = np.array([s_d])
            saved[f"{pfx}_subj_p"] = np.array([s_p])
            saved[f"{pfx}_lev_same"] = np.array([l_same.mean() if l_same.size else np.nan])
            saved[f"{pfx}_lev_diff"] = np.array([l_diff.mean() if l_diff.size else np.nan])
            saved[f"{pfx}_lev_d"] = np.array([l_d])
            saved[f"{pfx}_lev_p"] = np.array([l_p])
            summary.append((vname, space, s_ratio, s_d, s_p, l_ratio, l_d, l_p))

            L.append(f"## view = `{vname}` · space = `{space}`")
            L.append("")
            L.append("### (1) LEVEL 고정 → SUBJECT 단독 검정")
            L.append(f"- within-level **same-subject** mean cos = {s_same.mean():+.4f} "
                     f"(n_pairs={s_same.size}); **diff-subject** = {s_diff.mean():+.4f} "
                     f"(n_pairs={s_diff.size}); ratio = **{s_ratio:.3f}x**")
            L.append(f"- **Cohen's d (same−diff) = {s_d:+.3f}**; Mann–Whitney p(same>diff) = {s_mwu:.2e}")
            L.append(f"- **permutation**(level 고정, subject 셔플 ×{args.n_perm}): "
                     f"stat={s_obs:+.5f}, **p={s_p:.4f}** (blocks={s_nb})")
            L.append("- per-level (same / diff / n_same / n_diff):")
            for lv, n, sm, dm, ns, nd in per_lv:
                L.append(f"    L{lv} (n={n}): same={sm:+.4f} diff={dm:+.4f}  ({ns}/{nd})")
            L.append("")
            L.append("### (2) 대칭 대조 — SUBJECT 고정 → LEVEL 단독 검정")
            L.append(f"- within-subject **same-level** mean cos = {l_same.mean():+.4f} "
                     f"(n_pairs={l_same.size}); **diff-level** = {l_diff.mean():+.4f} "
                     f"(n_pairs={l_diff.size}); ratio = **{l_ratio:.3f}x**")
            L.append(f"- **Cohen's d = {l_d:+.3f}**; Mann–Whitney p = {l_mwu:.2e}")
            L.append(f"- **permutation**(subject 고정, level 셔플 ×{args.n_perm}): "
                     f"stat={l_obs:+.5f}, **p={l_p:.4f}** (blocks={l_nb})")
            L.append("")

            # (3) within-level subject centroid cosine 행렬 (raw space 만, ssg 재사용)
            if space == "raw":
                try:
                    S_A, _den, _info = ssg.within_level_subject_sim(
                        DA_c, md, subjects, layers, min_cell=5)
                    saved[f"{pfx}_subjsim"] = S_A
                    L.append("### (3a) within-level subject centroid cosine (level 가중평균)")
                    L.append("```\n" + ssg.fmt_mat(S_A, subjects) + "\n```")
                    L.append("- 각 subject 최근접 과목:")
                    for i, s in enumerate(subjects):
                        row = S_A[i].copy(); row[i] = -np.inf
                        j = int(np.argmax(np.where(np.isfinite(row), row, -np.inf)))
                        L.append(f"    {s} → {subjects[j]} ({S_A[i, j]:+.3f})")
                    L.append("")
                except Exception as e:
                    L.append(f"- within_level_subject_sim 실패: {e}\n")

            # (3b) subject silhouette (이 view/space)
            sil = subject_silhouette(C, md["subject"].to_numpy())
            saved[f"{pfx}_subj_silhouette"] = np.array([sil])
            L.append(f"### (3b) subject silhouette ({space}) = **{sil:+.4f}** "
                     f"(직전 unit-report subject raw=-0.160)")
            L.append("")
            del C; gc.collect()

    # ── 요약 표: 두 축 직접 비교 ────────────────────────────────────────────
    L.append("## 요약 — LEVEL 통제 후 SUBJECT vs (대칭) SUBJECT 통제 후 LEVEL")
    L.append("")
    L.append("| view | space | subj(level통제) ratio·d·p | level(subj통제) ratio·d·p |")
    L.append("|---|---|---|---|")
    for vname, space, sr, sd, sp, lr, ld, lp in summary:
        L.append(f"| `{vname}` | {space} | {sr:.3f}x · d={sd:+.3f} · p={sp:.4f} "
                 f"| {lr:.3f}x · d={ld:+.3f} · p={lp:.4f} |")
    L.append("")

    # ── 결론 ────────────────────────────────────────────────────────────────
    # layeravg/raw 기준 판정
    key = "layeravg__raw"
    sp_p = float(saved[f"{key}_subj_p"][0]); sp_d = float(saved[f"{key}_subj_d"][0])
    sp_ratio = (float(saved[f"{key}_subj_same"][0]) / float(saved[f"{key}_subj_diff"][0])
                if float(saved[f"{key}_subj_diff"][0]) != 0 else float("nan"))
    lv_p = float(saved[f"{key}_lev_p"][0]); lv_d = float(saved[f"{key}_lev_d"][0])
    L.append("## 결론")
    L.append("")
    L.append(f"1. **LEVEL 고정 후 subject 신호**(layeravg/raw): within/across = {sp_ratio:.3f}x, "
             f"Cohen's d = {sp_d:+.3f}, permutation p = {sp_p:.4f} → "
             f"{'유의: level 통제 시 subject 신호 존재(=level 교란에 묻혀 있었음)' if sp_p < 0.05 else '비유의: level 통제 후에도 subject 응집 약함'}.")
    L.append(f"2. **대칭 대조(subject 고정 후 level)**: Cohen's d = {lv_d:+.3f}, p = {lv_p:.4f}. "
             f"두 효과크기 비교(|d_subject|={abs(sp_d):.3f} vs |d_level|={abs(lv_d):.3f}) → "
             f"{'subject가 더 약함(=여전히 level 우위)' if abs(sp_d) < abs(lv_d) else 'subject가 동급 이상'}.")
    L.append("3. **silhouette**(위 §3b): level-residual 후 subject silhouette 가 raw 대비 개선되는지 "
             "각 view 항목 참조 — 음수 유지면 subject 군집은 여전히 약함.")
    L.append("")
    L.append("> 주의: 8과목/소표본 + 불균형 설계. permutation p 와 효과크기, raw/resid 일관성을 "
             "함께 보고 단일 수치로 강결론 금지. ratio≈1·d≈0·p≫0.05 면 'level 통제 후에도 "
             "subject 무신호' = 직전 결론을 (교란 통제 후에도) 재확인하는 것.")
    L.append("")
    L.append("### 부록: 직전 보고서 §3 정정 메모")
    L.append("- 직전 unit-report §3 \"고립=고난도\" 문구는 데이터(고립=저난도 Pcalc/Geom/Algebra)와 "
             "반대 → \"고립 과목은 **저난도** 경향\"으로 정정 필요(본 분석 범위 밖, 메모만).")

    out_md = ANALYSIS / f"REPORT_subject_controlled_{today}.md"
    np.savez(ANALYSIS / f"{TAG}_artifacts.npz", **saved)
    out_md.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"[OK] wrote {out_md}", flush=True)
    print(f"[OK] wrote {ANALYSIS / f'{TAG}_artifacts.npz'} "
          f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
