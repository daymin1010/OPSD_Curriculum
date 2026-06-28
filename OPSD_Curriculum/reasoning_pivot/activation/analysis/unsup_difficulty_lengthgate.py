#!/usr/bin/env python3
"""
unsup_difficulty_lengthgate.py — Unsupervised difficulty axis + length gate.
============================================================================
Goal (curriculum 난이도를 GPT label 없이 activation 으로 정의 가능한지):
  TASK 1  UNSUPERVISED difficulty axis : pooled THINKING ΔA(centered) 에서
          PCA top-5 sweep. 각 PC 를 GPT level / 1-shot is_correct / gen_len 와
          Spearman ρ 로 비교하여 난이도를 가장 잘 추적하는 PC 채택(부호 정렬).
          GPT level 은 PC 선택·부호·평가에만 쓰고 *방향 학습엔 안 씀* (공분산만).
          + unit-centroid 1D ordering proxy.
  TASK 2  LENGTH RESIDUALIZE GATE ★ : gen_len 성분을 (a) score-level,
          (b) feature-level 로 회귀 제거한 뒤 axis 재추출. 잔차 전/후 ρ(level)·
          ρ(is_correct) 비교. 살아남으면 activation 이 length 너머 난이도를 담음
          (게이트 통과). 거의 사라지면 length proxy 경고.
          (주의: 난이도-길이는 본질적 비례일 수 있어 confound 단정이 아니라
           "잔차 후 잔존 신호" 확인용.)
  TASK 3  SUBJECT 검증(측정용, partition 아님) : (i) unsupervised(PC) 공간에서
          subject 가 약함을 silhouette 로 정량, (ii) supervised subject LDA
          (mid-layer L11-15, pilot1 train / pilot2 test) macro-F1 로
          "subject 정보는 있으나 representation 이 작다" 확인.
  TASK 4  대조 + fallback 판정 : unsup difficulty vs GPT level vs supervised
          ridge_level 의 ρ 표 + rank 크게 갈리는 sample + fallback verdict.

OUTPUT : analysis/REPORT_unsup_difficulty_lengthgate.md  (+ unsupdiff_artifacts.npz)
         기존 산출물 미변경. 함수 재사용. CPU only. smoke(--max-n)→본런.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import similarity_analysis as sa          # spearman / centroids / sim_matrix / MIN_N / rng
import curriculum_materials as cm          # load_pooled / flatten_f32 (reuse, no side effects)

ANALYSIS = Path(__file__).resolve().parent
OUT_MD = ANALYSIS / "REPORT_unsup_difficulty_lengthgate.md"
TAG = "unsupdiff"
MID_LAYERS = list(range(11, 16))           # L11..L15
SEED = 42
N_PC = 5
LDA_PCA_DIM = 100                          # dim-reduce before LDA
# gate thresholds (adjustable)
GATE_RETAIN_FRAC = 0.5                     # ρ_after must keep >= this fraction of ρ_before
GATE_ABS_MIN = 0.30                        # and |ρ_after| must be at least this


# ───────────────────────────── small stats ────────────────────────────────
def srow(score, lev, gl, ic_mask, ic_val):
    """Return (ρ(level), ρ(is_correct), ρ(gen_len))."""
    r_lev = sa.spearman(score, lev)
    r_ic = (sa.spearman(score[ic_mask], ic_val[ic_mask])
            if ic_mask.sum() >= 4 else float("nan"))
    r_gl = sa.spearman(score, gl)
    return r_lev, r_ic, r_gl


def partial_spearman(a, b, c):
    """Spearman partial correlation ρ(a,b | c) via rank-residuals (OLS on rank c)."""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if m.sum() < 5:
        return float("nan")
    ra, rb, rc = rankdata(a[m]), rankdata(b[m]), rankdata(c[m])
    Rc = np.c_[np.ones_like(rc), rc]
    ba, *_ = np.linalg.lstsq(Rc, ra, rcond=None)
    bb, *_ = np.linalg.lstsq(Rc, rb, rcond=None)
    resa = ra - Rc @ ba
    resb = rb - Rc @ bb
    if resa.std() == 0 or resb.std() == 0:
        return float("nan")
    return float(np.corrcoef(resa, resb)[0, 1])


def residualize_inplace(X, g, chunk=256):
    """Remove linear gen_len component from each feature dim, IN PLACE.
    X (n,F) centered f32 ; g (n,). Returns beta (F,)."""
    gc = (g - g.mean()).astype(np.float32)
    denom = float(gc @ gc) + 1e-12
    beta = (X.T @ gc) / denom                       # (F,), no n×F temp
    for i in range(0, X.shape[0], chunk):
        X[i:i + chunk] -= np.outer(gc[i:i + chunk], beta)
    return beta


def pca_sweep(X, lev, gl, ic_mask, ic_val, tag, lines):
    """Fit PCA top-N on centered X (unsupervised). Return (scores NxN_PC oriented,
    rows list, best_k index, best_score oriented)."""
    from sklearn.decomposition import PCA
    t0 = time.time()
    pca = PCA(n_components=N_PC, svd_solver="randomized", random_state=SEED)
    sc = pca.fit_transform(X)                        # (n, N_PC)
    lines.append(f"- [{tag}] PCA top-{N_PC} fit (n={X.shape[0]}, F={X.shape[1]}); "
                 f"EVR={np.round(pca.explained_variance_ratio_,4).tolist()} "
                 f"({time.time()-t0:.0f}s)")
    lines.append(f"\n| PC | EVR | ρ(level) | ρ(is_correct) | ρ(gen_len) |")
    lines.append("|----|-----|----------|---------------|------------|")
    oriented = np.zeros_like(sc)
    rows = []
    for k in range(N_PC):
        s = sc[:, k].copy()
        if sa.spearman(s, lev) < 0:
            s = -s
        oriented[:, k] = s
        r_lev, r_ic, r_gl = srow(s, lev, gl, ic_mask, ic_val)
        rows.append((r_lev, r_ic, r_gl, float(pca.explained_variance_ratio_[k])))
        lines.append(f"| PC{k+1} | {pca.explained_variance_ratio_[k]:.4f} | "
                     f"{r_lev:+.3f} | {r_ic:+.3f} | {r_gl:+.3f} |")
    best_k = int(np.argmax([abs(r[0]) for r in rows]))
    lines.append(f"- **adopted PC = PC{best_k+1}** (|ρ(level)| max = {rows[best_k][0]:+.3f}).")
    return oriented, rows, best_k


# ───────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="cap per pilot (smoke)")
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)
    t_all = time.time()

    DAT, _DAF, md, n1 = cm.load_pooled(args.max_n, want_faithful=False)
    N = len(md)
    lev = md["level"].to_numpy(float)
    gl = md["gen_len"].to_numpy(float)
    subj = md["subject"].to_numpy()
    tr = (md["pilot"] == "pilot1").to_numpy()
    te = (md["pilot"] == "pilot2").to_numpy()
    ic = md["is_correct"]
    ic_mask = ic.notna().to_numpy()
    ic_val = ic.where(ic.notna(), np.nan).astype(float).to_numpy()

    lines = [f"# Unsupervised Difficulty Axis + Length Gate (tag={TAG})", ""]
    lines.append(f"- pooled N = **{N}** (pilot1={n1}, pilot2={N-n1}); "
                 f"subjects={md['subject'].nunique()}, "
                 f"levels={sorted(md['level'].unique().tolist())}")
    lines.append(f"- is_correct non-null = {ic.notna().sum()} "
                 f"(overall rate={ic.dropna().astype(float).mean():.3f})")
    lines.append("- Method: difficulty 방향은 **unsupervised PCA(공분산만)**; GPT level 은 "
                 "PC 선택·부호·평가에만 사용(학습 X) → circularity 회피. CPU only, THINKING ΔA.")

    saved = {"level": lev, "gen_len": gl, "subject": subj.astype(str), "pilot": md["pilot"].to_numpy().astype(str)}

    # mid-layer features for LDA (extract BEFORE freeing DAT)
    feat_mid = DAT[:, MID_LAYERS, :].reshape(N, -1).astype(np.float32)   # (N, 5*12288)

    # flatten + center (in place)
    X = cm.flatten_f32(DAT)                          # (N,F) f32
    del DAT
    mu = X.mean(axis=0, keepdims=True)
    X -= mu
    print(f"[mem] X {X.shape} f32, feat_mid {feat_mid.shape}", flush=True)

    # ── TASK 1: unsupervised difficulty axis ──────────────────────────────
    lines.append("\n## TASK 1 — Unsupervised difficulty axis (PCA, no GPT-label fitting)")
    oriented, rows, best_k = pca_sweep(X, lev, gl, ic_mask, ic_val, "raw", lines)
    diff_score = oriented[:, best_k].copy()
    saved["diff_score_unsup"] = diff_score
    saved["pc_scores_raw"] = oriented

    # unit-centroid 1D ordering proxy
    md_idx = md.reset_index(drop=True)
    units = sorted([u for u in md_idx["unit"].value_counts().index
                    if md_idx["unit"].value_counts()[u] >= sa.MIN_N])
    u_score = np.array([diff_score[md_idx.index[md_idx["unit"] == u]].mean() for u in units])
    u_level = np.array([int(u.split("|L")[1]) for u in units])
    rho_unit = sa.spearman(u_score, u_level)
    lines.append(f"- unit-centroid 1D ordering proxy: ρ(unit mean diff_score, unit level) "
                 f"= **{rho_unit:+.3f}** (units n≥{sa.MIN_N}: {len(units)}). "
                 "양수 = unsupervised score 가 난이도 순서를 재현.")

    # ── ridge_level (supervised, out-of-sample) for TASK 4 contrast ───────
    lines.append("\n## (보조) Supervised ridge_level — 대조용 (partition 정의엔 안 씀)")
    yc = lev[tr] - lev[tr].mean()
    Xtr = X[tr]
    K = Xtr @ Xtr.T
    ntr = K.shape[0]
    rng = np.random.default_rng(SEED)
    folds = rng.integers(0, 5, size=ntr)
    best_a, best_cv = 1e4, -2.0
    for a in [1e1, 1e2, 1e3, 1e4, 1e5]:
        preds = np.zeros(ntr)
        for f in range(5):
            tem = folds == f; trm = ~tem
            ad = np.linalg.solve(K[np.ix_(trm, trm)] + a * np.eye(trm.sum()), yc[trm])
            preds[tem] = K[np.ix_(tem, trm)] @ ad
        s = sa.spearman(preds, lev[tr])
        if s > best_cv:
            best_cv, best_a = s, a
    ad = np.linalg.solve(K + best_a * np.eye(ntr), yc)
    ridge_all = (X @ Xtr.T) @ ad
    if sa.spearman(ridge_all, lev) < 0:
        ridge_all = -ridge_all
    saved["ridge_level_all"] = ridge_all
    r_lev, r_ic, r_gl = srow(ridge_all[te], lev[te], gl[te], ic_mask[te], ic_val[te])
    lines.append(f"- ridge_level (α={best_a:g}, pilot1 5-fold cv ρ={best_cv:+.3f}); "
                 f"pilot2 test: ρ(level)={r_lev:+.3f}, ρ(is_correct)={r_ic:+.3f}, ρ(gen_len)={r_gl:+.3f}")
    del Xtr, K

    # ── TASK 2: length residualize gate ───────────────────────────────────
    lines.append("\n## TASK 2 — Length residualize gate ★")
    # (a) score-level
    r_before = sa.spearman(diff_score, lev)
    gc = gl - gl.mean()
    beta_s = float((diff_score - diff_score.mean()) @ gc) / float(gc @ gc + 1e-12)
    score_resid = diff_score - beta_s * gc
    r_after_s = sa.spearman(score_resid, lev)
    pr = partial_spearman(diff_score, lev, gl)
    ic_pr = partial_spearman(diff_score, ic_val, gl)
    lines.append("\n### (a) score-level residualize (adopted unsup difficulty)")
    lines.append(f"- ρ(diff_score, level)  before = {r_before:+.3f} → after gen_len residual = {r_after_s:+.3f}")
    lines.append(f"- partial ρ(diff_score, level | gen_len) = **{pr:+.3f}**")
    lines.append(f"- partial ρ(diff_score, is_correct | gen_len) = {ic_pr:+.3f}")

    # (b) feature-level: residualize X in place, re-PCA
    lines.append("\n### (b) feature-level residualize (각 차원에서 gen_len 회귀 후 PCA 재fit)")
    t0 = time.time()
    residualize_inplace(X, gl)
    lines.append(f"- residualized features in place ({time.time()-t0:.0f}s)")
    oriented_r, rows_r, best_kr = pca_sweep(X, lev, gl, ic_mask, ic_val, "residual", lines)
    r_after_f = rows_r[best_kr][0]
    saved["diff_score_resid_feat"] = oriented_r[:, best_kr]
    del X

    # gate verdict (use feature-level after as primary, score-level/partial as support)
    retain = (abs(r_after_f) >= GATE_RETAIN_FRAC * abs(rows[best_k][0])) and (abs(r_after_f) >= GATE_ABS_MIN)
    partial_ok = abs(pr) >= GATE_ABS_MIN
    gate_pass = retain and partial_ok
    lines.append("\n### GATE 판정")
    lines.append(f"- 기준: feature-level ρ(level) 잔차후 |{r_after_f:+.3f}| ≥ "
                 f"{GATE_RETAIN_FRAC}×|{rows[best_k][0]:+.3f}| 이고 ≥ {GATE_ABS_MIN}; "
                 f"AND partial ρ(score,level|gen_len) |{pr:+.3f}| ≥ {GATE_ABS_MIN}")
    lines.append(f"- **GATE = {'PASS ✅' if gate_pass else 'FAIL ⚠️'}** "
                 + ("→ activation 이 length 너머 난이도를 담음 (unsupervised 난이도 정당)."
                    if gate_pass else
                    "→ difficulty 가 상당 부분 length proxy. unsupervised 난이도 주의(아래 fallback 참조)."))
    lines.append("- 주의: 난이도-길이는 본질적 비례일 수 있으므로 이 게이트는 confound 단정이 아니라 "
                 "'gen_len 제거 후 잔존 난이도 신호' 확인용.")

    # ── TASK 3: subject verification ───────────────────────────────────────
    lines.append("\n## TASK 3 — Subject 검증 (측정용, partition 아님)")
    from sklearn.metrics import silhouette_score, f1_score
    from sklearn.decomposition import PCA as _PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # (i) unsupervised: silhouette of subject vs level in adopted-PC space (top-N PCs)
    try:
        sil_subj = silhouette_score(oriented, subj)
    except Exception:
        sil_subj = float("nan")
    lev_bin = lev.astype(int).astype(str)
    try:
        sil_lev = silhouette_score(oriented, lev_bin)
    except Exception:
        sil_lev = float("nan")
    lines.append(f"- unsupervised PC-{N_PC} 공간 silhouette: subject={sil_subj:+.3f} vs level={sil_lev:+.3f} "
                 "(subject ≪ level 이면 unsupervised 공간에서 subject 약함 = 예상된 결과).")
    # (ii) supervised subject LDA (mid-layer, pilot1 train / pilot2 test)
    pcad = _PCA(n_components=min(LDA_PCA_DIM, feat_mid.shape[0]//2, feat_mid.shape[1]),
                svd_solver="randomized", random_state=SEED)
    Ztr = pcad.fit_transform(feat_mid[tr])
    Zte = pcad.transform(feat_mid[te])
    lda = LinearDiscriminantAnalysis()
    lda.fit(Ztr, subj[tr])
    pred_te = lda.predict(Zte)
    f1m = f1_score(subj[te], pred_te, average="macro")
    f1w = f1_score(subj[te], pred_te, average="weighted")
    n_subj = len(set(subj[tr]))
    lines.append(f"- supervised subject LDA (mid-layer L{MID_LAYERS[0]}-{MID_LAYERS[-1]}, "
                 f"PCA→{Ztr.shape[1]}d, pilot1 train / pilot2 test): "
                 f"macro-F1=**{f1m:.3f}**, weighted-F1={f1w:.3f}, chance≈{1.0/n_subj:.3f} ({n_subj} subj).")
    # per-class F1
    f1_per = f1_score(subj[te], pred_te, average=None, labels=sorted(set(subj[tr])))
    lines.append("- per-subject F1: " + ", ".join(
        f"{s}={v:.2f}" for s, v in zip(sorted(set(subj[tr])), f1_per)))
    lines.append("- 해석: chance 보다 높으면 subject 정보는 존재하나(F1 작으면) representation 이 약함. "
                 "이는 검증·대조용이며 curriculum partition 정의가 아님(circularity 회피).")
    saved["subject_lda_macro_f1"] = np.array([f1m])

    # ── TASK 4: contrast + fallback verdict ───────────────────────────────
    lines.append("\n## TASK 4 — 대조 (unsup vs GPT level vs ridge) + fallback 판정")
    lines.append("\n### pairwise Spearman ρ")
    lines.append("| pair | ρ |")
    lines.append("|------|---|")
    lines.append(f"| unsup difficulty ↔ GPT level | {sa.spearman(diff_score, lev):+.3f} |")
    lines.append(f"| ridge_level ↔ GPT level | {sa.spearman(ridge_all, lev):+.3f} |")
    lines.append(f"| unsup difficulty ↔ ridge_level | {sa.spearman(diff_score, ridge_all):+.3f} |")
    lines.append(f"| unsup difficulty ↔ gen_len | {sa.spearman(diff_score, gl):+.3f} |")

    # where unsup and GPT level diverge most (rank diff)
    r_unsup = rankdata(diff_score)
    r_lev_rank = rankdata(lev)
    rank_gap = np.abs(r_unsup - r_lev_rank)
    top = np.argsort(-rank_gap)[:10]
    lines.append("\n### unsup difficulty 와 GPT level 의 rank 차 큰 sample top-10")
    lines.append("| problem_id | subject | level | gen_len | unsup_rank | level_rank |")
    lines.append("|---|---|---|---|---|---|")
    for i in top:
        lines.append(f"| {str(md_idx.loc[i,'problem_id'])[:12]} | {md_idx.loc[i,'subject']} | "
                     f"{int(lev[i])} | {int(gl[i])} | {int(r_unsup[i])} | {int(r_lev_rank[i])} |")

    # stability: split-half on pilots (PC adopted ρ on each pilot)
    rho_p1 = sa.spearman(diff_score[tr], lev[tr])
    rho_p2 = sa.spearman(diff_score[te], lev[te])
    stable = abs(rho_p1 - rho_p2) <= 0.20 and min(abs(rho_p1), abs(rho_p2)) >= GATE_ABS_MIN
    lines.append(f"\n### 안정성 (split-half by pilot)")
    lines.append(f"- ρ(diff_score, level): pilot1={rho_p1:+.3f}, pilot2={rho_p2:+.3f} "
                 f"→ {'안정 ✅' if stable else '불안정 ⚠️'} (|차|≤0.20 & 둘다≥{GATE_ABS_MIN})")

    adopt_unsup = gate_pass and stable
    lines.append("\n### Fallback 판정")
    lines.append(f"- (a) length 게이트 = {'PASS' if gate_pass else 'FAIL'}, "
                 f"(b) 안정성 = {'OK' if stable else 'NG'}.")
    if adopt_unsup:
        lines.append("- **권고: UNSUPERVISED difficulty 채택** — length 잔차 후에도 GPT level 과 "
                     "합리적 ρ 유지 + pilot 간 안정. GPT label 없이 난이도 정렬 가능.")
    else:
        lines.append("- **권고: GPT level FALLBACK** — unsupervised 난이도가 length proxy 거나 "
                     "불안정. curriculum 난이도는 GPT level 사용 권장.")
    lines.append("- (subject 는 원래 GPT mixing 이 기본 → fallback 대상 아님; TASK3 은 대조 측정용.)")

    # ── write ──────────────────────────────────────────────────────────────
    np.savez(ANALYSIS / f"{TAG}_artifacts.npz", **saved)
    OUT_MD.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}", flush=True)
    print(f"[OK] wrote {ANALYSIS / f'{TAG}_artifacts.npz'}", flush=True)
    print(f"[done] total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
