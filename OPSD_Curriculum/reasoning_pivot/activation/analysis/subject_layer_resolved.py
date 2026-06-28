#!/usr/bin/env python3
"""
subject_layer_resolved.py — SUBJECT 신호 단독 검정 (level 통제) + 레이어 분해.
================================================================================
배경(직전 세션): unit(=subject×level) 유사도에서 LEVEL이 표현 구조를 지배하고
SUBJECT는 거의 무신호(within/across≈1.0, silhouette 음수)로 나왔다. 다만 그 분석은
(1) 전 레이어 평균(layeravg) 위에서 (2) PCA(PC1 prototype) 기반이었다. 이 스크립트는
세 가지를 바꿔서 "level을 통제하면 subject 신호가 살아나는가"를 단독 검정한다:

  변경 1 — 레이어 스캔: 36개 레이어 각각에서 subject(와 level) 판별력을 측정해
           "subject를 가장 잘 나타내는 레이어/윈도우"를 데이터로 찾는다.
           (직전 경험상 mid L11–15 였으나 여기선 고정 X, 참고값으로만 비교.)
  변경 2 — PCA prototype 대신 *supervised 판별* 지표 사용:
           (a) Fisher 판별비 tr(Sb)/tr(Sw)  — 레이어별, 싸고 36개 전부.
           (b) 교차검증 선형 프로브(PCA-whiten 차원축소 → 다항 로지스틱),
               pilot1 train / pilot2 test, macro-F1 — 일반화 가능한 판별력.
  변경 3 — level 통제하에 subject 단독 검정:
           (A) within-level same-subject vs diff-subject *sample-pairwise* cosine
               분포 (level 고정 → level 오염 0), Cohen's d + Mann–Whitney.
           (B) block-permutation: level 블록 고정, subject 라벨만 셔플 → p값+효과크기.
           (C) within-level subject centroid cosine → 각 subject 최근접 과목 + dendro.
           (D) LDA confusion (pilot1 train/pilot2 test) → "헷갈리는(=유사) 과목".
  + UNIT 분해: per-layer 2-way 분산분해(η²_level / η²_subject / η²_interaction)로
    "unit 구조에서 subject 비중"을 정량화. 추가로 level-residualize 후 subject
    silhouette 재계산(직전의 음수 silhouette과 대조).

데이터/재사용: pooled(pilot1+pilot2) THINKING ΔA, canonical finite N=3025,
  μ_pooled(per-layer) centering. similarity_analysis(sa) primitive 와
  subject_similarity_gate(ssg) 의 within_level_subject_sim / level_centroid_residual
  / dendro / heatmap / fmt_mat / mat_corr 재사용. CPU only. 기존 산출물 미변경.

OUTPUT (analysis/):
  REPORT_subject_controlled_<date>.md, subjlayer_artifacts.npz, 곡선/heatmap/dendro PNG.
"""
from __future__ import annotations
import argparse
import gc
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

import similarity_analysis as sa
import pooled_analysis as pa
import subject_similarity_gate as ssg

ANALYSIS = Path(__file__).resolve().parent
TAG = "subjlayer"
LAYERS = 36
MID_LAYERS = list(range(11, 16))     # 직전 세션 subject 윈도우(참고값)
SEED = 42


# ───────────────────────────── 레이어 스캔 helpers ──────────────────────────
def fisher_ratio(X, labels, order):
    """레이어 한 장 X:(N,D) 에서 trace 기반 Fisher 판별비 = tr(Sb)/tr(Sw).
    tr(Sw) = Σ_i ||x_i - μ_{c(i)}||²,  tr(Sb) = Σ_c n_c ||μ_c - μ||².
    값↑ = 클래스가 그 레이어에서 잘 분리됨."""
    X = X.astype(np.float32)
    mu = X.mean(axis=0)
    sq_all = (X * X).sum()                       # Σ||x_i||²
    tr_within = 0.0
    tr_between = 0.0
    for g in order:
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            continue
        mc = X[idx].mean(axis=0)
        # Σ_{i∈c}||x_i-mc||² = Σ||x_i||² - n_c||mc||²
        tr_within += float((X[idx] * X[idx]).sum() - len(idx) * (mc @ mc))
        tr_between += float(len(idx) * ((mc - mu) @ (mc - mu)))
    if tr_within <= 0:
        return float("nan")
    return tr_between / tr_within


def probe_f1(Xtr, ytr, Xte, yte, order, pca_comps, seed=SEED):
    """PCA-whiten 차원축소 → 다항 로지스틱 회귀 프로브. macro-F1 (test).
    PCA 는 *전처리 차원축소* 일 뿐, 표현은 supervised 결정경계(=PC1 prototype 아님)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    nc = int(min(pca_comps, Xtr.shape[0] - 1, Xtr.shape[1]))
    pca = PCA(n_components=nc, whiten=True, svd_solver="randomized",
              random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(Ztr, ytr)
    pred = clf.predict(Zte)
    return float(f1_score(yte, pred, labels=order, average="macro", zero_division=0))


def select_window(score, frac=0.85):
    """score[l] 의 최고점 부근 연속 윈도우: score >= max - frac*(max-min) 인
    레이어들의 (min..max) 연속 span. 단일 best layer 도 반환."""
    s = np.asarray(score, float)
    finite = np.isfinite(s)
    if not finite.any():
        return int(np.argmax(np.nan_to_num(s))), [int(np.argmax(np.nan_to_num(s)))]
    best = int(np.nanargmax(s))
    thr = np.nanmax(s) - (1 - frac) * (np.nanmax(s) - np.nanmin(s))
    hot = [l for l in range(len(s)) if finite[l] and s[l] >= thr]
    if not hot:
        hot = [best]
    win = list(range(min(hot), max(hot) + 1))
    return best, win


# ───────────────────────────── pairwise cosine helpers ─────────────────────
def window_cos_matrix(DAn, layers):
    """layer-averaged cosine 행렬 (N×N) = (1/|layers|) Σ_l An[:,l,:] @ An[:,l,:].T.
    DAn 은 per-layer L2-normalized (sa.normalize_members)."""
    sub = np.ascontiguousarray(DAn[:, layers, :]).astype(np.float32)
    N = sub.shape[0]
    C = np.zeros((N, N), dtype=np.float32)
    for li in range(sub.shape[1]):
        Al = sub[:, li, :]
        C += Al @ Al.T
    C /= sub.shape[1]
    return C


def within_level_pair_dists(C, md, subjects):
    """각 level 블록 안에서 same-subject vs diff-subject *쌍별* cosine 수집.
    Returns (same_vals, diff_vals, per_level_stats)."""
    same_all, diff_all = [], []
    per_level = []
    levels = sorted(md["level"].unique().tolist())
    subj = md["subject"].to_numpy()
    for lv in levels:
        idx = md.index[md["level"] == lv].to_numpy()
        if len(idx) < 4:
            per_level.append((lv, len(idx), np.nan, np.nan, 0, 0))
            continue
        sub_subj = subj[idx]
        Csub = C[np.ix_(idx, idx)]
        n = len(idx)
        iu = np.triu_indices(n, 1)
        eq = (sub_subj[:, None] == sub_subj[None, :])[iu]
        vals = Csub[iu]
        sv = vals[eq]; dv = vals[~eq]
        same_all.append(sv); diff_all.append(dv)
        per_level.append((lv, n, float(sv.mean()) if sv.size else np.nan,
                          float(dv.mean()) if dv.size else np.nan,
                          int(sv.size), int(dv.size)))
    same = np.concatenate(same_all) if same_all else np.array([])
    diff = np.concatenate(diff_all) if diff_all else np.array([])
    return same, diff, per_level


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


def block_perm_subject(C, md, subjects, n_perm, seed=SEED):
    """level 블록 고정, subject 라벨만 블록 내 셔플 → 통계량 = level-가중 평균
    (mean_same - mean_diff). p = P(stat_perm >= stat_obs). 효과크기=관측 stat."""
    rng = np.random.default_rng(seed)
    levels = sorted(md["level"].unique().tolist())
    subj = md["subject"].to_numpy()
    # per-level 사전계산: 블록 cosine + upper-tri index + 블록 라벨
    blocks = []
    for lv in levels:
        idx = md.index[md["level"] == lv].to_numpy()
        if len(idx) < 4:
            continue
        n = len(idx)
        Csub = C[np.ix_(idx, idx)].astype(np.float32)
        iu = np.triu_indices(n, 1)
        blocks.append({"lab": subj[idx].copy(), "C": Csub, "iu": iu,
                       "vals": Csub[iu], "npairs": len(iu[0])})
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
        perm_labels = [rng.permutation(bk["lab"]) for bk in blocks]
        if stat(perm_labels) >= obs:
            ge += 1
        if (it + 1) % log_every == 0:
            print(f"    [perm] {it+1}/{n_perm} (ge={ge})", flush=True)
    p = (ge + 1) / (n_perm + 1)
    return float(obs), float(p), len(blocks)


# ───────────────────────────── UNIT 분산분해 ───────────────────────────────
def eta2_partition(X, lev, subj):
    """레이어 한 장 X:(N,D) 에서 2-way 주변평균 기반 분산분해.
    SS_total = Σ||x_i-μ||²;  SS_level = Σ_i||μ_{lev(i)}-μ||²;
    SS_subj = Σ_i||μ_{subj(i)}-μ||²;  SS_cell = Σ_i||μ_{cell(i)}-μ||²;
    SS_inter = SS_cell - SS_level - SS_subj (불균형 설계라 음수 가능 → 0 클립 표기).
    η²_x = SS_x / SS_total. 불균형 때문에 비직교 → 근사 분해임을 리포트에 명시."""
    X = X.astype(np.float32)
    mu = X.mean(axis=0)
    Xc = X - mu
    ss_total = float((Xc * Xc).sum())
    if ss_total <= 0:
        return {"level": np.nan, "subject": np.nan, "interaction": np.nan,
                "residual": np.nan, "ss_total": 0.0}

    def ss_factor(fac):
        ss = 0.0
        for g in np.unique(fac):
            idx = np.where(fac == g)[0]
            mg = X[idx].mean(axis=0) - mu
            ss += float(len(idx) * (mg @ mg))
        return ss

    ss_level = ss_factor(lev)
    ss_subj = ss_factor(subj)
    cell = np.array([f"{a}|{b}" for a, b in zip(lev, subj)])
    ss_cell = ss_factor(cell)
    ss_inter = ss_cell - ss_level - ss_subj
    return {"level": ss_level / ss_total, "subject": ss_subj / ss_total,
            "interaction": ss_inter / ss_total,
            "residual": 1.0 - ss_cell / ss_total, "ss_total": ss_total}


def subject_silhouette(C, subj):
    """precomputed distance(=1-cos) 로 subject silhouette."""
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


# ───────────────────────────── LDA confusion ───────────────────────────────
def lda_confusion(DA_c, md, subjects, layers, pca_comps, seed=SEED):
    try:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import confusion_matrix, f1_score
    except Exception as e:
        return None, f"sklearn import 실패: {e}"
    tr = (md["_pilot"] == "pilot1").to_numpy()
    te = (md["_pilot"] == "pilot2").to_numpy()
    if tr.sum() < 50 or te.sum() < 50:
        return None, "train/test 표본 부족(smoke?)"
    X = DA_c[:, layers, :].reshape(len(md), -1).astype(np.float32)
    nc = int(min(pca_comps, int(tr.sum()) - 1, X.shape[1]))
    pca = PCA(n_components=nc, svd_solver="randomized", random_state=seed).fit(X[tr])
    Ztr, Zte = pca.transform(X[tr]), pca.transform(X[te])
    ytr = md.loc[tr, "subject"].to_numpy(); yte = md.loc[te, "subject"].to_numpy()
    lda = LinearDiscriminantAnalysis().fit(Ztr, ytr)
    pred = lda.predict(Zte)
    order = sorted(set(ytr) & set(yte))
    f1 = f1_score(yte, pred, labels=order, average="macro", zero_division=0)
    cm = confusion_matrix(yte, pred, labels=order).astype(float)
    row = cm.sum(1, keepdims=True)
    cm_norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return {"order": order, "f1": float(f1), "cm_norm": cm_norm}, None


# ───────────────────────────── plotting ────────────────────────────────────
def plot_layer_scan(fisher_s, fisher_l, f1_s, f1_l, best_s, win_s, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    xs = np.arange(LAYERS)
    ax[0].plot(xs, fisher_s, "-o", ms=3, label="subject")
    ax[0].plot(xs, fisher_l, "-s", ms=3, label="level")
    ax[0].set_ylabel("Fisher tr(Sb)/tr(Sw)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Layer-wise discriminability (Fisher ratio)")
    ax[1].plot(xs, f1_s, "-o", ms=3, label="subject")
    ax[1].plot(xs, f1_l, "-s", ms=3, label="level")
    ax[1].set_ylabel("probe macro-F1 (p1→p2)"); ax[1].set_xlabel("layer")
    ax[1].legend(); ax[1].grid(alpha=.3)
    for a in ax:
        a.axvspan(min(win_s) - .4, max(win_s) + .4, color="orange", alpha=.15)
        a.axvline(best_s, color="red", ls="--", lw=1)
        for ml in (MID_LAYERS[0], MID_LAYERS[-1]):
            a.axvline(ml, color="gray", ls=":", lw=.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pair_hist(same, diff, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(same.min(), diff.min()), max(same.max(), diff.max()), 60)
    ax.hist(diff, bins=bins, alpha=.5, density=True, label=f"diff-subject (n={diff.size})")
    ax.hist(same, bins=bins, alpha=.5, density=True, label=f"same-subject (n={same.size})")
    ax.axvline(diff.mean(), color="C0", ls="--"); ax.axvline(same.mean(), color="C1", ls="--")
    ax.set_xlabel("within-level pairwise cosine"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_eta_curves(eta_level, eta_subj, eta_inter, win, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(LAYERS)
    ax.plot(xs, eta_level, "-o", ms=3, label="η² level")
    ax.plot(xs, eta_subj, "-s", ms=3, label="η² subject")
    ax.plot(xs, eta_inter, "-^", ms=3, label="η² interaction")
    ax.axvspan(min(win) - .4, max(win) + .4, color="orange", alpha=.15)
    ax.set_xlabel("layer"); ax.set_ylabel("variance fraction (η²)")
    ax.set_title("UNIT variance partition per layer (level vs subject)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fmt_curve(arr):
    return ", ".join(f"L{l}:{arr[l]:.3f}" if np.isfinite(arr[l]) else f"L{l}:-"
                     for l in range(len(arr)))


# ───────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    ap.add_argument("--n-perm", type=int, default=1000, help="block-permutation 횟수")
    ap.add_argument("--pca-comps", type=int, default=150, help="프로브/LDA PCA 차원")
    ap.add_argument("--probe-pca", type=int, default=100, help="레이어 스캔 프로브 PCA 차원")
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
    print(f"[load] N={N} subjects={subjects} levels={levels} ({time.time()-t0:.0f}s)", flush=True)

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    del DAT; gc.collect()

    saved = {"subjects": np.array(subjects), "levels": np.array(levels)}
    L = []
    today = date.today().isoformat()
    L.append(f"# SUBJECT 단독 검정 (level 통제) + 레이어 분해 — {today}  (tag={TAG})")
    L.append("")
    L.append("> **가설.** 직전 세션에서 unit(subject×level) 구조는 LEVEL이 지배하고 SUBJECT는")
    L.append("> 거의 무신호(within/across≈1.0, silhouette 음수)였다. 그 분석은 (1) 전 레이어")
    L.append("> 평균 + (2) PCA(PC1 prototype) 기반이었다. 여기선 (1) **레이어 스캔**으로 subject")
    L.append("> 판별 레이어를 찾고 (2) **supervised 판별(Fisher·프로브·LDA)** 로, level을 통제한")
    L.append("> 조건에서 subject 신호가 살아나는지, unit에서 subject 비중이 얼마인지 검정한다.")
    L.append("")
    L.append(f"**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. "
             f"finite **N={ninfo['n_final']}** (raw {ninfo['n_loaded']}, non-finite drop "
             f"{ninfo['n_nonfinite']}). CPU only. seed={SEED}.")
    pc = md["_pilot"].value_counts()
    L.append(f"- provenance: " + ", ".join(f"{k}={int(v)}" for k, v in pc.items()))
    L.append(f"- 참고: 직전 세션 subject 윈도우 = mid L{MID_LAYERS[0]}–{MID_LAYERS[-1]} (고정 X, 비교용).")
    L.append("")

    # ── A. 레이어 스캔 ──────────────────────────────────────────────────────
    print("[A] layer scan (Fisher + probe) ...", flush=True)
    fisher_s = np.full(LAYERS, np.nan); fisher_l = np.full(LAYERS, np.nan)
    f1_s = np.full(LAYERS, np.nan); f1_l = np.full(LAYERS, np.nan)
    tr = (md["_pilot"] == "pilot1").to_numpy()
    te = (md["_pilot"] == "pilot2").to_numpy()
    can_probe = tr.sum() >= 50 and te.sum() >= 50
    y_s_tr, y_s_te = subj_arr[tr], subj_arr[te]
    y_l_tr, y_l_te = lev_arr[tr], lev_arr[te]
    lev_order = sorted(set(y_l_tr) & set(y_l_te)) if can_probe else []
    subj_order = sorted(set(y_s_tr) & set(y_s_te)) if can_probe else []
    for l in range(LAYERS):
        Xl = DA_c[:, l, :]
        fisher_s[l] = fisher_ratio(Xl, subj_arr, subjects)
        fisher_l[l] = fisher_ratio(Xl, lev_arr, levels)
        if can_probe:
            Xtr, Xte = Xl[tr], Xl[te]
            f1_s[l] = probe_f1(Xtr, y_s_tr, Xte, y_s_te, subj_order, args.probe_pca)
            f1_l[l] = probe_f1(Xtr, y_l_tr, Xte, y_l_te, lev_order, args.probe_pca)
        if (l + 1) % 6 == 0:
            print(f"  layer {l+1}/{LAYERS} ({time.time()-t0:.0f}s)", flush=True)

    # best layer / window: probe-F1 우선(가능하면), 아니면 Fisher
    score_for_select = f1_s if can_probe and np.isfinite(f1_s).any() else fisher_s
    best_s, win_s = select_window(score_for_select)
    best_l_layer, _ = select_window(f1_l if can_probe and np.isfinite(f1_l).any() else fisher_l)
    saved.update({"fisher_subject": fisher_s, "fisher_level": fisher_l,
                  "f1_subject": f1_s, "f1_level": f1_l,
                  "best_subject_layer": np.array([best_s]),
                  "subject_window": np.array(win_s)})
    plot_layer_scan(fisher_s, fisher_l, f1_s, f1_l, best_s, win_s,
                    ANALYSIS / f"{TAG}_layerscan.png")

    L.append("## A. 레이어 스캔 — subject(와 level) 판별 레이어 찾기")
    L.append("")
    L.append("방법: (a) Fisher 판별비 tr(Sb)/tr(Sw) — 36레이어 전부(싸다); "
             "(b) PCA-whiten 차원축소→다항 로지스틱 프로브, pilot1 train→pilot2 test, "
             "macro-F1(=일반화 판별력, PC1 prototype 아님).")
    L.append("")
    L.append(f"- **best subject layer = L{best_s}**; subject 윈도우(>=85%) = "
             f"L{min(win_s)}–L{max(win_s)} {win_s}")
    L.append(f"- best level layer = L{best_l_layer}")
    if can_probe:
        L.append(f"- probe macro-F1: subject best = {np.nanmax(f1_s):.3f} @L{int(np.nanargmax(f1_s))} "
                 f"(chance≈{1/len(subjects):.3f}); level best = {np.nanmax(f1_l):.3f} "
                 f"@L{int(np.nanargmax(f1_l))} (chance≈{1/len(levels):.3f})")
    else:
        L.append("- (smoke: train/test 부족으로 프로브 생략 → Fisher 로 선정)")
    L.append(f"- Fisher subject best = {np.nanmax(fisher_s):.3f} @L{int(np.nanargmax(fisher_s))}; "
             f"level best = {np.nanmax(fisher_l):.3f} @L{int(np.nanargmax(fisher_l))}")
    L.append(f"- 직전 mid L{MID_LAYERS[0]}–{MID_LAYERS[-1]} 비교: Fisher_subj 평균="
             f"{np.nanmean(fisher_s[MID_LAYERS]):.3f}, "
             f"F1_subj 평균={np.nanmean(f1_s[MID_LAYERS]):.3f}" if can_probe else
             f"- 직전 mid 비교: Fisher_subj 평균={np.nanmean(fisher_s[MID_LAYERS]):.3f}")
    L.append("")
    L.append("Fisher subject per-layer: " + fmt_curve(fisher_s))
    if can_probe:
        L.append("")
        L.append("probe-F1 subject per-layer: " + fmt_curve(f1_s))
    L.append("")
    L.append("> 해석: subject 곡선의 봉우리 레이어/윈도우가 subject 신호 집중 구간. level 곡선과")
    L.append("> 비교해 어느 축이 어느 레이어에서 강한지 확인. (곡선 그림: "
             f"`{TAG}_layerscan.png`)")
    L.append("")

    # 분석 view 정의: layeravg(36) / subject best-window / mid(참고)
    VIEWS = {"layeravg": list(range(LAYERS)),
             f"subjwin_L{min(win_s)}-{max(win_s)}": win_s,
             f"mid_L{MID_LAYERS[0]}-{MID_LAYERS[-1]}": MID_LAYERS}

    # per-layer normalize 한 번 (pairwise cosine 용)
    print("[norm] per-layer L2-normalize members ...", flush=True)
    DAn = sa.normalize_members(DA_c)

    # ── B. level 통제하 subject 단독 검정 (view 별) ─────────────────────────
    L.append("## B. level 통제하 SUBJECT 단독 검정")
    L.append("")
    L.append("metric = per-layer L2-normalized 후 layer-averaged *sample-pairwise* cosine. "
             "level 블록 안에서만 same-subject vs diff-subject 를 비교(→ level 오염 0).")
    for vname, layers in VIEWS.items():
        print(f"[B] view={vname} ...", flush=True)
        C = window_cos_matrix(DAn, layers)
        same, diff, per_level = within_level_pair_dists(C, md, subjects)
        d = cohens_d(same, diff)
        try:
            u, p_mwu = mannwhitneyu(same, diff, alternative="greater", method="asymptotic")
        except Exception:
            p_mwu = float("nan")
        obs, p_perm, nblk = block_perm_subject(C, md, subjects, args.n_perm)

        saved[f"{vname}_same_mean"] = np.array([same.mean() if same.size else np.nan])
        saved[f"{vname}_diff_mean"] = np.array([diff.mean() if diff.size else np.nan])
        saved[f"{vname}_cohend"] = np.array([d])
        saved[f"{vname}_perm_p"] = np.array([p_perm])
        saved[f"{vname}_perm_stat"] = np.array([obs])

        L.append(f"\n### view = `{vname}` (layers={layers if len(layers)<=6 else f'{layers[0]}..{layers[-1]}'})")
        ratio = (same.mean() / diff.mean()) if (diff.size and diff.mean() != 0) else float("nan")
        L.append(f"- within-level **same-subject** mean cos = {same.mean():+.4f} "
                 f"(n_pairs={same.size}); **diff-subject** mean cos = {diff.mean():+.4f} "
                 f"(n_pairs={diff.size}); ratio = **{ratio:.3f}x**")
        L.append(f"- **Cohen's d (same−diff) = {d:+.3f}**; Mann–Whitney p(same>diff) = {p_mwu:.2e}")
        L.append(f"- **block-permutation**(level 고정, subject 셔플 ×{args.n_perm}): "
                 f"stat(mean_same−mean_diff)={obs:+.4f}, **p={p_perm:.4f}** (blocks={nblk})")
        L.append("- per-level (same_mean / diff_mean / n_same / n_diff):")
        for lv, n, sm, dm, ns, nd in per_level:
            L.append(f"    L{lv} (n={n}): same={sm:+.4f} diff={dm:+.4f}  ({ns}/{nd} pairs)")
        if same.size and diff.size:
            plot_pair_hist(same, diff, f"{vname}: within-level same vs diff subject",
                           ANALYSIS / f"{TAG}_pairhist_{vname}.png")

        # within-level subject centroid cosine (closeness) — ssg 재사용
        S_A, den_A, info_A = ssg.within_level_subject_sim(DA_c, md, subjects, layers,
                                                          min_cell=5)
        saved[f"{vname}_subjsim"] = S_A
        L.append("\n- within-level subject centroid cosine 행렬 (level 가중평균):")
        L.append("```\n" + ssg.fmt_mat(S_A, subjects) + "\n```")
        L.append("- 각 subject 최근접 과목(centroid cosine):")
        for i, s in enumerate(subjects):
            row = S_A[i].copy(); row[i] = -np.inf
            valid = np.where(np.isfinite(row), row, -np.inf)
            j = int(np.argmax(valid))
            L.append(f"    {s} → {subjects[j]} ({S_A[i, j]:+.3f})")
        try:
            _, cuts, _ = ssg.dendro(S_A, subjects, f"{vname} subject clustering",
                                    ANALYSIS / f"{TAG}_dendro_{vname}.png")
            for k, groups in cuts.items():
                L.append(f"- {k}-cluster cut: " +
                         " | ".join(f"G{g}={{{', '.join(m)}}}" for g, m in sorted(groups.items())))
        except Exception as e:
            L.append(f"- dendrogram 실패: {e}")
        ssg.heatmap(S_A, subjects, f"{vname} within-level subject cosine",
                    ANALYSIS / f"{TAG}_heatmap_{vname}.png")
        del C; gc.collect()

    # ── B-LDA: confusion (closeness, supervised) ────────────────────────────
    L.append("\n### B-LDA — supervised confusion (어느 과목끼리 헷갈리나=유사)")
    for vname, layers in VIEWS.items():
        res, err = lda_confusion(DA_c, md, subjects, layers, args.pca_comps)
        if res is None:
            L.append(f"- `{vname}`: LDA 생략 ({err})"); continue
        order = res["order"]; cm = res["cm_norm"]
        saved[f"{vname}_lda_cm"] = cm; saved[f"{vname}_lda_order"] = np.array(order)
        saved[f"{vname}_lda_f1"] = np.array([res["f1"]])
        L.append(f"\n- `{vname}`: PCA({args.pca_comps})→LDA, pilot1→pilot2, "
                 f"**macro-F1={res['f1']:.3f}** (chance≈{1/len(order):.3f})")
        L.append("  confusion (row-normalized, test):")
        L.append("```\n" + ssg.fmt_mat(cm, order) + "\n```")
        # 가장 큰 off-diagonal 혼동쌍 top5
        cms = (cm + cm.T) / 2.0; np.fill_diagonal(cms, 0.0)
        iu = np.triu_indices(len(order), 1)
        pairs = sorted(zip(cms[iu], [(order[a], order[b]) for a, b in zip(*iu)]),
                       reverse=True)[:5]
        L.append("  top 혼동쌍(=내부 유사): " +
                 "; ".join(f"{a}↔{b}={v:.3f}" for v, (a, b) in pairs))

    # ── C. UNIT 분산분해 + level-residual silhouette ───────────────────────
    print("[C] unit variance partition ...", flush=True)
    eta_level = np.full(LAYERS, np.nan); eta_subj = np.full(LAYERS, np.nan)
    eta_inter = np.full(LAYERS, np.nan)
    for l in range(LAYERS):
        e = eta2_partition(DA_c[:, l, :], lev_arr, subj_arr)
        eta_level[l] = e["level"]; eta_subj[l] = e["subject"]; eta_inter[l] = e["interaction"]
    saved.update({"eta_level": eta_level, "eta_subject": eta_subj, "eta_interaction": eta_inter})
    plot_eta_curves(eta_level, eta_subj, eta_inter, win_s, ANALYSIS / f"{TAG}_eta.png")

    L.append("\n## C. UNIT(subject×level) 구조에서 SUBJECT 비중 — 2-way 분산분해")
    L.append("")
    L.append("주변평균 기반 η²(불균형 설계라 비직교 → 근사 분해; interaction 음수면 0 취급 표기). "
             "η²_level vs η²_subject 로 'unit 구조를 누가 이끄는가'를 수치화.")
    L.append(f"- 전체 레이어 평균: η²_level={np.nanmean(eta_level):.3f}, "
             f"η²_subject={np.nanmean(eta_subj):.3f}, η²_interaction={np.nanmean(eta_inter):.3f}")
    L.append(f"- subject 윈도우(L{min(win_s)}–{max(win_s)}) 평균: "
             f"η²_level={np.nanmean(eta_level[win_s]):.3f}, "
             f"η²_subject={np.nanmean(eta_subj[win_s]):.3f}, "
             f"η²_interaction={np.nanmean(eta_inter[win_s]):.3f}")
    L.append(f"- best subject layer L{best_s}: η²_level={eta_level[best_s]:.3f}, "
             f"η²_subject={eta_subj[best_s]:.3f}")
    L.append("")
    L.append("η² level per-layer:   " + fmt_curve(eta_level))
    L.append("")
    L.append("η² subject per-layer: " + fmt_curve(eta_subj))
    L.append(f"\n(곡선 그림: `{TAG}_eta.png`)")

    # level-residualize → subject silhouette (window + layeravg)
    L.append("\n### C-silhouette — level 제거 전/후 subject silhouette")
    L.append("self-level centroid(per-layer) 차감으로 level 주효과 제거 후, subject 라벨 "
             "silhouette(1-cos 거리) 재계산. 직전 unit-report 의 음수 subject silhouette 과 대조.")
    resid = ssg.level_centroid_residual(DA_c, md)
    DAn_resid = sa.normalize_members(resid)
    for vname, layers in VIEWS.items():
        C_raw = window_cos_matrix(DAn, layers)
        sil_raw = subject_silhouette(C_raw, subj_arr); del C_raw
        C_res = window_cos_matrix(DAn_resid, layers)
        sil_res = subject_silhouette(C_res, subj_arr); del C_res
        gc.collect()
        saved[f"{vname}_sil_subject_raw"] = np.array([sil_raw])
        saved[f"{vname}_sil_subject_residlevel"] = np.array([sil_res])
        L.append(f"- `{vname}`: subject silhouette raw = {sil_raw:+.4f} → "
                 f"level-residual 후 = {sil_res:+.4f} "
                 f"({'개선' if (np.isfinite(sil_res) and np.isfinite(sil_raw) and sil_res>sil_raw) else '비개선'})")
    del resid, DAn_resid; gc.collect()

    # ── 결론 ────────────────────────────────────────────────────────────────
    L.append("\n## 결론 (요약)")
    win_name = f"subjwin_L{min(win_s)}-{max(win_s)}"
    pperm = float(saved[f"{win_name}_perm_p"][0])
    dwin = float(saved[f"{win_name}_cohend"][0])
    L.append(f"1. **레이어 스캔**: subject 판별 봉우리 = L{min(win_s)}–L{max(win_s)} "
             f"(best L{best_s}); 직전 mid L11–15 와의 정합성은 위 §A 수치 참조.")
    L.append(f"2. **level 통제 subject 신호**(best window): same vs diff Cohen's d={dwin:+.3f}, "
             f"block-permutation p={pperm:.4f} → "
             f"{'유의(level 고정 시 subject 신호 존재)' if pperm < 0.05 else '비유의(level 통제 후 subject 신호 약함)'}.")
    L.append(f"3. **UNIT 비중**: η²_level vs η²_subject (윈도우 "
             f"{np.nanmean(eta_level[win_s]):.3f} vs {np.nanmean(eta_subj[win_s]):.3f}) — "
             "level이 더 크면 여전히 level 우위지만, subject η²가 유의 비중이면 subject로도 부분 구분 가능.")
    L.append("4. **silhouette**: level-residual 후 subject silhouette 변화는 위 §C-silhouette 참조.")
    L.append("")
    L.append("> 주의: η² 분해는 불균형 설계의 근사이며, 8그룹/소표본 행렬 하나로 강결론 금지. "
             "permutation p 와 효과크기, 그리고 pilot1/pilot2 일반화(프로브·LDA)를 함께 본다.")

    # ── write ───────────────────────────────────────────────────────────────
    out_md = ANALYSIS / f"REPORT_subject_controlled_{today}.md"
    np.savez(ANALYSIS / f"{TAG}_artifacts.npz", **saved)
    out_md.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"[OK] wrote {out_md}", flush=True)
    print(f"[OK] wrote {ANALYSIS / f'{TAG}_artifacts.npz'}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
