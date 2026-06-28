#!/usr/bin/env python3
"""
level_unit_resolved.py — LEVEL / UNIT 신호 정밀검증 (subject 통제) + 레이어 분해.
================================================================================
배경. 직전 세션 결론: unit(=subject×level) 표현 구조는 LEVEL이 지배하고 SUBJECT는
거의 무신호(within/across≈1.0, silhouette 음수)였다. 그리고 직후 세션에서 SUBJECT를
*level 통제하*에 단독검정하는 `subject_layer_resolved.py`(레이어스캔 Fisher/프로브,
block-permutation, η² 2-way 분산분해, LDA confusion, level-residual silhouette)를
구현했다.

이 스크립트는 **그 subject 분석 코드(기법)를 그대로 재사용**하되, 주효과 축을
subject→**level/unit** 으로 대칭 전환하여, 지배적이라 알려진 LEVEL 신호와 UNIT
구조가 같은 정밀검증 도구(supervised 판별 + permutation + 분산분해)로 봐도 견고한지
정량 검증한다. 새 통계 로직은 추가하지 않는다 — `subject_layer_resolved`(=slr)의
헬퍼를 import 해서 호출하거나, 라벨/블록 컬럼만 파라미터화한 대칭 미러를 쓴다.

  §A 레이어 스캔 — slr.fisher_ratio / slr.probe_f1 / slr.select_window 재사용.
       36레이어 Fisher 판별비 + CV 선형프로브(pilot1→pilot2, macro-F1)로 LEVEL(과
       subject 참조) 판별 레이어/윈도우를 데이터로 찾는다.
  §B subject 통제하 LEVEL 단독 검정 — slr 의 within-block pairwise + block-perm 을
       *대칭 미러*(블록=subject 고정, 라벨=level 셔플)로 적용.
       (A) within-subject same-level vs diff-level sample-pairwise cosine 분포,
           Cohen's d(slr.cohens_d) + Mann–Whitney.
       (B) block-permutation: subject 블록 고정, level 라벨만 셔플 → p값+효과크기.
       (C) LDA confusion (pilot1 train/pilot2 test) → 헷갈리는(=유사) level (인접?).
  §C UNIT(subject×level) 구조 정밀검증 —
       (1) per-layer 2-way η² 분산분해(slr.eta2_partition; level/subject/interaction),
       (2) same-unit vs diff-unit pairwise cosine(within/across ratio) + unit silhouette,
       (3) subject-residualize 후 level silhouette / level-residualize 후 unit silhouette
           대조(slr.subject_silhouette = 라벨 무관 generic silhouette 재사용).

데이터/재사용: pooled(pilot1+pilot2) THINKING ΔA, canonical finite N=3025,
  μ_pooled(per-layer) centering. pooled_analysis(pa)/similarity_analysis(sa)/
  subject_similarity_gate(ssg) 및 subject_layer_resolved(slr) 재사용. CPU only.
  기존 산출물 미변경.

OUTPUT (analysis/):
  REPORT_levelunit_controlled_<date>.md, levunit_artifacts.npz, 곡선/heatmap/hist PNG.
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
import subject_layer_resolved as slr   # 기법(헬퍼) 재사용

ANALYSIS = Path(__file__).resolve().parent
TAG = "levunit"
LAYERS = slr.LAYERS
MID_LAYERS = slr.MID_LAYERS           # 직전 subject 윈도우(참고값)
SEED = slr.SEED


# ───────────── 대칭 미러: 블록/라벨 컬럼 파라미터화 (slr 로직 동일) ─────────────
def within_block_pair_dists(C, md, block_col, label_col):
    """slr.within_level_pair_dists 의 대칭 일반화.
    각 `block_col` 블록 안에서 same-`label_col` vs diff-`label_col` *쌍별* cosine 수집.
    (block=subject, label=level 로 호출하면 'subject 고정 후 level 검정')."""
    same_all, diff_all = [], []
    per_block = []
    blocks = sorted(md[block_col].unique().tolist())
    lab = md[label_col].to_numpy()
    for bk in blocks:
        idx = md.index[md[block_col] == bk].to_numpy()
        if len(idx) < 4:
            per_block.append((bk, len(idx), np.nan, np.nan, 0, 0))
            continue
        sub_lab = lab[idx]
        Csub = C[np.ix_(idx, idx)]
        n = len(idx)
        iu = np.triu_indices(n, 1)
        eq = (sub_lab[:, None] == sub_lab[None, :])[iu]
        vals = Csub[iu]
        sv = vals[eq]; dv = vals[~eq]
        same_all.append(sv); diff_all.append(dv)
        per_block.append((bk, n, float(sv.mean()) if sv.size else np.nan,
                          float(dv.mean()) if dv.size else np.nan,
                          int(sv.size), int(dv.size)))
    same = np.concatenate(same_all) if same_all else np.array([])
    diff = np.concatenate(diff_all) if diff_all else np.array([])
    return same, diff, per_block


def block_perm(C, md, block_col, label_col, n_perm, seed=SEED):
    """slr.block_perm_subject 의 대칭 일반화.
    `block_col` 블록 고정, `label_col` 라벨만 블록 내 셔플 → 통계량 = 블록-가중 평균
    (mean_same - mean_diff). p = P(stat_perm >= stat_obs)."""
    rng = np.random.default_rng(seed)
    blocks_v = sorted(md[block_col].unique().tolist())
    lab = md[label_col].to_numpy()
    blocks = []
    for bk in blocks_v:
        idx = md.index[md[block_col] == bk].to_numpy()
        if len(idx) < 4:
            continue
        n = len(idx)
        Csub = C[np.ix_(idx, idx)].astype(np.float32)
        iu = np.triu_indices(n, 1)
        blocks.append({"lab": lab[idx].copy(), "C": Csub, "iu": iu,
                       "vals": Csub[iu], "npairs": len(iu[0])})
    if not blocks:
        return float("nan"), float("nan"), 0

    def stat(label_lists):
        num = 0.0; den = 0.0
        for bk, lb in zip(blocks, label_lists):
            eq = (lb[:, None] == lb[None, :])[bk["iu"]]
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


def lda_confusion_col(DA_c, md, layers, label_col, pca_comps, seed=SEED):
    """slr.lda_confusion 의 대칭 일반화 (label_col 로 train/test)."""
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
    ytr = md.loc[tr, label_col].to_numpy(); yte = md.loc[te, label_col].to_numpy()
    lda = LinearDiscriminantAnalysis().fit(Ztr, ytr)
    pred = lda.predict(Zte)
    order = sorted(set(ytr) & set(yte))
    f1 = f1_score(yte, pred, labels=order, average="macro", zero_division=0)
    cm = confusion_matrix(yte, pred, labels=order).astype(float)
    row = cm.sum(1, keepdims=True)
    cm_norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return {"order": order, "f1": float(f1), "cm_norm": cm_norm}, None


def unit_pair_dists(C, md):
    """UNIT(=subject×level) cohesion: same-unit vs diff-unit 전체 쌍별 cosine.
    (블록 없음 — unit 라벨이 같은 쌍 vs 다른 쌍.)"""
    unit = (md["subject"].astype(str) + "|" + md["level"].astype(str)).to_numpy()
    n = len(md)
    iu = np.triu_indices(n, 1)
    eq = (unit[:, None] == unit[None, :])[iu]
    vals = C[iu]
    return vals[eq], vals[~eq]


# ───────────────────────────── plotting ────────────────────────────────────
def plot_layer_scan(fisher_l, fisher_s, f1_l, f1_s, best_l, win_l, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    xs = np.arange(LAYERS)
    ax[0].plot(xs, fisher_l, "-s", ms=3, label="level")
    ax[0].plot(xs, fisher_s, "-o", ms=3, label="subject")
    ax[0].set_ylabel("Fisher tr(Sb)/tr(Sw)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Layer-wise discriminability (Fisher ratio)")
    ax[1].plot(xs, f1_l, "-s", ms=3, label="level")
    ax[1].plot(xs, f1_s, "-o", ms=3, label="subject")
    ax[1].set_ylabel("probe macro-F1 (p1->p2)"); ax[1].set_xlabel("layer")
    ax[1].legend(); ax[1].grid(alpha=.3)
    for a in ax:
        a.axvspan(min(win_l) - .4, max(win_l) + .4, color="orange", alpha=.15)
        a.axvline(best_l, color="red", ls="--", lw=1)
        for ml in (MID_LAYERS[0], MID_LAYERS[-1]):
            a.axvline(ml, color="gray", ls=":", lw=.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pair_hist(same, diff, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(same.min(), diff.min()), max(same.max(), diff.max()), 60)
    ax.hist(diff, bins=bins, alpha=.5, density=True, label=f"diff (n={diff.size})")
    ax.hist(same, bins=bins, alpha=.5, density=True, label=f"same (n={same.size})")
    ax.axvline(diff.mean(), color="C0", ls="--"); ax.axvline(same.mean(), color="C1", ls="--")
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_eta_curves(eta_level, eta_subj, eta_inter, win, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(LAYERS)
    ax.plot(xs, eta_level, "-o", ms=3, label="eta2 level")
    ax.plot(xs, eta_subj, "-s", ms=3, label="eta2 subject")
    ax.plot(xs, eta_inter, "-^", ms=3, label="eta2 interaction")
    ax.axvspan(min(win) - .4, max(win) + .4, color="orange", alpha=.15)
    ax.set_xlabel("layer"); ax.set_ylabel("variance fraction (eta2)")
    ax.set_title("UNIT variance partition per layer (level vs subject)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ───────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    ap.add_argument("--n-perm", type=int, default=1000, help="block-permutation 횟수")
    ap.add_argument("--pca-comps", type=int, default=150, help="LDA PCA 차원")
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
    unit_arr = (md["subject"].astype(str) + "|" + md["level"].astype(str)).to_numpy()
    n_units = len(np.unique(unit_arr))
    print(f"[load] N={N} subjects={subjects} levels={levels} units={n_units} "
          f"({time.time()-t0:.0f}s)", flush=True)

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    del DAT; gc.collect()

    saved = {"subjects": np.array(subjects), "levels": np.array(levels),
             "n_units": np.array([n_units])}
    L = []
    today = date.today().isoformat()
    L.append(f"# LEVEL / UNIT 정밀검정 (subject 통제) + 레이어 분해 — {today}  (tag={TAG})")
    L.append("")
    L.append("> **목적.** 직전 세션에서 unit(subject×level) 구조는 LEVEL이 지배(within/across")
    L.append("> 비율 level 1.46x vs subject 0.99x)했다. subject는 `subject_layer_resolved.py`로")
    L.append("> level 통제하 단독검정을 마쳤다. 여기서는 **그 분석 코드(기법)를 그대로 재사용**해")
    L.append("> 주효과 축만 level/unit 으로 대칭 전환하여, 지배적이라 알려진 LEVEL/UNIT 신호가")
    L.append("> supervised 판별(Fisher·프로브·LDA) + block-permutation + η² 분산분해로 봐도")
    L.append("> 견고한지 정밀 검증한다. (subject 분석의 대칭 통제: 블록=subject, 라벨=level.)")
    L.append("")
    L.append(f"**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. "
             f"finite **N={ninfo['n_final']}** (raw {ninfo['n_loaded']}, non-finite drop "
             f"{ninfo['n_nonfinite']}). units={n_units}. CPU only. seed={SEED}.")
    pc = md["_pilot"].value_counts()
    L.append(f"- provenance: " + ", ".join(f"{k}={int(v)}" for k, v in pc.items()))
    L.append(f"- 재사용: subject_layer_resolved(slr).fisher_ratio/probe_f1/select_window/"
             f"window_cos_matrix/cohens_d/eta2_partition/subject_silhouette(=generic).")
    L.append("")

    # ── A. 레이어 스캔 ──────────────────────────────────────────────────────
    print("[A] layer scan (Fisher + probe) ...", flush=True)
    fisher_l = np.full(LAYERS, np.nan); fisher_s = np.full(LAYERS, np.nan)
    f1_l = np.full(LAYERS, np.nan); f1_s = np.full(LAYERS, np.nan)
    tr = (md["_pilot"] == "pilot1").to_numpy()
    te = (md["_pilot"] == "pilot2").to_numpy()
    can_probe = tr.sum() >= 50 and te.sum() >= 50
    y_l_tr, y_l_te = lev_arr[tr], lev_arr[te]
    y_s_tr, y_s_te = subj_arr[tr], subj_arr[te]
    lev_order = sorted(set(y_l_tr) & set(y_l_te)) if can_probe else []
    subj_order = sorted(set(y_s_tr) & set(y_s_te)) if can_probe else []
    for l in range(LAYERS):
        Xl = DA_c[:, l, :]
        fisher_l[l] = slr.fisher_ratio(Xl, lev_arr, levels)
        fisher_s[l] = slr.fisher_ratio(Xl, subj_arr, subjects)
        if can_probe:
            Xtr, Xte = Xl[tr], Xl[te]
            f1_l[l] = slr.probe_f1(Xtr, y_l_tr, Xte, y_l_te, lev_order, args.probe_pca)
            f1_s[l] = slr.probe_f1(Xtr, y_s_tr, Xte, y_s_te, subj_order, args.probe_pca)
        if (l + 1) % 6 == 0:
            print(f"  layer {l+1}/{LAYERS} ({time.time()-t0:.0f}s)", flush=True)

    score_for_select = f1_l if can_probe and np.isfinite(f1_l).any() else fisher_l
    best_l, win_l = slr.select_window(score_for_select)
    best_s_layer, _ = slr.select_window(f1_s if can_probe and np.isfinite(f1_s).any() else fisher_s)
    saved.update({"fisher_level": fisher_l, "fisher_subject": fisher_s,
                  "f1_level": f1_l, "f1_subject": f1_s,
                  "best_level_layer": np.array([best_l]),
                  "level_window": np.array(win_l)})
    plot_layer_scan(fisher_l, fisher_s, f1_l, f1_s, best_l, win_l,
                    ANALYSIS / f"{TAG}_layerscan.png")

    L.append("## A. 레이어 스캔 — LEVEL(과 subject 참조) 판별 레이어 찾기")
    L.append("")
    L.append("방법(slr 재사용): (a) Fisher 판별비 tr(Sb)/tr(Sw) — 36레이어 전부; "
             "(b) PCA-whiten→다항 로지스틱 프로브, pilot1 train→pilot2 test, macro-F1.")
    L.append("")
    L.append(f"- **best level layer = L{best_l}**; level 윈도우(>=85%) = "
             f"L{min(win_l)}–L{max(win_l)} {win_l}")
    L.append(f"- best subject layer = L{best_s_layer} (참조)")
    if can_probe:
        L.append(f"- probe macro-F1: level best = {np.nanmax(f1_l):.3f} @L{int(np.nanargmax(f1_l))} "
                 f"(chance≈{1/len(levels):.3f}); subject best = {np.nanmax(f1_s):.3f} "
                 f"@L{int(np.nanargmax(f1_s))} (chance≈{1/len(subjects):.3f})")
    else:
        L.append("- (smoke: train/test 부족으로 프로브 생략 → Fisher 로 선정)")
    L.append(f"- Fisher level best = {np.nanmax(fisher_l):.3f} @L{int(np.nanargmax(fisher_l))}; "
             f"subject best = {np.nanmax(fisher_s):.3f} @L{int(np.nanargmax(fisher_s))}")
    L.append("")
    L.append("Fisher level per-layer: " + slr.fmt_curve(fisher_l))
    if can_probe:
        L.append("")
        L.append("probe-F1 level per-layer: " + slr.fmt_curve(f1_l))
    L.append("")
    L.append(f"> 해석: level 곡선의 봉우리 레이어/윈도우가 level 신호 집중 구간. "
             f"(곡선 그림: `{TAG}_layerscan.png`)")
    L.append("")

    # 분석 view: layeravg / level best-window / mid(참고)
    VIEWS = {"layeravg": list(range(LAYERS)),
             f"levwin_L{min(win_l)}-{max(win_l)}": win_l,
             f"mid_L{MID_LAYERS[0]}-{MID_LAYERS[-1]}": MID_LAYERS}

    print("[norm] per-layer L2-normalize members ...", flush=True)
    DAn = sa.normalize_members(DA_c)

    # ── B. subject 통제하 LEVEL 단독 검정 (대칭 미러) ───────────────────────
    L.append("## B. subject 통제하 LEVEL 단독 검정 (대칭: 블록=subject, 라벨=level)")
    L.append("")
    L.append("metric = per-layer L2-normalized 후 layer-averaged *sample-pairwise* cosine. "
             "subject 블록 안에서만 same-level vs diff-level 비교(→ subject 오염 0). "
             "이는 slr 의 'level 고정·subject 검정' 의 정확한 대칭(subject 고정·level 검정).")
    for vname, layers in VIEWS.items():
        print(f"[B] view={vname} ...", flush=True)
        C = slr.window_cos_matrix(DAn, layers)
        same, diff, per_block = within_block_pair_dists(C, md, "subject", "level")
        d = slr.cohens_d(same, diff)
        try:
            u, p_mwu = mannwhitneyu(same, diff, alternative="greater", method="asymptotic")
        except Exception:
            p_mwu = float("nan")
        obs, p_perm, nblk = block_perm(C, md, "subject", "level", args.n_perm)

        saved[f"{vname}_same_mean"] = np.array([same.mean() if same.size else np.nan])
        saved[f"{vname}_diff_mean"] = np.array([diff.mean() if diff.size else np.nan])
        saved[f"{vname}_cohend"] = np.array([d])
        saved[f"{vname}_perm_p"] = np.array([p_perm])
        saved[f"{vname}_perm_stat"] = np.array([obs])

        L.append(f"\n### view = `{vname}` (layers={layers if len(layers)<=6 else f'{layers[0]}..{layers[-1]}'})")
        ratio = (same.mean() / diff.mean()) if (diff.size and diff.mean() != 0) else float("nan")
        L.append(f"- within-subject **same-level** mean cos = {same.mean():+.4f} "
                 f"(n_pairs={same.size}); **diff-level** mean cos = {diff.mean():+.4f} "
                 f"(n_pairs={diff.size}); ratio = **{ratio:.3f}x**")
        L.append(f"- **Cohen's d (same−diff) = {d:+.3f}**; Mann–Whitney p(same>diff) = {p_mwu:.2e}")
        L.append(f"- **block-permutation**(subject 고정, level 셔플 ×{args.n_perm}): "
                 f"stat(mean_same−mean_diff)={obs:+.4f}, **p={p_perm:.4f}** (blocks={nblk})")
        L.append("- per-subject (same_mean / diff_mean / n_same / n_diff):")
        for bk, n, sm, dm, ns, nd in per_block:
            L.append(f"    {bk} (n={n}): same={sm:+.4f} diff={dm:+.4f}  ({ns}/{nd} pairs)")
        if same.size and diff.size:
            plot_pair_hist(same, diff, f"{vname}: within-subject same vs diff level",
                           "within-subject pairwise cosine",
                           ANALYSIS / f"{TAG}_pairhist_{vname}.png")
        del C; gc.collect()

    # ── B-LDA: level confusion (인접 level 혼동?) ───────────────────────────
    L.append("\n### B-LDA — supervised LEVEL confusion (어느 level끼리 헷갈리나=유사)")
    for vname, layers in VIEWS.items():
        res, err = lda_confusion_col(DA_c, md, layers, "level", args.pca_comps)
        if res is None:
            L.append(f"- `{vname}`: LDA 생략 ({err})"); continue
        order = res["order"]; cm = res["cm_norm"]
        saved[f"{vname}_lda_cm"] = cm; saved[f"{vname}_lda_order"] = np.array(order)
        saved[f"{vname}_lda_f1"] = np.array([res["f1"]])
        L.append(f"\n- `{vname}`: PCA({args.pca_comps})→LDA, pilot1→pilot2, "
                 f"**macro-F1={res['f1']:.3f}** (chance≈{1/len(order):.3f})")
        L.append("  confusion (row-normalized, test):")
        L.append("```\n" + ssg.fmt_mat(cm, [str(o) for o in order]) + "\n```")
        cms = (cm + cm.T) / 2.0; np.fill_diagonal(cms, 0.0)
        iu = np.triu_indices(len(order), 1)
        pairs = sorted(zip(cms[iu], [(order[a], order[b]) for a, b in zip(*iu)]),
                       reverse=True)[:5]
        L.append("  top 혼동쌍(=내부 유사): " +
                 "; ".join(f"{a}↔{b}={v:.3f}" for v, (a, b) in pairs))

    # ── C. UNIT 정밀검증 ────────────────────────────────────────────────────
    print("[C] unit variance partition ...", flush=True)
    eta_level = np.full(LAYERS, np.nan); eta_subj = np.full(LAYERS, np.nan)
    eta_inter = np.full(LAYERS, np.nan)
    for l in range(LAYERS):
        e = slr.eta2_partition(DA_c[:, l, :], lev_arr, subj_arr)
        eta_level[l] = e["level"]; eta_subj[l] = e["subject"]; eta_inter[l] = e["interaction"]
    saved.update({"eta_level": eta_level, "eta_subject": eta_subj, "eta_interaction": eta_inter})
    plot_eta_curves(eta_level, eta_subj, eta_inter, win_l, ANALYSIS / f"{TAG}_eta.png")

    L.append("\n## C. UNIT(subject×level) 구조 정밀검증")
    L.append("")
    L.append("### C-1. 2-way 분산분해 (slr.eta2_partition)")
    L.append("주변평균 기반 η²(불균형 설계라 비직교 → 근사 분해; interaction 음수면 0 취급). "
             "η²_level vs η²_subject 로 'unit 구조를 누가 이끄는가' 수치화.")
    L.append(f"- 전체 레이어 평균: η²_level={np.nanmean(eta_level):.3f}, "
             f"η²_subject={np.nanmean(eta_subj):.3f}, η²_interaction={np.nanmean(eta_inter):.3f}")
    L.append(f"- level 윈도우(L{min(win_l)}–{max(win_l)}) 평균: "
             f"η²_level={np.nanmean(eta_level[win_l]):.3f}, "
             f"η²_subject={np.nanmean(eta_subj[win_l]):.3f}, "
             f"η²_interaction={np.nanmean(eta_inter[win_l]):.3f}")
    L.append(f"- best level layer L{best_l}: η²_level={eta_level[best_l]:.3f}, "
             f"η²_subject={eta_subj[best_l]:.3f}")
    L.append("")
    L.append("η² level per-layer:   " + slr.fmt_curve(eta_level))
    L.append("")
    L.append("η² subject per-layer: " + slr.fmt_curve(eta_subj))
    L.append(f"\n(곡선 그림: `{TAG}_eta.png`)")

    # C-2. same-unit vs diff-unit cohesion + unit silhouette
    L.append("\n### C-2. UNIT cohesion — same-unit vs diff-unit pairwise cosine + silhouette")
    for vname, layers in VIEWS.items():
        print(f"[C-2] view={vname} ...", flush=True)
        C = slr.window_cos_matrix(DAn, layers)
        su, du = unit_pair_dists(C, md)
        du_mean = du.mean() if du.size else np.nan
        ratio_u = (su.mean() / du_mean) if (du.size and du_mean != 0) else float("nan")
        d_u = slr.cohens_d(su, du)
        sil_unit = slr.subject_silhouette(C, unit_arr)
        sil_level = slr.subject_silhouette(C, lev_arr)
        sil_subj = slr.subject_silhouette(C, subj_arr)
        saved[f"{vname}_unit_same_mean"] = np.array([su.mean() if su.size else np.nan])
        saved[f"{vname}_unit_diff_mean"] = np.array([du_mean])
        saved[f"{vname}_unit_cohend"] = np.array([d_u])
        saved[f"{vname}_sil_unit"] = np.array([sil_unit])
        saved[f"{vname}_sil_level"] = np.array([sil_level])
        saved[f"{vname}_sil_subject"] = np.array([sil_subj])
        L.append(f"- `{vname}`: same-unit cos={su.mean():+.4f} (n={su.size}) / "
                 f"diff-unit cos={du_mean:+.4f} (n={du.size}); ratio=**{ratio_u:.3f}x**; "
                 f"Cohen's d={d_u:+.3f}")
        L.append(f"    silhouette: unit={sil_unit:+.4f}, level={sil_level:+.4f}, "
                 f"subject={sil_subj:+.4f}")
        if su.size and du.size and vname.startswith("levwin"):
            plot_pair_hist(su, du, f"{vname}: same vs diff UNIT",
                           "pairwise cosine",
                           ANALYSIS / f"{TAG}_unithist_{vname}.png")
        del C; gc.collect()

    # C-3. residualize 대조: subject 제거 후 level silhouette / level 제거 후 unit
    L.append("\n### C-3. residualize 대조 silhouette")
    L.append("level-residual(self-level centroid 차감, ssg.level_centroid_residual) 후 "
             "subject/unit silhouette 변화로 'level 제거 시 잔여 구조'를 확인.")
    resid = ssg.level_centroid_residual(DA_c, md)
    DAn_resid = sa.normalize_members(resid)
    for vname, layers in VIEWS.items():
        C_res = slr.window_cos_matrix(DAn_resid, layers)
        sil_subj_res = slr.subject_silhouette(C_res, subj_arr)
        sil_unit_res = slr.subject_silhouette(C_res, unit_arr)
        del C_res; gc.collect()
        saved[f"{vname}_sil_subject_residlevel"] = np.array([sil_subj_res])
        saved[f"{vname}_sil_unit_residlevel"] = np.array([sil_unit_res])
        sil_subj_raw = float(saved[f"{vname}_sil_subject"][0])
        sil_unit_raw = float(saved[f"{vname}_sil_unit"][0])
        L.append(f"- `{vname}`: subject silhouette {sil_subj_raw:+.4f} → "
                 f"(level-residual) {sil_subj_res:+.4f}; "
                 f"unit silhouette {sil_unit_raw:+.4f} → {sil_unit_res:+.4f}")
    del resid, DAn_resid; gc.collect()

    # ── 결론 ────────────────────────────────────────────────────────────────
    L.append("\n## 결론 (요약)")
    win_name = f"levwin_L{min(win_l)}-{max(win_l)}"
    pperm = float(saved[f"{win_name}_perm_p"][0])
    dwin = float(saved[f"{win_name}_cohend"][0])
    su_ratio_win = (float(saved[f"{win_name}_unit_same_mean"][0]) /
                    float(saved[f"{win_name}_unit_diff_mean"][0])
                    if saved[f"{win_name}_unit_diff_mean"][0] not in (0, np.nan) else float("nan"))
    L.append(f"1. **레이어 스캔**: level 판별 봉우리 = L{min(win_l)}–L{max(win_l)} "
             f"(best L{best_l}); subject best L{best_s_layer} 과 비교는 §A 수치 참조.")
    L.append(f"2. **subject 통제 LEVEL 신호**(best window): same vs diff Cohen's d={dwin:+.3f}, "
             f"block-permutation p={pperm:.4f} → "
             f"{'유의(subject 고정해도 level 신호 견고)' if pperm < 0.05 else '비유의'}.")
    L.append(f"3. **UNIT**: η²_level vs η²_subject (윈도우 "
             f"{np.nanmean(eta_level[win_l]):.3f} vs {np.nanmean(eta_subj[win_l]):.3f}); "
             f"same/diff-unit ratio≈{su_ratio_win:.3f}x. silhouette(unit/level/subject)는 §C-2 참조.")
    L.append("4. **residualize 대조**: level 제거 후 subject/unit silhouette 변화는 §C-3 참조.")
    L.append("")
    L.append("> 주의: η² 분해는 불균형 설계의 근사이며 소표본 행렬 하나로 강결론 금지. "
             "permutation p·효과크기·pilot1/pilot2 일반화(프로브·LDA)를 함께 본다.")

    out_md = ANALYSIS / f"REPORT_levelunit_controlled_{today}.md"
    np.savez(ANALYSIS / f"{TAG}_artifacts.npz", **saved)
    out_md.write_text("\n".join(str(x) for x in L), encoding="utf-8")
    print(f"[OK] wrote {out_md}", flush=True)
    print(f"[OK] wrote {ANALYSIS / f'{TAG}_artifacts.npz'}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
