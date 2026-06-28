#!/usr/bin/env python3
"""
unit_analysis.py
================
Activation-shift (ΔA) analysis for the thinking-mode pilot, in TWO views:

  (1) SELECTED UNIT view   : unit = (subject, level) cell  -> within/between-unit
                             dispersion, per-unit correct-incorrect contrast
                             consistency.
  (2) POOLED / GLOBAL view : ignore unit boundaries -> global PCA of per-layer
                             |dA| profile + LAYERWISE linear probes
                             (is_correct / level / subject) vs chance.

Both dA_faithful (answer span) and dA_thinking (think span) are analyzed.
Length / truncation confounds are controlled (faithful uses finish=stop subset;
partial Spearman of |dA| vs level controlling gen_len).

CPU only. Memory-frugal: ΔA loaded once as float16 (N,36,12288)*2 ~ 2.7 GB.

Usage:
  PY unit_analysis.py \
     --shifts-dir .../outputs/pilot/shifts \
     --out-dir   .../analysis \
     --tag pilot_partial [--max-n N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

BASE = Path("/scratch/lami2026/personal/jimin_2782")
PROBE_PCA_DIM = 50
N_FOLDS = 5
SEED = 42


# ────────────────────────────── helpers ──────────────────────────────────
def spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return float("nan")
    ra, rb = rankdata(a[m]), rankdata(b[m])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman ρ(x, y) controlling z, via rank residuals."""
    x = rankdata(x); y = rankdata(y); z = rankdata(z)

    def resid(a, ctrl):
        A = np.c_[np.ones_like(ctrl), ctrl]
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta

    rx, ry = resid(x, z), resid(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def layer_probe_classif(X_layer, y, groups_for_strat):
    """PCA->logistic, 5-fold CV balanced accuracy. y: int labels."""
    n = len(y)
    k = min(N_FOLDS, np.bincount(y).min()) if len(np.unique(y)) > 1 else 0
    if k < 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X_layer.astype(np.float32))
    d = min(PROBE_PCA_DIM, Xs.shape[1], n - 1)
    Xp = PCA(n_components=d, random_state=SEED).fit_transform(Xs)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    try:
        pred = cross_val_predict(clf, Xp, y, cv=skf)
    except Exception:
        return float("nan")
    return float(balanced_accuracy_score(y, pred))


def layer_probe_ordinal(X_layer, y):
    """PCA->ridge regression, CV predictions, return Spearman(pred, y)."""
    n = len(y)
    Xs = StandardScaler().fit_transform(X_layer.astype(np.float32))
    d = min(PROBE_PCA_DIM, Xs.shape[1], n - 1)
    Xp = PCA(n_components=d, random_state=SEED).fit_transform(Xs)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    try:
        pred = cross_val_predict(Ridge(alpha=10.0), Xp, y.astype(float), cv=kf)
    except Exception:
        return float("nan")
    return spearman(pred, y)


# ────────────────────────────── load ─────────────────────────────────────
def load_pilot(shifts_dir: Path, max_n: int | None):
    files = sorted(shifts_dir.glob("*.pt"))
    if max_n:
        files = files[:max_n]
    n = len(files)
    print(f"[load] {n} .pt files")

    DAF = np.zeros((n, 36, 12288), dtype=np.float16)
    DAT = np.zeros((n, 36, 12288), dtype=np.float16)
    meta = []
    keep = 0
    for i, pf in enumerate(files):
        try:
            d = torch.load(pf, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [skip] {pf.name}: {e}")
            continue
        DAF[keep] = d["dA_faithful"].to(torch.float16).numpy()
        DAT[keep] = d["dA_thinking"].to(torch.float16).numpy()
        meta.append({
            "problem_id": d.get("problem_id"),
            "subject": d.get("subject"),
            "level": int(d.get("level", -1)),
            "is_correct": d.get("is_correct", None),
            "finish_reason": d.get("finish_reason"),
            "truncated": bool(d.get("truncated", False)),
            "think_valid": bool(d.get("think_valid", False)),
            "gen_len": int(d.get("gen_len", -1)),
            "r1_cot_token_count": float(d.get("r1_cot_token_count", -1) or -1),
            "think_span_len": int(d.get("think_span_len", -1)),
        })
        keep += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{n}")
    DAF = DAF[:keep]; DAT = DAT[:keep]
    md = pd.DataFrame(meta)
    md["unit"] = md["subject"].astype(str) + "|L" + md["level"].astype(str)
    print(f"[load] kept {keep} (faithful/thinking fp16 arrays ready)")
    return DAF, DAT, md


# ────────────────────────────── analyses ─────────────────────────────────
def per_layer_norms(DA):
    # L2 over hidden dim -> (N, 36)
    return np.linalg.norm(DA.astype(np.float32), axis=2)


def analyze_magnitude(L, md, name, lines):
    lines.append(f"\n## [{name}] magnitude (per-layer |dA| L2)")
    mag = L.mean(axis=1)  # layer-avg
    lev = md["level"].values
    cot = md["r1_cot_token_count"].values
    gl = md["gen_len"].values
    lines.append(f"- layer-avg |dA|: mean={mag.mean():.2f} std={mag.std():.2f}")
    lines.append(f"- ρ(|dA|, level)        = {spearman(mag, lev):+.3f}")
    lines.append(f"- ρ(|dA|, r1_cot_tokens)= {spearman(mag, cot):+.3f}")
    lines.append(f"- ρ(|dA|, gen_len)      = {spearman(mag, gl):+.3f}")
    lines.append(f"- partial ρ(|dA|, level | gen_len) = "
                 f"{partial_spearman(mag, lev, gl):+.3f}")
    # correct vs incorrect
    ic = md["is_correct"]
    m = ic.notna().values
    if m.sum() > 8:
        cb = ic[m].astype(bool).values
        fm = mag[m]
        if cb.any() and (~cb).any():
            lines.append(f"- |dA| correct={fm[cb].mean():.2f} (n={cb.sum()}) "
                         f"vs incorrect={fm[~cb].mean():.2f} (n={(~cb).sum()})")
    # per-level
    dfm = pd.DataFrame({"level": lev, "mag": mag})
    lines.append("\n  |dA| by level:")
    lines.append("  " + dfm.groupby("level")["mag"].mean().round(2).to_string().replace("\n", "\n  "))
    # which layer best correlates with level
    rs = [spearman(L[:, l], lev) for l in range(L.shape[1])]
    bl = int(np.nanargmax(np.abs(rs)))
    lines.append(f"\n- best layer for ρ(|dA_layer|, level): L{bl} ρ={rs[bl]:+.3f}")
    return mag


def analyze_global_pca(L, md, name, lines):
    lines.append(f"\n## [{name}] global PCA of per-layer |dA| profile (36-d)")
    X = L - L.mean(0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        ev = s ** 2
        vr = ev / ev.sum()
        lines.append(f"- PC var-explained: " +
                     " ".join(f"PC{i+1}={vr[i]:.3f}" for i in range(min(5, len(vr)))))
        pc1 = u[:, 0] * s[0]
        lines.append(f"- ρ(PC1, level)  = {spearman(pc1, md['level'].values):+.3f}")
        lines.append(f"- ρ(PC1, gen_len)= {spearman(pc1, md['gen_len'].values):+.3f}")
        lines.append("  (PC1이 gen_len과만 강상관 & level과 약하면 = 길이 교란 신호)")
    except Exception as e:
        lines.append(f"- PCA failed: {e}")


def analyze_probes(DA, md, name, lines, subset_mask=None):
    lines.append(f"\n## [{name}] layerwise linear probe (full {DA.shape[2]}-d → PCA{PROBE_PCA_DIM})")
    idx = np.arange(len(md)) if subset_mask is None else np.where(subset_mask)[0]
    sub = md.iloc[idx].reset_index(drop=True)
    lines.append(f"- probe N = {len(idx)}")

    # is_correct
    ic = sub["is_correct"]
    cm = ic.notna().values
    acc_curve = []
    if cm.sum() > 30 and ic[cm].astype(bool).nunique() == 2:
        y = ic[cm].astype(int).values
        ridx = idx[cm]
        base = max(np.bincount(y)) / len(y)
        for l in range(DA.shape[1]):
            Xl = DA[ridx, l, :].astype(np.float32)
            acc_curve.append(layer_probe_classif(Xl, y, y))
        acc_curve = np.array(acc_curve)
        bl = int(np.nanargmax(acc_curve))
        lines.append(f"\n### is_correct probe (chance bal-acc=0.50; majority={base:.2f})")
        lines.append(f"- best layer L{bl}: bal-acc={acc_curve[bl]:.3f}")
        lines.append(f"- mean over layers: {np.nanmean(acc_curve):.3f}")
        lines.append("- curve(every4): " +
                     " ".join(f"L{l}:{acc_curve[l]:.2f}" for l in range(0, DA.shape[1], 4)))
    else:
        lines.append("\n### is_correct probe: skipped (insufficient class balance)")

    # level (ordinal)
    lev = sub["level"].values
    lev_curve = []
    for l in range(DA.shape[1]):
        lev_curve.append(layer_probe_ordinal(DA[idx, l, :], lev))
    lev_curve = np.array(lev_curve)
    bl = int(np.nanargmax(np.abs(lev_curve)))
    lines.append(f"\n### level probe (Spearman of CV-pred vs level; chance≈0)")
    lines.append(f"- best layer L{bl}: ρ={lev_curve[bl]:+.3f}")
    lines.append(f"- mean |ρ| over layers: {np.nanmean(np.abs(lev_curve)):.3f}")
    lines.append("- curve(every4): " +
                 " ".join(f"L{l}:{lev_curve[l]:+.2f}" for l in range(0, DA.shape[1], 4)))

    # subject (multiclass)
    subj = sub["subject"].astype("category")
    ys = subj.cat.codes.values
    vc = pd.Series(ys).value_counts()
    keep_cls = vc[vc >= N_FOLDS].index
    sm = np.isin(ys, keep_cls)
    subj_curve = []
    if sm.sum() > 50 and len(keep_cls) > 1:
        ys2 = ys[sm]; sidx = idx[sm]
        base = vc.max() / vc.sum()
        for l in range(DA.shape[1]):
            subj_curve.append(layer_probe_classif(DA[sidx, l, :], ys2, ys2))
        subj_curve = np.array(subj_curve)
        bl = int(np.nanargmax(subj_curve))
        lines.append(f"\n### subject probe ({len(keep_cls)} classes; majority={base:.2f})")
        lines.append(f"- best layer L{bl}: bal-acc={subj_curve[bl]:.3f}")
        lines.append(f"- mean over layers: {np.nanmean(subj_curve):.3f}")
        lines.append("- curve(every4): " +
                     " ".join(f"L{l}:{subj_curve[l]:.2f}" for l in range(0, DA.shape[1], 4)))
    else:
        lines.append("\n### subject probe: skipped")
        subj_curve = np.array([])

    return {"is_correct": np.array(acc_curve) if len(acc_curve) else np.array([]),
            "level": lev_curve, "subject": subj_curve}


def analyze_units(DA, md, name, lines):
    """SELECTED UNIT view: within/between dispersion + correct-incorrect contrast."""
    lines.append(f"\n## [{name}] SELECTED UNIT view (unit=subject×level)")
    units = md["unit"].values
    uniq = pd.Series(units).value_counts()
    big = uniq[uniq >= 5].index.tolist()
    lines.append(f"- units total={len(uniq)}, units with n≥5={len(big)}")

    # representative layers across depth
    layers = list(range(0, DA.shape[1], 6)) + [DA.shape[1] - 1]
    layers = sorted(set(layers))

    lines.append("\n### within- vs between-unit dispersion (cosine) per layer")
    lines.append("  layer | within_mean_cos | between_mean_cos | sep(=bet-with, lower=more separable... ) ")
    for l in layers:
        # unit centroids
        cents = {}
        within = []
        for u in big:
            V = DA[units == u, l, :].astype(np.float32)
            c = V.mean(0)
            cents[u] = c
            # within cosine to centroid
            Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
            cn = c / (np.linalg.norm(c) + 1e-8)
            within.append((Vn @ cn).mean())
        C = np.stack([cents[u] for u in big])
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        cos_bet = Cn @ Cn.T
        bet = cos_bet[np.triu_indices(len(big), 1)].mean()
        wmean = float(np.mean(within))
        lines.append(f"  L{l:<3d} | {wmean:+.3f}          | {bet:+.3f}          | {wmean-bet:+.3f}")
    lines.append("  (within≫between 이면 unit이 응집/분리됨. 둘 다 ~1 이면 공통 성분 지배=신호 약함)")

    # correct - incorrect contrast vector per unit, cross-unit consistency at mid layer
    mid = DA.shape[1] // 2
    lines.append(f"\n### correct−incorrect contrast consistency @ L{mid}")
    contrasts = []
    for u in big:
        sel = (units == u)
        ics = md.loc[sel, "is_correct"]
        m = ics.notna().values
        if m.sum() < 6:
            continue
        cb = ics[m].astype(bool).values
        if cb.sum() < 2 or (~cb).sum() < 2:
            continue
        V = DA[sel, mid, :].astype(np.float32)[m]
        vec = V[cb].mean(0) - V[~cb].mean(0)
        contrasts.append(vec / (np.linalg.norm(vec) + 1e-8))
    if len(contrasts) >= 2:
        Cc = np.stack(contrasts)
        cosm = Cc @ Cc.T
        off = cosm[np.triu_indices(len(contrasts), 1)]
        lines.append(f"- usable units={len(contrasts)}; mean pairwise cos of "
                     f"(correct−incorrect) direction = {off.mean():+.3f} "
                     f"(std={off.std():.3f})")
        lines.append("  (>0.2 이면 난이도 통제 후에도 일관된 'competence' 축 존재 시사)")
    else:
        lines.append(f"- usable units={len(contrasts)} (<2) → contrast consistency 평가 불가")


def verdict(lines, probe_results):
    lines.append("\n## SIGNAL VERDICT (heuristic)")
    msgs = []
    for view, pr in probe_results.items():
        lev = pr.get("level", np.array([]))
        ic = pr.get("is_correct", np.array([]))
        if lev.size:
            best = np.nanmax(np.abs(lev))
            msgs.append(f"- [{view}] level probe best |ρ|={best:.3f} "
                        f"({'STRONG' if best>0.4 else 'moderate' if best>0.2 else 'WEAK'})")
        if ic.size:
            best = np.nanmax(ic)
            msgs.append(f"- [{view}] is_correct probe best bal-acc={best:.3f} "
                        f"({'STRONG' if best>0.65 else 'moderate' if best>0.57 else 'WEAK(~chance)'})")
    lines += msgs
    lines.append("\n→ 모든 probe가 WEAK/~chance 이고 PC1이 길이 교란이면: 방향 전환 권고 "
                 "(token 위치 재선정: A_t1_think / A_tK_think / A_prompt_last 대비, 혹은 "
                 "thinking-vs-nonthinking 추출).")


# ────────────────────────────── main ─────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="pilot_partial")
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    DAF, DAT, md = load_pilot(Path(args.shifts_dir), args.max_n)

    # save compact features
    LF = per_layer_norms(DAF)
    LT = per_layer_norms(DAT)
    feat = md.copy()
    feat["faithful_mag"] = LF.mean(1)
    feat["thinking_mag"] = LT.mean(1)
    feat.to_parquet(out_dir / f"unit_features_{args.tag}.parquet", index=False)

    lines = [f"# ΔA Unit Analysis — {args.tag}", ""]
    lines.append(f"- N loaded: **{len(md)}**")
    lines.append(f"- subject×level units: {md['unit'].nunique()}")
    lines.append(f"- finish_reason: {md['finish_reason'].value_counts().to_dict()}")
    lines.append(f"- truncated: {int(md['truncated'].sum())}/{len(md)}")
    ic = md['is_correct'].dropna()
    if len(ic):
        lines.append(f"- correct rate: {ic.astype(bool).mean():.3f} (n={len(ic)})")
    lines.append("\n### counts by (subject, level)")
    ct = pd.crosstab(md["subject"], md["level"])
    lines.append("```\n" + ct.to_string() + "\n```")
    lines.append("\n### truncation% by level (confound watch)")
    tr = md.groupby("level")["truncated"].mean().round(2)
    lines.append("```\n" + tr.to_string() + "\n```")

    # clean subset for faithful (finish=stop)
    clean = (md["finish_reason"] == "stop").values
    lines.append(f"\n- faithful clean subset (finish=stop): {clean.sum()}/{len(md)}")

    probe_results = {}
    # FAITHFUL
    analyze_magnitude(LF, md, "FAITHFUL", lines)
    analyze_global_pca(LF, md, "FAITHFUL", lines)
    probe_results["FAITHFUL(clean)"] = analyze_probes(DAF, md, "FAITHFUL(clean)", lines,
                                                      subset_mask=clean)
    analyze_units(DAF, md, "FAITHFUL", lines)
    # THINKING
    analyze_magnitude(LT, md, "THINKING", lines)
    analyze_global_pca(LT, md, "THINKING", lines)
    probe_results["THINKING(all)"] = analyze_probes(DAT, md, "THINKING(all)", lines)
    analyze_units(DAT, md, "THINKING", lines)

    verdict(lines, probe_results)

    rep = out_dir / f"REPORT_unit_{args.tag}.md"
    rep.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    # save probe curves
    np.savez(out_dir / f"probe_curves_{args.tag}.npz",
             **{f"{v}__{k}": arr for v, pr in probe_results.items()
                for k, arr in pr.items() if isinstance(arr, np.ndarray)})
    print(f"[OK] wrote {rep}")
    print("\n".join(str(x) for x in lines))


if __name__ == "__main__":
    main()
