#!/usr/bin/env python3
"""
curriculum_materials.py — Curriculum MATERIAL analysis (Task B, pre-stage).
==========================================================================
Goal: produce the *materials* needed to later define a unit-joint-clustering
curriculum (stage count / boundaries / schedule are decided by the USER after
reviewing this report — NOT here).

INPUT  : pooled (pilot1 + pilot2) centered ΔA, THINKING primary / FAITHFUL aux.
         GPT labels (subject, level), 1-shot is_correct (read from .pt),
         unit = subject|L{level}.
OUTPUT : analysis/REPORT_curriculum_materials.md  (+ currmat_*.npz, PNGs)
         Does NOT modify any existing artifact. CPU only.

TASKS
  1. Difficulty axis (2 candidates), fair comparison on the pilot2 TEST sample:
       (a) unsupervised PCA top-3 PCs (fit on pilot1 with μ_train centering,
           projected onto pilot2; pooled-PC reported as side sanity);
       (b) supervised level RIDGE direction via DUAL form
           w = Xᵀ (XXᵀ + αI)⁻¹ y   (avoids 442k×442k normal equations),
           α chosen by pilot1-internal CV, pilot2 = pure out-of-sample.
     Each candidate scored vs GPT level / is_correct / gen_len (Spearman ρ) on
     pilot2 test → pick the most monotone axis as the difficulty score
     (report all; PC1 first, escalate to PC2/PC3 if weak).
  2. Two-axis (subject / level) decomposition of unit-centroid cosine:
       same-level/diff-subject, same-subject/diff-level (per Δlevel), both-diff.
       + sample-level conditional separability (block-restricted perm, N=1000):
         within-level/between-subject gap ; within-subject/between-level gap.
  3. Unit joint clustering (Ward, cosine distance) — TWO views:
       (i) 36 layer-averaged centroid ; (ii) mid-layer (L11-15) subset centroid.
       K sweep 4..8 + silhouette + dendrogram.
  4. Subject-branching diagnostic: cluster×{level,subject} cross-tab, per-cluster
     subject/level entropy, fraction of same-level units landing in different
     clusters; compare (i) vs (ii).
  5. Stage MATERIALS: per-cluster difficulty score (axis projection mean) →
     candidate difficulty ordering (NOT final stages).

Sparse units (n < MIN_N, e.g. L8) are EXCLUDED from clustering (or noted as
nearest-cluster absorption) — no standalone conclusions.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import similarity_analysis as sa  # validated metric functions + MIN_N + rng

ANALYSIS = Path(__file__).resolve().parent
ACT = ANALYSIS.parent
PILOT1_DIR = ACT / "outputs" / "pilot" / "shifts"
PILOT2_DIR = ACT / "outputs" / "pilot2" / "shifts"
OUT_MD = ANALYSIS / "REPORT_curriculum_materials.md"
TAG = "currmat"
LAYERS = 36
MID_LAYERS = list(range(11, 16))  # L11..L15 subject-channel window
N_PERM_COND = 1000
SEED = 42


# ───────────────────────────── load (with is_correct) ─────────────────────
def load_dir(shifts_dir: Path, pilot_name: str, max_n, want_faithful: bool):
    files = sorted(shifts_dir.glob("*.pt"))
    if max_n:
        files = files[:max_n]
    n = len(files)
    print(f"[load:{pilot_name}] {n} .pt files", flush=True)
    DAT = np.zeros((n, LAYERS, 12288), dtype=np.float16)
    DAF = np.zeros((n, LAYERS, 12288), dtype=np.float16) if want_faithful else None
    meta = []
    keep = 0
    for i, pf in enumerate(files):
        try:
            d = torch.load(pf, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [skip] {pf.name}: {e}", flush=True); continue
        t = d["dA_thinking"]
        if not torch.isfinite(t).all():
            print(f"  [skip non-finite] {pf.name}", flush=True); continue
        DAT[keep] = t.to(torch.float16).numpy()
        if want_faithful:
            DAF[keep] = d["dA_faithful"].to(torch.float16).numpy()
        ic = d.get("is_correct", None)
        meta.append({
            "problem_id": d.get("problem_id"),
            "subject": str(d.get("subject")),
            "level": int(d.get("level", -1)),
            "gen_len": int(d.get("gen_len", -1)),
            "is_correct": (bool(ic) if isinstance(ic, (bool, np.bool_)) else
                           (ic if ic is None else None)),
            "pilot": pilot_name,
        })
        keep += 1
        if (i + 1) % 400 == 0:
            print(f"  ...{i+1}/{n} (kept {keep})", flush=True)
    DAT = DAT[:keep]
    if want_faithful:
        DAF = DAF[:keep]
    md = pd.DataFrame(meta)
    print(f"[load:{pilot_name}] kept {keep}", flush=True)
    return DAT, DAF, md


def load_pooled(max_n, want_faithful):
    t0 = time.time()
    DAT1, DAF1, md1 = load_dir(PILOT1_DIR, "pilot1", max_n, want_faithful)
    DAT2, DAF2, md2 = load_dir(PILOT2_DIR, "pilot2", max_n, want_faithful)
    DAT = np.concatenate([DAT1, DAT2], axis=0)
    DAF = (np.concatenate([DAF1, DAF2], axis=0) if want_faithful else None)
    md = pd.concat([md1, md2], ignore_index=True)
    md["unit"] = md["subject"] + "|L" + md["level"].astype(str)
    n1 = len(md1)
    print(f"[load] pooled N={len(md)} (pilot1={n1}, pilot2={len(md2)}) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return DAT, DAF, md, n1


# ───────────────────────────── small stats ────────────────────────────────
def spearman(a, b):
    return sa.spearman(a, b)


def flatten_f32(DA):
    """(N,36,D) f16 -> (N, 36*D) f32 view-ish copy."""
    n = DA.shape[0]
    return DA.reshape(n, -1).astype(np.float32)


def entropy(counts):
    p = np.asarray(counts, float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


# ───────────────────────────── TASK 1: difficulty axis ────────────────────
def task1_axis(DAT, md, n1, lines):
    lines.append("\n## TASK 1 — Difficulty axis (compared on pilot2 TEST)")
    tr = (md["pilot"] == "pilot1").to_numpy()
    te = (md["pilot"] == "pilot2").to_numpy()
    Xtr = flatten_f32(DAT[tr])
    mu_train = Xtr.mean(axis=0, keepdims=True)        # μ_train (pilot1)
    Xtr_c = Xtr - mu_train
    Xte_c = flatten_f32(DAT[te]) - mu_train
    lev_tr = md.loc[tr, "level"].to_numpy(float)
    lev_te = md.loc[te, "level"].to_numpy(float)
    gl_te = md.loc[te, "gen_len"].to_numpy(float)
    ic_te = md.loc[te, "is_correct"]
    ic_te_mask = ic_te.notna().to_numpy()
    ic_te_val = ic_te.where(ic_te.notna(), np.nan).astype(float).to_numpy()

    # ---- (a) unsupervised PCA top-3 (fit pilot1, project pilot2) ----
    from sklearn.decomposition import PCA
    t0 = time.time()
    pca = PCA(n_components=3, svd_solver="randomized", random_state=SEED)
    pca.fit(Xtr_c)
    proj_te = pca.transform(Xte_c)                    # (n_te, 3)
    lines.append(f"- PCA fit on pilot1 (n={tr.sum()}), projected pilot2 "
                 f"(n={te.sum()}); EVR={np.round(pca.explained_variance_ratio_,4).tolist()} "
                 f"({time.time()-t0:.0f}s)")
    pc_scores = {}
    lines.append("\n### unsupervised PC candidates (ρ on pilot2 test)")
    lines.append("| PC | EVR | ρ(level) | ρ(is_correct) | ρ(gen_len) |")
    lines.append("|----|-----|----------|---------------|------------|")
    for k in range(3):
        s = proj_te[:, k]
        # orient so that higher score => higher level (difficulty)
        if spearman(s, lev_te) < 0:
            s = -s
        pc_scores[f"PC{k+1}"] = s
        r_lev = spearman(s, lev_te)
        r_ic = (spearman(s[ic_te_mask], ic_te_val[ic_te_mask])
                if ic_te_mask.sum() >= 4 else float("nan"))
        r_gl = spearman(s, gl_te)
        lines.append(f"| PC{k+1} | {pca.explained_variance_ratio_[k]:.4f} | "
                     f"{r_lev:+.3f} | {r_ic:+.3f} | {r_gl:+.3f} |")

    # ---- (b) supervised level ridge, DUAL form ----
    # w = Xᵀ (K + αI)⁻¹ y, K = Xtr_c Xtr_cᵀ ; score_te = Xte_c w = Kte α_dual
    t0 = time.time()
    yc = lev_tr - lev_tr.mean()
    K = Xtr_c @ Xtr_c.T                                # (n_tr, n_tr)
    Kte = Xte_c @ Xtr_c.T                              # (n_te, n_tr)
    ntr = K.shape[0]
    # alpha CV inside pilot1 (5-fold), pick best ρ(level) on held-out
    rng = np.random.default_rng(SEED)
    folds = rng.integers(0, 5, size=ntr)
    alphas = [1e1, 1e2, 1e3, 1e4, 1e5]
    best_a, best_score = alphas[0], -2.0
    for a in alphas:
        preds = np.zeros(ntr)
        for f in range(5):
            temask = folds == f; trmask = ~temask
            Ktt = K[np.ix_(trmask, trmask)]
            alpha_dual = np.linalg.solve(Ktt + a * np.eye(trmask.sum()),
                                         yc[trmask])
            preds[temask] = K[np.ix_(temask, trmask)] @ alpha_dual
        sc = spearman(preds, lev_tr)
        if sc > best_score:
            best_score, best_a = sc, a
    alpha_dual = np.linalg.solve(K + best_a * np.eye(ntr), yc)
    ridge_te = Kte @ alpha_dual
    if spearman(ridge_te, lev_te) < 0:
        ridge_te = -ridge_te
    r_lev = spearman(ridge_te, lev_te)
    r_ic = (spearman(ridge_te[ic_te_mask], ic_te_val[ic_te_mask])
            if ic_te_mask.sum() >= 4 else float("nan"))
    r_gl = spearman(ridge_te, gl_te)
    lines.append(f"\n### supervised ridge (dual; α={best_a:g} via pilot1 5-fold CV, "
                 f"cv ρ={best_score:+.3f}; {time.time()-t0:.0f}s)")
    lines.append("| axis | ρ(level) | ρ(is_correct) | ρ(gen_len) |")
    lines.append("|------|----------|---------------|------------|")
    lines.append(f"| ridge_level | {r_lev:+.3f} | {r_ic:+.3f} | {r_gl:+.3f} |")

    # ---- pick most monotone (|ρ(level)| primary) ----
    cands = {**{k: spearman(v, lev_te) for k, v in pc_scores.items()},
             "ridge_level": r_lev}
    best_axis = max(cands, key=lambda k: abs(cands[k]))
    lines.append(f"\n**ADOPTED difficulty axis (pilot2 test, |ρ(level)| max): "
                 f"`{best_axis}` (ρ(level)={cands[best_axis]:+.3f}).**")
    lines.append("- ridge = honest out-of-sample (pilot1→pilot2); PCs centered "
                 "by μ_train(pilot1) then projected to pilot2 for the SAME-sample "
                 "fair comparison. PC1 considered first; PC2/PC3 reported in case "
                 "PC1 tracks a non-difficulty common component (see ρ(gen_len)).")

    # full-pooled PC1 as side sanity (different sample -> sanity only)
    Xall = flatten_f32(DAT)
    Xall_c = Xall - Xall.mean(axis=0, keepdims=True)
    p2 = PCA(n_components=1, svd_solver="randomized", random_state=SEED).fit_transform(Xall_c)[:, 0]
    if spearman(p2, md["level"].to_numpy(float)) < 0:
        p2 = -p2
    lines.append(f"- [side sanity] pooled-all PC1 ρ(level)="
                 f"{spearman(p2, md['level'].to_numpy(float)):+.3f}, "
                 f"ρ(gen_len)={spearman(p2, md['gen_len'].to_numpy(float)):+.3f} "
                 "(different sample than the comparison above; sanity only).")

    # return adopted axis as a FULL-pooled score (for task5 cluster ordering):
    # use ridge direction projected on ALL samples (consistent axis) if adopted,
    # else the chosen pooled PC. Keep it simple: project all on adopted.
    if best_axis == "ridge_level":
        Kall = (Xall - mu_train) @ Xtr_c.T
        score_all = Kall @ alpha_dual
        if spearman(score_all, md["level"].to_numpy(float)) < 0:
            score_all = -score_all
    else:
        kk = int(best_axis[2]) - 1
        score_all = pca.transform(Xall - mu_train)[:, kk]
        if spearman(score_all, md["level"].to_numpy(float)) < 0:
            score_all = -score_all
    del Xtr, Xtr_c, Xte_c, Xall, Xall_c, K, Kte
    return best_axis, np.asarray(score_all, float)


# ───────────────────────────── TASK 2: two-axis decomposition ─────────────
def cos_within_between(B_sub, labels):
    """B_sub: (m,36,D) per-layer-normalized members. labels: (m,).
    within = mean over groups of mean member layeravg-cos to own centroid.
    between = mean over group-pairs of layeravg-cos of centroids."""
    order = [g for g in pd.unique(labels)]
    order = [g for g in order if (labels == g).sum() >= 2]
    if len(order) < 2:
        return None
    cents_n = {}
    withins = []
    for g in order:
        idx = np.where(labels == g)[0]
        c = B_sub[idx].mean(axis=0)                     # (36,D)
        cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
        cents_n[g] = cn
        cos_ml = np.einsum("mld,ld->ml", B_sub[idx], cn)
        withins.append(float(cos_ml.mean()))
    within = float(np.mean(withins))
    bet = []
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            bet.append(float((cents_n[order[a]] * cents_n[order[b]]).sum(axis=1).mean()))
    between = float(np.mean(bet))
    return within, between, within - between


def conditional_gap(B, md, group_col, block_col, n_perm=N_PERM_COND):
    """Block-restricted: per block, compute group within/between gap; weighted
    mean over blocks. Permutation shuffles group labels WITHIN each block."""
    blocks = sorted(md[block_col].unique(), key=lambda x: (int(x) if str(x).lstrip('-').isdigit() else str(x)))

    idx_by_block = {b: md.index[md[block_col] == b].to_numpy() for b in blocks}

    def weighted_gap(perm_labels=None):
        gaps, ws = [], []
        for b in blocks:
            idx = idx_by_block[b]
            if len(idx) < 4:
                continue
            lab = (perm_labels[idx] if perm_labels is not None
                   else md.loc[idx, group_col].to_numpy())
            r = cos_within_between(B[idx], lab)
            if r is None:
                continue
            gaps.append(r[2]); ws.append(len(idx))
        if not gaps:
            return float("nan")
        return float(np.average(gaps, weights=ws))

    obs = weighted_gap()
    base = md[group_col].to_numpy().copy()
    ge = 0; done = 0
    step = max(1, n_perm // 5)
    for it in range(n_perm):
        perm = base.copy()
        for b in blocks:
            idx = idx_by_block[b]
            perm[idx] = sa.rng.permutation(perm[idx])
        g = weighted_gap(perm)
        if np.isfinite(g):
            done += 1
            if g >= obs:
                ge += 1
        if (it + 1) % step == 0:
            print(f"    cond[{group_col}|{block_col}] {it+1}/{n_perm} ge={ge}", flush=True)
    p = (ge + 1) / (done + 1)
    return obs, p, done



def task2_decomposition(DAT_c, md, lines, saved, cond_perm=N_PERM_COND):

    lines.append("\n## TASK 2 — Two-axis (subject / level) decomposition")
    # unit centroids (n>=MIN_N) on centered THINKING
    vc = md["unit"].value_counts()
    units = sorted([u for u in vc.index if vc[u] >= sa.MIN_N])
    idxg = {u: md.index[md["unit"] == u].to_numpy() for u in units}
    cents = sa.centroids(DAT_c, idxg)                   # raw centroids (36,D)
    S = sa.sim_matrix(cents, units)                     # layer-avg cosine
    saved["unit_order"] = np.array(units)
    saved["unit_cos"] = S
    usub = [u.split("|L")[0] for u in units]
    ulev = [int(u.split("|L")[1]) for u in units]
    lines.append(f"- units (n≥{sa.MIN_N}): {len(units)}")

    # pair classification
    sl_ds, ss_dl_by_delta, both = [], {}, []
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            c = S[a, b]
            same_s = usub[a] == usub[b]
            same_l = ulev[a] == ulev[b]
            if same_l and not same_s:
                sl_ds.append(c)
            elif same_s and not same_l:
                ss_dl_by_delta.setdefault(abs(ulev[a]-ulev[b]), []).append(c)
            elif not same_s and not same_l:
                both.append(c)
    lines.append("\n### unit-centroid cosine by pair type (lower = more separated)")
    lines.append(f"- same-level / diff-subject : mean cos = {np.mean(sl_ds):+.3f} "
                 f"(n_pairs={len(sl_ds)})  ← subject가 같은 난이도 안에서 가르는 정도")
    lines.append("- same-subject / diff-level (by Δlevel):")
    deltas = sorted(ss_dl_by_delta)
    dvals = []
    for d in deltas:
        m = float(np.mean(ss_dl_by_delta[d]))
        dvals.append(m)
        lines.append(f"    Δ={d}: mean cos = {m:+.3f} (n={len(ss_dl_by_delta[d])})")
    if len(deltas) >= 3:
        lines.append(f"    ordinality ρ(cos, -Δ) = {spearman(dvals, [-d for d in deltas]):+.3f} "
                     "(positive => 가까운 level이 더 유사)")
    lines.append(f"- both-diff (baseline)      : mean cos = {np.mean(both):+.3f} "
                 f"(n_pairs={len(both)})")

    # conditional separability (sample-level, block-restricted perm)
    lines.append("\n### conditional separability (sample-level, block-restricted "
                 f"perm N={cond_perm})")
    B = sa.normalize_members(DAT_c)
    # restrict to levels with enough subjects / subjects with enough levels handled inside
    obs1, p1, d1 = conditional_gap(B, md, "subject", "level", n_perm=cond_perm)
    lines.append(f"- within-level / between-SUBJECT gap = {obs1:+.4f} "
                 f"(p={p1:.4f}, blocks_used={d1}) → 같은 level 안에서 subject 분리도")
    obs2, p2, d2 = conditional_gap(B, md, "level", "subject", n_perm=cond_perm)

    lines.append(f"- within-subject / between-LEVEL gap = {obs2:+.4f} "
                 f"(p={p2:.4f}, blocks_used={d2}) → 같은 subject 안에서 level 분리도")
    del B
    return units, usub, ulev, cents, S


# ───────────────────────────── TASK 3+4: clustering + branching ───────────
def cluster_view(name, vecs, units, usub, ulev, score_units, lines, saved, out_dir):
    """vecs: (U, F) unit feature matrix; cosine Ward clustering + K sweep."""
    from sklearn.metrics import silhouette_score
    # cosine distance
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    D = 1.0 - (vn @ vn.T)
    D = np.clip((D + D.T) / 2, 0, None)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="ward")
    lines.append(f"\n### [{name}] Ward clustering (U={len(units)} units, "
                 f"feat dim={vecs.shape[1]})")
    lines.append("| K | silhouette(cosine) |")
    lines.append("|---|--------------------|")
    best_k, best_sil = None, -2
    sil_by_k = {}
    for K in range(4, 9):
        lab = fcluster(Z, K, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        try:
            sil = silhouette_score(D, lab, metric="precomputed")
        except Exception:
            sil = float("nan")
        sil_by_k[K] = (lab, sil)
        lines.append(f"| {K} | {sil:.3f} |")
        if np.isfinite(sil) and sil > best_sil:
            best_sil, best_k = sil, K
    lines.append(f"- best K by silhouette = **{best_k}** (sil={best_sil:.3f})")

    # dendrogram
    fig, ax = plt.subplots(figsize=(max(8, len(units) * 0.25), 5))
    dendrogram(Z, labels=units, leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title(f"{name} unit dendrogram (cosine/Ward)")
    fig.tight_layout()
    png = out_dir / f"dendro_{name}_{TAG}.png"
    fig.savefig(png, dpi=120); plt.close(fig)
    lines.append(f"- dendrogram: {png.name}")

    # describe best-K clustering + subject-branching diagnostics (TASK 4)
    lab = sil_by_k[best_k][0]
    df = pd.DataFrame({"unit": units, "subject": usub, "level": ulev,
                       "cluster": lab, "score": score_units})
    saved[f"cluster_{name}_labels"] = lab
    lines.append(f"\n#### [{name}] cluster composition (K={best_k})")
    for c in sorted(set(lab)):
        sub = df[df.cluster == c]
        levs = sorted(sub.level.unique())
        subj_counts = sub.subject.value_counts().to_dict()
        lines.append(f"- C{c} (n_units={len(sub)}): levels={levs} | "
                     f"subjects={subj_counts} | "
                     f"subj_H={entropy(list(subj_counts.values())):.2f} "
                     f"level_H={entropy(sub.level.value_counts().values):.2f} | "
                     f"mean difficulty score={sub.score.mean():+.3f}")

    # cross-tabs
    lines.append(f"\n#### [{name}] cluster × level")
    lines.append("```\n" + pd.crosstab(df.cluster, df.level).to_string() + "\n```")
    lines.append(f"#### [{name}] cluster × subject")
    lines.append("```\n" + pd.crosstab(df.cluster, df.subject).to_string() + "\n```")

    # subject-branching: fraction of same-level unit-pairs in DIFFERENT clusters
    same_lvl_pairs, split_pairs = 0, 0
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if ulev[i] == ulev[j]:
                same_lvl_pairs += 1
                if lab[i] != lab[j]:
                    split_pairs += 1
    frac = split_pairs / same_lvl_pairs if same_lvl_pairs else float("nan")
    lines.append(f"\n- **subject-branching index** = same-level unit-pairs in "
                 f"DIFFERENT clusters = {split_pairs}/{same_lvl_pairs} = {frac:.3f} "
                 "(높을수록 같은 난이도가 subject별로 갈라짐 = joint 구조)")
    # mean per-cluster entropies
    subjH = np.mean([entropy(df[df.cluster == c].subject.value_counts().values)
                     for c in set(lab)])
    levH = np.mean([entropy(df[df.cluster == c].level.value_counts().values)
                    for c in set(lab)])
    lines.append(f"- mean per-cluster subject entropy = {subjH:.2f} bits, "
                 f"level entropy = {levH:.2f} bits "
                 "(subjectH 높고 levelH 낮으면 cluster가 주로 level띠)")
    return df, frac, best_k


# ───────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="cap per pilot (smoke)")
    ap.add_argument("--with-faithful", action="store_true")
    ap.add_argument("--cond-perm", type=int, default=200,
                    help="block-restricted permutation budget for Task2 conditional gaps")
    args = ap.parse_args()

    sa.rng = np.random.default_rng(SEED)

    DAT, DAF, md, n1 = load_pooled(args.max_n, args.with_faithful)
    N = len(md)
    lines = [f"# Curriculum Materials — pooled THINKING ΔA (tag={TAG})", ""]
    lines.append(f"- pooled N = **{N}** (pilot1={n1}, pilot2={N-n1}); "
                 f"subjects={md['subject'].nunique()}, "
                 f"levels={sorted(md['level'].unique().tolist())}")
    # is_correct availability (CONFIRM, not "maybe")
    ic = md["is_correct"]
    n_ic1 = md[(md.pilot == 'pilot1')]["is_correct"].notna().sum()
    n_ic2 = md[(md.pilot == 'pilot2')]["is_correct"].notna().sum()
    lines.append(f"- is_correct non-null: pilot1={n_ic1}, pilot2={n_ic2}, "
                 f"total={ic.notna().sum()} (axis comparison uses pilot2 test).")
    if ic.notna().sum():
        lines.append(f"- overall 1-shot correct rate (non-null) = "
                     f"{ic.dropna().astype(float).mean():.3f}")
    lines.append("\n_Method: group-similarity / centering / perm = OUR diagnostic, "
                 "not the NAIT PCA-scoring. CPU only. THINKING primary._")

    saved = {}
    # TASK 1
    best_axis, score_all = task1_axis(DAT, md, n1, lines)
    saved["difficulty_score_all"] = score_all
    saved["adopted_axis"] = np.array([best_axis])

    # center THINKING by pooled mean for tasks 2-5
    mu = DAT.astype(np.float32).mean(axis=0, keepdims=True)
    DAT_c = DAT.astype(np.float32) - mu
    del DAT

    # TASK 2
    units, usub, ulev, cents, S = task2_decomposition(DAT_c, md, lines, saved,
                                                       cond_perm=args.cond_perm)


    # per-unit difficulty score (mean of sample scores in that unit)
    score_units = np.array([score_all[md.index[md["unit"] == u]].mean() for u in units])

    # TASK 3+4: view (i) 36 layer-avg centroid ; (ii) mid-layer subset
    lines.append("\n## TASK 3+4 — Unit joint clustering & subject branching")
    feat_full = np.stack([cents[u].mean(axis=0) for u in units])           # (U, D) layer-avg
    feat_mid = np.stack([cents[u][MID_LAYERS].reshape(-1) for u in units]) # (U, 5*D)
    out_dir = ANALYSIS
    df_i, frac_i, k_i = cluster_view("layeravg", feat_full, units, usub, ulev,
                                     score_units, lines, saved, out_dir)
    df_ii, frac_ii, k_ii = cluster_view(f"midL{MID_LAYERS[0]}-{MID_LAYERS[-1]}",
                                        feat_mid, units, usub, ulev,
                                        score_units, lines, saved, out_dir)

    lines.append("\n### subject-branching verdict")
    lines.append(f"- branching index: layeravg={frac_i:.3f} vs mid-layer={frac_ii:.3f}")
    stronger = "mid-layer" if frac_ii > frac_i else "layeravg"
    lines.append(f"- {stronger} view shows stronger same-level splitting. "
                 "판정 가이드: branching index가 충분히 크고 cluster×subject가 비대각이면 "
                 "joint(subject 분기) 구조 = novelty; 거의 level 띠(낮은 branching, "
                 "level별 정렬)이면 정직하게 'level-driven'으로 보고.")

    # TASK 5: cluster difficulty ordering (materials only)
    lines.append("\n## TASK 5 — Cluster difficulty ordering (MATERIALS, NOT final stages)")
    for nm, df in [("layeravg", df_i), (f"midL{MID_LAYERS[0]}-{MID_LAYERS[-1]}", df_ii)]:
        order = (df.groupby("cluster")["score"].mean().sort_values())
        lines.append(f"\n### [{nm}] candidate difficulty order (easy→hard by axis `{best_axis}`)")
        for rank, (c, sc) in enumerate(order.items(), 1):
            sub = df[df.cluster == c]
            lines.append(f"  {rank}. C{c}: score={sc:+.3f}, "
                         f"levels={sorted(sub.level.unique())}, "
                         f"n_units={len(sub)}")
    lines.append("\n**stage 개수·경계·schedule은 본 재료를 사용자 review 후 확정.**")

    # sparse units note
    vc = md["unit"].value_counts()
    sparse = sorted([u for u in vc.index if vc[u] < sa.MIN_N])
    lines.append(f"\n## Sparse units (n<{sa.MIN_N}, EXCLUDED from clustering)")
    lines.append(f"- {len(sparse)} units: " + ", ".join(f"{u}({vc[u]})" for u in sparse))
    lines.append("- 단독 결론 금지; 추후 nearest-cluster 흡수 대상으로만 표기.")

    np.savez(ANALYSIS / f"{TAG}_artifacts.npz", **saved)
    OUT_MD.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}", flush=True)
    print(f"[OK] wrote {ANALYSIS / f'{TAG}_artifacts.npz'}", flush=True)


if __name__ == "__main__":
    main()
