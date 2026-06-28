#!/usr/bin/env python3
"""
clusterderive.py — pooled THINKING ΔA, unit-centroid 클러스터 도출 & 검증
===============================================================================
배경. SUBJECT 우세 윈도우 W_SUBJ=[9,10,11,12,14] 와 LEVEL 우세 윈도우
W_LEV=[20,25,26,27,29,30,31] 가 서로 다른 레이어 대역에 분리되어 있다는 사실은
`bestlayer_resolved.py`/`REPORT_bestlayer_resolved_2026-06-22.md` 에서 확정됨.

이번 스크립트는 **새 통계 0줄**(헬퍼만 호출). cos 클립 절대 금지.
1) cross-level binding (cluster≠level 직접검정): unit-centroid 행렬에서
   A=same-subject/diff-level vs B=diff-subject/same-level 비교 + level-gap 분해.
2) unit-centroid Ward 클러스터(W_SUBJ, raw & level-residual), k∈{3,4,5},
   cluster×level / cluster×subject 교차표, silhouette(cluster/subject/level),
   pilot1 vs pilot2 ARI, raw vs residual ARI.
3) ‖ΔA‖ depth profile + length redundancy (ρ(z,gen_len), partial ρ(z,level|gen_len)).

OUTPUT:
  REPORT_clusterderive_<YYYY-MM-DD>.md
  clusterderive_artifacts.npz
  clusterderive_dendro_{raw,resid}.png (옵션)
  clusterderive_norm_layer.png (옵션)
"""
from __future__ import annotations
import argparse
import gc
import time
from datetime import date
from pathlib import Path
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score

import similarity_analysis as sa
import pooled_analysis as pa
import subject_similarity_gate as ssg
import subject_layer_resolved as slr

ANALYSIS = Path(__file__).resolve().parent
TAG = "clusterderive"
LAYERS = slr.LAYERS
SEED = slr.SEED

# data-derived windows (bestlayer_resolved 보고서 §A.1)
W_SUBJ = [9, 10, 11, 12, 14]
W_LEV = [20, 25, 26, 27, 29, 30, 31]
W_ALL = list(range(LAYERS))

MIN_UNIT_N = 30


# ──────────────────────── helpers (no new stats; just bookkeeping) ─────────
def unit_centroid_matrix(DAn, unit_arr, units, layers):
    """unit-centroid cosine 행렬. DAn 은 per-layer L2-normalized.
    centroid = mean of L2-normalized members over window-layers, then
    final L2-normalize for cosine == dot. (cluster centroid norm 통제용)."""
    sub = np.ascontiguousarray(DAn[:, layers, :]).astype(np.float32)
    sub = sub.mean(axis=1)                                       # (N, D) layer-mean
    cents = np.zeros((len(units), sub.shape[1]), dtype=np.float32)
    for k, u in enumerate(units):
        idx = np.where(unit_arr == u)[0]
        cents[k] = sub[idx].mean(axis=0)
    n = np.linalg.norm(cents, axis=1, keepdims=True)
    n[n == 0] = 1.0
    cents = cents / n
    M = cents @ cents.T
    np.fill_diagonal(M, 1.0)
    return M, cents


def ward_labels(M, k):
    """Ward(1-cos). cos 음수는 정보 보존 → 절대 클립 금지.
    distance = 1 − cos ∈ [0, 2]. 부동소수 numerical noise 만 max(0,·) 로 처리."""
    D = 1.0 - M
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)            # numerical safety only; cos∈[-1,1] 이미 D∈[0,2]
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z


def silhouette_from_M(M, labels):
    """precomputed dist=1-M, no clip."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    D = 1.0 - M
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)
    try:
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception as e:
        print(f"    [silhouette] 실패: {e}", flush=True)
        return float("nan")


def cross_tab(rows, cols, row_order, col_order):
    """integer cross-tab; rows/cols 동일 길이 array, order 별로 정렬."""
    M = np.zeros((len(row_order), len(col_order)), dtype=np.int64)
    ri = {r: i for i, r in enumerate(row_order)}
    ci = {c: i for i, c in enumerate(col_order)}
    for r, c in zip(rows, cols):
        if r in ri and c in ci:
            M[ri[r], ci[c]] += 1
    return M


def md_table(M, row_order, col_order, head):
    """markdown 표 문자열."""
    cols = [str(c) for c in col_order]
    out = ["| " + head + " | " + " | ".join(cols) + " | sum |",
           "|" + "---|" * (len(cols) + 2)]
    for i, r in enumerate(row_order):
        s = M[i].sum()
        out.append("| " + str(r) + " | " + " | ".join(str(int(v)) for v in M[i]) + f" | {s} |")
    tot = M.sum(axis=0)
    out.append("| **sum** | " + " | ".join(str(int(v)) for v in tot) + f" | {int(tot.sum())} |")
    return "\n".join(out)


def partial_spearman(x, y, z):
    """ρ(x,y|z) via Spearman on rank-residuals (numpy 인라인; 새 통계 아님)."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b0 = b - b.mean()
        beta = (a * b0).sum() / (b0 * b0).sum() if (b0 * b0).sum() > 0 else 0.0
        return a - beta * b
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


# ───────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None, help="smoke: pilot 당 .pt 제한")
    ap.add_argument("--n-perm", type=int, default=1000, help="(STEP1 exploratory perm)")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    DAF, DAT, md, ninfo = pa.load_pooled(args.max_n)
    del DAF; gc.collect()
    N = len(md)
    subjects = sorted(md["subject"].unique().tolist())
    levels = sorted(md["level"].unique().tolist())
    subj_arr = md["subject"].to_numpy()
    lev_arr = md["level"].to_numpy()
    unit_arr = (md["subject"].astype(str) + "|L" + md["level"].astype(str)).to_numpy()
    cnt = Counter(unit_arr)
    units_all = sorted(cnt.keys())
    units_keep = sorted([u for u, n in cnt.items() if n >= MIN_UNIT_N])
    units_drop = sorted([u for u, n in cnt.items() if n < MIN_UNIT_N])

    print(f"[load] N={N} subjects={len(subjects)} levels={levels} "
          f"units(all)={len(units_all)} units(keep n>={MIN_UNIT_N})={len(units_keep)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if args.max_n is None and N != 3025:
        raise SystemExit(f"[GATE] expected N=3025 (pooled THINKING finite); got {N}")
    if W_SUBJ != [9, 10, 11, 12, 14] or W_LEV != [20, 25, 26, 27, 29, 30, 31] \
            or W_ALL != list(range(LAYERS)):
        raise SystemExit("[GATE] window constants drifted")

    DAT = DAT.astype(np.float32)
    mu = DAT.mean(axis=0, keepdims=True)
    DA_c = DAT - mu
    print("[norm] per-layer L2-normalize members ...", flush=True)
    DAn = sa.normalize_members(DA_c)

    # residual (level-centroid 빼기)
    print("[resid] level_centroid_residual ...", flush=True)
    DA_resid = ssg.level_centroid_residual(DA_c, md)
    DAn_r = sa.normalize_members(DA_resid)
    del DA_resid; gc.collect()

    # unit → (subject, level) lookup on keep set
    unit_subj = {u: u.split("|L")[0] for u in units_keep}
    unit_lev = {u: int(u.split("|L")[1]) for u in units_keep}

    saved = {
        "subjects": np.array(subjects),
        "levels": np.array(levels, dtype=int),
        "W_SUBJ": np.array(W_SUBJ),
        "W_LEV": np.array(W_LEV),
        "W_ALL": np.array(W_ALL),
        "units_all": np.array(units_all),
        "units_keep": np.array(units_keep),
        "units_drop": np.array(units_drop),
        "n_per_unit_keep": np.array([cnt[u] for u in units_keep], dtype=int),
        "N_total": np.array([N], dtype=int),
    }

    L = []
    today = date.today().isoformat()
    L.append(f"# unit-centroid 클러스터 도출/검증 — {today} (tag={TAG})")
    L.append("")
    L.append("> **metric pipeline 검증.** `DAT → DA_c = DAT − μ_pooled(per-layer,per-feature)` "
             "→ `DAn = per-sample per-layer L2-normalize` (sa.normalize_members) → "
             "window 평균 후 dot-product = cosine. cos 음수는 정보 → **클립 금지**, "
             "Ward(1−cos) 만 numerical noise 방어용 max(0,·).")
    L.append("")
    L.append(f"**데이터.** pooled(pilot1+pilot2) THINKING ΔA, N={ninfo['n_final']} "
             f"(raw {ninfo['n_loaded']}, non-finite drop {ninfo['n_nonfinite']}). "
             f"subjects={len(subjects)} levels={levels} units(all)={len(units_all)} "
             f"units(keep n≥{MIN_UNIT_N})={len(units_keep)}.")
    if units_drop:
        L.append(f"- units(drop n<{MIN_UNIT_N}, n={len(units_drop)}): "
                 + ", ".join(f"{u}(n={cnt[u]})" for u in units_drop))
    L.append("")
    L.append(f"**windows.** W_SUBJ={W_SUBJ}, W_LEV={W_LEV}, W_ALL=range({LAYERS}).")
    L.append("")

    # ╭──────────────────────────── STEP 1 ───────────────────────────────╮
    # A=same-subj/diff-lev vs B=diff-subj/same-lev on unit-centroid cos
    # ╰───────────────────────────────────────────────────────────────────╯
    L.append("## §1. cross-level binding (cluster ≠ level 직접검정)")
    L.append("")
    L.append("unit-centroid(layer 평균 후 L2-norm) cosine 위에서, off-diagonal 쌍을 분류:")
    L.append("- **A** = same-subject / diff-level  (subject가 level을 가로질러 묶이는가)")
    L.append("- **B** = diff-subject / same-level  (level이 subject를 가로질러 묶이는가)")
    L.append("- A>B → level penalty 를 이긴 강한 cluster≠level 증거.")
    L.append("- A 는 |Δlevel|∈{1,2,≥3} 으로 분해해 long-range subject binding 검사.")
    L.append("")

    step1_results = {}
    for wname, layers in [("Wall", W_ALL), ("Wsubj", W_SUBJ), ("Wlev", W_LEV)]:
        print(f"  [S1/{wname}] unit centroid matrix |layers|={len(layers)} ...", flush=True)
        M, _ = unit_centroid_matrix(DAn, unit_arr, units_keep, layers)
        n = len(units_keep)
        iu = np.triu_indices(n, 1)
        cos_vals = M[iu]
        ui = iu[0]; vj = iu[1]
        s_u = np.array([unit_subj[units_keep[i]] for i in ui])
        s_v = np.array([unit_subj[units_keep[j]] for j in vj])
        l_u = np.array([unit_lev[units_keep[i]] for i in ui])
        l_v = np.array([unit_lev[units_keep[j]] for j in vj])
        same_subj = (s_u == s_v); same_lev = (l_u == l_v)
        A_mask = same_subj & (~same_lev)
        B_mask = (~same_subj) & same_lev
        C_mask = same_subj & same_lev  # 트리비얼한 same/same 쌍 (작지만 기록)
        D_mask = (~same_subj) & (~same_lev)
        A = cos_vals[A_mask]; B = cos_vals[B_mask]
        C_vals = cos_vals[C_mask]; D_vals = cos_vals[D_mask]
        d_AB = slr.cohens_d(A, B)
        # level-gap 분해
        gap = np.abs(l_u - l_v)
        A_g1 = cos_vals[A_mask & (gap == 1)]
        A_g2 = cos_vals[A_mask & (gap == 2)]
        A_g3 = cos_vals[A_mask & (gap >= 3)]
        # exploratory permutation: subject 라벨을 unit centroid 행에 셔플
        rng = np.random.default_rng(SEED)
        obs = (A.mean() if A.size else 0.0) - (B.mean() if B.size else 0.0)
        ge = 0
        units_arr = np.array(units_keep)
        unit_lev_vec = np.array([unit_lev[u] for u in units_keep])
        unit_subj_vec = np.array([unit_subj[u] for u in units_keep])
        for it in range(args.n_perm):
            perm = rng.permutation(unit_subj_vec)
            ss_u = perm[ui]; ss_v = perm[vj]
            same_s_p = (ss_u == ss_v)
            Ap = cos_vals[same_s_p & (~same_lev)]
            Bp = cos_vals[(~same_s_p) & same_lev]
            stat_p = (Ap.mean() if Ap.size else 0.0) - (Bp.mean() if Bp.size else 0.0)
            if stat_p >= obs:
                ge += 1
        p_perm = (ge + 1) / (args.n_perm + 1)

        step1_results[wname] = dict(
            A_mean=float(A.mean()) if A.size else float("nan"),
            B_mean=float(B.mean()) if B.size else float("nan"),
            delta=float(obs), d=float(d_AB),
            nA=int(A.size), nB=int(B.size),
            C_mean=float(C_vals.mean()) if C_vals.size else float("nan"),
            D_mean=float(D_vals.mean()) if D_vals.size else float("nan"),
            nC=int(C_vals.size), nD=int(D_vals.size),
            A_g1_mean=float(A_g1.mean()) if A_g1.size else float("nan"),
            A_g2_mean=float(A_g2.mean()) if A_g2.size else float("nan"),
            A_g3_mean=float(A_g3.mean()) if A_g3.size else float("nan"),
            nA_g1=int(A_g1.size), nA_g2=int(A_g2.size), nA_g3=int(A_g3.size),
            perm_p=float(p_perm),
        )
        saved[f"step1_{wname}_M"] = M.astype(np.float32)
        for k, v in step1_results[wname].items():
            saved[f"step1_{wname}_{k}"] = np.array([v])

    L.append("| window | A mean (same-subj/diff-lev) | B mean (diff-subj/same-lev) | Δ=A−B | "
             "Cohen's d | nA | nB | C (same/same) | D (diff/diff) | perm p* |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for wname in ["Wall", "Wsubj", "Wlev"]:
        r = step1_results[wname]
        L.append(f"| {wname} | {r['A_mean']:+.4f} | {r['B_mean']:+.4f} | {r['delta']:+.4f} | "
                 f"{r['d']:+.3f} | {r['nA']} | {r['nB']} | "
                 f"{r['C_mean']:+.4f}(n={r['nC']}) | {r['D_mean']:+.4f}(n={r['nD']}) | "
                 f"{r['perm_p']:.4f} |")
    L.append("")
    L.append("\\* perm p = subject 라벨을 unit-centroid 행에 셔플한 exploratory null. "
             "결정은 effect size(Δ, Cohen's d)에 의함.")
    L.append("")
    L.append("**A 의 level-gap 분해 (same-subject 결합이 먼 level까지 살아남는가)**")
    L.append("")
    L.append("| window | A |Δlev|=1 | A |Δlev|=2 | A |Δlev|≥3 |")
    L.append("|---|---|---|---|")
    for wname in ["Wall", "Wsubj", "Wlev"]:
        r = step1_results[wname]
        L.append(f"| {wname} | {r['A_g1_mean']:+.4f}(n={r['nA_g1']}) | "
                 f"{r['A_g2_mean']:+.4f}(n={r['nA_g2']}) | "
                 f"{r['A_g3_mean']:+.4f}(n={r['nA_g3']}) |")
    L.append("")

    # ╭──────────────────────────── STEP 2 ───────────────────────────────╮
    # unit-centroid Ward clustering on W_SUBJ (raw + level-residual)
    # ╰───────────────────────────────────────────────────────────────────╯
    L.append("## §2. unit-centroid Ward 클러스터 (W_SUBJ)")
    L.append("")
    L.append("거리 = 1 − cos. **cos 음수 클립 절대 없음** (정보 보존).")
    L.append("silhouette 라벨 후보 = {cluster, subject, level} (unit 자체는 self-라벨이라 무의미).")
    L.append("")

    if len(units_keep) < 4:
        L.append(f"### §2 SKIPPED — units_keep n={len(units_keep)} (<4). smoke 한정.")
        L.append("")
        ari_pilot_k = {}; ari_raw_resid_k = {}
        # placeholder silhouette artifacts for the conclusion box
        for ver in ("raw", "resid"):
            for k in (3, 4, 5):
                saved[f"sil_cluster_{ver}_k{k}"] = np.array([np.nan])
                saved[f"sil_subject_{ver}_k{k}"] = np.array([np.nan])
                saved[f"sil_level_{ver}_k{k}"] = np.array([np.nan])
        # jump to §3
        _SKIP_S2 = True
    else:
        _SKIP_S2 = False

    if _SKIP_S2:
        # write tail markers and jump
        L.append("### §2.ARI 안정성  SKIPPED (smoke)")
        L.append("")
        ari_pilot_k = {}
        # placeholder for conclusion reference
        M_raw = None
    else:
        M_raw, cents_raw = unit_centroid_matrix(DAn, unit_arr, units_keep, W_SUBJ)
        M_res, cents_res = unit_centroid_matrix(DAn_r, unit_arr, units_keep, W_SUBJ)
        saved["step2_Wsubj_raw_M"] = M_raw.astype(np.float32)

    saved["step2_Wsubj_resid_M"] = M_res.astype(np.float32)

    unit_lev_vec = np.array([unit_lev[u] for u in units_keep])
    unit_subj_vec = np.array([unit_subj[u] for u in units_keep])
    lev_order = sorted(set(unit_lev_vec.tolist()))
    subj_order = sorted(set(unit_subj_vec.tolist()))

    labels_store = {}
    Z_store = {}
    for ver, M in [("raw", M_raw), ("resid", M_res)]:
        for k in (3, 4, 5):
            labels, Z = ward_labels(M, k)
            labels_store[(ver, k)] = labels
            Z_store[(ver, k)] = Z
            saved[f"labels_{ver}_k{k}"] = labels.astype(np.int32)
            sil_clu = silhouette_from_M(M, labels)
            sil_subj = silhouette_from_M(M, unit_subj_vec)
            sil_lev = silhouette_from_M(M, unit_lev_vec)
            saved[f"sil_cluster_{ver}_k{k}"] = np.array([sil_clu])
            saved[f"sil_subject_{ver}_k{k}"] = np.array([sil_subj])
            saved[f"sil_level_{ver}_k{k}"] = np.array([sil_lev])
            ctab_lev = cross_tab(labels.tolist(), unit_lev_vec.tolist(),
                                 list(range(1, k + 1)), lev_order)
            ctab_subj = cross_tab(labels.tolist(), unit_subj_vec.tolist(),
                                  list(range(1, k + 1)), subj_order)
            saved[f"ctab_lev_{ver}_k{k}"] = ctab_lev
            saved[f"ctab_subj_{ver}_k{k}"] = ctab_subj
            L.append(f"### §2.{ver}.k={k}  silhouette(cluster)={sil_clu:+.3f}  "
                     f"silhouette(subject)={sil_subj:+.3f}  silhouette(level)={sil_lev:+.3f}")
            L.append("")
            L.append("**cluster × level**")
            L.append("")
            L.append(md_table(ctab_lev, list(range(1, k + 1)), lev_order, "cluster\\level"))
            L.append("")
            L.append("**cluster × subject** (unit 수)")
            L.append("")
            L.append(md_table(ctab_subj, list(range(1, k + 1)), subj_order, "cluster\\subject"))
            L.append("")

    # dendrogram PNG
    if not args.no_png:
        for ver, M in [("raw", M_raw), ("resid", M_res)]:
            Z = Z_store[(ver, 4)]
            fig, ax = plt.subplots(figsize=(12, 4))
            dendrogram(Z, labels=units_keep, leaf_font_size=7, color_threshold=0)
            ax.set_title(f"unit-centroid Ward dendrogram ({ver}, W_SUBJ, k=4 cut shown)")
            ax.set_ylabel("1 − cos")
            plt.tight_layout()
            p_png = ANALYSIS / f"clusterderive_dendro_{ver}.png"
            fig.savefig(p_png, dpi=100); plt.close(fig)
            L.append(f"![dendro_{ver}]({p_png.name})")
            L.append("")

    # ── ARI 안정성 (pilot1 vs pilot2) — 공통 unit (n≥30 각 pilot에서) ──
    pilot1_mask = (md["_pilot"].to_numpy() == "pilot1")
    pilot2_mask = (md["_pilot"].to_numpy() == "pilot2")
    cnt_p1 = Counter(unit_arr[pilot1_mask])
    cnt_p2 = Counter(unit_arr[pilot2_mask])
    units_common = [u for u in units_keep
                    if cnt_p1.get(u, 0) >= MIN_UNIT_N and cnt_p2.get(u, 0) >= MIN_UNIT_N]
    saved["units_common_p1p2"] = np.array(units_common)
    saved["n_per_unit_p1"] = np.array([cnt_p1.get(u, 0) for u in units_keep], dtype=int)
    saved["n_per_unit_p2"] = np.array([cnt_p2.get(u, 0) for u in units_keep], dtype=int)

    L.append(f"### §2.ARI 안정성 (raw, W_SUBJ)")
    L.append("")
    L.append(f"- pilot1·pilot2 각각에서 n≥{MIN_UNIT_N} 인 unit 의 **교집합** = "
             f"{len(units_common)} units (전체 keep {len(units_keep)} 중).")

    ari_pilot_k = {}
    ari_raw_resid_k = {}
    if len(units_common) >= 4:
        # pilot1-only / pilot2-only centroid (raw, W_SUBJ) on common units
        DAn_p1 = DAn[pilot1_mask]; unit_p1 = unit_arr[pilot1_mask]
        DAn_p2 = DAn[pilot2_mask]; unit_p2 = unit_arr[pilot2_mask]
        M_p1, _ = unit_centroid_matrix(DAn_p1, unit_p1, units_common, W_SUBJ)
        M_p2, _ = unit_centroid_matrix(DAn_p2, unit_p2, units_common, W_SUBJ)
        # raw·resid full 의 common subset
        idx_common = [units_keep.index(u) for u in units_common]
        M_raw_c = M_raw[np.ix_(idx_common, idx_common)]
        M_res_c = M_res[np.ix_(idx_common, idx_common)]
        for k in (3, 4, 5):
            l1, _ = ward_labels(M_p1, k)
            l2, _ = ward_labels(M_p2, k)
            ari12 = float(adjusted_rand_score(l1, l2))
            lr, _ = ward_labels(M_raw_c, k)
            lrs, _ = ward_labels(M_res_c, k)
            ari_rr = float(adjusted_rand_score(lr, lrs))
            ari_pilot_k[k] = ari12
            ari_raw_resid_k[k] = ari_rr
            saved[f"ari_pilot1_pilot2_k{k}"] = np.array([ari12])
            saved[f"ari_raw_vs_resid_k{k}"] = np.array([ari_rr])
        L.append("")
        L.append("| k | ARI(pilot1 vs pilot2, raw) | ARI(raw vs residual, full keep ∩ common) |")
        L.append("|---|---|---|")
        for k in (3, 4, 5):
            L.append(f"| {k} | {ari_pilot_k[k]:+.3f} | {ari_raw_resid_k[k]:+.3f} |")
    else:
        L.append("- 공통 unit 부족(<4) → ARI 계산 생략.")
    L.append("")

    # ╭──────────────────────────── STEP 3 ───────────────────────────────╮
    # norm magnitude profile + length redundancy
    # ╰───────────────────────────────────────────────────────────────────╯
    L.append("## §3. ‖ΔA‖ depth profile + length redundancy")
    L.append("")
    L.append("normalize 이전(centered) 의 magnitude 분포가 깊은 레이어에서 큰지, "
             "그리고 그 z-score 가 gen_len 으로 얼마나 설명되는지.")
    L.append("")

    N_il_raw = np.linalg.norm(DAT.astype(np.float32) if False else (DA_c + mu).astype(np.float32),
                              axis=2) if False else None
    # 위 라인 dead — DA_c 만 사용 (raw DAT은 메모리 절약상 생략, centered 가 충분히 직접적)
    N_il = np.linalg.norm(DA_c, axis=2)                      # (N, L)
    layer_mean_norm = N_il.mean(axis=0)
    layer_std_norm = N_il.std(axis=0)
    z_il = (N_il - layer_mean_norm[None, :]) / np.where(layer_std_norm > 0, layer_std_norm, 1.0)
    z_i = z_il.mean(axis=1)
    gl = md["gen_len"].to_numpy(float) if "gen_len" in md.columns else np.full(N, np.nan)
    finite_gl = np.isfinite(gl) & (gl > 0)
    if not finite_gl.all():
        print(f"[warn] gen_len finite={finite_gl.sum()}/{N}", flush=True)

    rho_z_lev, _ = spearmanr(z_i, lev_arr.astype(float))
    rho_z_gl, _ = spearmanr(z_i[finite_gl], gl[finite_gl]) if finite_gl.any() else (float("nan"), 1)
    # partial ρ(z, level | gen_len)
    if finite_gl.sum() >= 10:
        pr = partial_spearman(z_i[finite_gl], lev_arr[finite_gl].astype(float), gl[finite_gl])
    else:
        pr = float("nan")
    # η² (subject) on z (1D ANOVA-like)
    grand = z_i.mean()
    ss_t = float(((z_i - grand) ** 2).sum())
    ss_b = 0.0
    for s in subjects:
        idx = np.where(subj_arr == s)[0]
        if len(idx) == 0: continue
        ss_b += len(idx) * (z_i[idx].mean() - grand) ** 2
    eta2_subj_z = (ss_b / ss_t) if ss_t > 0 else float("nan")

    saved["step3_norm_layermean"] = layer_mean_norm.astype(np.float32)
    saved["step3_norm_layerstd"] = layer_std_norm.astype(np.float32)
    saved["step3_z_per_sample"] = z_i.astype(np.float32)
    saved["step3_rho_z_level"] = np.array([float(rho_z_lev)])
    saved["step3_rho_z_genlen"] = np.array([float(rho_z_gl)])
    saved["step3_partial_rho_z_level_given_genlen"] = np.array([float(pr)])
    saved["step3_eta2_subject_on_z"] = np.array([float(eta2_subj_z)])

    L.append(f"- per-layer mean ‖DA_c‖ (depth profile, L0..L{LAYERS-1}):")
    L.append("")
    L.append("| layer | mean ‖DA_c‖ |")
    L.append("|---|---|")
    for l in range(LAYERS):
        L.append(f"| L{l} | {layer_mean_norm[l]:.3f} |")
    L.append("")
    L.append(f"- ρ(z, level)               = **{rho_z_lev:+.3f}**")
    L.append(f"- ρ(z, gen_len)             = **{rho_z_gl:+.3f}**")
    L.append(f"- partial ρ(z, level|gen_len)= **{pr:+.3f}**")
    L.append(f"- η²(subject) on z          = **{eta2_subj_z:.4f}** (1-D, 불균형 근사)")
    L.append("")

    if not args.no_png:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(LAYERS), layer_mean_norm, yerr=layer_std_norm, capsize=2)
        ax.set_xlabel("layer index")
        ax.set_ylabel("mean ‖DA_c‖")
        ax.set_title("per-layer L2 norm of centered ΔA (depth profile)")
        plt.tight_layout()
        p_png = ANALYSIS / "clusterderive_norm_layer.png"
        fig.savefig(p_png, dpi=100); plt.close(fig)
        L.append(f"![norm_layer]({p_png.name})")
        L.append("")

    # ╭──────────────────────────── 결론 박스 ─────────────────────────────╮
    L.append("## 결론 박스 (수치-only)")
    L.append("")
    r_all = step1_results["Wall"]
    L.append(f"- **§1 Wall**: Δ=A−B={r_all['delta']:+.4f}, Cohen's d={r_all['d']:+.3f} "
             f"(A 분해: gap1={r_all['A_g1_mean']:+.3f}, gap2={r_all['A_g2_mean']:+.3f}, "
             f"gap≥3={r_all['A_g3_mean']:+.3f}). "
             "Δ>0 & 먼 gap에서도 양수 유지 → cluster≠level.")
    L.append(f"- **§2 raw k=4**: sil(cluster)={float(saved['sil_cluster_raw_k4'][0]):+.3f}, "
             f"sil(subject)={float(saved['sil_subject_raw_k4'][0]):+.3f}, "
             f"sil(level)={float(saved['sil_level_raw_k4'][0]):+.3f}. "
             "subject > level 이면 unit이 subject 로 더 뭉친 것.")
    if ari_pilot_k:
        L.append(f"- **§2 ARI k=4**: pilot1↔pilot2={ari_pilot_k[4]:+.3f}, "
                 f"raw↔resid={ari_raw_resid_k[4]:+.3f}.")
    L.append(f"- **§3**: ρ(z,gen_len)={rho_z_gl:+.3f} (∼0.74면 redundant), "
             f"partial ρ(z,level|gen_len)={pr:+.3f} (length-controlled 난이도 신호 잔존 여부).")
    L.append("")
    L.append("---")
    L.append(f"elapsed = {time.time()-t0:.0f}s")
    L.append("")

    # save
    out_md = ANALYSIS / f"REPORT_clusterderive_{today}.md"
    out_md.write_text("\n".join(L))
    np.savez(ANALYSIS / "clusterderive_artifacts.npz", **saved)
    print(f"[done] {out_md.name}  + clusterderive_artifacts.npz "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
