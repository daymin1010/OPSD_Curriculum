# Critical Analysis — prompt-only activation vs ΔA
## Setup
- Source: `outputs/prompt_act/{id}.pt`, 36 layers × 12288 dims (bfloat16, MLP down_proj input @ last prompt token, no generation).
- Pilot: N=36 layers from prompt cache; reusing 8 MATH subjects.
- Probe pipeline identical to Phase A: PCA-256 → 5-fold CV LDA/Ridge.

## Headline numbers (prompt-only activation)
| signal | best layer | value |
|---|---|---|
| subject macro-F1                | L13 | **0.788** |
| level Ridge R²                  | L21 | **+0.908** |
| level LDA-1 \|ρ\| vs level     | L18 | **0.912** |
| pass_rate Ridge R²              | L27 | **+0.326** |

## ΔA vs A_prompt — head-to-head
|   layer |   deltaA_subj_F1 |   prompt_subj_F1 |   deltaA_lvl_ridge_R2 |   prompt_lvl_ridge_R2 |   deltaA_lvl_lda_rho |   prompt_lvl_lda_rho |
|--------:|-----------------:|-----------------:|----------------------:|----------------------:|---------------------:|---------------------:|
|  11.000 |            0.757 |            0.770 |                 0.809 |                 0.820 |                0.864 |                0.891 |
|  14.000 |            0.779 |            0.782 |                 0.858 |                 0.846 |                0.885 |                0.899 |
|  17.000 |            0.735 |            0.778 |                 0.879 |                 0.893 |                0.881 |                0.910 |
|  18.000 |            0.747 |            0.777 |                 0.879 |                 0.905 |                0.891 |                0.912 |
|  21.000 |            0.703 |            0.774 |                 0.818 |                 0.908 |                0.886 |                0.912 |
|  28.000 |            0.715 |            0.772 |                 0.862 |                 0.905 |                0.880 |                0.902 |

### Best-layer comparison
| signal | ΔA best | A_prompt best | gap |
|---|---:|---:|---:|
| subject F1 | 0.779 | 0.788 | -0.009 |
| level R²   | +0.879 | +0.908 | -0.029 |

### Interpretation guide
- subject F1 gap = -0.009: ΔA's subject info is *essentially the prompt's keyword footprint*. The Phase-A claim of "subject signal in ΔA" should be reframed as "subject is already linearly readable from the prompt activation at t_1".
- For level/pass: A_prompt cannot see generation outcome, so a high A_prompt R²_pass would imply pass-rate is predictable from prompt features alone (length, vocabulary), not model reasoning.
