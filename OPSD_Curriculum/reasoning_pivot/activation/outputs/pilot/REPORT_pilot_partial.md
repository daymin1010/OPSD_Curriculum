# Extraction Report — pilot_partial

- meta rows: **1506** (output_dir=`src/OPSD_Curriculum/reasoning_pivot/activation/outputs/pilot`)
- status counts: {'ok': 1506}
- shifts/*.pt files: **1506**

- finish_reason: {'stop': 1204, 'length': 302}
- think_status: {'ok': 1294, 'ok_truncated': 212}
- truncated: 302/1506

## is_correct (n scored = 1505)
- overall correct rate: **0.809**

### by level

           mean  count
level                 
1      0.933333    195
2      0.910314    223
3      0.876106    226
4      0.830986    213
5      0.815385    195
6      0.772727    198
7      0.587629    194
8       0.52459     61

- ρ(is_correct, level) = -0.275 (expect NEGATIVE: harder → lower)

### by subject

                            mean  count
subject                                
Algebra                 0.827411    197
Counting & Probability  0.807692    208
Geometry                0.783654    208
Intermediate Algebra    0.842932    191
Number Theory           0.821101    218
Other                   0.743842    203
Prealgebra                  0.88    100
Precalculus             0.805556    180

## ΔA signal (N=1506, layers=36)
- mean |dA_faithful| (layer-avg): 406.701
- mean |dA_thinking| (layer-avg): 69.281
- ρ(|dA_faithful|, level)          = 0.080
- ρ(|dA_faithful|, r1_cot_tokens)  = 0.064
- ρ(|dA_thinking|, level)          = -0.371
- ρ(|dA_thinking|, r1_cot_tokens)  = -0.396

### |dA| by level (layer-avg mean)

         faithful   thinking
level                       
1      405.868073  71.568588
2      406.172577  71.486839
3      406.181763  70.762550
4      406.518494  69.768997
5      406.866638  68.846664
6      407.375031  67.973663
7      407.654297  65.601952
8      408.095764  64.113258

### correct vs incorrect |dA_faithful|
- correct  : mean=406.451 (n=1218)
- incorrect: mean=407.731 (n=287)

- global PC1 var-explained (|dA_faithful| layer profile): 0.524
