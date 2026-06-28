# Extraction Report — smoke2

- meta rows: **9** (output_dir=`src/OPSD_Curriculum/reasoning_pivot/activation/outputs/smoke2`)
- status counts: {'ok': 9}
- shifts/*.pt files: **9**

- finish_reason: {'stop': 7, 'length': 2}
- think_status: {'ok': 7, 'ok_truncated': 2}
- truncated: 2/9

## is_correct (n scored = 9)
- overall correct rate: **0.889**

### by level

       mean  count
level             
1       1.0      1
2       1.0      1
3       1.0      1
4       1.0      1
5       1.0      1
6       1.0      1
7       1.0      2
8       0.0      1

- ρ(is_correct, level) = -0.550 (expect NEGATIVE: harder → lower)

### by subject

                      mean  count
subject                          
Algebra                1.0      1
Geometry               1.0      1
Intermediate Algebra   1.0      1
Number Theory          0.8      5
Prealgebra             1.0      1

## ΔA signal (N=9, layers=36)
- mean |dA_faithful| (layer-avg): 406.760
- mean |dA_thinking| (layer-avg): 69.008
- ρ(|dA_faithful|, level)          = 0.126
- ρ(|dA_faithful|, r1_cot_tokens)  = -0.033
- ρ(|dA_thinking|, level)          = -0.285
- ρ(|dA_thinking|, r1_cot_tokens)  = 0.317

### |dA| by level (layer-avg mean)

         faithful   thinking
level                       
1      406.791199  73.710480
2      407.158752  72.070572
3      405.975555  70.104942
4      405.560272  72.649033
5      408.686707  56.989033
6      406.046631  71.260399
7      406.075073  72.828987
8      408.468994  58.626865

### correct vs incorrect |dA_faithful|
- correct  : mean=406.546 (n=8)
- incorrect: mean=408.469 (n=1)

- global PC1 var-explained (|dA_faithful| layer profile): 0.911
