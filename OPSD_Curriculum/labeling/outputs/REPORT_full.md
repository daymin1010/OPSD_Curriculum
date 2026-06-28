# OpenThoughts 30K — Full Labeling Report

- file: `src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels.csv`
- rows: **29,434** (expected 29,434)
- cols: 20

## 1. Basic integrity
- error non-empty: **0**
- subject null: **0**
- level null: **0**
- raw_response empty: **0**
- row_index unique: True
- row_index range matches 0..29433: missing=0 extra=0
- finish_reason `length` (truncation): **0**

## 2. Category validity
- subjects outside 8 allowed: **66**
- level dtype: `int64`
- level outside [1,8]: **0**

## 3. raw_response valid JSON: **29434/29434**

## 4. Tokens / latency / retries
- prompt_tokens median/max: **1656** / 3139
- completion_tokens median/max: **13** / 14
- total prompt_tokens: 49,055,219
- total completion_tokens: 385,711
- attempts >1: **0**, max attempts: 1
- latency mean/p50/p95/p99 (s): 0.83 / 0.68 / 1.35 / 3.50
- cost estimate (no caching): **$20.24**  (실제 ~$9 → caching 효과)

## 5. Meta consistency
- models: ['gpt-4.1-mini-2025-04-14']
- prompt_sha: ['208fbdb6202f']

## 6. Distributions

### 6.1 subject
```
subject
Algebra                   6823
Number Theory             5436
Geometry                  5266
Counting & Probability    3859
Prealgebra                2903
Intermediate Algebra      2289
Precalculus               2195
Other                      597
Calculus                    32
Logic                       16
Physics                     11
Linear Algebra               5
Trigonometry                 1
Functional Equations         1
```

### 6.2 level
```
level
1    1343
2    4328
3    5914
4    6172
5    6110
6    3682
7    1819
8      66
```

### 6.3 source
```
source
olympiads     21315
math           5351
aops_forum     2291
amc_aime        477
```

### 6.4 subject × level cross-tab
```
level                      1     2     3     4     5     6     7   8    All
subject                                                                    
Algebra                  165  1190  1805  1724  1365   495    79   0   6823
Calculus                   1     5     7     3    12     2     2   0     32
Counting & Probability   106   561   961   713   720   480   308  10   3859
Functional Equations       0     0     0     0     0     0     1   0      1
Geometry                  37   356   797  1317  1389   870   491   9   5266
Intermediate Algebra      16    85   231   464   725   517   240  11   2289
Linear Algebra             0     0     2     0     0     3     0   0      5
Logic                      0     0     2     3     3     8     0   0     16
Number Theory             89   445   909  1101  1271  1033   565  23   5436
Other                     23    92   112    73   100    78   106  13    597
Physics                    0     1     1     2     6     1     0   0     11
Prealgebra               888  1460   538    17     0     0     0   0   2903
Precalculus               18   133   549   755   518   195    27   0   2195
Trigonometry               0     0     0     0     1     0     0   0      1
All                     1343  4328  5914  6172  6110  3682  1819  66  29434
```

### 6.5 source × level cross-tab
```
level          1     2     3     4     5     6     7   8    All
source                                                         
amc_aime      59   167    94    66    55    29     7   0    477
aops_forum    12   135   282   410   607   485   337  23   2291
math         712  1875  1332   769   456   178    28   1   5351
olympiads    560  2151  4206  4927  4992  2990  1447  42  21315
All         1343  4328  5914  6172  6110  3682  1819  66  29434
```

### 6.6 source × subject cross-tab
```
subject     Algebra  Calculus  Counting & Probability  Functional Equations  Geometry  Intermediate Algebra  Linear Algebra  Logic  Number Theory  Other  Physics  Prealgebra  Precalculus  Trigonometry    All
source                                                                                                                                                                                                         
amc_aime        113         0                      42                     0        43                    36               0      1             76      8        0         148           10             0    477
aops_forum      371         5                     379                     0       443                   213               0      1            624     70        2         100           83             0   2291
math           1378         0                     613                     0       795                   417               2      0            885      1        0         883          377             0   5351
olympiads      4961        27                    2825                     1      3985                  1623               3     14           3851    518        9        1772         1725             1  21315
All            6823        32                    3859                     1      5266                  2289               5     16           5436    597       11        2903         2195             1  29434
```

## 7. Sparse cells (subject × level, count < 30)
- # sparse cells: **64** of 112
```
subject                 level
Algebra                 8         0
Functional Equations    6         0
                        4         0
                        5         0
                        3         0
                        2         0
                        1         0
Calculus                8         0
Linear Algebra          4         0
                        5         0
                        7         0
                        8         0
Logic                   2         0
Linear Algebra          2         0
                        1         0
Functional Equations    8         0
Logic                   1         0
Prealgebra              6         0
                        7         0
                        8         0
Precalculus             8         0
Trigonometry            3         0
                        2         0
                        1         0
                        6         0
                        7         0
                        8         0
Physics                 7         0
                        1         0
                        8         0
Logic                   8         0
                        7         0
Trigonometry            4         0
Prealgebra              5         0
Physics                 2         1
                        6         1
Trigonometry            5         1
Calculus                1         1
Functional Equations    7         1
Physics                 3         1
Logic                   3         2
Linear Algebra          3         2
Calculus                7         2
                        6         2
Physics                 4         2
Calculus                4         3
Logic                   5         3
Linear Algebra          6         3
Logic                   4         3
Calculus                2         5
Physics                 5         6
Calculus                3         7
Logic                   6         8
Geometry                8         9
Counting & Probability  8        10
Intermediate Algebra    8        11
Calculus                5        12
Other                   8        13
Intermediate Algebra    1        16
Prealgebra              4        17
Precalculus             1        18
Other                   1        23
Number Theory           8        23
Precalculus             7        27
```
- cells with n=0: **34**

## 8. Difficulty signal — Spearman ρ with level
```
level                   1.000
r1_cot_token_count      0.610
problem_qwen_tok_len    0.325
problem_char_len        0.310
solution_char_len       0.544
```
- level vs r1_cot_token_count overall: ρ = **0.61**

Per-source level vs r1_cot ρ:
```
  amc_aime        n=   477  ρ=0.578
  aops_forum      n=  2291  ρ=0.498
  math            n=  5351  ρ=0.684
  olympiads       n= 21315  ρ=0.528
```

## 9. Smoke (200) vs Full (29,434)
```
level distribution (%)
       smoke_pct  full_pct
level                     
1            2.5       4.6
2            7.5      14.7
3           24.0      20.1
4           28.0      21.0
5           20.0      20.8
6           13.5      12.5
7            4.5       6.2
8            0.0       0.2
```

## 10. Qualitative samples — one per level
- **L1 Prealgebra**: `Determine the nearest integer to (a) $\frac{19}{15}+\frac{19}{3}$ (b) $\frac{85}{42}+\frac{43}{21}+\frac{29}{14}+\frac{15}{7}$ (c) $-\frac{1`
- **L2 Prealgebra**: `Give the value of \(0 - 1 + 2 - 3 + 4 - 5 + \ldots - 49 + 50\). Only a numerical answer is expected.`
- **L3 Counting & Probability**: `Given that \(1 \leq x, y, z \leq 6\), how many cases are there in which the product of natural numbers \(x, y, z\) is divisible by 10?`
- **L4 Number Theory**: `Let \( p = 2^{3009}, q = 3^{2006}, \) and \( r = 5^{1003} \). Which of the following statements is true? (A) \( p < q < r \) (B) \( p < r < `
- **L5 Algebra**: `Given two linear functions \( f(x) \) and \( g(x) \) such that the graphs \( y = f(x) \) and \( y = g(x) \) are parallel lines that are not `
- **L6 Geometry**: ` Vasya cut a triangle out of cardboard and numbered its vertices with the digits $1, 2, 3$. It turned out that if Vasya rotates the triangle`
- **L7 Intermediate Algebra**: `Given real numbers \( a, b, c \) and a positive number \( \lambda \) such that the polynomial \( f(x) = x^3 + a x^2 + b x + c \) has three r`
- **L8 Counting & Probability**: `Consider an infinite grid of unit squares. An $n$-omino is a subset of $n$ squares that is connected. Two $n$-ominoes are considered equival`

## 11. Issues
- ⚠️ category invalid: bad_subject=66 bad_level=0