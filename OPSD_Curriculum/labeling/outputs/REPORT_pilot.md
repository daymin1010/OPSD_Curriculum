# Pilot Universe Candidate

- source CSV : `outputs/openthoughts_30k_labels.csv`
- size       : **3,000** problems (target 3000)
- stratifier : (subject, level), 8×8 = 64 cells (subject ∈ 8 canonical)
- sampler    : sqrt(n)*5, capped 5–80, seed=42

## Subject × Level after normalization (full 29,434)
```
level                     1     2     3     4     5     6    7   8
subject                                                           
Algebra                 165  1190  1805  1724  1365   495   79   0
Counting & Probability  106   561   961   713   720   480  308  10
Geometry                 37   356   797  1317  1389   870  491   9
Intermediate Algebra     16    85   231   464   725   517  240  11
Number Theory            89   445   909  1101  1271  1033  565  23
Other                    24    98   124    81   122    92  109  13
Prealgebra              888  1460   538    17     0     0    0   0
Precalculus              18   133   549   755   518   195   27   0
```

## Pilot subject × level
```
level                    1   2   3   4   5   6   7   8
subject                                               
Algebra                 54  71  66  71  72  70  36   0
Counting & Probability  44  65  63  67  65  72  62   9
Geometry                23  61  68  69  72  68  65   9
Intermediate Algebra    13  41  53  66  66  70  68  10
Number Theory           36  71  68  67  67  70  64  19
Other                   22  44  48  36  44  38  42  11
Prealgebra              64  69  68  14   0   0   0   0
Precalculus             14  50  64  61  63  53  24   0
```

## Per-cell pilot sampling diagnostics
| subject | level | full_n | pilot_n |
|---|---|---:|---:|
| Algebra | 1 | 165 | 64 |
| Algebra | 2 | 1190 | 80 |
| Algebra | 3 | 1805 | 80 |
| Algebra | 4 | 1724 | 80 |
| Algebra | 5 | 1365 | 80 |
| Algebra | 6 | 495 | 80 |
| Algebra | 7 | 79 | 44 |
| Counting & Probability | 1 | 106 | 51 |
| Counting & Probability | 2 | 561 | 80 |
| Counting & Probability | 3 | 961 | 80 |
| Counting & Probability | 4 | 713 | 80 |
| Counting & Probability | 5 | 720 | 80 |
| Counting & Probability | 6 | 480 | 80 |
| Counting & Probability | 7 | 308 | 80 |
| Counting & Probability | 8 | 10 | 10 |
| Geometry | 1 | 37 | 30 |
| Geometry | 2 | 356 | 80 |
| Geometry | 3 | 797 | 80 |
| Geometry | 4 | 1317 | 80 |
| Geometry | 5 | 1389 | 80 |
| Geometry | 6 | 870 | 80 |
| Geometry | 7 | 491 | 80 |
| Geometry | 8 | 9 | 9 |
| Intermediate Algebra | 1 | 16 | 16 |
| Intermediate Algebra | 2 | 85 | 46 |
| Intermediate Algebra | 3 | 231 | 76 |
| Intermediate Algebra | 4 | 464 | 80 |
| Intermediate Algebra | 5 | 725 | 80 |
| Intermediate Algebra | 6 | 517 | 80 |
| Intermediate Algebra | 7 | 240 | 77 |
| Intermediate Algebra | 8 | 11 | 11 |
| Number Theory | 1 | 89 | 47 |
| Number Theory | 2 | 445 | 80 |
| Number Theory | 3 | 909 | 80 |
| Number Theory | 4 | 1101 | 80 |
| Number Theory | 5 | 1271 | 80 |
| Number Theory | 6 | 1033 | 80 |
| Number Theory | 7 | 565 | 80 |
| Number Theory | 8 | 23 | 23 |
| Other | 1 | 24 | 24 |
| Other | 2 | 98 | 49 |
| Other | 3 | 124 | 56 |
| Other | 4 | 81 | 45 |
| Other | 5 | 122 | 55 |
| Other | 6 | 92 | 48 |
| Other | 7 | 109 | 52 |
| Other | 8 | 13 | 13 |
| Prealgebra | 1 | 888 | 80 |
| Prealgebra | 2 | 1460 | 80 |
| Prealgebra | 3 | 538 | 80 |
| Prealgebra | 4 | 17 | 17 |
| Precalculus | 1 | 18 | 18 |
| Precalculus | 2 | 133 | 58 |
| Precalculus | 3 | 549 | 80 |
| Precalculus | 4 | 755 | 80 |
| Precalculus | 5 | 518 | 80 |
| Precalculus | 6 | 195 | 70 |
| Precalculus | 7 | 27 | 26 |

## Length / difficulty sanity in pilot
- ρ(level, r1_cot_token_count) in pilot: **0.668**
- ρ(level, problem_qwen_tok_len)   in pilot: **0.438**