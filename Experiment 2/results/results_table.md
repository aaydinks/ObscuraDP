| Configuration | Accuracy | Loss | Wall-clock/round (s) | Agg latency/round (ms) | Ciphertext size (MB) |
|---|---:|---:|---:|---:|---:|
| Standard FL | 93.44% | 0.1726 | 0.192 | 0.102 | — |
| DP-FL (eps=14.13) | 91.57% | 0.2116 | 0.283 | 0.061 | — |
| DP-FL (eps=8.65) | 91.80% | 0.2123 | 0.254 | 0.058 | — |
| DP-FL (eps=5.74) | 89.93% | 0.2349 | 0.275 | 0.061 | — |
| HE-FL | 93.44% | 0.1726 | 0.130 | 21.658 | 0.334 |

_Plaintext update = 3.72 KB/client; CKKS ciphertext = 0.334 MB/client (90x blow-up). HE encryption = 2.56 ms/client; HE aggregation is 212x slower than the plaintext mean._
