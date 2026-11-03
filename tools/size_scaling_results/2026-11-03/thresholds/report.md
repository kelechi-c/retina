# 768-tied thresholds at 512 / 768 / 1,024

A/B on 18 images (3 each from div2k, met, oi, sa1b, synth, wikiart; 768-1024 px).
Delta = plain minus variant; positive = the variant (warp on / refiner on) is better.

## Warp on vs off (dLPIPS, dPSNR)

| side | tokens | n | dLPIPS | dPSNR dB |
|---|---|---|---|---|
| 512 | 64 | 18 | +0.0240 | +0.42 |
| 512 | 81 | 18 | +0.0179 | +0.55 |
| 512 | 121 | 18 | +0.0117 | +0.50 |
| 512 | 196 | 18 | +0.0114 | +0.60 |
| 512 | 256 | 18 | +0.0068 | +0.66 |
| 768 | 64 | 18 | +0.0212 | +0.44 |
| 768 | 121 | 18 | +0.0144 | +0.46 |
| 768 | 196 | 18 | +0.0034 | +0.49 |
| 768 | 256 | 18 | +0.0036 | +0.56 |
| 768 | 400 | 18 | +0.0047 | +0.59 |
| 768 | 576 | 18 | +0.0004 | +0.69 |
| 1024 | 121 | 18 | +0.0098 | +0.46 |
| 1024 | 196 | 18 | +0.0127 | +0.46 |
| 1024 | 361 | 18 | +0.0004 | +0.56 |
| 1024 | 529 | 18 | -0.0002 | +0.66 |
| 1024 | 784 | 18 | +0.0001 | +0.65 |
| 1024 | 1225 | 18 | -0.0002 | +0.50 |

## S1 refiner on vs off (dLPIPS, dPSNR, stats attached)

| side | tokens | n | dLPIPS | dPSNR dB | stats |
|---|---|---|---|---|---|
| 512 | 121 | 18 | +0.1333 | -0.64 | 14/18 |
| 512 | 256 | 18 | +0.0939 | -1.32 | 14/18 |
| 512 | 529 | 18 | +0.0497 | -0.71 | 14/18 |
| 768 | 121 | 18 | +0.1607 | -0.84 | 15/18 |
| 768 | 256 | 18 | +0.1278 | -1.45 | 15/18 |
| 768 | 576 | 18 | +0.0863 | -1.00 | 14/18 |
| 1024 | 256 | 18 | +0.1734 | -1.60 | 15/18 |
| 1024 | 576 | 18 | +0.1188 | -1.22 | 15/18 |
| 1024 | 1024 | 18 | +0.0905 | -0.95 | 14/18 |

## Readout

- **Warp:** helps below ~200 tokens at every side and is flat above ~250; the fixed 200-token cutoff is
  empirically right. At 512 the gain is still present at 196-256 tokens (+0.011, +0.007) so the cutoff
  could go up slightly there; at 1,024 it is gone by 361 tokens. No rescaling needed.
- **Refiner:** the 768-trained S1 refiner improves LPIPS at 512 and 1,024 too, and by more there
  (+0.13 at 512/121, +0.17 at 1,024/256), paying 0.6-1.6 dB PSNR. So it generalises; the 600-token
  limit is a policy choice, not a cliff: it still helps at 529 tokens (512), 576 (768) and 1,024 (1,024).
- **22 dB floor** (from `run1` sweeps, floor excluded there): median tokens to clear 22 dB are 9 / 16 / 16
  at 512 / 768 / 1,024, i.e. ~1% of the grid; the floor bites for 2-25% of images (mostly flats) and never
  fails at the max grid. It is scale-robust and needs no rescale.
- **6-px filter window** (`split.structure(r=6)`, a 13-px window): already handled by `BUDGET_SIDE=768`;
  residual |dtexture| 0.0012 at 512 (the 768 copy is an *upscale* there) and 0.0004 at 1,024, both under
  the 0.002 check. The 512 asymmetry is bicubic upsampling smoothing, 60% of the tolerance.
- **Mechanical bug found in the refiner gate:** `_encode_fast` tests `length(t) + (S//T)**2 <= learned_max`,
  i.e. it adds the *tile count* (4 / 9 / 16) instead of the packed statistics tokens (1 / 2 / 4). At 1,024
  that is 12 tokens stricter than intended. Fix before scaling the limit.
