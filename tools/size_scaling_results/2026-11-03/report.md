# Size scaling of the token budget

251 images (209 training, 42 held out), sides 512, 768, 1024. Model: `log(N_S/N_768) = (g0 + g1*t + g2*d) * log(S/768)`.

## Budget inputs read at each side (should match 768)

| side | median abs change in d | median abs change in texture |
|---|---|---|
| 512 | 0.0044 | 0.00120 |
| 768 | 0.0000 | 0.00000 |
| 1024 | 0.0007 | 0.00038 |

## Sanity at 768: measured tokens against the shipped grid

| criterion | median measured / shipped | share censored |
|---|---|---|
| lpips_display | 1.000 | 2% |
| lpips_native | 1.000 | 2% |
| psnr_display | 1.000 | 0% |
| psnr_native | 1.000 | 0% |

## Fits

| criterion | g0 | g1 (t) | g2 (d) | one gamma | R² train | points | censored below / above |
|---|---|---|---|---|---|---|---|
| lpips_display | 0.342 | 1.469 | -0.232 | 0.846 | -0.106 | 400 | 22 / 1 |
| lpips_native | 1.286 | 1.728 | -1.006 | 1.571 | 0.506 | 407 | 11 / 1 |
| psnr_display | 0.006 | 1.047 | 0.837 | 0.764 | -0.270 | 414 | 3 / 1 |
| psnr_native | 0.290 | 2.448 | -0.683 | 0.981 | 0.333 | 410 | 3 / 5 |

## Held out: median absolute error in log tokens (lower is better)

| criterion | no scaling | area (gamma 2) | one gamma | fitted |
|---|---|---|---|---|
| lpips_display | 0.228 | 0.583 | 0.301 | 0.324 |
| lpips_native | 0.466 | 0.333 | 0.293 | 0.236 |
| psnr_display | 0.231 | 0.644 | 0.284 | 0.226 |
| psnr_native | 0.181 | 0.542 | 0.252 | 0.171 |

## Gamma by group (median per image, training and held out)

| criterion | group | side | n | median gamma | p10 | p90 |
|---|---|---|---|---|---|---|
| lpips_display | div2k | 512 | 40 | 1.71 | 0.99 | 2.52 |
| lpips_display | div2k | 1024 | 40 | -0.25 | -0.61 | 0.03 |
| lpips_display | met | 512 | 40 | 2.11 | 1.39 | 3.00 |
| lpips_display | met | 1024 | 40 | -0.49 | -0.84 | -0.19 |
| lpips_display | oi | 512 | 40 | 1.20 | 0.57 | 1.94 |
| lpips_display | oi | 1024 | 40 | -0.32 | -0.75 | -0.08 |
| lpips_display | procedural | 512 | 40 | 0.55 | -1.99 | 1.52 |
| lpips_display | procedural | 1024 | 40 | -0.14 | -0.87 | 0.19 |
| lpips_display | sa1b | 512 | 40 | 1.77 | 1.14 | 2.52 |
| lpips_display | sa1b | 1024 | 40 | -0.29 | -0.53 | -0.04 |
| lpips_display | synth | 512 | 2 | 1.26 | 0.21 | 2.30 |
| lpips_display | synth | 1024 | 3 | -0.19 | -2.88 | -0.07 |
| lpips_display | wikiart | 512 | 35 | 1.01 | 0.31 | 2.79 |
| lpips_display | wikiart | 1024 | 39 | -0.28 | -1.10 | 0.09 |
| lpips_native | div2k | 512 | 40 | 1.82 | 1.48 | 2.44 |
| lpips_native | div2k | 1024 | 40 | 1.83 | 1.32 | 3.16 |
| lpips_native | met | 512 | 40 | 2.08 | 1.18 | 2.90 |
| lpips_native | met | 1024 | 40 | 2.63 | 1.45 | 5.06 |
| lpips_native | oi | 512 | 40 | 1.26 | 0.56 | 2.01 |
| lpips_native | oi | 1024 | 40 | 0.79 | 0.10 | 1.72 |
| lpips_native | procedural | 512 | 40 | 0.40 | -0.65 | 1.02 |
| lpips_native | procedural | 1024 | 40 | 0.24 | -0.86 | 1.73 |
| lpips_native | sa1b | 512 | 40 | 2.00 | 1.41 | 2.43 |
| lpips_native | sa1b | 1024 | 40 | 1.94 | 1.44 | 3.25 |
| lpips_native | synth | 512 | 2 | 1.23 | 0.16 | 2.31 |
| lpips_native | synth | 1024 | 8 | 12.96 | -0.26 | 17.49 |
| lpips_native | wikiart | 512 | 39 | 1.07 | 0.24 | 2.12 |
| lpips_native | wikiart | 1024 | 39 | 0.62 | -0.15 | 1.63 |
| psnr_display | div2k | 512 | 39 | 1.88 | 0.86 | 3.30 |
| psnr_display | div2k | 1024 | 40 | -0.66 | -1.26 | -0.33 |
| psnr_display | met | 512 | 39 | 2.25 | 0.91 | 3.71 |
| psnr_display | met | 1024 | 40 | -0.89 | -1.97 | -0.32 |
| psnr_display | oi | 512 | 40 | 0.97 | 0.48 | 1.76 |
| psnr_display | oi | 1024 | 40 | -0.40 | -0.70 | -0.03 |
| psnr_display | procedural | 512 | 40 | 0.40 | -0.12 | 0.92 |
| psnr_display | procedural | 1024 | 40 | -0.19 | -1.03 | 0.00 |
| psnr_display | sa1b | 512 | 40 | 2.21 | 0.89 | 3.20 |
| psnr_display | sa1b | 1024 | 40 | -0.79 | -1.27 | -0.21 |
| psnr_display | synth | 512 | 10 | 0.14 | -0.58 | 0.45 |
| psnr_display | synth | 1024 | 11 | 0.00 | -0.58 | 1.34 |
| psnr_display | wikiart | 512 | 39 | 0.84 | 0.15 | 2.41 |
| psnr_display | wikiart | 1024 | 40 | -0.29 | -1.33 | 0.19 |
| psnr_native | div2k | 512 | 39 | 1.20 | 0.47 | 2.20 |
| psnr_native | div2k | 1024 | 40 | 1.23 | 0.41 | 2.61 |
| psnr_native | met | 512 | 39 | 1.42 | 0.53 | 2.80 |
| psnr_native | met | 1024 | 40 | 1.47 | 0.41 | 4.91 |
| psnr_native | oi | 512 | 40 | 0.46 | -0.04 | 1.02 |
| psnr_native | oi | 1024 | 40 | 0.26 | 0.00 | 1.49 |
| psnr_native | procedural | 512 | 40 | 0.20 | -0.70 | 0.46 |
| psnr_native | procedural | 1024 | 40 | 0.23 | -0.05 | 0.83 |
| psnr_native | sa1b | 512 | 40 | 1.48 | 0.52 | 2.21 |
| psnr_native | sa1b | 1024 | 40 | 1.13 | 0.34 | 2.88 |
| psnr_native | synth | 512 | 10 | 0.13 | -0.60 | 0.32 |
| psnr_native | synth | 1024 | 10 | 0.41 | -0.04 | 13.24 |
| psnr_native | wikiart | 512 | 37 | 0.31 | -0.28 | 1.17 |
| psnr_native | wikiart | 1024 | 39 | 0.14 | -0.03 | 1.13 |

## Example paintings under lpips_native

| image | d | t | gamma | budget at 512 | budget at 768 | budget at 1024 |
|---|---|---|---|---|---|---|
| mondrian_piet | 0.148 | 0.186 | 1.46 | 19 (cap 1024) | 34 (cap 2304) | 52 (cap 4096) |
| mondrian_b144 | 0.234 | 0.229 | 1.45 | 30 (cap 1024) | 53 (cap 2304) | 81 (cap 4096) |
| bruegel | 0.776 | 0.721 | 1.75 | 780 (cap 1024) | 1586 (cap 2304) | 2626 (cap 4096) |
| veronese | 0.831 | 0.738 | 1.73 | 1027 (cap 1024) | 2066 (cap 2304) | 3395 (cap 4096) |
