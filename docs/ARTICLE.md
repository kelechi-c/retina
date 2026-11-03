# A Mondrian in 75 tokens: content-aware image tokens from a warp, a small decoder and six numbers

Numbers are from 2026-11-03, on the released system: FLUX.2 VAE, tile DReX, warp and S1 refiner, plus a 57M MAR trained on its tokens. This is a research prototype.

## The result

At 768 px, a plain grid VAE spends the same 2,304 tokens on every image. A blank wall costs as much as a Where's Waldo page.

Our tokenizer gives the median Mondrian **75 tokens** and the Waldo book page **1,764**, a gap of **23.5x** (95% bootstrap interval 19.2-29.4x, over the 13 Mondrians). 10 of 13 Mondrians clear the 16x bar of the bounty. Packed for an AR model, the extra tokens shrink and the median Mondrian is **67**, or 26.3x. Encoding takes **.226 s** and decoding **.094 s** at the median on one L4. Tokens are ordinary FLUX.2 latents of 128 floats, and a save-and-reload round trip is exact.

| Group | Images | Median tokens | Range | Page / group |
|---|---|---|---|---|
| Clean synthetic shapes | 8 | 44 | 38-66 | 39.6x |
| Mondrian | 13 | 75 | 47-370 | 23.5x |
| Rotated Mondrian, 15 and 45 degrees | 26 | 60 | 36-2,304 | 29.4x |
| Simple images | 8 | 172 | 18-784 | 10.3x |
| Unseen eval images, 5 complexity bins | 70 | 333 | 60-1,681 | 5.3x |
| Texture | 13 | 841 | 155-2,304 | 2.1x |
| Waldo-like scenes | 3 | 841 | 784-1,764 | 2.1x |
| Ornate | 6 | 930 | 409-1,296 | 1.9x |
| Dense paintings | 9 | 1,521 | 784-2,304 | 1.2x |
| Real document pages | 63 | 2,304 | 409-2,304 | below 1x |

Across the 52 main test images, tokens follow DReX complexity with a Spearman correlation of .85. The eval bins rise in order: 132, 234, 333, 538 and 841 tokens. The system is capped at the grid, so no image costs more than a plain VAE.

## How to read that number

Four things limit the headline, and the article should say so.

1. **The gap is partly chosen.** Two viewing verdicts anchor the curve: what the Mondrians and the page need to look right. A steepness setting (gap 1.25) then widens it, and a quality floor sets the page. The system decides where every other image falls. The Mondrian-to-page ratio itself is not an independent measurement.
2. **The eye is the rule, and it is forgiving.** The metrics are harsher than the eye on simple images. Mondrian whole-image DreamSim is .022, against .001 on dense paintings. Canvas weave softens and PSNR falls, but flat colour and lines hold. The project's reviewer passed Mondrians at about 90 tokens, called sheets down to about 47 "similar to the original", and saw no difference between the gap settings at 92, 60 and 47 tokens. The current default of 75 sits in that range.
3. **It is not grid quality by the metrics.** Median Mondrian DreamSim is .113 at 75 tokens and .010 at the full grid; PSNR is 33.8 against 38.9 dB. Side by side, lines and colour hold and the first four sheets look close to the grid. What goes is canvas cracks and weave. "Close to the grid" holds by eye, on the images we viewed, and not in scores. Broadway Boogie Woogie at 234 tokens rearranges some small squares.
4. **FLUX.2 cost some gap.** FLUX.1 reached 30.7x with its page at 2,304. FLUX.2 keeps more per token, so its page clears the quality floor at 1,764, and the ratio falls while each token holds twice the floats.

## The system

Image in, variable-length tokens out, image back.

**1. Budget.** Two cheap signals, DReX (a human-calibrated complexity model, with its last layers tuned for text) and texture energy, go through a six-weight curve fit on training images only. This costs milliseconds.

**2. Warp, below 200 tokens.** This is what makes the gap possible. Think of an accordion: rows and columns with no edges fold up, and rows and columns crossing edges stay at native scale. A Mondrian's flat paint folds away and its lines survive. The fold positions go into a header, and the image is resampled before the VAE, so latents are never edited. Above 200 tokens the warp mostly stops helping, so it is skipped.

**3. Real VAE.** The squashed image goes through the frozen FLUX.2 encoder as an ordinary image. Four latent cells pack into one 128-float token.

**4. Quality floor.** One decode and a PSNR check catch images the budget underrates: below 22 dB, the size grows 1.4x per side and it tries again.

**5. Learned decoder, at or below 600 tokens.** A 1.52M network, which we call S1, runs after the frozen decoder and repairs what stretching lost. It reads the decode, a map of how much each pixel was stretched, noise, and one statistics token per 256 px tile. Those 9 tokens are the only extra cost, and S1 is skipped on clean images, where it would invent texture.

Decoding needs nothing but the tokens.

## Tensor shapes, stage by stage

Printed from the running code for two images on the earlier FLUX.1 path: a Mondrian at 75 tokens and a page at the full grid. The batch size is 1 and the image side is 768. Budget and warp shapes are unchanged on FLUX.2; latent shapes are given after the tables.

**Budget.** The shapes are the same for every image, and the result is one number.

| Stage | Shape |
|---|---|
| Image, in [-1, 1] | [1, 3, 768, 768] |
| 9 tiles for texture energy | [9, 3, 256, 256] |
| DINOv3 at 224 px: 1 class, 4 register and 196 patch tokens | [1, 201, 384] |
| Class token kept | [1, 384] |
| ResNet-50 at 512 px, four stages | [1, 256, 128, 128], [1, 512, 64, 64], [1, 1024, 32, 32], [1, 2048, 16, 16] |
| Stages pooled and joined | [1, 3840] |
| Fusion | [1, 384] |
| Head: DReX score | [1, 1] |

The score and texture energy give a budget. The grid side is then g = round(sqrt(budget - 2)), so the VAE sees n = 2g latent cells per side and an image of N = 8n pixels per side. The Mondrian's budget of 59.5 gives g = 8, n = 16, N = 128. The page's budget of 1,739 first gives N = 672, and the floor grows it to 768.

**Encode.**

| Stage | Mondrian (warped) | Page (no warp) |
|---|---|---|
| Edge energy and density, per axis | [768] | not used |
| Knots, per axis | [65] | not used |
| Image entering the VAE | [1, 3, 128, 128] | [1, 3, 768, 768] |
| VAE latent | [1, 16, 16, 16] | [1, 16, 96, 96] |
| 2x2 cells packed | [1, 64, 8, 8] | [1, 64, 48, 48] |
| Latent tokens | [64, 64] | [2304, 64] |
| Stored knots, inner positions on 2 axes | [2, 63] | none |
| Plain decode, used by the floor and statistics | [1, 3, 768, 768] | [1, 3, 768, 768] |
| Statistics, 27 floats per tile | [9, 27] | none |
| **Tokens counted** | **64 + 2 + 9 = 75** | **2,304** |

**Packing for an AR model.** Inside the tokenizer the parts stay as three fields, and the count of 75 adds their costs on the old basis, 2 for knots and 9 for statistics. `Tokenizer.pack` writes one sequence: knots, statistics, latents, every token one VAE-token wide with unused floats zero. A layout written first (grid side, warped, statistics) says which is which. On FLUX.2, 126 knot floats take 1 token and 243 statistics floats take 2, so 75 becomes 67. The headline keeps the conservative count.

**Decode.**

| Stage | Mondrian | Page |
|---|---|---|
| Tokens back to a grid | [1, 64, 8, 8] | [1, 64, 48, 48] |
| Unpacked latent | [1, 16, 16, 16] | [1, 16, 96, 96] |
| VAE decode | [1, 3, 128, 128] | [1, 3, 768, 768] |
| Unwarped or upsampled: plain decode | [1, 3, 768, 768] | [1, 3, 768, 768] |
| S1 input: stretch map | [1, 2, 768, 768] | skipped |
| S1 input: statistics map | [1, 27, 768, 768] | skipped |
| S1 input: noise, fixed seed | [1, 1, 768, 768] | skipped |
| **Final image** | **[1, 3, 768, 768]** | **[1, 3, 768, 768]** |

**On FLUX.2.** The VAE latent has 32 channels, so the same Mondrian grid gives a latent of [1, 32, 16, 16], packed to [1, 128, 8, 8], and tokens of [64, 128]. Packed, the whole sequence is [67, 128]; the page is [1764, 128].

## What the learned decoder earned

S1 is a 1.52M network trained with L1, LPIPS, DISTS and a GAN; DreamSim and DINOv3 stayed out of the loss. It was retrained on FLUX.2 on the curated 6,000 images for 10,000 steps on an L4. Through the real encoder, plain against S1:

| Median | Plain | S1 |
|---|---|---|
| Mondrian tokens | 66 | 75 |
| Mondrian DreamSim, tiles (13 of 13 better) | .145 | .113 |
| Mondrian DINOv3, tiles (10 of 13 better) | .095 | .071 |
| Mondrian DreamSim, whole image (12 of 13 better) | .032 | .022 |
| Rotated 45 degrees, DreamSim | .212 | .133 |
| Eval, simplest bin, DreamSim | .155 | .125 |

Against those gains, the middle eval bin gets worse on DINOv3 (.031 to .034), and on non-axis shapes S1 fires on 9 of 16 and worsens DreamSim, although they look sharper. On a clean-shape control S1 fires 0 of 8 times. Above 600 tokens it loses, so it stays off there. A known artifact: a rippled red line on the Suprematist.

## Text was the blind spot

Text has low DReX and little texture energy, so the budget called it simple. At that budget OCR recovered 1-3% of the words. The VAE was not the cause: at the full grid, Tesseract reads 97-100% of the words at 15 px and above.

Text at low budget scores 15-18 dB, while Mondrians sit above 30 dB, so a PSNR floor separates them.

| Page (synthetic) | Tokens | Words recovered | Original |
|---|---|---|---|
| Slide, 44 px | 132 | 100% | 100% |
| Text with a picture, 15 px | 1,521 | 96% | 97% |
| Sans, 20 px | 1,681 | 98% | 99% |
| Serif, 15 px | 1,681 | 97% | 99% |
| Serif, 11 px | 2,304 | 45% | 58% |

The floor overshoots. An earlier version reached 95% on the serif page at 902 tokens.

**That fix did not reach real documents.** On 63 held-out pages (DocLayNet, FUNSD, synthdog-en and our own), the floor alone read very poorly. PSNR averages over all pixels, so thin ink on white can clear 22 dB and still be unreadable. So we tuned DReX to score text by what it needs. Score below is words recovered by IndraOCR against the original.

| On the same 63 pages | Pages at .9+ | Median words | Median tokens |
|---|---|---|---|
| Original DReX + floor | 8% | .43 | 784 |
| Tuned DReX + floor (default) | 60% | .94 | 2,304 |
| Plain full grid | 65% | .94 | 2,304 |

Those rows are on FLUX.1. On the final FLUX.2 system with S1, 45 of 62 pages read at .9+ (73%), median .94, against 44 plain.

The tune retrained DReX's last two parts on OCR-derived token needs, with a guard holding other images to their original score. It reads within .05 of the full grid on 52 of 63 pages, against 6 of 63 before. It failed its frozen rule on both tries, and the project's reviewer adopted it anyway on this evidence: the 80% bar sat above the full grid's own 65%, which we found only after the result. The cost is Mondrians moving from 60 to 75 tokens. Weights are public at [tensorkelechi/DReX-text-tuned](https://huggingface.co/tensorkelechi/DReX-text-tuned). Printed documents only: scene text and receipts are untested, and Indra uses language models, so it may forgive errors a reader would not.

## Speed

Measured on one L4, one 768 px image at a time, full FLUX.2 VAE, with the tile DReX pass and S1.

| | Median | p90 |
|---|---|---|
| Encode | .226 s | .74 s |
| Decode | .094 s | .43 s |

Encode was .088 s before tile DReX, on a faster path; the project's reviewer accepted the cost, and it can be optimized later. Busy images are slower because the floor retries at larger sizes.

## Choice of VAE

The same tokenizer ran on three VAEs at equal tokens, using the earlier budget curve. Round-trip difference was 0 on all three.

| DreamSim, tiles | FLUX.1 | Qwen-Image 2.1 | TAEQI 2.1 |
|---|---|---|---|
| Mondrian | .136 | .115 | .150 |
| Simple | .073 | .076 | .122 |
| Texture | .047 | .076 | .075 |
| Ornate | .010 | .023 | .043 |
| Dense | .030 | .042 | .060 |

This comparison predates FLUX.2, which is now the default and beats FLUX.1 at equal tokens below. Qwen is close on simple images and weaker on complex ones. TAEQI is weaker in most groups but very fast to decode.

## An AR model can write these tokens

**Overfit.** A 4.84M plain AR on packed tokens, overfit on 4 images in 262 s, wrote exact layouts and lengths from a label. Generated decodes sat at DreamSim .115 against .101 for true tokens, DINOv3 .083 against .081, and read correctly by eye.

**A first MAR.** A 23.7M MAR-style model conditioned on CLIP image embeddings passed its numeric bars: lengths varied 17-961, re-encoding agreed with a Spearman of .90, held-out loss matched training. It failed where it mattered. True, wrong and empty conditions gave held-out losses within .03% of each other, and one seed under different conditions gave nearly the same picture: CLIP set the length and nothing else. Sampling settings did not fix that.

**The released MAR.** A 57.1M model at 512 px, conditioned on four content labels (simple, moderate, textured, dense) that were assigned from source, style and genre before anything was tokenized: procedural flats, Minimalism and Color Field for simple; abstract and cubist painting for moderate; DTD textures, Impressionism and landscapes for textured; Baroque, Northern Renaissance, Mannerism and Romanticism for dense. The label is added to every token and to the flow head's condition, so it cannot be lost among hundreds of tokens. Other changes from the first MAR: a 1024-wide head, an EMA of the weights (.9999), warmup then a constant rate, temperature .95 at sampling, 10% label dropout. It writes a layout first, which fixes the token count, then knots and statistics, then latents in random order over 64 steps, each token drawn with 32 flow steps.

Data: 12,000 images encoded with their flips for the first 30,000 steps on one A100, then 34,795 more for steps 44,000 to 100,000, 92,789 sequences in all against 800 held out. 9.8 steps a second at 12,000 tokens a batch, about 3 hours and 25 Colab units in all.

| Bar, frozen before training | EMA weights at 100k |
|---|---|
| All 64 samples decode | yes |
| Median length rises across the labels | 44, 212, 650.5, 625: textured and dense tie within 4%, fail |
| p75/p25 at least 1.5 in 3 labels | 3.49, 2.43, 3.16, 2.07 |
| Re-encoding agrees on length, Spearman at least .5 | .96 |
| Held-out loss within 20% of training | 1.219 against 1.270 |
| The true label beats a wrong one on held-out loss | by 2.6% |
| One seed under each label gives different pictures | yes |

At 30,000 steps the EMA still carried much of the untrained start (held-out 1.410 against 1.329 for the live weights); by 86,000 it had passed them (1.233 against 1.253), and at 100,000 it is the better model on every measure (live: Spearman .82, medians 44.5, 212, 362, 291). By eye, simple samples are colour fields, stripes and Mondrian-like black grids with red and blue blocks, moderate ones abstract figure compositions, textured ones landscapes with trees and skies, dense ones dark portraits and crowded figure groups whose faces are almost there. There are no clean recognizable objects, which is about what 57M parameters and 47,000 images allow. What it shows is that a small model can learn to choose its own length from content and write all three kinds of token so they decode.

**A correction worth keeping.** The first MAR gate passed its median while warped images failed badly, DreamSim up to .658. Knots came first in the order, so they were rarely a training target, and one knot token was averaged among hundreds of latents. Giving them their own training phase and loss term fixed most of it. Warped images stay 22-31% above true decodes in the overfit, and PSNR falls hard there (25.2 to 16.7 on one), because small knot errors move edges.

## How close to the full grid

Every image at 768, 259 in all, went through the plain FLUX.2 grid and through `Tokenizer()`, and the two decodes were compared after rounding to 8 bit: DINOv3 ViT-B/16 cosine averaged over patches and on the class token, LPIPS (AlexNet), PSNR. The worse tail is the 10th percentile of similarity and PSNR and the 90th of LPIPS. Tokens here are packed counts.

| Group | n | Tokens | DINOv3 patch, median / worse 10% | DINOv3 global | LPIPS | PSNR dB |
|---|---|---|---|---|---|---|
| Clean shapes | 8 | 44 | .990 / .982 | .992 / .981 | .017 / .026 | 35.5 / 34.2 |
| Rotated Mondrian | 26 | 52 | .895 / .826 | .743 / .615 | .180 / .288 | 30.6 / 26.5 |
| Mondrian | 13 | 67 | .930 / .909 | .850 / .775 | .156 / .226 | 33.7 / 26.2 |
| Simple | 8 | 165 | .952 / .906 | .898 / .803 | .169 / .235 | 29.1 / 23.1 |
| Unseen eval | 70 | 326 | .965 / .943 | .924 / .887 | .140 / .215 | 26.1 / 22.4 |
| Texture | 13 | 784 | .977 / .939 | .952 / .793 | .188 / .302 | 26.3 / 22.8 |
| Waldo-like | 3 | 841 | .990 / .989 | .986 / .974 | .104 / .238 | 27.8 / 23.3 |
| Ornate | 6 | 930 | .992 / .986 | .989 / .978 | .030 / .089 | 35.1 / 30.1 |
| Dense paintings | 9 | 1,521 | .993 / .988 | .993 / .988 | .063 / .143 | 31.0 / 28.3 |
| Document pages | 63 | 2,304 | identical / .990 | identical / .977 | 0 / .056 | identical / 28.2 |
| All | 259 | 326 | .972 / .912 | .948 / .776 | .102 / .223 | 30.4 / 23.4 |

Busy images keep most of their tokens and sit close to the grid by every measure. Cheap images keep lines and colour, so PSNR stays above 30 dB on Mondrians, while global DINOv3 and LPIPS see the lost canvas weave and cracks. Against the original, our decode scores nearly the same as against the grid (LPIPS .103 against .102), because the grid itself is near the original (LPIPS .007): the gap is the tokens we drop, not the VAE. Images at the full 2,304 decode exactly as the grid does.

DReX, as the budget reads it, rises with the groups: medians .20 for rotated Mondrians, .22 Mondrians, .27 clean shapes, .35 simple, .43 texture, .48 eval, .56 ornate, .71 Waldo-like, .80 dense and 1.04 documents, which score highest because DReX was tuned on text. Tokens are not DReX alone: one texture photo at DReX .39 takes all 2,304 tokens through texture energy and the PSNR floor.

## FLUX.2 baseline

Before retraining on FLUX.2, we measured it plain, on 52 test images, 70 eval images and 63 text pages, at equal token counts. One FLUX.2 token holds 128 floats against 64.

| Median at 676 tokens, plain decoder | FLUX.1 DreamSim | FLUX.2 DreamSim | FLUX.1 PSNR | FLUX.2 PSNR |
|---|---|---|---|---|
| Mondrian | .056 | .055 | 33.2 | 33.2 |
| Simple | .040 | .036 | 32.5 | 33.0 |
| Texture | .073 | .059 | 23.8 | 24.3 |
| Waldo-like | .071 | .068 | 26.6 | 27.3 |
| Dense | .089 | .065 | 25.9 | 26.8 |
| Eval images, middle bin | .065 | .050 | 27.3 | 28.0 |

| Pages reading at OCR .9+ | FLUX.1 | FLUX.2 |
|---|---|---|
| 1,296 tokens | 23% | 37% |
| Full grid | 66% | 74% |

Speed is the same. FLUX.2's weakness is on documents at the full grid: DINOv3 scores it worse on 58 of 63 pages, because blank white carries a faint tint and mottling, visible only when amplified 12x. Without S1 on the default path, FLUX.2 gives a median Mondrian of 66 tokens and a page of 1,764, or 26.7x (interval 21.3-34.6x), and 10 of 13 clear 16x. The page clears the floor at 22.3 dB and looks right at native crop.

## Testing localized allocation

The budget is one number per image, so a page with a small dense region can be underrated. We tested two fixes against rules frozen beforehand, on 62 text pages, with near-grid meaning within .05 of the full grid's OCR.

| FLUX.2 | Pages near grid | Under .5 | Simple tokens | Eval tokens | Encode |
|---|---|---|---|---|---|
| Current budget | 55 | 3 | 136 | 272 | .088 s |
| Highest of 9 tile DReX scores | 55 | 2 | 164 | 324 | .175 s |
| Warp at all sizes | 56 | 1 | 136 | 274 | .092 s |
| Both | 58 | 1 | 164 | 326 | .184 s |

No option passed. Warping above 200 tokens lowers DreamSim and DINOv3 across many groups, so it is out. Tile DReX misses on encode time, against a .15 s target on an L4. It also misses on one texture score, though that comes from one image landing lower on the floor's growth steps, and 4 of the 5 changed texture images improve. Viewing favours it: a garbled manual page at 400 tokens goes to 1,444 and OCR rises from .02 to .90. It costs simple images 21% more tokens and leaves Mondrians unchanged.

## Should the encoder be tuned?

No encoder can beat latents optimized directly per image against the frozen decoder, so 300 Adam steps per image bound what tuning could gain, for about an hour of compute. The bars were frozen beforehand, and a score in a loss could not judge that loss.

| Median, non-text | DreamSim | DINOv3 | LPIPS | DISTS | PSNR |
|---|---|---|---|---|---|
| Encoder + S1 | .100 | .044 | .332 | .144 | 28.3 |
| Optimized, DreamSim + DINOv3 | .016 | .007 | .311 | .205 | 28.1 |
| Optimized, DreamSim + DISTS | .027 | .033 | .291 | .051 | 27.5 |

An earlier run, optimizing LPIPS + DISTS on 20 non-text images, is not in this table: it took DreamSim from .064 to .070 and DINOv3 from .027 to .022.

All failed. LPIPS + DISTS got worse on DreamSim, unseen in that loss, and Mondrians got clearly worse. DreamSim + DINOv3 got worse on held-out DISTS, better on 3 of 15. DreamSim + DISTS got worse on held-out DINOv3 on Mondrians, .119 against .069. Optimized latents plus S1 lost 1.7-2.2 dB of PSNR. OCR rose at most .03, against the .10 asked. By eye, scores halve while flat areas gain mottled blotches, so this looks more like fitting the judge than a better picture. The encoder stays as FLUX.2 ships it. These are three losses and not every loss, though the same pattern held for each.

## What failed

These are worth reporting, because they shaped the design.

- **Learned warp (S2).** Direct optimization of one image's warp gained 0-1.5% at equal tokens, against a 10% bar, and adding rotation made it worse. The optimizer itself proved weak: from a uniform warp it reached q .627 where the classical warp has .532. A search over the warp's own settings, without gradients, gained about 2% at the median. The classical warp stays. A bendable 2D warp is untested and would cost about 8 header tokens.
- **Neural budget head (S3).** A 100k head got 1.08x on training images and 1.86x on held-out ones, against a 1.3x bar. Six weights on DReX and texture beat it on consistency, with Mondrian spread falling from 14.5x to 5.1x. They still missed the bar at 1.53x. On flat images quality barely changes with budget, so small errors in quality become large errors in tokens.
- **Classical rotation fix.** It found the right angle and scored worse at the same budget, since rotating resamples the image.
- **Per-tile complexity.** DReX tracks tokens at image level (about +.75) but barely within an image (under +.24), and splitting by tile complexity added nothing over an even split.
- **Stability of the earlier best.** An older 13x best collapsed to about 4x under a small tilt, and the median sat near 10x. It failed its frozen rule and led to the current whole-image warp.
- **LLM viewers.** Haiku pass rates stayed flat from 145 to 400 tokens, and one passed a Broadway decode we saw fail. Viewing stayed with a person.

## Known weaknesses

- **Allocation is still one number per image.** A few pages slip through with a small text region or thin ink: 2 of 62 pages read under half their words on the final system. A worst-tile PSNR check catches more of the unreadable decodes (77% against 56%) on saved images, but fires on half the simple images at that threshold, so it is not adopted.
- **Rotated outliers.** Median 60, but a rotated van Doesburg goes to the full 2,304, because the floor reacts to diagonal lattices. Whole-image DreamSim on rotated Mondrians is .023-.026 with S1, about the same as upright.
- **Blunt floor.** Gravel and an aerial field also go to the full grid, which may be more than the eye needs.
- **One frozen check narrowly failed.** Removing judges from encoding was meant to keep every group within 10%. Simple images miss on DINOv3 (.060 vs .050), and synthetic grain misses on DreamSim (.055 vs .048).
- **Settings tuned on test images.** The 200-token warp cutoff and the PSNR floor were set while looking at these images.
- **Round trip checked lightly.** Save-and-reload was exact on the final default for 10 images, difference 0.
- **Close to the grid by eye, not by every score.** On cheap images LPIPS against the grid decode is .14-.18 at the median, and rotated Mondrians fall to .74 global DINOv3. Only Mondrians and text pages were viewed against grid decodes on the final system.
- **FLUX.2 tints blank pages.** A faint tint and mottling on white, visible only amplified 12x, costs it on DINOv3 on documents.
- **Licences.** FLUX.2-dev weights carry their own licence, which also bears on using S1 and the MAR, both trained on its latents.

## Method notes

- **Frozen rules.** Pass bars were written down before runs, and every amendment made after seeing a result is labelled as such.
- **Overfit first.** Every trained part had to overfit before it could train for real. That saved the full run on several parts.
- **Judges as screens.** DreamSim and DINOv3 are in no shipped model's loss, only in the one test of what encoder tuning could gain. Viewing has the final say.
- **Budget.** About $23 of Modal and about 44 Colab compute units over the project, 25 of them for the released MAR.

## Pointers

The repository is [github.com/kelechi-c/retina](https://github.com/kelechi-c/retina): `retina/` (the tokenizer, `Tokenizer()`), `mar/` (the MAR model, encoding, training, sampling), `eval/evaluate.py` (tokens and closeness to the grid on any grouped folder), `tests/`, and a README with commands that were run as written on a Colab L4 and on Modal. Weights: `s1_flux2.pt` and `mar_v2.pt` in [tensorkelechi/retina-tokenizer](https://huggingface.co/tensorkelechi/retina-tokenizer), and [tensorkelechi/DReX-text-tuned](https://huggingface.co/tensorkelechi/DReX-text-tuned).

Default call: `Tokenizer()`.
