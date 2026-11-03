# Size scaling of the token budget: run notes

These notes are for whoever runs `tools/fit_size_scaling.py`, person or agent. They say what the script measures, what it needs, how to run it, what to check, and what to hand back. The script only measures; it changes no code.

## Why this exists

The budget curve in `retina/tokenizer.py` (`REFIT`, `PIVOT`, `gap`) was fit at 768 px. It returns a token count, not a share of the grid, so at other sizes the grid grows or shrinks around a fixed number:

| Bruegel, before this work | 512 px | 768 px | 1,024 px |
|---|---|---|---|
| Budget, tokens | 2,018 | 1,700 | 1,504 |
| Share of the grid | 100% (capped at 1,024) | 73% | 37% |

Part of that came from the budget's inputs drifting with size: texture energy uses a 13-pixel window and tiles are 256 px, so both read differently at other sizes. That part is fixed: `Tokenizer.budget` now reads DReX and texture energy from a 768 px copy of the image (`BUDGET_SIDE`), so the inputs are the same at every size.

What remains is the output: how many more (or fewer) tokens an image should get at a side S than at 768. Flat paint should need about the same at any size; fine texture should need roughly as many more as there are extra pixels, if it is to be kept. This script measures that.

## What the script measures

For each image:

1. At 768 px it runs the shipped budget (DReX and texture energy, the curve, the gap) to get the shipped grid, decodes at that grid, and records the decode's quality against the original. That quality is the target. It is what the shipped system delivers at 768, so 768 stays exactly as it is.
2. At each side S (default 512, 768, 1,024) it sweeps grid sizes from 2 × 2 up to the full grid (about 30 sizes, a factor 1.12 apart), decodes at each, and finds the fewest tokens N_S whose decode reaches the target. It interpolates between the two grids either side, on the envelope of the sweep so that a dip from the warp cutoff cannot count twice.
3. The decode path is the shipped one with the floor and the refiner off: warp below 200 tokens (`WARP_MAX`), FLUX.2 encode and decode, unwarp. The refiner and the 22 dB floor act after the budget and are not what is being fit.

Quality is judged by two criteria and two views, four in all:

- `lpips_*`: LPIPS (AlexNet), lower is better. `psnr_*`: PSNR in dB, higher is better.
- `*_native`: decode and original compared at the image's own size.
- `*_display`: both resized to 768 first, as if shown at the same size on screen. At 768 the two views are the same.

Then it fits, on training images only (four in five, chosen by a hash of the image id):

```
log(N_S / N_768) = (g0 + g1 * t + g2 * d) * log(S / 768)
```

with `d` the budget's DReX score and `t` = 10 × texture energy. `gamma = g0 + g1 t + g2 d` is the size exponent for that image: 0 means the same tokens at every size, 2 means tokens grow with pixel count. It reports the fit for each of the four criteria, with three baselines on the held-out images: no scaling, area scaling (gamma 2), and one gamma for all images.

Censoring: if even the smallest grid reaches the target, the point is "below"; if even the full grid misses it, "above". Both are left out of the fit and counted in the report. Busy images at 512 px are often "above", because the 1,024-token grid at 512 cannot match what 1,700 tokens do at 768. That is expected.

## Requirements

- A CUDA GPU with at least 24 GB (an L4 is enough; an A100 is about twice as fast).
- `uv sync` in the repository root.
- Hugging Face access to `black-forest-labs/FLUX.2-dev`, which is gated: accept the licence on its model page, then `uv run hf auth login`.
- Network access to huggingface.co (FLUX.2 VAE, `tensorkelechi/DReX-text-tuned`, and `huggan/wikiart` if `--wikiart` is used), media.githubusercontent.com (DReX weights), and download.pytorch.org (AlexNet weights for LPIPS).
- Disk: a few GB if `--wikiart` is used (parquet shards of about 400 MB each are downloaded until each label has its quota).

## Choosing images

This decides whether the fit means anything. Coverage from flat to dense matters more than count.

- Every image must be at least as large as the biggest side measured (1,024 px on its short side by default). Smaller images are skipped and logged; nothing is upsampled.
- Best: the training images the 768 curve was fit on, from the project's local archive, if they exist at 1,024 px or more. Pass the folder with `--images DIR`; its first level of subfolders becomes the group names in the report.
- Otherwise: `--procedural 40 --wikiart 160`, plus a folder of your own at 1,024 px or more covering what those lack: texture photos (gravel, foliage, fabric, bark), scanned or rendered document pages, and plain photographs. Keep them in subfolders by kind (`textures/`, `documents/`, `photos/`).
- Aim for 200 to 300 images in all.
- Do not use the test images: `assets/examples` (outside the smoke test), the 52 main test images, the 70 eval images or the 63 document pages from the write-up. They are for checking the result afterwards.

## Running

Smoke test first, about 5 to 10 minutes. It checks downloads, the GPU path and LPIPS; its numbers mean nothing.

```bash
uv run python tools/fit_size_scaling.py size_out/smoke --images assets/examples --procedural 4 --limit 6
```

`mondrian_b144.jpg` is 978 px on its short side and is skipped; that line in the log is expected. The run ends by printing `report.md`.

The real run:

```bash
uv run python tools/fit_size_scaling.py size_out/run1 --images /path/to/images --procedural 40 --wikiart 160
```

- Expect roughly 20 to 60 seconds per image on an L4, so 200 images take about 1.5 to 3.5 hours. `--ratio 1.2` sweeps fewer grid sizes and roughly halves the time, at some cost in precision.
- If it stops, run the same command again. Images already in `images.jsonl` are skipped.
- To refit without a GPU from finished measurements: `uv run python tools/fit_size_scaling.py size_out/run1 --fit-only`.
- Optional, if enough images are 1,536 px or larger: add `--sizes 512,768,1024,1536` in a separate output folder.

## Checks before handing back

Read `report.md` and check each of these. If one fails, report it as it is; do not change the script, the model form or the images to make it pass.

1. **Budget inputs read at each side.** Median absolute change in d under 0.02 and in texture under 0.002 at every side. If not, the 768 copy in `Tokenizer.budget` is not working; stop and report.
2. **Sanity at 768.** Median measured / shipped between 0.8 and 1.0, censored share near 0%. Below 1 is possible because a smaller grid can occasionally match the shipped one.
3. **Censoring.** Note the below / above counts. Many "below" (more than a fifth of points) would mean the target is too easy and is unexpected. "Above" at 512 for busy images is expected.
4. **Held-out errors.** Note whether "fitted" beats "no scaling" and "one gamma" for each criterion. Do not tune anything to make it win.
5. **Gamma by group.** For the native view, flat groups (procedural, Mondrian-like) are expected near 0 and texture-heavy or dense groups between 0.5 and 2. The display view should give smaller values. Report anything far outside this rather than adjusting.
6. **One sweep by eye.** In `sweep.jsonl`, look at one flat and one dense image at 1,024: quality should mostly improve as tokens grow, with at most a small step at the 200-token warp cutoff.

## What not to do

- Do not edit anything in `retina/`, including the curve (`REFIT`, `PIVOT`, `gap`), and do not apply the fitted numbers to the tokenizer. That is the next step and is reviewed separately.
- Do not push to `main`.
- Do not commit images, model weights or `sweep.jsonl` if it is over 20 MB.

## What to hand back

From the output folder: `report.md`, `fit.json`, `per_image.csv`, `images.jsonl`, `settings.json` and `examples.json` (and `sweep.jsonl` if under 20 MB). Either give them to the person who asked for the run, or commit them under `tools/size_scaling_results/<date>/` on a new branch named `size-scaling-results` and push that branch.

Add a short note with:

- the GPU, the total time, and the exact command;
- the images used: where they came from, how many per group, and how many were skipped and why;
- the result of each check above, in one line each;
- anything unexpected: errors, very slow images, odd sweeps.

## What happens with the results

Not part of this run, for context only. One criterion will be chosen (LPIPS native is the likely default). `Tokenizer.budget` will then multiply its result by `(S / 768) ** gamma(d, t)`, and the thresholds tied to 768 will be checked at other sizes: the 200-token warp cutoff, the 600-token refiner limit (the refiner was trained at 768), the 22 dB floor and the 6-pixel filter radius. After that come the GPU test suite and the write-up numbers at the new sizes.
