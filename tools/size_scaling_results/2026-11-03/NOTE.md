# Size-scaling run, 2026-11-03

Run of `tools/fit_size_scaling.py` per `tools/size_scaling.md`. Measures only; no tokenizer code
was changed, and nothing here is applied. Primary criterion (the script's default order):
**`lpips_native`**.

## Setup

- **GPU / wall time**: one Modal L4 (ephemeral), ~65 min total: ~25 min cold start downloads
  (FLUX.2-dev VAE, DReX, tuned DReX, LPIPS AlexNet into an empty `/cache` volume) and ~38.5 min
  measuring at **9.2 s/image**. That is far below the notes' 20–60 s/image estimate, so a run of
  this size is cheaper than planned; `--ratio 1.2` is what made it fast.
- **Exact command**:
  ```
  # stage the images (see below) inside the container first:
  python tools/size_stage_images.py --shards 0,3 --per-group 40 --out /runs/size_images2
  # then the measurement:
  modal run -d tools/modal_run.py \
    --cmd "python tools/fit_size_scaling.py OUT --images /runs/size_images2 --procedural 40 \
           --sizes 512,768,1024 --ratio 1.2" \
    --run size_run1 --gpu L4
  ```
- **Images**: 251 — 211 curated + 40 procedural (`mar.data.procedural`, 1024 px, flat).
  - Source of the 211: the project's own curated fine-tune pool, private HF
    `tensorkelechi/retina-inputs:curated/shard-00000.tar` and `shard-00003.tar`. Not the test,
    eval or document images the notes exclude.
  - Per group: div2k 40, met 40, oi 40, sa1b 40, wikiart 40, synth 11, procedural 40.
    The 12 shards contain 2,774 images ≥1024 px; two shards hold 455, of which the 40-per-group
    cap keeps 211 for balance. Not used: 545 members of those shards skipped for short side
    <1024 px (DIV2K/Synth are the PNGs that mostly pass; small OI/SA-1B crops do not).
  - 209 training / 42 held out (hash of the image id, as the script does).
- **Artifacts**: `report.md`, `fit.json`, `per_image.csv`, `images.jsonl`, `settings.json`,
  `examples.json`, `sweep.jsonl` (2.9 MB, under the 20 MB limit).

## The six checks from the notes

1. **Budget inputs read at each side — pass.** Median |Δd| 0.0044 at 512 and 0.0007 at 1024
   (limit 0.02); median |Δtexture| 0.00120 and 0.00038 (limit 0.002). The `BUDGET_SIDE = 768`
   copy in `Tokenizer.budget` is doing its job.
2. **Sanity at 768 — pass.** Median measured/shipped tokens 1.000 for all four criteria,
   2% of points censored (lpips criteria), 0% for psnr.
3. **Censoring — pass.** "below" is 2–5% of points (largest: 13/251 at 512 for `lpips_display`),
   well under the fifth that would mean the target is too easy. "above" is rare: 1–3 points at 512,
   ≤2 at 1024. Fewer "above" at 512 than expected for busy images: at 512 the full grid is 1,024
   tokens, which nearly reconstructs the image, so it almost always beats the 768 target — the fit
   is effectively driven by the 1,024 side.
4. **Held-out error — fitted wins 3 of 4 criteria, loses `lpips_display`.**
   Fitted vs "no scaling" vs "one gamma" (median |log tokens|): `lpips_native` 0.236 / 0.466 / 0.293,
   `psnr_native` 0.171 / 0.181 / 0.252, `psnr_display` 0.226 / 0.231 / 0.284, `lpips_display`
   **0.324 / 0.228 / 0.301** (fitted worst; train R² is negative, −0.106). The display view is not
   explained by this linear form; the primary criterion is the native view, where the fit is clearly
   better than both baselines.
5. **Gamma by group — native view mostly in range, one model-form problem to report.**
   Measured per-image gamma (median, 512 / 1024): procedural **0.40 / 0.24** (flat, near 0, as
   expected), div2k 1.82 / 1.83, met 2.08 / 2.63, sa1b 2.00 / 1.94, oi 1.26 / 0.79,
   wikiart 1.07 / 0.62 — busy groups inside or just above the 0.5–2 range. Synth is a wild outlier
   (1.23 / 12.96) on n=2–8, ignored. Display view is smaller and turns negative at 1,024
   (−0.19 to −0.89), which the fit does not represent.
   **Report, not tuned:** the fitted `g0` is 1.286, so the model predicts gamma ≈ 1.20 for flat
   images whose measured gamma is 0.24–0.40 — at 1,024 it would give a Mondrian ~1.4x more tokens
   than the measurement says it needs. The fit is dominated by busy images and the `(t, d)` plane
   cannot make gamma vanish as `t, d → 0`. Before the apply step: clamp gamma (e.g. to [0, 2]) or
   add a low-complexity intercept/floor, and check the example budgets it implies
   (report table: mondrian_piet 34 → 52 tokens at 768 → 1,024, bruegel 1,586 → 2,626).
6. **One sweep by eye — pass.** At 1,024: flat `synth-0266` LPIPS 0.007 → 0.005 (flat, upticks only
   at 0.00–0.01); dense `div2k-0198` 0.847 → 0.005, monotone, 0 upticks. No visible step at the
   200-token warp cutoff.

## Unexpected

- Two duplicate detached apps ran because the Modal client on this host kept timing out on the
  stream (`ConnectionError: Deadline exceeded`); one was stopped, the surviving run produced these
  artifacts, and `images.jsonl` was checked for duplicates: 251 rows, 251 unique ids.
- `modal run` with a `bash -lc "…"` wrapper loses the image's uv venv PATH (no `huggingface_hub`).
  Run each step as its own Modal command.
- The 512 side is nearly uncensored "above", so the fit effectively comes from 768 vs 1,024; a
  second run at 512/768/1024/1536 on larger images would test the extrapolation.
