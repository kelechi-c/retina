# Does the encoder still need its PSNR floor? Run notes

These notes are for whoever runs `tools/test_floor.py`, person or agent. The script only measures; it changes no code.

## The question

After the budget picks a grid, the encoder decodes once and checks PSNR against the original. Under 22 dB it grows each side 1.4x and tries again, up to the full grid (`Tokenizer._encode_fast`). The floor was added because text read as simple to the budget. DReX has since been tuned for text, so the floor may now be doing little.

What the code and the write-up already say:

- **The original is only needed at encode time**, where the encoder has it anyway. Decoding never needs it, and the MAR's samples never go through the floor.
- **The first check is free for images that get refiner statistics** (600 tokens or fewer). The statistics are computed from the original minus the plain decode, so that decode happens anyway. The floor's real cost is the retries, plus one decode for images above 600 tokens, which would otherwise need none.
- **Tuned DReX alone is close to the full grid for a typical page.** At the median document score, DReX 1.04, the curve gives 1,936 to 2,304 tokens depending on texture energy. So on most pages the floor has little left to add.
- **But a page scored lower is not.** At DReX 0.6 to 0.9 the curve gives 225 to 1,024 tokens, and only the floor would catch such a page if it is unreadable there.
- **The write-up never ran tuned DReX without the floor.** Its document table compares original DReX + floor, tuned DReX + floor, and the plain full grid.
- **The floor also acts on images that are not text.** Gravel, an aerial field and a rotated van Doesburg went to the full grid through it, which the write-up lists as possibly more than the eye needs.

## What the script does

For every image, at 768 px by default:

1. Encodes with the floor on (the shipped encoder), counting the tries and the time.
2. Encodes with the floor off (`floor=0`): the budget's grid is final.
3. Decodes both and scores the final images, with the refiner where it applies: PSNR against the original and against the full-grid decode, and LPIPS likewise. Optionally DreamSim, and optionally OCR (Tesseract): the share of the original's words found in each decode, and in the full-grid decode for reference.
4. Records a third policy, **free**: keep the check only where the decode is needed anyway (the budget's grid gets refiner statistics), and drop it elsewhere.

Times for "off" leave out the one decode the current code still does when no statistics are needed, since a floorless encoder would skip it.

The report gives, per group: how many images the floor moved, by how many steps, tokens and time with and without it, and quality on the moved images. Then a table of the three policies, and every moved image ranked by how much the floor helped it.

## Requirements

- A CUDA GPU with at least 24 GB (an L4 is enough), `uv sync`, and Hugging Face access to the gated `black-forest-labs/FLUX.2-dev` (`uv run hf auth login`). Network access to huggingface.co, media.githubusercontent.com and download.pytorch.org.
- For `--ocr`: the Tesseract binary (`apt install tesseract-ocr`, or `brew install tesseract`) and `uv pip install pytesseract`. The write-up used IndraOCR for its document numbers; Tesseract is stricter, so compare on and off with each other, not with the write-up.
- For `--dreamsim`: `uv sync --extra judges`.

## Images

This is an evaluation, so the write-up's test sets are the right images here. Put them in one folder with a subfolder per group, for example:

```
eval_images/
  mondrian/  rotated_mondrian/  clean_shapes/  simple/  texture/  ornate/  dense/  waldo/
  eval/        the 70 unseen images
  documents/   the 63 pages (DocLayNet, FUNSD, synthdog-en, our own)
  hard/        gravel, the aerial field, the rotated van Doesburg, any page that read badly
```

The documents group matters most. Include the pages the write-up found reading under half their words.

## Running

Smoke test first, a few minutes:

```bash
uv run python tools/test_floor.py floor_out/smoke --images assets/examples --limit 4
```

The real run:

```bash
uv run python tools/test_floor.py floor_out/run1 --images /path/to/eval_images --ocr --save
```

- Expect roughly 1 to 3 seconds per image on an L4, more with `--ocr` on documents. About 300 images take 15 to 30 minutes.
- If it stops, run the same command again; finished images are skipped.
- `--report-only` rebuilds `report.md` from `results.jsonl`.
- `--save` writes both decodes of every moved image to `moved/`, for looking at by eye.

## Decision rules: agree on these before running

Proposed. Adjust them if you want, but write them down before seeing any results and do not change them afterwards.

1. **Documents.** Under "off", the number of pages reading at OCR .9 or better is within 1 of "on", the median OCR is within .02 of "on", and no page drops by more than .2.
2. **Everything else.** On images the floor moved, the median LPIPS against the full-grid decode is at most .01 worse with the floor off, and no single image is more than .05 worse.
3. **By eye.** The five moved images that the floor helped most (top of the report's last table, files in `moved/`) still look acceptable with the floor off.

- If "off" passes all three, the floor can go, or become an option that is off by default.
- If only "free" passes, keep the check only where its decode is free.
- If neither passes, keep the floor as it is.

## What not to do

- Do not change anything in `retina/`, and do not change the decision rules after seeing results.
- Do not push to `main`. Do not commit images, model weights or the `moved/` folder.

## What to hand back

`report.md`, `results.jsonl` and `settings.json` from the output folder, and the `moved/` images separately if there are any. Either give them to the person who asked, or commit the three files under `tools/floor_results/<date>/` on a new branch named `floor-results`.

Add a short note with:

- the GPU, the time and the exact command;
- the image groups and their counts;
- the result of each decision rule, in one line each, including your by-eye verdict;
- anything odd.
