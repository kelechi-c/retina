# Is the 22 dB PSNR floor still needed? (2026-11-04)

A/B with the shipped default `Tokenizer()` (text-tuned DReX, S1 refiner): same image, `floor=22`
vs `floor=0`, at 768. 419 image/side pairs: the 90 DocLayNet pages, 90 FUNSD forms, 90 SynthDog text
docs, the 5 rendered text pages, the 52 test images, 68 rotations and 24 synthetics from
`retina-inputs:inputs.tar`. Modal A100, ~12 min. Script `check_floor.py`.

## Answer

**Keep it.** It changes the budget on only **10/419 pairs (2.4%)**, but on those it is doing real work:

- median on the active ones: tokens 784 -> 2,304, PSNR 19.4 -> 22.2 dB, **LPIPS 0.434 -> 0.138**
- worst cases: gravel road texture 538 -> 2,304 (PSNR 15.4 -> 22.2, LPIPS 0.194 -> 0.042);
  Dülmen fields 784 -> 2,304 (LPIPS 0.663 -> 0.080); van Doesburg rotations 900 -> 2,304
  (LPIPS 0.43 -> 0.14); rendered text pages 625-961 -> 1,225-1,849 (LPIPS 0.39 -> 0.09).

The text tuning did remove the floor's original reason for **documents**: all 270 DocLayNet/FUNSD/
SynthDog pages are inactive (already at 2,304 tokens, PSNR 32.5-35 dB) without the floor. But it is
still active for **rendered text pages** (3/5), texture (2/13), rotations (4/52) and one simple image.

## Cost

- Encode: median 123 -> 139 ms (**+12%**), i.e. the extra decode per growth iteration; on active
  cases +40-90% (text 104 -> 190 ms; gravel used 4 probes). **Decoding and generation are
  untouched** — the floor lives only in `encode` and nothing about it is in the token payload.
- It needs the original image, but only at encode time (it decodes and compares to the input).
- Independent waste worth fixing: `_encode_fast` always decodes once *before* checking whether the
  grid is already at max or whether the refiner needs the decode for its statistics. Reordering that
  check (skip the decode when `cell*n >= S` and the refiner will not use it) would cut ~40% off
  encode time for the document/busy class (229 ms -> ~140 ms measured components) without changing
  any output. Do this instead of removing the floor.

## What removing it would change

Nothing in the public headline numbers: Mondrians (75), documents (2,304) and Waldo pages (841) are
all floor-inactive, so the README/article numbers do not depend on it. Removing it would only change
the 2.4% above — the texture / rendered-text / line-art / rotated classes in the eval table — for the
worse (e.g. gravel LPIPS 0.042 -> 0.194, rendered text 0.09 -> 0.39).
