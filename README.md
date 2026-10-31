# retina

An image tokenizer whose token count follows the image's content, built on a frozen grid VAE. A simple image takes far fewer tokens than a busy one at the same quality, and never more than the VAE's own grid.

Pipeline: Qwen-Image 2.1 VAE, channel KLT, CDF 9/7 wavelet, one rounding step per image. A token is one non-zero block of 16 numbers, sent coarse to fine.

## Install

```bash
pip install -r requirements.txt
pip install --no-deps dreamsim DISTS-pytorch
```

The DINOv3 weights are gated on Hugging Face, so request access and log in first. Use of the Qwen-Image 2.1 weights and DINOv3 is subject to their own licences.

If DISTS cannot find `weights.pt`, copy it from the installed package to `sys.prefix`.

## Data

Run from a folder holding test images at `data/<group>/*.jpg`. The groups used are `mondrian`, `simple`, `ornate`, `texture`, `dense` and `waldo`. Fitting the KLT needs a separate tar of 256 px training images that doesn't overlap the test set.

## Run

```bash
export PYTHONPATH=src

# check the VAE: shapes, alpha, speed, memory
python src/vae_check.py runs/vae.json

# reconstruct the test images with the plain VAE, score them, freeze the bars
python src/floor.py recon runs/floor
for j in dinov3 dists dreamsim colour; do python src/floor.py score runs/floor $j; done
python src/freeze_floor.py runs/floor
python src/readout_floor.py runs/floor

# run the codec against those bars
python src/run_codec.py runs/codec runs/floor/floor.json --tar images.tar
python src/readout_codec.py runs/codec

# tokens and judge scores at every step
python src/ladder.py runs/ladder runs/floor/floor.json --tar images.tar
python src/readout_ladder.py runs/ladder
```

## Evaluation

All judges are distances where 0 is identical, scored in bfloat16 against the original:

- DINOv3 at native resolution
- DISTS at native resolution
- DreamSim per 256 px tile
- Delta E at 1/16 scale, for colour

An image passes only if every judge stays within its bar, on the worst 256 px tile. Each bar is the plain VAE's score on that image plus a small fixed margin.

Status: research prototype. The learned filters and the small autoregressive model aren't in this repository yet.
