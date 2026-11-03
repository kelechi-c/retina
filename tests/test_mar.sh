#!/usr/bin/env bash
# MAR end to end at toy size: encode a few images of each label, train 300 steps, sample from the checkpoint, sample from the released weights.
# usage: bash tests/test_mar.sh OUT        (PY=python on a machine without uv)
set -euo pipefail
OUT=${1:-test_out/mar}
PY=${PY:-uv run python}
$PY -m mar.encode "$OUT/enc" --caps 0/procedural=32,2/dtd=32,1/wikiart=32,3/wikiart=32 --p 1,1,1,1 --workers 4
$PY -m mar.train "$OUT/train" "$OUT/enc/tokens.pt" --steps 300 --budget 4000 --warmup 50 --every 100 --samples 1
$PY -m mar.sample "$OUT/own" --ckpt "$OUT/train/mar.pt" --per 1 --live
$PY -m mar.sample "$OUT/released" --per 2
ls "$OUT/own" "$OUT/released"
echo MAR_TEST_DONE
