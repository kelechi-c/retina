"""CPU checks that need no weights: the packed sequence and the MAR's slot layout agree at every side, packing round-trips exactly, half-precision sequences unpack, and the warp is sane.

usage: uv run python tests/test_cpu.py
"""
import itertools
import sys

import torch

from mar.model import W as MAR_W
from mar.model import slots
from retina import warp as Wp
from retina.tokenizer import T, Tokenizer

fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


torch.manual_seed(0)
for side, warped, stats in itertools.product((256, 512, 768, 1024), (False, True), (False, True)):
    n = 16
    t = dict(side=side, n=n, lat=torch.randn((n // 2) ** 2, MAR_W), knots=torch.rand(2, 63) if warped else None)
    if stats:
        t["stats"] = torch.randn((side // T) ** 2, 27)
    p = Tokenizer.pack(t)
    u = Tokenizer.unpack(p)
    same = all(torch.equal(t[k], u[k]) for k in ("lat", "knots", "stats") if t.get(k) is not None)
    same = same and (u["knots"] is None) == (not warped) and ("stats" in u) == stats
    tag = f"side {side} warped {warped} stats {stats}"
    check(same, f"pack round trip, {tag}")
    check(len(slots(p["layout"])) == len(p["seq"]), f"MAR slots match the packed length ({len(p['seq'])}), {tag}")
    h = Tokenizer.unpack(dict(seq=p["seq"].half(), layout=p["layout"]))
    check(h["lat"].shape == t["lat"].shape, f"half-precision sequence unpacks, {tag}")

S = 256
x = torch.rand(1, 3, S, S) * 2 - 1
k = Wp.knots(torch.ones(S), S)
check((Wp.forward(x, k, k, S, S) - x).abs().max().item() < 1e-4, "a native-rate warp is the identity")
flat = torch.zeros(1, 3, 768, 768)
for N in (32, 128):
    ky = Wp.knots(Wp.density(Wp.energy(flat, 2), N, 2, .03), N)
    y = Wp.forward(flat, ky, ky, N, N)
    check(torch.isfinite(y).all().item() and (ky.diff() > 0).all().item(), f"flat image warps to {N} px with increasing knots")

print("FAILURES", fails)
print("TEST_PASSED" if not fails else "TEST_FAILED", flush=True)
sys.exit(1 if fails else 0)
