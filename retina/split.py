"""Classical structure/texture split, and the statistics token.

Split: a self-guided filter (He et al.) keeps strong edges and removes low-contrast detail; what it removes is the texture that texture energy measures.
Statistics token: power of the signal a decode is missing in 4 frequency rings x 6 orientations, plus 3 colour gains (27 floats). The lowest frequencies are left out. The S1 refiner reads it to put texture back."""
import math

import torch
import torch.nn.functional as F


def box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, 1)


def guided(x, r, eps):
    m = box(x, r)
    v = box(x * x, r) - m * m
    a = v / (v + eps)
    return box(a, r) * x + box((1 - a) * m, r)


def structure(x, r=6, eps=.02, passes=2):
    s = x
    for _ in range(passes):
        s = guided(s, r, eps)
    return s


def _grid(n, dev):
    f = torch.fft.fftfreq(n, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    rad, ang = (fy * fy + fx * fx).sqrt(), torch.atan2(fy, fx) % math.pi
    ring = torch.bucketize(rad, torch.tensor([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.], device=dev)) - 1  # -1 below 1/32: not measured
    sector = (ang / math.pi * 6).long().clamp(max=5)
    return ring, sector


def measure(res, m):
    """res [1,3,n,n] = what the decode is missing, m its mask -> 24 band powers and 3 colour gains."""
    n = res.shape[-1]
    ring, sector = _grid(n, res.device)
    w = m[0, 0]
    area = w.mean().clamp_min(1e-3)
    luma = (res * m).mean(1)[0]
    P = torch.fft.fft2(luma).abs().pow(2) / (n * n) / area
    bands = torch.zeros(24, device=res.device)
    for rr in range(4):
        for ss in range(6):
            sel = (ring == rr) & (sector == ss)
            if sel.any():
                bands[rr * 6 + ss] = P[sel].mean()
    gain = ((res * m).flatten(2).std(2)[0] / (luma * 1).std().clamp_min(1e-5)).clamp(0, 3)
    return bands, gain
