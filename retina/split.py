"""Classical structure/texture split and a texture token.

Split: a self-guided filter (He et al.) keeps strong edges and removes low-contrast detail. Only low-contrast residual counts as texture; where the residual is strong (faces, figures, text) the original stays in the coded image, so busy content is protected.
Texture token: power of the missing signal in 4 frequency rings x 6 orientations, plus 3 colour gains (27 floats). The decoder draws random-phase noise with that spectrum, which matches second-order statistics, the part early vision relies on for stationary texture (Julesz). The lowest frequencies are left out, so it cannot paint blotches."""
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


def texture_mask(x, s, tau, r=8):
    """1 where the residual is low contrast, so it may be replaced by statistics."""
    e = (x - s).pow(2).mean(1, keepdim=True)
    e = box(e, r).sqrt()
    return box(torch.sigmoid((tau - e) / (.15 * tau)), 4)


def _grid(n, dev):
    f = torch.fft.fftfreq(n, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    rad, ang = (fy * fy + fx * fx).sqrt(), torch.atan2(fy, fx) % math.pi
    ring = torch.bucketize(rad, torch.tensor([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.], device=dev)) - 1  # -1 below 1/32: never synthesised
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


def synth(r, m, bands, gain, seed=0):
    n = r.shape[-1]
    ring, sector = _grid(n, r.device)
    amp = torch.zeros((n, n), device=r.device)
    ok = ring >= 0
    amp[ok] = bands[(ring[ok] * 6 + sector[ok])].clamp_min(0).sqrt()
    g = torch.Generator(device=r.device).manual_seed(seed)
    w = torch.randn(n, n, device=r.device, generator=g)
    tex = torch.fft.ifft2(torch.fft.fft2(w) * amp).real
    return (r + tex[None, None] * gain.view(1, 3, 1, 1) * m).clamp(-1, 1)
