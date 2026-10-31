"""The fixed codec on a latent grid: channel KLT, multi-level CDF 9/7 in Mallat layout, one dead-zone step, and the token count.

A token is 16 floats and exists only if not all zero. Two shapes are counted on the same coefficients: 16 components x 1 position,
and 4 components x 2x2 positions. Length adds one skip token per run of empty blocks, in coarse-to-fine order.
"""
import numpy as np
import torch

LA, LB, LG, LD, R = -1.586134342, -0.05298011854, 0.8829110762, 0.4435068522, 1.149604398


def quant(c, step, offset=.33):
    return c.sign() * torch.floor(c.abs() / step + offset)


class Wavelet:
    """KLT fit on zs ([C,H,W] latents), then the wavelet over space. fwd and inv are exact inverses up to float error."""

    def __init__(self, zs, levels, dev):
        self.levels = levels
        v = torch.cat([z.flatten(1).T for z in zs]).double()
        ev, U = torch.linalg.eigh(torch.cov(v.T))
        self.mu = v.mean(0).float().to(dev)
        self.U = U.flip(1).float().to(dev)
        self.ev = ev.flip(0).float().clamp_min(0).to(dev)

    @staticmethod
    def _lift(x, inverse=False):
        """One 1D 9/7 level along the last axis, symmetric extension, even length."""

        def nb(v, odd):  # the two neighbours of each sample in the other subband
            if odd:
                return v + torch.cat([v[..., 1:], v[..., -1:]], -1)
            return v + torch.cat([v[..., :1], v[..., :-1]], -1)

        if not inverse:
            s, d = x[..., ::2].clone(), x[..., 1::2].clone()
            d = d + LA * nb(s, True)
            s = s + LB * nb(d, False)
            d = d + LG * nb(s, True)
            s = s + LD * nb(d, False)
            return torch.cat([s * R, d / R], -1)  # near-orthonormal, so one step means the same on every band
        n = x.shape[-1] // 2
        s, d = x[..., :n] / R, x[..., n:] * R
        s = s - LD * nb(d, False)
        d = d - LG * nb(s, True)
        s = s - LB * nb(d, False)
        d = d - LA * nb(s, True)
        out = torch.empty_like(x)
        out[..., ::2], out[..., 1::2] = s, d
        return out

    def fwd(self, z):
        C, H, W = z.shape
        x = (self.U.T @ (z.flatten(1).T - self.mu).T).view(C, H, W)
        for lv in range(self.levels):
            h, w = H >> lv, W >> lv
            b = self._lift(x[:, :h, :w])
            x[:, :h, :w] = self._lift(b.transpose(1, 2)).transpose(1, 2)
        return x

    def inv(self, c):
        C, H, W = c.shape
        x = c.clone()
        for lv in reversed(range(self.levels)):
            h, w = H >> lv, W >> lv
            b = self._lift(x[:, :h, :w].transpose(1, 2), True).transpose(1, 2)
            x[:, :h, :w] = self._lift(b, True)
        z = (self.U @ x.flatten(1)).T + self.mu
        return z.T.view(C, H, W)


def scan(H, W, levels, cell):
    """Block positions in Mallat layout, coarse band first, raster inside a band; cell is the block's side in positions."""
    order, seen = [], torch.zeros(H, W, dtype=torch.bool)
    for h, w in [(H >> levels, W >> levels)] + [(H >> lv, W >> lv) for lv in range(levels - 1, -1, -1)]:
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                if not seen[y, x]:
                    seen[y : y + cell, x : x + cell] = True
                    order.append((y, x))
    return order


def tokens(q, levels):
    """Non-zero blocks and sequence length with skips, for both block shapes. q is [C,H,W]."""
    C, H, W = q.shape
    out = dict(nonzero=int((q != 0).sum()))
    for name, comps, cell in (("t16x1", 16, 1), ("t4x2", 4, 2)):
        nz = []
        for y, x in scan(H, W, levels, cell):
            for c in range(0, C, comps):
                nz.append(bool((q[c : c + comps, y : y + cell, x : x + cell] != 0).any()))
        nz = np.array(nz)
        idx = np.flatnonzero(nz)
        skips = int(np.count_nonzero(np.diff(np.r_[-1, idx]) > 1))
        out[name] = dict(blocks=int(nz.sum()), length=int(nz.sum()) + skips, grid=len(nz))
    return out
