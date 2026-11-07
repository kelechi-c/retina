"""The tokenizer as one object. Image -> variable-length tokens -> image.

Encode (policy "v18", the default): about twenty content features from DReX, texture and gradient statistics give the image a budget through a fit on the user's own accept/reject labels. At every size the image is encoded both folded (so edges keep their resolution) and plain, and the better decode is kept. A learned check, fitted on the user's blind ratings, accepts that decode or steps up one size; when the content is concentrated, each busy region must pass on its own, so a small dense island is not averaged away. Images at LEARNED_MAX tokens or fewer also get statistics for the S1 refiner. Decode needs only the tokens.
Policy "v16" keeps v1's budget with fold-if-better at every size, a PSNR floor capped below the grid's own PSNR and a per-tile LPIPS floor; policy "v1" is the released tokenizer.
One token is 128 floats on FLUX.2, the default. `pack` puts knots, statistics and latents into one sequence of equal tokens for an AR model."""
import math
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from . import common as C
from . import refiner as R
from . import split as X
from . import warp as W
from .drex import drex_score, load_drex
from .vae import VAES

T = 256
# log budget = w . (1, d, t, d^2, t^2, d t), d = DReX, t = 10 * texture energy. Fit on training images only: tokens needed at the quality the viewing anchors set, q = .633 - .547 DReX. The user passed its tokens by eye.
REFIT = [3.149, 3.091, 2.111, 0.792, 0.836, -1.624]
# gap above 1 widens simple vs complex around this log budget, about 356 tokens, midway between the median Mondrian and the page.
PIVOT = 5.876
# Held out, the S1 refiner beats the plain decode at or below this many tokens and loses above it.
LEARNED_MAX = 600
# The side the budget's inputs are measured at: the filter window is fixed in pixels and tiles are fixed at 256 px, so at other sides both signals drift.
BUDGET_SIDE = 768
WARP_MAX = 200
# How much each side grows when a decode misses the floor.
GROW = 1.4
# v1.6: grow 1.2x per side, never ask for more PSNR than the grid gives minus this, and no 256 px tile more than this much LPIPS worse than the grid.
V16_GROW, V16_MARGIN, V16_TAU = 1.2, 1.5, .35
# v1.8: the grid sides (tokens per side) the budget snaps to and the check steps through.
V18_SIDES = [4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30, 34, 38, 43, 48]
POLICIES = ("v18", "v16", "v1")
# The user adopted this over the original DReX, 2026-11-01: on real documents it reads close to the full grid where the original does not.
DREXT = "tensorkelechi/DReX-text-tuned"
# Default weights: S1 for FLUX.2, and the small MAR in mar.py.
WEIGHTS = "tensorkelechi/retina-tokenizer"
S1_FLUX2 = "s1_flux2.pt"


class Tokenizer:
    def __init__(self, vae="flux2", dev="cuda", learned=S1_FLUX2, gap=1.25, floor=22., drext=DREXT, tile=1, warp_max=WARP_MAX, policy="v18"):
        """policy: v18 (learned budget and check, fold-if-better at every size), v16 (v1's budget, fold-if-better at every size, capped PSNR floor and tile floor) or v1 (the released tokenizer; floor and warp_max apply only to it). vae: flux2, flux or qwen. learned: S1 refiner weights for this VAE, a file, or the published name for FLUX.2; None decodes plain. gap above 1 widens the spread between simple and busy images. floor: PSNR in dB the first decode must reach before the grid stops growing; 0 turns it off. drext: tuned DReX weights, a file or a Hugging Face repo; None keeps the original. tile k: DReX can rise to the kth highest of the 9 tile scores, so one dense region counts; 0 is whole-image only. warp_max: warp below this many tokens."""
        self.dev, self.v = dev, VAES[vae](dev)
        self.gap, self.floor, self.tile, self.warp_max = gap, floor, tile, warp_max
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy}")
        self.policy = policy
        if policy == "v18":
            from .check import Checker
            self.checker = Checker(dev)
            self.lp = self.checker.lp  # v1.6's tile floor, for sides other than 768
        if policy == "v16":
            import lpips
            self.lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev).eval().requires_grad_(False)
        self.drex = load_drex(dev)
        if drext:
            # Fusion and head tuned so text scores for the tokens it needs; other images stay near the original score.
            w = torch.load(drext if os.path.exists(drext) else hf_hub_download(drext, "drext.pt"), map_location=dev, weights_only=True)
            self.drex[2].load_state_dict(w["fusion"])
            self.drex[3].load_state_dict(w["head"])
        self.net = None
        if learned == S1_FLUX2 and vae != "flux2":
            learned = None
        if learned:
            assert vae in ("flux", "flux2"), "S1 needs weights trained on this VAE"
            self.net = R.StatRefiner().to(dev).eval().requires_grad_(False)
            self.net.load_state_dict(torch.load(learned if os.path.exists(learned) else hf_hub_download(WEIGHTS, learned), map_location=dev))

    def _check(self, x):
        """One square RGB image whose side is a multiple of 256: the budget, the statistics tokens and decode all cut it into 256 px tiles."""
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3 or x.shape[-1] != x.shape[-2]:
            raise ValueError(f"expected one square RGB image of shape [1, 3, S, S], got {tuple(x.shape)}")
        if x.shape[-1] % T:
            raise ValueError(f"the image side must be a multiple of {T}, got {x.shape[-1]}")
        return x.to(self.dev)

    @torch.no_grad()
    def budget(self, x):
        """-> (budget in tokens, DReX, texture energy). Both signals are read from a 768 px copy, the size the curve was fit at, so a picture gets the same inputs at any size. The budget is not yet scaled for size: tools/fit_size_scaling.py measures that."""
        x = self._check(x)
        if x.shape[-1] != BUDGET_SIDE:
            x = C.down(x, BUDGET_SIDE).clamp(-1, 1)
        b, d, tex, _, _ = self._budget(x)
        return b, d, tex

    def _budget(self, x):
        """x at BUDGET_SIDE -> (budget, DReX as the budget reads it, texture energy, whole-image DReX, the tile scores)"""
        S = x.shape[-1]
        tiles = torch.cat([x[..., i : i + T, j : j + T] for i in range(0, S, T) for j in range(0, S, T)])
        tex = (tiles - X.structure(tiles)).abs().mean().item()
        dw = float(drex_score(self.drex, x)[0])
        dt = [float(v) for v in drex_score(self.drex, tiles)] if self.tile or self.policy == "v18" else []
        d = max(dw, sorted(dt)[-min(self.tile, len(dt))]) if self.tile else dw
        t = 10 * tex
        b = sum(w * v for w, v in zip(REFIT, (1, d, t, d * d, t * t, d * t)))
        return math.exp(PIVOT + self.gap * (b - PIVOT)), d, tex, dw, dt

    def size_of(self, b, S):
        """Budget -> cells per side, a whole number of tokens per side."""
        g = max(round(max(b - 2, 4) ** .5), 2)
        return min(g * self.v.pack, S // self.v.cell)

    @torch.no_grad()
    def _code(self, x, n, warped):
        S, N = x.shape[-1], self.v.cell * n
        if warped and N < S:
            ky = W.knots(W.density(W.energy(x, 2), N, 2, .03), N)
            kx = W.knots(W.density(W.energy(x, 3), N, 2, .03), N)
            z = self.v.enc(W.forward(x, ky, kx, N, N))
            return z, torch.stack((ky[1:-1], kx[1:-1])) / S
        return self.v.enc(C.down(x, N) if N != S else x), None

    @torch.no_grad()
    def encode(self, x, n=None):
        """x [1,3,S,S] in [-1,1], S a multiple of 256 -> token dict. n, if given, fixes the cells per side and skips the floor.
        Text and other fine high-contrast detail read as simple to the budget, so one decode is checked against the PSNR floor and the grid grows until it passes."""
        x = self._check(x)
        S, step, top = x.shape[-1], self.v.pack, x.shape[-1] // self.v.cell
        given = n is not None
        if given and (n % step or not step <= n <= top):
            raise ValueError(f"n must be a multiple of {step} from {step} to {top} at side {S}, got {n}")
        if not given and self.policy == "v18" and S == BUDGET_SIDE:
            return self._encode_v18(x)
        if not given and self.policy in ("v16", "v18"):  # v1.8's check is laid out for 768 px; other sides use v1.6
            return self._encode_v16(x)
        b, d, tex = self.budget(x)
        n = n or self.size_of(b, S)
        while True:
            t = self._pack(*self._code(x, n, (n // step) ** 2 < self.warp_max), n, S)
            c = self.decode(t)
            if given or n >= top or (-10 * ((c - x) / 2).pow(2).mean().log10()).item() >= self.floor:
                break
            n = min(max(int(n * GROW / step + .5) * step, n + step), top)
        t.update(budget=b, drex=d, tex=tex)
        # Statistics for the refiner, one token per 256 tile as length() counts them; skipped when the decode lost almost nothing.
        if self.net is not None and self.length(t) + (S // T) ** 2 <= LEARNED_MAX:
            s = self._stats(x, c).float()
            if s[:, :24].mean().item() >= 0:
                t["stats"] = s
        return t

    @staticmethod
    def _psnr(a, b):
        return (-10 * ((a - b) / 2).pow(2).mean().log10()).item()

    @torch.no_grad()
    def _best(self, x, n):
        """Encode at n cells per side, folded and plain, keep the decode with the higher PSNR, add S1 statistics where encode would.
        -> (tokens, final decode, PSNR of the plain decode)"""
        S, top, best = x.shape[-1], x.shape[-1] // self.v.cell, None
        for w in ((True, False) if n < top else (False,)):
            t = self._pack(*self._code(x, n, w), n, S)
            c = self.decode(t)
            p = self._psnr(c, x)
            if best is None or p > best[0]:
                best = (p, t, c)
        p, t, c = best
        if self.net is not None and self.length(t) + (S // T) ** 2 <= LEARNED_MAX:
            s = self._stats(x, c).float()
            if s[:, :24].mean().item() >= 0:
                t["stats"] = s
                c = self.decode(t)
        return t, c, p

    @torch.no_grad()
    def _encode_v18(self, x):
        """One-pass learned budget, then one decode per size until the learned check accepts it."""
        from . import check as K
        b, d, tex, dw, dt = self._budget(x)
        f, ct = K.content(x, dw, dt, tex, b)
        pred = K.budget(f)
        k = min(range(len(V18_SIDES)), key=lambda i: abs(math.log(V18_SIDES[i] ** 2 / pred)))
        while True:
            t, c, _ = self._best(x, V18_SIDES[k] * self.v.pack)
            p = self.checker(x, c, ct, self.length(t))
            if p > self.checker.q or k == len(V18_SIDES) - 1:
                break
            k += 1
        t.update(budget=pred, drex=d, tex=tex, accept=p)
        return t

    @torch.no_grad()
    def _encode_v16(self, x):
        """v1's budget; grow until the plain decode reaches the floor (capped at the grid's own PSNR minus V16_MARGIN) and no 256 px tile of the final decode is more than V16_TAU LPIPS worse than the grid's."""
        S, step, top = x.shape[-1], self.v.pack, x.shape[-1] // self.v.cell
        b, d, tex = self.budget(x)
        g = self.v.dec(self.v.enc(x)).float().clamp(-1, 1)
        tl = lambda c: [self.lp(x[..., i : i + T, j : j + T], c[..., i : i + T, j : j + T]).mean().item() for i in range(0, S, T) for j in range(0, S, T)]
        ref, fl = tl(g), min(self.floor, self._psnr(g, x) - V16_MARGIN)
        n = self.size_of(b, S)
        while True:
            t, c, p = self._best(x, n)
            if (p >= fl and max(a - r for a, r in zip(tl(c), ref)) <= V16_TAU) or n >= top:
                break
            n = min(max(int(n * V16_GROW / step + .5) * step, n + step), top)
        t.update(budget=b, drex=d, tex=tex)
        return t

    def _stats(self, x, c):
        """One statistics token per 256 tile: what the plain decode is missing, 27 floats of the 64."""
        S, ones, o = x.shape[-1], torch.ones(1, 1, T, T, device=self.dev), []
        for i in range(0, S, T):
            for j in range(0, S, T):
                b, g = X.measure((x - c)[..., i : i + T, j : j + T], ones)
                o.append(torch.cat(((b.clamp_min(1e-8).log() + 8) / 8, g)))
        return torch.stack(o)

    def _pack(self, z, k, n, S):
        p = self.v.pack
        lat = F.pixel_unshuffle(z, p) if p > 1 else z
        return dict(side=S, n=n, lat=lat[0].flatten(1).T.float(), knots=None if k is None else k.float())

    def _edges(self, k, S):
        """Stored inner knots (0..1) -> every knot as a source position, 0 to S."""
        return torch.cat((torch.zeros(1, device=self.dev), k.to(self.dev).float() * S, torch.full((1,), float(S), device=self.dev)))

    @torch.no_grad()
    def decode(self, t):
        c = self._plain(t)
        s = t.get("stats")
        if s is None:
            return c
        S, N = t["side"], self.v.cell * t["n"]
        if t["knots"] is None:
            st = torch.full((1, 2, S, S), math.log2(max(S / N, 1)) / 4, device=self.dev)
        else:
            st = (W.local_scale(self._edges(t["knots"][0], S), self._edges(t["knots"][1], S), N, N, S, S).clamp_min(1).log2() / 4).float()
        q = S // T
        m = s.to(self.dev).float().T.reshape(1, 27, q, q).repeat_interleave(T, 2).repeat_interleave(T, 3)
        noise = torch.randn((1, 1, S, S), generator=torch.Generator(self.dev).manual_seed(0), device=self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.net(c, st, m, noise).float().clamp(-1, 1)

    @torch.no_grad()
    def _plain(self, t):
        S, n, p = t["side"], t["n"], self.v.pack
        g = n // p
        z = t["lat"].T.reshape(1, -1, g, g).to(self.dev).float()  # mar.encode stores sequences in half precision
        z = F.pixel_shuffle(z, p) if p > 1 else z
        N = self.v.cell * n
        y = self.v.dec(z).float()
        if t["knots"] is None:
            return (C.up(y, S) if N != S else y).clamp(-1, 1)
        return W.inverse(y, self._edges(t["knots"][0], S), self._edges(t["knots"][1], S), S, S).clamp(-1, 1)

    @staticmethod
    def pack(t):
        """Token dict -> one sequence, all tokens one latent token wide: knots, statistics, latents, unused floats zero. The layout says which tokens are which."""
        W = t["lat"].shape[1]
        wrap = lambda v: F.pad(v.flatten(), (0, -(-v.numel() // W) * W - v.numel())).view(-1, W)
        parts = [wrap(t[k].to(t["lat"])) for k in ("knots", "stats") if t.get(k) is not None]
        return dict(seq=torch.cat(parts + [t["lat"]]), layout=dict(side=t["side"], n=t["n"], warped=t["knots"] is not None, stats=t.get("stats") is not None))

    @staticmethod
    def unpack(p):
        q, L, W, c = p["seq"], p["layout"], p["seq"].shape[1], 0
        def take(m, shape):
            nonlocal c
            k = -(-m // W)
            v = q[c : c + k].flatten()[:m].view(shape)
            c += k
            return v
        knots = take(126, (2, 63)) if L["warped"] else None
        tiles = (L["side"] // T) ** 2
        stats = take(27 * tiles, (tiles, 27)) if L["stats"] else None
        t = dict(side=L["side"], n=L["n"], lat=q[c:], knots=knots)
        if stats is not None:
            t["stats"] = stats
        return t

    @staticmethod
    def length(t):
        return len(t["lat"]) + (2 if t["knots"] is not None else 0) + (len(t["stats"]) if t.get("stats") is not None else 0)
