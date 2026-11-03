"""The tokenizer as one object. Image -> variable-length tokens -> image.

Encode: DReX + texture energy give the image a budget along a curve anchored to viewing verdicts. One warp over the whole image, or none if that scores better, goes through the VAE. Decode needs only the tokens.
One token is 128 floats on FLUX.2, the default. An image can also carry knots and statistics; `pack` puts all of it into one sequence of equal tokens for an AR model."""
import math
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from . import common as C
from . import refiner as R
from . import split as X
from . import vae as FV
from . import warp as W
from .drex import drex_score, load_drex

T = 256
# The earlier curve, kept with refit=False. log budget = alpha + beta * (drex * DReX + tex * texture energy). Weights from the whole-image search on the 52; alpha and beta set so the median Mondrian gets 146 tokens and the book page 1,600, the viewing anchors.
CURVE = {'drex': 0.8502, 'tex': 15.6389, 'alpha': 3.5886, 'beta': 2.2292}
# log budget = w . (1, d, t, d^2, t^2, d t), d = DReX, t = 10 * texture energy. Fit on training images only: tokens needed at the quality the viewing anchors set, q = .633 - .547 DReX. The default; the user passed its tokens by eye.
REFIT = [3.149, 3.091, 2.111, 0.792, 0.836, -1.624]
# gap above 1 widens simple vs complex around this log budget, about 356 tokens, midway between the median Mondrian and the page.
PIVOT = 5.876
# Held out, the S1 decoder beats the plain one at or below this budget and loses above it.
LEARNED_MAX = 600
# The side the budget's inputs are measured at: the filter window is fixed in pixels and tiles are fixed at 256 px, so at other sides both signals drift.
BUDGET_SIDE = 768
WARP_MAX = 200
# The user adopted this over the original DReX, 2026-11-01: on real documents it reads close to the full grid where the original does not.
DREXT = "tensorkelechi/DReX-text-tuned"
# Default weights: S1 for FLUX.2, and the small MAR in mar.py.
WEIGHTS = "tensorkelechi/retina-tokenizer"
S1_FLUX2 = "s1_flux2.pt"


class Flux:
    """Real FLUX.1: f8 with 16 channels, four cells to a token."""

    cell, pack = 8, 2

    def __init__(self, dev):
        self.fx = FV.Flux1(dev)
        self.enc, self.dec = self.fx.encode, self.fx.dec_raw


class Flux2VAE:
    """Real FLUX.2: f8 with 32 channels, four cells to a 128-float token."""

    cell, pack = 8, 2

    def __init__(self, dev):
        self.fx = FV.Flux2(dev)
        self.enc, self.dec = self.fx.encode, self.fx.dec_raw


class QwenVAE:
    """Real Qwen-Image 2.1: f16 with 64 channels, one cell to a token."""

    cell, pack = 16, 1

    def __init__(self, dev):
        self.q = FV.Qwen(dev)
        self.enc, self.dec = self.q.enc, self.q.dec


class TAEQI:
    """Ollin's TAEQI 2.1, the tiny stand-in for Qwen."""

    cell, pack = 16, 1

    def __init__(self, dev):
        self.c = C.Codec("taeqi", dev)
        self.enc, self.dec = self.c.enc, self.c.dec


VAES = dict(flux=Flux, flux2=Flux2VAE, qwen=QwenVAE, taeqi=TAEQI)


class Tokenizer:
    def __init__(self, vae="flux2", dev="cuda", judges=None, learned=S1_FLUX2, scale=1., refit=True, gap=1.25, floor=22., grow=1.4, s1_min=0., drext=DREXT, tile=1, warp_max=WARP_MAX, tile_res=(224, 512), learned_max=LEARNED_MAX):
        """learned: S1 decoder weights for this VAE, a file, or the published name for FLUX.2; None decodes plain. Used at or below LEARNED_MAX tokens. scale multiplies the budget of those images, so it does nothing without S1. gap only acts on the refit curve (refit=True). drext: tuned DReX weights, a file or a Hugging Face repo; None keeps the original. tile k: DReX can rise to the kth highest of the 9 tile scores, so one dense region counts; 0 is whole-image only. warp_max: warp below this many tokens."""
        self.dev, self.v, self.J = dev, VAES[vae](dev), judges
        self.c = CURVE
        self.drex = load_drex(dev)
        if drext:
            # Fusion and head tuned so text scores for the tokens it needs; other images stay near the original score.
            w = torch.load(drext if os.path.exists(drext) else hf_hub_download(drext, "drext.pt"), map_location=dev, weights_only=True)
            self.drex[2].load_state_dict(w["fusion"])
            self.drex[3].load_state_dict(w["head"])
        self.net, self.scale, self.refit, self.gap = None, scale, refit, gap
        self.floor, self.grow, self.s1_min, self.tile, self.warp_max, self.tile_res, self.learned_max = floor, grow, s1_min, tile, warp_max, tile_res, learned_max
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
        S = x.shape[-1]
        tiles = torch.cat([x[..., i : i + T, j : j + T] for i in range(0, S, T) for j in range(0, S, T)])
        tex = (tiles - X.structure(tiles)).abs().mean().item()
        d = drex_score(self.drex, x)[0]
        if self.tile:
            d = max(d, sorted(drex_score(self.drex, tiles, *self.tile_res))[-min(self.tile, len(tiles))])
        if self.refit:
            t = 10 * tex
            b = sum(w * v for w, v in zip(REFIT, (1, d, t, d * d, t * t, d * t)))
            return math.exp(PIVOT + self.gap * (b - PIVOT)), d, tex
        z = self.c["drex"] * d + self.c["tex"] * tex
        return float(torch.tensor(self.c["alpha"] + self.c["beta"] * z).exp()), d, tex

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
        """x [1,3,S,S] in [-1,1], S a multiple of 256 -> token dict. n, if given, fixes the cells per side and skips the floor. With judges, warp or no warp is settled by one decode of each, never the budget."""
        x = self._check(x)
        S = x.shape[-1]
        given = n is not None
        if given and (n % self.v.pack or not self.v.pack <= n <= S // self.v.cell):
            raise ValueError(f"n must be a multiple of {self.v.pack} from {self.v.pack} to {S // self.v.cell} at side {S}, got {n}")
        b, d, tex = self.budget(x)
        use = self.net is not None and b <= self.learned_max
        n = n or self.size_of(b * self.scale if use else b, S)
        if self.J is None:
            return self._encode_fast(x, n, b, d, tex, given)
        opts = [self._code(x, n, True)] + ([self._code(x, n, False)] if self.v.cell * n < S else [])
        best = opts[0]
        if self.J is not None and len(opts) > 1:
            self.J.set_ref(x)
            sc = [self.J.score(self.decode(self._pack(z, k, n, S))) for z, k in opts]
            best = min(zip(sc, opts), key=lambda t: t[0]["ds_mean"] + t[0]["di_mean"])[1]
        t = self._pack(*best, n, S)
        t.update(budget=b, drex=d, tex=tex)
        if use:
            c = self.decode(t)
            t["stats"] = self._stats(x, c).float()
            if self.J is not None:
                # The S1 decoder invents texture on clean images, so it has to win on this image. Same judges as the warp choice.
                self.J.set_ref(x)
                a, l = self.J.score(c), self.J.score(self.decode(t))
                if l["ds_mean"] + l["di_mean"] >= a["ds_mean"] + a["di_mean"]:
                    del t["stats"]
        return t

    def _psnr(self, x, c):
        return (-10 * ((c - x) / 2).pow(2).mean().log10()).item()

    def _encode_fast(self, x, n, b, d, tex, given):
        """No judges and one code per try. Warp below WARP_MAX tokens, where the judges mostly prefer it. Text and other fine high-contrast detail read as simple to the budget, so a PSNR floor raises the size until it clears."""
        S, step = x.shape[-1], self.v.pack
        while True:
            t = self._pack(*self._code(x, n, (n // step) ** 2 < self.warp_max), n, S)
            c = self.decode(t)
            if given or self._psnr(x, c) >= self.floor or self.v.cell * n >= S:
                break
            n = min(max(int(n * self.grow / step + .5) * step, n + step), S // self.v.cell)
        t.update(budget=b, drex=d, tex=tex)
        # one statistics token per 256 tile, counted as length() counts them
        if self.net is not None and self.length(t) + (S // T) ** 2 <= self.learned_max:
            s = self._stats(x, c).float()
            if s[:, :24].mean().item() >= self.s1_min:
                t["stats"] = s
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
            e = lambda v: torch.cat((torch.zeros(1, device=self.dev), v.to(self.dev).float() * S, torch.full((1,), float(S), device=self.dev)))
            st = (W.local_scale(e(t["knots"][0]), e(t["knots"][1]), N, N, S, S).clamp_min(1).log2() / 4).float()
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
        e = lambda v: torch.cat((torch.zeros(1, device=self.dev), v.to(self.dev).float() * S, torch.full((1,), float(S), device=self.dev)))
        return W.inverse(y, e(t["knots"][0]), e(t["knots"][1]), S, S).clamp(-1, 1)

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
