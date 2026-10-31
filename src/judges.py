"""The judges: DINOv3 and DISTS at native resolution, DreamSim per 256 tile at its own 224 input, and a colour check at 1/16 scale.
All are distances, 0 = identical. Images are [1,3,H,W] in [-1,1], H and W multiples of 256, scored in bfloat16.

The tile score is the worst 256 tile, since a mean over tiles lets a crowd with melted faces pass. Models load one at a time, so
they fit a 6 GB GPU, and reference features are cached, so one image's variants cost one forward pass each.
"""
import torch
import torch.nn.functional as F

TILE = 256
DINO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
PATCH = 16


def tiles(x):
    """[1,3,H,W] -> [n,3,256,256], row-major."""
    return x.unfold(2, TILE, TILE).unfold(3, TILE, TILE).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TILE, TILE)


class Dino:
    name = "dinov3"

    def __init__(self, dev):
        from transformers import AutoModel

        self.m, self.dev, self.ref = AutoModel.from_pretrained(DINO).eval().to(dev).requires_grad_(False), dev, None
        self.mean = torch.tensor((.485, .456, .406), device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor((.229, .224, .225), device=dev).view(1, 3, 1, 1)

    def feats(self, x):
        x = (((x.to(self.dev) + 1) / 2) - self.mean) / self.std
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = self.m(pixel_values=x).last_hidden_state.float()
        return F.normalize(t[:, -h * w :], dim=-1).view(len(x), h, w, -1)

    @torch.no_grad()
    def set_ref(self, x):
        self.ref = self.feats(x)

    @torch.no_grad()
    def score(self, r):
        d = 1 - (self.feats(r) * self.ref).sum(-1)
        t = F.avg_pool2d(d[None], TILE // PATCH).flatten()
        return dict(mean=d.mean().item(), q95=d.flatten().quantile(.95).item(), tile_worst=t.max().item(), tile_mean=t.mean().item())


class Dists:
    name = "dists"

    def __init__(self, dev):
        from DISTS_pytorch import DISTS

        self.m, self.dev, self.ref = DISTS().to(dev).eval(), dev, None

    def d(self, a, b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m((a + 1) / 2, (b + 1) / 2, batch_average=False).float()

    @torch.no_grad()
    def set_ref(self, x):
        self.ref = x.to(self.dev)
        self.ref_t = tiles(self.ref)

    @torch.no_grad()
    def score(self, r):
        r = r.to(self.dev)
        t = self.d(tiles(r), self.ref_t)
        return dict(native=self.d(r, self.ref).item(), tile_worst=t.max().item(), tile_mean=t.mean().item())


class DreamSim:
    name = "dreamsim"

    def __init__(self, dev, cache="~/.cache/dreamsim"):
        import os

        from dreamsim import dreamsim

        self.m, _ = dreamsim(pretrained=True, device=dev, cache_dir=os.path.expanduser(cache))
        self.dev, self.ref = dev, None

    def prep(self, x):
        return F.interpolate((tiles(x.to(self.dev)) + 1) / 2, size=224, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)

    @torch.no_grad()
    def set_ref(self, x):
        self.ref = self.prep(x)

    @torch.no_grad()
    def score(self, r):
        a = self.prep(r)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = torch.cat([self.m(a[i : i + 4], self.ref[i : i + 4]) for i in range(0, len(a), 4)]).float()
        return dict(tile_worst=t.max().item(), tile_mean=t.mean().item())


def lab(x):
    """[..,3,H,W] sRGB in [0,1] -> CIELAB, D65."""
    c = torch.where(x > .04045, ((x + .055) / 1.055) ** 2.4, x / 12.92)
    r, g, b = c.unbind(-3)
    X, Y, Z = (.4124 * r + .3576 * g + .1805 * b) / .95047, .2126 * r + .7152 * g + .0722 * b, (.0193 * r + .1192 * g + .9505 * b) / 1.08883
    f = lambda t: torch.where(t > .008856, t.clamp_min(1e-9) ** (1 / 3), 7.787 * t + 16 / 116)
    fx, fy, fz = f(X), f(Y), f(Z)
    return torch.stack((116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)), -3)


def colour(r, x, scale=16):
    """Low-frequency colour error at 1/16 scale: PSNR and CIELAB Delta E."""
    a, b = F.avg_pool2d((r.float() + 1) / 2, scale).clamp(0, 1), F.avg_pool2d((x.float() + 1) / 2, scale).clamp(0, 1)
    e = (lab(a) - lab(b)).pow(2).sum(1).sqrt()
    return dict(psnr=(-10 * torch.log10(F.mse_loss(a, b) + 1e-12)).item(), de_mean=e.mean().item(), de_max=e.max().item())
