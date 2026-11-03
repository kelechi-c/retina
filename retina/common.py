"""Shared pieces: TAEQI2.1 (f16c64, madebyollin), test images, and the two judges the user named (DreamSim, DINOv3), scored per 256 tile."""
import glob
import os

import numpy as np
import safetensors.torch as stt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from .taesd import TAESD

DINO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
PATCH, TILE = 16, 256
GROUPS = ("mondrian", "simple", "ornate", "texture", "dense", "waldo")


def load_tae(dev, repo="madebyollin/taeqi2_1", name="taeqi2_1.safetensors", arch="f16"):
    m = TAESD(encoder_path=None, decoder_path=None, latent_channels=64, arch_variant=arch, image_channels=4)
    sd = {}
    for k, v in stt.load_file(hf_hub_download(repo, name)).items():
        ed, _, i, *s = k.split(".")
        sd[".".join([ed, str(int(i) + (ed == "decoder")), *s])] = v  # model card's key conversion
    m.load_state_dict(sd)
    return m.eval().to(dev).requires_grad_(False)


def enc(tae, x):
    """[B,3,H,W] in [-1,1] -> latent [B,64,H/16,W/16]. Opaque alpha."""
    x = (x + 1) / 2
    return tae.encoder(torch.cat((x, torch.ones_like(x[:, :1])), 1))


def dec(tae, z):
    """Latent -> [B,3,H,W] in [-1,1], alpha dropped."""
    return tae.decoder(z)[:, :3].clamp(0, 1) * 2 - 1


def down(x, side):
    return F.interpolate(x, size=side, mode="bicubic", antialias=True, align_corners=False)


def up(x, side):
    return F.interpolate(x, size=side, mode="bicubic", align_corners=False).clamp(-1, 1)


def to8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1


def save_png(x, path):
    a = ((x[0].clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(a).save(path)


def load_img(path, side):
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    l, u = (im.width - s) // 2, (im.height - s) // 2
    a = np.asarray(im.crop((l, u, l + s, u + s)).resize((side, side), Image.LANCZOS)).copy()
    return torch.from_numpy(a).permute(2, 0, 1).float()[None] / 127.5 - 1


def test_files(root):
    out = []
    for d in ("data", "data2"):
        out += sorted(glob.glob(f"{root}/{d}/*/*.jpg"))
    return out


def key(f):
    return f"{os.path.basename(os.path.dirname(f))}__{os.path.basename(f)[:-4]}"


class Judges:
    """Distances, 0 = identical. DreamSim on each 256 tile at its 224 input; DINOv3 ViT-B/16 at native resolution, patch cosine distance pooled per tile."""

    def __init__(self, dev, dream_cache=None):
        from dreamsim import dreamsim
        from transformers import AutoModel

        self.dev = dev
        self.dino = AutoModel.from_pretrained(DINO).eval().to(dev).requires_grad_(False)
        dream_cache = dream_cache or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "dreamsim")
        self.dream, _ = dreamsim(pretrained=True, device=dev, cache_dir=dream_cache)
        self.mean = torch.tensor((.485, .456, .406), device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor((.229, .224, .225), device=dev).view(1, 3, 1, 1)

    def dfeat(self, x):
        x = (((x.to(self.dev) + 1) / 2) - self.mean) / self.std
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = self.dino(pixel_values=x).last_hidden_state.float()
        return F.normalize(t[:, -h * w :], dim=-1).view(len(x), h, w, -1)

    def tiles(self, x):
        return x.unfold(2, TILE, TILE).unfold(3, TILE, TILE).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TILE, TILE)

    def prep(self, x):
        return F.interpolate((self.tiles(x.to(self.dev)) + 1) / 2, size=224, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)

    @torch.no_grad()
    def set_ref(self, x):
        self.x = x.to(self.dev)
        self.rd = self.dfeat(self.x)
        self.rt = self.prep(self.x)

    @torch.no_grad()
    def score(self, r):
        """r [1,3,S,S] -> dict of tile arrays and summaries; r is rounded to 8 bit first, as a saved PNG would be."""
        r = to8(r.to(self.dev))
        d = 1 - (self.dfeat(r) * self.rd).sum(-1)
        di = F.avg_pool2d(d[None], TILE // PATCH).flatten()
        a = self.prep(r)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ds = torch.cat([self.dream(a[i : i + 8], self.rt[i : i + 8]).reshape(-1) for i in range(0, len(a), 8)]).float()
        mse = ((r - self.x) / 2).pow(2).mean().item()
        return dict(ds_worst=ds.max().item(), ds_mean=ds.mean().item(), di_worst=di.max().item(), di_mean=di.mean().item(), di_q95=d.flatten().quantile(.95).item(), psnr=-10 * float(np.log10(max(mse, 1e-12))), ds=ds.cpu().tolist(), di=di.cpu().tolist())


CODECS = dict(
    taeqi=dict(repo="madebyollin/taeqi2_1", file="taeqi2_1.safetensors", arch="f16", z=64, f=16, rgba=True),
    taef2=dict(repo="madebyollin/taef2", file="taef2.safetensors", arch="flux_2", z=32, f=8, rgba=False),
    taef1=dict(repo="madebyollin/taef1", file="diffusion_pytorch_model.safetensors", arch=None, z=16, f=8, rgba=False),
)


class Codec:
    """One of Ollin's tiny autoencoders. Images in [-1,1]; `floats` counts what a grid of side n costs."""

    def __init__(self, name, dev):
        c = CODECS[name]
        self.name, self.z, self.f, self.rgba = name, c["z"], c["f"], c["rgba"]
        m = TAESD(encoder_path=None, decoder_path=None, latent_channels=c["z"], arch_variant=c["arch"], image_channels=4 if c["rgba"] else 3)
        sd = {}
        for k, v in stt.load_file(hf_hub_download(c["repo"], c["file"])).items():
            ed, _, i, *s = k.split(".")
            sd[".".join([ed, str(int(i) + (ed == "decoder")), *s])] = v
        m.load_state_dict(sd)
        self.m = m.eval().to(dev).requires_grad_(False)

    def enc(self, x):
        x = (x + 1) / 2
        return self.m.encoder(torch.cat((x, torch.ones_like(x[:, :1])), 1) if self.rgba else x)

    def dec(self, z):
        return self.m.decoder(z)[:, :3].clamp(0, 1) * 2 - 1
