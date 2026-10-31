"""Qwen-Image 2.1 VAE, RGB in and out, one image as one video frame. Input gets an opaque alpha channel, output drops it."""
import numpy as np
import torch
from PIL import Image

REPO = "Qwen/Qwen-Image-2.1-Turbo"


def load_vae(dev, dtype=torch.bfloat16):
    from diffusers import AutoencoderKLQwenImage21

    return AutoencoderKLQwenImage21.from_pretrained(REPO, subfolder="vae", torch_dtype=dtype).eval().to(dev).requires_grad_(False)


def load_img(path, side):
    """Square centre crop, Lanczos resize, [3,H,W] in [-1,1]."""
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    l, u = (im.width - s) // 2, (im.height - s) // 2
    a = np.asarray(im.crop((l, u, l + s, u + s)).resize((side, side), Image.LANCZOS)).copy()
    return torch.from_numpy(a).permute(2, 0, 1).float() / 127.5 - 1


@torch.no_grad()
def encode(vae, x):
    """[B,3,H,W] in [-1,1] -> raw latent [B,64,H/16,W/16], the posterior mean."""
    p = next(vae.parameters())
    rgba = torch.cat((x, torch.ones_like(x[:, :1])), 1)[:, :, None].to(p.device, p.dtype)
    return vae.encode(rgba).latent_dist.mode()[:, :, 0]


@torch.no_grad()
def decode(vae, z):
    """Raw latent -> ([B,3,H,W] RGB in [-1,1], [B,1,H,W] alpha), float32."""
    p = next(vae.parameters())
    y = vae.decode(z[:, :, None].to(p.device, p.dtype)).sample[:, :, 0].float()
    return y[:, :3].clamp(-1, 1), y[:, 3:]


def norm_stats(vae):
    c = vae.config
    return torch.tensor(c.latents_mean).view(1, -1, 1, 1), torch.tensor(c.latents_std).view(1, -1, 1, 1)
