"""The VAEs behind the tokenizer: FLUX.2-dev (default), FLUX.1 and Qwen-Image 2.1. Each says how many pixels one latent cell covers (`cell`) and how many cells per side make one token (`pack`)."""
import torch
from diffusers import AutoencoderKL


class Flux1:
    """FLUX.1: f8 with 16 channels, 2 x 2 cells to a 64-float token."""

    cell, pack = 8, 2

    def __init__(self, dev):
        self.vae = self.load().to(dev).eval().requires_grad_(False)

    def load(self):
        return AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="vae", torch_dtype=torch.float32)

    @torch.no_grad()
    def enc(self, x):
        return self.vae.encode(x).latent_dist.mode()

    @torch.no_grad()
    def dec(self, z):
        return self.vae.decode(z).sample.clamp(-1, 1)


class Flux2(Flux1):
    """FLUX.2: f8 with 32 channels, 2 x 2 cells to a 128-float token."""

    def load(self):
        from diffusers import AutoencoderKLFlux2

        return AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.float32)


class Qwen:
    """Qwen-Image 2.1: f16 with 64 channels, one cell to a token. RGB in and out, one image as one frame; alpha is dropped."""

    cell, pack = 16, 1

    def __init__(self, dev):
        from diffusers import AutoencoderKLQwenImage21

        self.dev = dev
        self.vae = AutoencoderKLQwenImage21.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="vae", torch_dtype=torch.bfloat16).eval().to(dev).requires_grad_(False)

    @torch.no_grad()
    def enc(self, x):
        rgba = torch.cat((x, torch.ones_like(x[:, :1])), 1)[:, :, None].to(self.dev, torch.bfloat16)
        return self.vae.encode(rgba).latent_dist.mode()[:, :, 0]

    @torch.no_grad()
    def dec(self, z):
        return self.vae.decode(z[:, :, None].to(self.dev, torch.bfloat16)).sample[:, :, 0].float()[:, :3].clamp(-1, 1)


VAES = dict(flux=Flux1, flux2=Flux2, qwen=Qwen)
