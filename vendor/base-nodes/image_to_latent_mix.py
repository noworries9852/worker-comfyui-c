# ComfyUI/custom_nodes/BaseNodes/image_to_latent_mix.py
import torch

class ImageToLatentMix:
    """
    Encode an IMAGE to LATENT and blend with random noise.

    Controls:
      - keep_amount (0..1): 0 -> pure random latent, 1 -> pure encoded image latent.
      - noise_injection (0..1): add extra Gaussian noise on top of the blend.

    Notes:
      - Expects a VAE object compatible with ComfyUI (encode BCHW -> latent).
      - Output is a standard LATENT dict: {"samples": tensor[B,4,H/8,W/8]}.
    """
    CATEGORY = "BaseNodes/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),  # BHWC in [0..1]
                "keep_amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"
                }),
                "noise_injection": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"

    # ---- helpers ----
    def _to_bchw_rgb(self, img_bhwc: torch.Tensor) -> torch.Tensor:
        """
        Ensure BCHW with exactly 3 channels (drop alpha if present, replicate if grayscale).
        """
        if img_bhwc.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor BHWC, got shape {tuple(img_bhwc.shape)}")
        b, h, w, c = img_bhwc.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image spatial size: H={h}, W={w}")

        if c >= 3:
            rgb = img_bhwc[..., :3]
        else:
            # c == 1 (MASK/gray) or c == 2 -> replicate first channel to reach 3
            first = img_bhwc[..., :1]
            rgb = first.repeat(1, 1, 1, 3)

        # BHWC -> BCHW (contiguous)
        return rgb.permute(0, 3, 1, 2).contiguous()

    def _encode_vae(self, vae, x_bchw: torch.Tensor) -> torch.Tensor:
        """
        Call VAE.encode and normalize return to a latent tensor.
        Some VAEs return a dict with "samples"; others may return a tensor directly.
        """
        out = vae.encode(x_bchw)
        if isinstance(out, dict):
            # Comfy standard: {"samples": latent}
            lat = out.get("samples", None)
            if lat is None:
                # some variants use "latent"
                lat = out.get("latent", None)
            if lat is None:
                raise RuntimeError("VAE.encode returned dict without 'samples'/'latent'.")
            return lat
        elif torch.is_tensor(out):
            return out
        else:
            raise RuntimeError(f"Unexpected VAE.encode return type: {type(out)}")

    # ---- op ----
    def apply(self, vae, image, keep_amount: float, noise_injection: float):
        # Clamp UI parameters
        k  = float(max(0.0, min(1.0, keep_amount)))
        ni = float(max(0.0, min(1.0, noise_injection)))

        # Convert BHWC -> BCHW RGB (3ch)
        x = self._to_bchw_rgb(image)

        # Encode to latent (B,4,H/8,W/8)
        latent = self._encode_vae(vae, x)
        latent = latent.contiguous()

        # Blend with random noise (same shape)
        noise = torch.randn_like(latent)
        mixed = k * latent + (1.0 - k) * noise

        # Optional extra additive noise
        if ni > 0.0:
            mixed = mixed + ni * torch.randn_like(mixed)

        mixed = mixed.contiguous()
        return ({"samples": mixed},)
