import torch

class ImageToLatentMix:
    """
    Encode an IMAGE to LATENT and blend with random noise.

    Controls:
      - keep_amount (0..1): 0 -> pure random latent, 1 -> pure encoded image latent.
      - noise_injection (0..1): add extra Gaussian noise on top of the blend.

    Notes:
      - Requires a VAE to encode the image into a latent space.
      - Output is a standard ComfyUI LATENT dict: {"samples": tensor[B,4,H/8,W/8]}.
    """
    CATEGORY = "BaseNodes/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
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

    def apply(self, vae, image, keep_amount: float, noise_injection: float):
        # Clamp UI parameters
        k  = float(max(0.0, min(1.0, keep_amount)))
        ni = float(max(0.0, min(1.0, noise_injection)))

        # Encode image to latent: BHWC -> BCHW -> latent[B,4,H/8,W/8]
        x = image.movedim(-1, 1)             # BHWC -> BCHW
        latent = vae.encode(x)               # tensor on VAE device/dtype
        device, dtype = latent.device, latent.dtype

        # Blend with random noise (same shape)
        noise = torch.randn_like(latent, device=device, dtype=dtype)
        mixed = k * latent + (1.0 - k) * noise

        # Optional extra additive noise (post-blend)
        if ni > 0.0:
            mixed = mixed + ni * torch.randn_like(mixed, device=device, dtype=dtype)

        mixed = mixed.contiguous()
        return ({"samples": mixed},)
