# ComfyUI/custom_nodes/BaseNodes/image_to_latent_blend.py
import torch

class ImageToLatentBlend:
    """
    Encode an IMAGE to LATENT using a VAE, then blend with noise:
      base  = keep_image * encoded + (1 - keep_image) * noise
      final = base + noise_injection * N(0,1)

    Notes:
      - Feed BHWC [0..1] directly to vae.encode (matches Comfy's built-in node).
      - Normalize channels to 3 (grayscale -> RGB, RGBA -> drop A).
      - Optional crop to multiples of 64 (disabled by default).
    """

    CATEGORY = "BaseNodes/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "keep_image": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "noise_injection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "crop_mult64": ("BOOLEAN", {"default": False}),
                "debug_shapes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode_and_blend"

    # ------------- helpers (BHWC path) -------------
    def _ensure_bhwc_rgb(self, img_bhwc: torch.Tensor) -> torch.Tensor:
        """Ensure BHWC with exactly 3 channels in [0..1]."""
        if img_bhwc.ndim != 4:
            raise ValueError(f"[ImageToLatentBlend] Expected BHWC 4D IMAGE, got {tuple(img_bhwc.shape)}")
        b, h, w, c = img_bhwc.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"[ImageToLatentBlend] Invalid H/W: {h}x{w} (upstream produced zero).")
        if c == 1:
            img_bhwc = img_bhwc.repeat(1, 1, 1, 3)
        elif c >= 3:
            img_bhwc = img_bhwc[..., :3]
        return img_bhwc

    def _crop_mult64_bhwc(self, x: torch.Tensor, debug: bool) -> torch.Tensor:
        b, h, w, c = x.shape
        h2 = (h // 64) * 64
        w2 = (w // 64) * 64
        if h2 == 0 or w2 == 0:
            raise ValueError(f"[ImageToLatentBlend] Too small to crop to 64-multiple: {h}x{w}.")
        if h2 != h or w2 != w:
            x = x[:, :h2, :w2, :].contiguous()
            if debug:
                print(f"[ImageToLatentBlend] cropped BHWC to mult64: {(b,h2,w2,c)}")
        return x

    def encode_and_blend(self, image, vae, keep_image, noise_injection, crop_mult64, debug_shapes):
        keep = float(max(0.0, min(1.0, keep_image)))
        ninj = float(max(0.0, min(1.0, noise_injection)))
        debug = bool(debug_shapes)

        # Use VAE device if available, else image.device
        device = getattr(vae, "device", image.device)

        # Keep BHWC → normalize channels to 3 → float32 on the right device
        x = self._ensure_bhwc_rgb(image).to(device=device, dtype=torch.float32)

        if debug:
            print(f"[ImageToLatentBlend] BHWC input to VAE: {tuple(x.shape)}, "
                  f"contig={x.is_contiguous()}, strides={x.stride()}")

        if bool(crop_mult64):
            x = self._crop_mult64_bhwc(x, debug)

        # IMPORTANT: Feed BHWC to vae.encode (Comfy handles BHWC→BCHW internally)
        with torch.no_grad():
            samples = vae.encode(x)  # -> BCHW latent (B,4,H/8,W/8)

        # Blend latent with noise
        noise_base = torch.randn_like(samples)
        base = keep * samples + (1.0 - keep) * noise_base
        if ninj > 0.0:
            base = base + ninj * torch.randn_like(base)

        if debug:
            print(f"[ImageToLatentBlend] latent shape: {tuple(base.shape)}")

        return ({"samples": base},)
