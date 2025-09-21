import torch

class ThresholdQuantize:
    """
    Threshold + quantização de tons de cinza.
    Saída: RGB (3 canais), tensor BHWC contíguo, 0..1, mesmo dtype/device da imagem.
    """
    CATEGORY = "BaseNodes/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider"}),
                "intermediate_tones": ("INT", {"default": 0, "min": 0, "max": 254, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    def _luma(self, img):
        # img: [B,H,W,C]
        if img.shape[-1] >= 3:
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        else:
            y = img[..., 0]
        return y.clamp(0.0, 1.0)  # [B,H,W]

    def apply(self, image, threshold: float, intermediate_tones: int):
        device = image.device
        dtype  = image.dtype
        b, h, w, c = image.shape

        t  = float(max(0.0, min(1.0, threshold)))
        it = int(max(0, min(254, int(intermediate_tones))))
        eps = 1e-6

        y = self._luma(image).to(device=device, dtype=dtype)  # [B,H,W]

        if it == 0:
            # binário
            out_gray = (y >= t).to(dtype)
        else:
            # remap para pivotar em t -> 0.5 e quantizar
            denom_low  = max(t, eps)
            denom_high = max(1.0 - t, eps)
            remap = torch.where(
                y < t, 0.5 * (y / denom_low),
                0.5 + 0.5 * ((y - t) / denom_high),
            ).clamp(0.0, 1.0)

            levels = it + 2  # inclui preto e branco
            step = 1.0 / (levels - 1)
            q = torch.round(remap / step) * step
            q = q.clamp(0.0, 1.0)

            out_gray = torch.where(
                q <= 0.5, (q / 0.5) * t,
                t + ((q - 0.5) / 0.5) * (1.0 - t),
            ).clamp(0.0, 1.0)

        # Converte para RGB real (sem expand) e contíguo
        out_rgb = torch.stack([out_gray, out_gray, out_gray], dim=-1)  # [B,H,W,3]
        out_rgb = out_rgb.to(device=device, dtype=dtype).contiguous()

        return (out_rgb,)
