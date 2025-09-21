
import torch

class AdjustSaturation:
    """
    Ajusta a saturação preservando luminância (luma Rec.709).
    Fórmula: out_rgb = y + (rgb - y) * saturation
      - saturation = 0.0  -> dessaturado (P&B)
      - saturation = 1.0  -> original
      - saturation > 1.0  -> mais saturado
    Mantém alfa se existir.

    Inputs:
      - image: IMAGE [B,H,W,C>=3] (0..1)
      - saturation: FLOAT [0..2] (padrão 1.0)
    Output:
      - image: IMAGE (mesmo nº de canais da entrada; se RGBA, preserva A)
    """
    CATEGORY = "BaseNodes/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    def _luma(self, rgb):
        # rgb: [B,H,W,3]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y.unsqueeze(-1)  # [B,H,W,1]

    def apply(self, image, saturation: float):
        sat = float(max(0.0, min(2.0, saturation)))

        b, h, w, c = image.shape
        device, dtype = image.device, image.dtype

        # separa canais
        if c >= 4:
            rgb  = image[..., :3]
            alpha = image[..., 3:4]
        else:
            rgb  = image[..., :3]
            alpha = None

        # luma e ajuste
        y = self._luma(rgb)
        out_rgb = (y + (rgb - y) * sat).clamp(0.0, 1.0)

        if alpha is not None:
            out = torch.cat([out_rgb, alpha], dim=-1).to(device=device, dtype=dtype).contiguous()
        else:
            out = out_rgb.to(device=device, dtype=dtype).contiguous()

        return (out,)
