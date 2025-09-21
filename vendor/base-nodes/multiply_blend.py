import torch
import torch.nn.functional as F

class MultiplyBlend:
    """
    Faz o blend Multiply entre duas imagens, com controle de intensidade (0..1).
    out = lerp(A, A*B, amount) = (1-amount)*A + amount*(A*B)
    - Ajusta automaticamente HxW e batch (se um dos batches for 1).
    - Mantém valores em 0..1.
    Inputs:
      - image_a: IMAGE [B,H,W,C] (0..1)
      - image_b: IMAGE [B?,H?,W?,C] (0..1)
      - amount: FLOAT [0..1] (força do efeito)
    Outputs:
      - image: IMAGE resultante
    """
    CATEGORY = "BaseNodes/Blend"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"

    def _match_hw(self, img, target_h, target_w):
        b, h, w, c = img.shape
        if h == target_h and w == target_w:
            return img
        bchw = img.movedim(-1, 1)            # BHWC -> BCHW
        bchw = F.interpolate(bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return bchw.movedim(1, -1)           # BCHW -> BHWC

    def _match_batch(self, a, b):
        ba, ha, wa, ca = a.shape
        bb, hb, wb, cb = b.shape
        if ba == bb:
            return a, b
        if bb == 1:
            b = b.expand(ba, -1, -1, -1)
            return a, b
        if ba == 1:
            a = a.expand(bb, -1, -1, -1)
            return a, b
        raise ValueError(f"Batch mismatch: {ba} vs {bb} (nenhum é 1).")

    def blend(self, image_a, image_b, amount: float):
        # Garante limites do amount
        t = float(max(0.0, min(1.0, amount)))

        # Ajusta dimensões
        ba, ha, wa, ca = image_a.shape
        image_b = self._match_hw(image_b, ha, wa)
        image_a, image_b = self._match_batch(image_a, image_b)

        # BHWC -> operar no mesmo layout
        # Multiply + lerp: out = (1-t)*A + t*(A*B)
        out = (1.0 - t) * image_a + t * (image_a * image_b)

        # Segurança numérica
        out = out.clamp(0.0, 1.0)
        return (out,)
