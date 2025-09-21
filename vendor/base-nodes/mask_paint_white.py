import torch
import torch.nn.functional as F

class MaskPaintWhite:
    """
    Pinta de branco (1.0) os pixels onde a máscara é 1, preservando o restante.
    - Ajusta automaticamente HxW (redimensiona a mask) e batch (se um dos batches for 1).
    - Mantém valores em 0..1.
    Inputs:
      - image: IMAGE [B,H,W,C] (0..1)
      - mask : MASK  [B?,H?,W?] (0..1)
    Output:
      - image: IMAGE resultante
    """
    CATEGORY = "BaseNodes/Mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    def _match_hw_mask(self, mask, target_h, target_w):
        # mask: [B,H,W] -> [B,1,H,W] -> resize -> [B,H,W]
        b, h, w = mask.shape
        if h == target_h and w == target_w:
            return mask
        m = mask.unsqueeze(1)
        m = F.interpolate(m, size=(target_h, target_w), mode="nearest")
        return m.squeeze(1)

    def _match_batch(self, a, m):
        ba, ha, wa, ca = a.shape
        bm, hm, wm = m.shape
        if ba == bm:
            return a, m
        if bm == 1:
            m = m.expand(ba, -1, -1)
            return a, m
        if ba == 1:
            a = a.expand(bm, -1, -1, -1)
            return a, m
        raise ValueError(f"Batch mismatch: {ba} vs {bm} (nenhum é 1).")

    def apply(self, image, mask):
        # Assegura device/dtype
        device = image.device
        mask = mask.to(device=device, dtype=image.dtype)

        # Ajusta dimensões
        b, h, w, c = image.shape
        mask = self._match_hw_mask(mask, h, w)
        image, mask = self._match_batch(image, mask)  # BHWC, [B,H,W]

        # Para operar: BHWC->BCHW e mask -> [B,1,H,W]
        img_bchw = image.movedim(-1, 1)              # [B,C,H,W]
        mask_bchw = mask.unsqueeze(1).clamp(0.0, 1.0)# [B,1,H,W]

        # out = (1 - mask)*img + mask*1.0
        out_bchw = (1.0 - mask_bchw) * img_bchw + mask_bchw

        out = out_bchw.movedim(1, -1).clamp(0.0, 1.0)  # volta p/ BHWC
        return (out,)
