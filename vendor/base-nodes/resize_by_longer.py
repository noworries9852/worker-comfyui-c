import torch
import torch.nn.functional as F

class ResizeByLongerSide:
    """
    Redimensiona preservando aspecto, definindo o lado MAIS LONGO = new_size.
    Inputs:
      - image: IMAGE (tensor [B,H,W,C], faixas 0..1)
      - new_size: INT (tamanho alvo do lado mais longo)
    Outputs:
      - image: IMAGE redimensionada
      - width: INT
      - height: INT
    """
    CATEGORY = "BaseNodes/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "new_size": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"

    def resize(self, image, new_size: int):
        # image: [B,H,W,C]
        b, h, w, c = image.shape
        target_long = int(max(1, min(int(new_size), 8192)))
        longer = max(h, w)

        if longer == target_long:
            # Sem mudança
            return (image, w, h)

        scale = target_long / float(longer)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        # BHWC -> BCHW -> interp -> BHWC
        img_bchw = image.movedim(-1, 1)
        out_bchw = F.interpolate(img_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        out = out_bchw.movedim(1, -1)

        return (out, new_w, new_h)
