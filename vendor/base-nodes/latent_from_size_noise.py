import torch

class LatentFromSizeNoise:
    """
    Gera um LATENT pelo tamanho (W,H) com injeção de ruído (0..1).
    NOTA: a imagem de entrada é ignorada (só garante a ordem do workflow).
    - batch do latent segue o batch da imagem.
    - width/height são ajustados para múltiplos de 8.
    - 'noise_amount' mistura 0=sem ruído, 1=ruído gaussiano completo.

    Inputs:
      - image: IMAGE [B,H,W,C] (não usado no cálculo)
      - width: INT
      - height: INT
      - noise_amount: FLOAT [0..1]
    Output:
      - latent: LATENT {"samples": tensor [B,4,H/8,W/8]}
    """
    CATEGORY = "BaseNodes/Latent"

    @classmethod
    def INPUT_TYPES(cls):
      return {
          "required": {
              "image": ("IMAGE",),
              "width": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 8}),
              "height": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 8}),
              "noise_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
          }
      }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "make"

    def _to_mult8(self, v: int) -> int:
        v = int(max(8, v))
        return v - (v % 8)

    def make(self, image, width, height, noise_amount):
        # Dispositivo/precisão: usa o da imagem por conveniência
        device = image.device
        dtype  = torch.float32  # latents geralmente em fp32

        B = int(image.shape[0])
        W = self._to_mult8(width)
        H = self._to_mult8(height)

        h8, w8 = H // 8, W // 8
        latent = torch.zeros((B, 4, h8, w8), dtype=dtype, device=device)

        amt = float(max(0.0, min(1.0, noise_amount)))
        if amt > 0.0:
            noise = torch.randn_like(latent)
            latent = latent + amt * noise

        return ({"samples": latent.contiguous()},)
