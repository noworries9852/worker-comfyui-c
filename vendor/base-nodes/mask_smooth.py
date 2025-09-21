import math
import torch
import torch.nn.functional as F

class MaskSmooth:
    """
    Suaviza bordas de máscara:
      1) (opcional) Closing morfológico (dilatação→erosão) para fechar rebarbas
      2) (opcional) Feather (blur gaussiano) para bordas mais macias
      3) (opcional) Threshold suave para voltar a binário sem serrilhado
    Entradas:
      - mask: MASK [B,H,W] (0..1)
      - morph_radius: INT (px do kernel morfológico; 0=desliga)
      - feather_px  : INT (px de feather/blur; 0=desliga)
      - apply_threshold: BOOL (se True, binariza suavemente)
      - threshold   : FLOAT (0..1) ponto de corte
      - softness    : FLOAT (0..0.5) "largura" da transição no threshold suave
    Saída:
      - mask: MASK [B,H,W] (0..1), contígua
    """
    CATEGORY = "BaseNodes/Mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "morph_radius": ("INT", {"default": 1, "min": 0, "max": 32, "step": 1}),
                "feather_px":   ("INT", {"default": 6, "min": 0, "max": 128, "step": 1}),
                "apply_threshold": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "softness":  ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"

    # ----------------- helpers -----------------
    def _dilate(self, m, r):
        if r <= 0: return m
        # m: [B,1,H,W]
        return F.max_pool2d(m, kernel_size=2*r+1, stride=1, padding=r)

    def _erode(self, m, r):
        if r <= 0: return m
        return -F.max_pool2d(-m, kernel_size=2*r+1, stride=1, padding=r)

    def _closing(self, m, r):
        if r <= 0: return m
        return self._erode(self._dilate(m, r), r)

    def _gaussian_kernel1d(self, sigma, device, dtype):
        # radius ~ 3*sigma
        r = max(1, int(math.ceil(3.0 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum().clamp_min(1e-8)
        return k, r

    def _gaussian_blur(self, m, sigma):
        # m: [B,1,H,W]
        if sigma <= 0: return m
        B, C, H, W = m.shape
        device, dtype = m.device, m.dtype
        k1d, r = self._gaussian_kernel1d(sigma, device, dtype)

        # separable: vertical, depois horizontal
        ky = k1d.view(1, 1, -1, 1)                # [1,1,Ky,1]
        kx = k1d.view(1, 1, 1, -1)                # [1,1,1,Kx]

        m = F.pad(m, (0,0, r, r), mode="reflect")
        m = F.conv2d(m, ky, padding=0, groups=1)

        m = F.pad(m, (r, r, 0, 0), mode="reflect")
        m = F.conv2d(m, kx, padding=0, groups=1)

        return m

    def _smoothstep(self, x, edge0, edge1):
        # transição suave (0..1)
        t = ((x - edge0) / max(edge1 - edge0, 1e-6)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # ----------------- op -----------------
    def process(self, mask, morph_radius, feather_px, apply_threshold, threshold, softness):
        # ensure tensor BHWC->BCHW-like shape for ops
        b, h, w = mask.shape
        m = mask.to(dtype=torch.float32).unsqueeze(1)  # [B,1,H,W], float32
        m = m.clamp(0.0, 1.0)

        # 1) closing morfológico
        if morph_radius > 0:
            m = self._closing(m, int(morph_radius)).clamp(0.0, 1.0)

        # 2) feather (blur gaussiano)
        if feather_px > 0:
            # interprete feather_px como sigma diretamente (bom controle visual)
            m = self._gaussian_blur(m, float(feather_px)).clamp(0.0, 1.0)

        # 3) threshold suave opcional
        if bool(apply_threshold):
            t = float(max(0.0, min(1.0, threshold)))
            s = float(max(0.0, min(0.5, softness)))
            edge0 = max(0.0, t - s)
            edge1 = min(1.0, t + s)
            m = self._smoothstep(m, edge0, edge1)

            # se quiser máscara binária dura, pode adicionar: m = (m >= 0.5).float()
            # mas mantemos suave para anti-serrilhado.

        m = m.clamp(0.0, 1.0).squeeze(1).contiguous()  # [B,H,W]
        return (m,)
