import torch
import torch.nn.functional as F

# --- util: batch/size helpers -------------------------------------------------
def _match_hw(img, target_h, target_w):
    b, h, w, c = img.shape
    if (h, w) == (target_h, target_w):
        return img
    bchw = img.movedim(-1, 1)  # BHWC -> BCHW
    bchw = F.interpolate(bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return bchw.movedim(1, -1)  # BCHW -> BHWC

def _match_batch(a, b):
    ba, _, _, _ = a.shape
    bb, _, _, _ = b.shape
    if ba == bb:
        return a, b
    if bb == 1:
        return a, b.expand(ba, -1, -1, -1)
    if ba == 1:
        return a.expand(bb, -1, -1, -1), b
    raise ValueError(f"Batch mismatch: {ba} vs {bb} (nenhum é 1)")

# --- util: color space (sRGB <-> Linear <-> XYZ <-> Lab, D65) ----------------
def _srgb_to_linear(u):
    a = 0.055
    return torch.where(u <= 0.04045, u / 12.92, ((u + a) / (1.0 + a)).pow(2.4))

def _linear_to_srgb(u):
    a = 0.055
    return torch.where(u <= 0.0031308, 12.92 * u, (1.0 + a) * u.clamp(min=0.0).pow(1.0 / 2.4) - a)

def _rgb_to_xyz(rgb):
    # rgb: [...,3] in linear
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.stack([x, y, z], dim=-1)

def _xyz_to_rgb(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r =  3.2404542 * x + (-1.5371385) * y + (-0.4985314) * z
    g = -0.9692660 * x +  1.8760108  * y +  0.0415560  * z
    b =  0.0556434 * x + (-0.2040259) * y +  1.0572252  * z
    return torch.stack([r, g, b], dim=-1)

def _f_lab(t):
    eps = 216.0 / 24389.0  # ~0.008856
    k   = 24389.0 / 27.0   # ~903.3
    return torch.where(t > eps, t.pow(1.0/3.0), (k * t + 16.0) / 116.0)

def _finv_lab(ft):
    eps = 216.0 / 24389.0
    return torch.where((ft ** 3) > eps, ft ** 3, (116.0 * ft - 16.0) / (24389.0 / 27.0))

# D65 reference white
_Xn, _Yn, _Zn = 0.95047, 1.0, 1.08883

def _rgb_to_lab(rgb_srgb):
    # rgb_srgb: [B,H,W,3] in sRGB 0..1
    lin = _srgb_to_linear(rgb_srgb.clamp(0.0, 1.0))
    xyz = _rgb_to_xyz(lin)
    xr = xyz[..., 0] / _Xn
    yr = xyz[..., 1] / _Yn
    zr = xyz[..., 2] / _Zn
    fx, fy, fz = _f_lab(xr), _f_lab(yr), _f_lab(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)  # L in [0..100] typically

def _lab_to_rgb(lab):
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    xr = _finv_lab(fx)
    yr = _finv_lab(fy)
    zr = _finv_lab(fz)

    xyz = torch.stack([xr * _Xn, yr * _Yn, zr * _Zn], dim=-1)
    lin = _xyz_to_rgb(xyz)
    srgb = _linear_to_srgb(lin)
    return srgb.clamp(0.0, 1.0)

# --- node ---------------------------------------------------------------------
class ColorBlendLAB:
    """
    Substitui o canal L* (luminosidade) da imagem colorida pelo L* da imagem P&B.
    - Ajusta automaticamente HxW (redimensiona bw p/ color) e batch (se um for 1).
    - Preserva alfa se a imagem colorida tiver 4 canais.
    - Saída: IMAGE BHWC contígua, 0..1, dtype/device da entrada colorida.
    Inputs:
      - bw_layer   : IMAGE [B?,H?,W?,C≥1] (0..1)
      - color_layer: IMAGE [B,H,W,C≥3]    (0..1)
    Output:
      - image: IMAGE (RGB ou RGBA se color_layer tinha alfa)
    """
    CATEGORY = "BaseNodes/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bw_layer": ("IMAGE",),
                "color_layer": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    def apply(self, bw_layer, color_layer):
        # Garantir device/dtype do resultado = do color
        device = color_layer.device
        dtype  = color_layer.dtype

        # Ajuste de tamanho e batch
        bc, hc, wc, cc = color_layer.shape
        bw_layer = _match_hw(bw_layer, hc, wc)
        color_layer, bw_layer = _match_batch(color_layer, bw_layer)

        # Se color tiver alfa, separe
        if cc >= 4:
            color_rgb = color_layer[..., :3]
            alpha     = color_layer[..., 3:4]
        else:
            color_rgb = color_layer[..., :3]
            alpha     = None

        # bw em RGB (se vier 1 canal, repete; se vier 3/4 usa os 3 primeiros)
        if bw_layer.shape[-1] == 1:
            bw_rgb = bw_layer.repeat(1, 1, 1, 3)
        else:
            bw_rgb = bw_layer[..., :3]

        # Converte para Lab
        color_lab = _rgb_to_lab(color_rgb.to(device=device, dtype=dtype))
        bw_lab    = _rgb_to_lab(bw_rgb.to(device=device, dtype=dtype))

        # Substitui L (mantendo a*, b* originais)
        # (Opcional: clampa L ao range típico 0..100)
        L_new = bw_lab[..., 0].clamp(0.0, 100.0)
        lab_mixed = torch.stack([L_new, color_lab[..., 1], color_lab[..., 2]], dim=-1)

        # Volta pra RGB
        out_rgb = _lab_to_rgb(lab_mixed).to(device=device, dtype=dtype).contiguous()

        # Reanexa alfa se existia
        if alpha is not None:
            out = torch.cat([out_rgb, alpha], dim=-1).contiguous()
        else:
            out = out_rgb

        return (out.clamp(0.0, 1.0),)
