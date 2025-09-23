from .tag_manager import TagManager
from .resize_by_longer import ResizeByLongerSide
from .multiply_blend import MultiplyBlend
from .mask_paint_white import MaskPaintWhite
from .threshold_quantize import ThresholdQuantize
from .color_blend_lab import ColorBlendLAB
from .mask_smooth import MaskSmooth
from .color_saturation import AdjustSaturation
from .latent_from_size_noise import LatentFromSizeNoise  # ⬅️ novo

NODE_CLASS_MAPPINGS = {
    "BaseNodesTagManager": TagManager,
    "BaseNodesResizeByLongerSide": ResizeByLongerSide,
    "BaseNodesMultiplyBlend": MultiplyBlend,
    "BaseNodesMaskPaintWhite": MaskPaintWhite,
    "BaseNodesThresholdQuantize": ThresholdQuantize,
    "BaseNodesColorBlendLAB": ColorBlendLAB,
    "BaseNodesMaskSmooth": MaskSmooth,
    "BaseNodesAdjustSaturation": AdjustSaturation,
    "BaseNodesLatentFromSizeNoise": LatentFromSizeNoise,  # ⬅️ novo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaseNodesTagManager": "BN • Tag Manager",
    "BaseNodesResizeByLongerSide": "BN • Resize by Longer Side",
    "BaseNodesMultiplyBlend": "BN • Multiply (Blend)",
    "BaseNodesMaskPaintWhite": "BN • Mask → Paint White",
    "BaseNodesThresholdQuantize": "BN • Threshold (Quantize)",
    "BaseNodesColorBlendLAB": "BN • Color Blend (LAB)",
    "BaseNodesMaskSmooth": "BN • Mask Smooth",
    "BaseNodesAdjustSaturation": "BN • Saturation",
    "BaseNodesLatentFromSizeNoise": "BN • Latent (Size+Noise)",  # ⬅️ novo
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
