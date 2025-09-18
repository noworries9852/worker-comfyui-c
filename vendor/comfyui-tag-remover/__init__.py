"""
ComfyUI Tag Remover Custom Node

テキストから指定されたタグとその内容を除去するカスタムノード
LM StudioのThinkモデルの<think>タグ除去などに使用可能
"""

from .tag_remover_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ComfyUIに公開するマッピング
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']