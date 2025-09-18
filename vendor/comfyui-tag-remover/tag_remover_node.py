import re


class TagRemoverNode:
    """
    ComfyUIカスタムノード：テキストから指定されたタグとその内容を除去する
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "tag_name": ("STRING", {"default": "think"}),
            },
            "optional": {
                "remove_empty_lines": ("BOOLEAN", {"default": True}),
                "trim_whitespace": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "remove_tags"
    CATEGORY = "text/processing"
    
    def remove_tags(self, text, tag_name, remove_empty_lines=True, trim_whitespace=True):
        """
        指定されたタグとその内容をテキストから除去する
        
        Args:
            text: 処理対象のテキスト
            tag_name: 除去するタグ名（例: "think"）
            remove_empty_lines: 空行を除去するかどうか
            trim_whitespace: 前後の空白を除去するかどうか
        
        Returns:
            処理済みのテキスト
        """
        if not text or not tag_name:
            return (text,)
        
        # タグのパターンを作成（大文字小文字を区別しない）
        pattern = f"<{re.escape(tag_name)}\\b[^>]*>.*?</{re.escape(tag_name)}>"
        
        # タグとその内容を除去（改行も含めて処理）
        processed_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        
        # 空行を除去
        if remove_empty_lines:
            lines = processed_text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            processed_text = '\n'.join(non_empty_lines)
        
        # 前後の空白を除去
        if trim_whitespace:
            processed_text = processed_text.strip()
        
        return (processed_text,)


# ノードのマッピング
NODE_CLASS_MAPPINGS = {
    "TagRemoverNode": TagRemoverNode
}

# ノードの表示名
NODE_DISPLAY_NAME_MAPPINGS = {
    "TagRemoverNode": "Tag Remover"
}