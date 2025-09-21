import re

def _tokenize(text: str):
    # remove quebras internas e colapsa espaços
    text = text.replace("\r", " ").replace("\n", " ")
    # separa por vírgulas ou espaços (1+), eliminando vazios e trims
    parts = re.split(r"[,\s]+", text)
    return [p.strip() for p in parts if p.strip()]

def _clean_prompt(text: str, blacklist_ci: set[str]) -> str:
    tokens = _tokenize(text)
    seen = set()  # dedupe case-insitive (mantém 1ª ocorrência)
    out = []
    for t in tokens:
        key = t.lower()
        if key in blacklist_ci:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    # junta com vírgula + espaço
    return ", ".join(out) if out else ""

class TagManager:
    """
    Limpa e normaliza tags no formato: "tag1, tag2 tag2, tag3, ...".
    Regras:
    - remove espaços duplos e quebras dentro de CADA prompt (mas mantém 1 linha por prompt no output)
    - remove vírgulas vazias (", ,")
    - deduplica tags (case-insensitive, preserva a 1ª)
    - remove tags presentes na blacklist (case-insensitive)
    - garante vírgula no fim de cada linha exceto a última
    Saída: STRING com linhas separadas por '\n' (não lista).
    """
    CATEGORY = "BaseNodes/Prompts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt1": ("STRING", {"multiline": True, "default": ""}),
                "prompt2": ("STRING", {"multiline": True, "default": ""}),
                "prompt3": ("STRING", {"multiline": True, "default": ""}),
                "prompt4": ("STRING", {"multiline": True, "default": ""}),
                "blacklist": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process"

    def process(self, prompt1, prompt2, prompt3, prompt4, blacklist):
        bl_set = {t.lower() for t in _tokenize(blacklist)}
        inputs = [prompt1, prompt2, prompt3, prompt4]

        lines = []
        for s in inputs:
            cleaned = _clean_prompt(s, bl_set)
            if cleaned:
                lines.append(cleaned)

        # vírgula no final de cada linha, exceto a última
        for i in range(len(lines) - 1):
            if not lines[i].endswith(","):
                lines[i] += ","

        # garante que a última não termina em vírgula
        if lines and lines[-1].endswith(","):
            lines[-1] = lines[-1].rstrip(",").rstrip()

        return ("\n".join(lines),)
