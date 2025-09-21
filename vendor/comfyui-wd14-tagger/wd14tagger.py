# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags
# Modified to optionally read models from an env var (WD14_MODELS_DIR),
# and to fail gracefully (log but do not interrupt execution) when models or
# dependencies are missing. The download function is a no-op for compatibility.

import asyncio
import csv
import os
import sys
import numpy as np
from PIL import Image
import aiohttp
from aiohttp import web

import comfy.utils
from server import PromptServer
import folder_paths
from .pysssss import (
    get_ext_dir,
    get_comfy_dir,
    download_to_file,              # kept for compatibility
    update_node_status,
    wait_for_async,
    get_extension_config,
    log,
)

# Try to import onnxruntime, but don't crash if it's unavailable.
try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
    _ORT_IMPORT_ERROR = None
except Exception as _e:
    ort = None
    InferenceSession = None
    _ORT_IMPORT_ERROR = _e

# Ensure "comfy" local path is available (kept from original)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# ---- Configuration ---------------------------------------------------------

config = get_extension_config()
if not isinstance(config, dict):
    config = {}
config.setdefault("models", {})
config.setdefault("settings", {})

defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "HF_ENDPOINT": "https://huggingface.co",
}
defaults.update(config.get("settings", {}))

# Resolve models_dir with environment override
# If WD14_MODELS_DIR is set, use it. Otherwise follow the original behavior.
models_dir_env = os.getenv("WD14_MODELS_DIR")
if models_dir_env:
    models_dir = models_dir_env
    try:
        os.makedirs(models_dir, exist_ok=True)
    except Exception as e:
        log(f"Unable to create WD14_MODELS_DIR at '{models_dir}': {e}", "WARN", True)
elif "wd14_tagger" in folder_paths.folder_names_and_paths:
    try:
        models_dir = folder_paths.get_folder_paths("wd14_tagger")[0]
        os.makedirs(models_dir, exist_ok=True)
    except Exception as e:
        # Fallback to extension-local models directory
        log(f"Failed to use folder_paths 'wd14_tagger': {e}. Falling back to extension models dir.", "WARN", True)
        models_dir = get_ext_dir("models", mkdir=True)
else:
    models_dir = get_ext_dir("models", mkdir=True)

known_models = list(config["models"].keys())

# Log provider info (if onnxruntime is importable)
if _ORT_IMPORT_ERROR is not None:
    log(f"onnxruntime is not available: {_ORT_IMPORT_ERROR}", "WARN", True)
else:
    try:
        log("Available ORT providers: " + ", ".join(ort.get_available_providers()), "DEBUG", True)
    except Exception as e:
        log(f"Failed to query ORT providers: {e}", "WARN", True)
    log("Using ORT providers: " + ", ".join(defaults["ortProviders"]), "DEBUG", True)


# ---- Helpers ----------------------------------------------------------------

def _has_model_files(prefix_dir: str, model_name: str) -> bool:
    """Check if both .onnx and .csv files exist for a given model."""
    onnx_path = os.path.join(prefix_dir, f"{model_name}.onnx")
    csv_path = os.path.join(prefix_dir, f"{model_name}.csv")
    return os.path.isfile(onnx_path) and os.path.isfile(csv_path)

def get_installed_models():
    """Return a list of model filenames ('.onnx') that also have a matching '.csv'."""
    try:
        if not os.path.isdir(models_dir):
            return []
        items = [f for f in os.listdir(models_dir) if f.endswith(".onnx")]
        items = [m for m in items if os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
        return items
    except Exception as e:
        log(f"Failed to list models in '{models_dir}': {e}", "WARN", True)
        return []


# ---- Core Tagging -----------------------------------------------------------

async def tag(image, model_name, threshold=0.35, character_threshold=0.85,
              exclude_tags="", replace_underscore=True, trailing_comma=False,
              client_id=None, node=None):
    """
    Run tagging over a single PIL image.
    If requirements or model files are missing, log and return an empty string.
    """
    # Normalize model name (strip .onnx suffix if present)
    if model_name.endswith(".onnx"):
        model_name = model_name[:-5]

    # Ensure ORT is available
    if InferenceSession is None or ort is None:
        log("onnxruntime is unavailable; skipping WD14 tagging and returning empty result.", "WARN", True)
        return ""

    # Ensure model files exist
    installed = list(get_installed_models())
    if not any(model_name + ".onnx" == m for m in installed):
        log(f"Model '{model_name}' not found in '{models_dir}'. "
            f"Expected files: '{model_name}.onnx' and '{model_name}.csv'. Skipping inference.", "WARN", True)
        # Keep UI responsive without raising
        return ""

    # Try to build inference session
    model_path = os.path.join(models_dir, model_name + ".onnx")
    try:
        model = InferenceSession(model_path, providers=defaults["ortProviders"])
    except Exception as e:
        log(f"Failed to create InferenceSession for '{model_path}': {e}. Skipping inference.", "WARN", True)
        return ""

    # Try to infer expected input size
    try:
        input_meta = model.get_inputs()[0]
        height = input_meta.shape[1]
        if height is None:
            # Fallback to a common default if the model doesn't expose a static shape
            height = 448
    except Exception as e:
        log(f"Failed to read model input shape: {e}. Using default height=448.", "WARN", True)
        height = 448

    # Preprocess: reduce to max size and pad with white to a square
    try:
        ratio = float(height) / max(image.size)
        new_size = tuple(int(x * ratio) for x in image.size)
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (height, height), (255, 255, 255))
        square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
        image = np.array(square).astype(np.float32)
        image = image[:, :, ::-1]  # RGB -> BGR
        image = np.expand_dims(image, 0)
    except Exception as e:
        log(f"Failed during image preprocessing: {e}. Returning empty result.", "WARN", True)
        return ""

    # Read tags from CSV
    tags = []
    general_index = None
    character_index = None
    csv_path = os.path.join(models_dir, model_name + ".csv")
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if general_index is None and len(row) > 2 and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and len(row) > 2 and row[2] == "4":
                    character_index = reader.line_num - 2
                tag_name = row[1] if len(row) > 1 else ""
                if replace_underscore:
                    tags.append(tag_name.replace("_", " "))
                else:
                    tags.append(tag_name)
    except Exception as e:
        log(f"Failed to read CSV '{csv_path}': {e}. Returning empty result.", "WARN", True)
        return ""

    # Run inference
    try:
        label_name = model.get_outputs()[0].name
        probs = model.run([label_name], {model.get_inputs()[0].name: image})[0]
    except Exception as e:
        log(f"Inference failed: {e}. Returning empty result.", "WARN", True)
        return ""

    # Postprocess
    try:
        result = list(zip(tags, probs[0]))
        # rating = max(result[:general_index], key=lambda x: x[1])  # unused
        if general_index is None or character_index is None:
            # Fallback: treat everything as general if CSV didn't expose indices
            general_index = 0 if general_index is None else general_index
            character_index = len(result) if character_index is None else character_index

        general = [item for item in result[general_index:character_index] if item[1] > threshold]
        character = [item for item in result[character_index:] if item[1] > character_threshold]

        all_tags = character + general
        remove = [s.strip() for s in (exclude_tags or "").lower().split(",") if s.strip()]
        all_tags = [tag for tag in all_tags if tag[0] not in remove]

        res = ("" if trailing_comma else ", ").join(
            (item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all_tags)
        )
        print(res)
        return res
    except Exception as e:
        log(f"Failed during postprocessing: {e}. Returning empty result.", "WARN", True)
        return ""


# ---- Download (Fake / No-Op) -----------------------------------------------

async def download_model(model, client_id, node):
    """
    Fake download function kept only for compatibility.
    It logs a message and returns HTTP 200 without downloading.
    """
    log(f"Download disabled: please place '{model}.onnx' and '{model}.csv' in '{models_dir}'.", "INFO", True)
    update_node_status(client_id, node, None)
    return web.Response(status=200)


# ---- HTTP Endpoint ----------------------------------------------------------

@PromptServer.instance.routes.get("/pysssss/wd14tagger/tag")
async def get_tags(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    typ = request.query.get("type", "output")
    if typ not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = get_comfy_dir(typ)
    image_path = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    c = os.path.commonpath((image_path, target_dir))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    image = Image.open(image_path)

    models = get_installed_models()
    default = defaults["model"] + ".onnx"
    # Choose default, or first available, or skip gracefully if none
    if len(models) == 0:
        log(f"No WD14 models found in '{models_dir}'. Returning empty tags.", "WARN", True)
        return web.json_response(await tag(image, defaults["model"], client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))

    model_choice = default if default in models else os.path.splitext(models[0])[0]
    return web.json_response(await tag(image, model_choice, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))


# ---- ComfyUI Node -----------------------------------------------------------

class WD14Tagger:
    @classmethod
    def INPUT_TYPES(s):
        # Include known models from config plus any extra .onnx present in the directory
        try:
            extra = [name for name, _ in (os.path.splitext(m) for m in get_installed_models()) if name not in known_models]
        except Exception:
            extra = []
        models = known_models + extra
        if defaults["model"] not in models:
            models = [defaults["model"]] + models
        return {"required": {
            "image": ("IMAGE", ),
            "model": (models, {"default": defaults["model"]}),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(self, image, model, threshold, character_threshold,
            exclude_tags="", replace_underscore=False, trailing_comma=False):
        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []
        for i in range(tensor.shape[0]):
            img = Image.fromarray(tensor[i])
            tags.append(wait_for_async(lambda: tag(
                img, model, threshold, character_threshold, exclude_tags,
                replace_underscore, trailing_comma
            )))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}


NODE_CLASS_MAPPINGS = {
    "WD14Tagger|pysssss": WD14Tagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14Tagger|pysssss": "WD14 Tagger 🐍",
}
