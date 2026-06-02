import torch
import torchvision
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import time
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()
torchvision.set_video_backend("pyav")

# --- Diagnostics Block ---
try:
    import triton
    print(f"[Diagnostics] Triton version installed: {triton.__version__}")
except ImportError:
    print("[Diagnostics] Triton is not installed (This is normal for Windows).")

import torchvision.io
print(f"[Diagnostics] Torchvision C++ Video Fast Path Compiled: {getattr(torchvision.io, '_HAS_VIDEO_OPT', False)}")
# -------------------------

start_time = time.time()

def resize_with_pad(image, target_size=(768, 768)):
    h, w, = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


model = AutoModelForCausalLM.from_pretrained(
    "NemoStation/Marlin-2B",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
    token=os.getenv("HF_TOKEN"),
)
processor = AutoProcessor.from_pretrained("NemoStation/Marlin-2B", trust_remote_code=True)


def resolve_pad_token_id(model, processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "pad_token_id", None) is not None:
        return generation_config.pad_token_id

    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id

    if generation_config is not None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            return eos_token_id[0] if eos_token_id else None
        return eos_token_id

    return None


pad_token_id = resolve_pad_token_id(model, processor)
generation_config = getattr(model, "generation_config", None)
if pad_token_id is not None and generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
    generation_config.pad_token_id = pad_token_id

messages = [{"role": "user", "content": [
    {"type": "video", "video": "/mnt/d/qontexvods/part11.mp4"},
    {"type": "text", "text": "Provide a spatial description of this clip followed by time-ranged events.\nFor each event, give the time range as <start - end> and a short description."},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to(model.device)

generation_kwargs = {"max_new_tokens": 512, "do_sample": False}
if pad_token_id is not None:
    generation_kwargs["pad_token_id"] = pad_token_id

with torch.inference_mode():
    out = model.generate(**inputs, **generation_kwargs)
out = out[:, inputs["input_ids"].shape[1]:]
text = processor.batch_decode(out, skip_special_tokens=True)[0]
print(text)

#CAPTION_PROMPT: str = (
#    "Provide a spatial description of this clip followed by time-ranged events.\n"
#    "For each event, give the time range as <start - end> and a short description."
#)

elapsed_time = time.time() - start_time
print(f"Script ran for {elapsed_time:.2f} seconds")
