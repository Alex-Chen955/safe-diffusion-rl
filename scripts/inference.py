#!/usr/bin/env python
"""
Inference script for a Stable‑Diffusion model

Supported modes
===============
1. Base model without negative prompt
   Standard sampling from a base Stable Diffusion model.

2. Base model with negative prompt
   Guide the generation away from unwanted concepts by specifying a negative prompt.

3. Base model + LoRA checkpoint
   Apply fine‑tuned LoRA weights on top of the base model.

The script samples five images per prompt and saves them to:
``outputs/<timestamp>/<slugified‑prompt>/<idx>.png``

Example – Base model + LoRA checkpoint:
```
python scripts/inference.py \
  --base_model_id runwayml/stable-diffusion-v1-5\
  --lora_path save/all_2025-04-27_17-24-30/checkpoints/checkpoint_146
  --fuse_lora
  --dataset_name all
```

Example – Base model without negative prompt
```
python scripts/inference.py --model_id runwayml/stable-diffusion-v1-5 --dataset_name all
```

Example – Base model with negative prompt
```
python scripts/inference.py --model_id runwayml/stable-diffusion-v1-5 --negative_prompt "nsfw, watermark" --dataset_name all
```
"""

import argparse
import pickle as pkl
import re
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
from trl import DefaultDDPOStableDiffusionPipeline

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def slugify(text: str, maxlen: int = 60) -> str:
    """Filesystem‑safe slug (lowercase, dash‑separated)."""
    text = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return text[:maxlen] or "prompt"


def load_prompts(pkl_path: str, subset: str) -> List[str]:
    df = pkl.load(open(pkl_path, "rb"))
    match subset:
        case "all":
            return df["prompt"].tolist()
        case "all_positive":
            return df[df["safe_label"] == 1]["prompt"].tolist()
        case "all_negative":
            return df[df["safe_label"] == 0]["prompt"].tolist()
        case _:
            return df[df["dataset_name"] == subset]["prompt"].tolist()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Checkpoints -------------------------------------------------------------
    parser.add_argument("--model_id", type=str, default=None,
                        help="Folder or Hub repo with the *full* DDPO pipeline.")
    parser.add_argument("--base_model_id", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Base SD model when --lora_path is given.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Directory holding pytorch_lora_weights.safetensors.")

    # Data / I/O --------------------------------------------------------------
    parser.add_argument("--dataset_path", type=str,
                        default="unsafe_prompt_dataset/test.pkl",
                        help="Pickle with a 'prompt' column (test split).")
    parser.add_argument("--dataset_name", type=str, default="all",
                        choices=[
                            "all", "all_positive", "all_negative",
                            "4chan", "Lexica", "Template", "coco",
                        ],
                        help="Subset of prompts to use.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Root directory for generated images.")

    # Misc --------------------------------------------------------------------
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Debug: only process the first N prompts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fuse_lora", action="store_true",
                        help="Fuse LoRA weights for faster inference.")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Optional negative prompt for classifier‑free guidance.")

    args = parser.parse_args()

    if args.model_id is None and args.lora_path is None:
        parser.error("Provide either --model_id or --lora_path.")
    
    if args.negative_prompt is not None and args.lora_path is not None:
        parser.error("--negative_prompt is only supported when using the base model without fine-tuning.")

    # ---------------------------------------------------------------------
    # Seeds / device / dtype
    # ---------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ---------------------------------------------------------------------
    # Prompts
    # ---------------------------------------------------------------------
    prompts = load_prompts(args.dataset_path, args.dataset_name)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    # ---------------------------------------------------------------------
    # Load pipeline
    #   ⚠️ `DefaultDDPOStableDiffusionPipeline` does *not* implement
    #      `from_pretrained`; instead we instantiate it directly with the
    #      model identifier. This mirrors the official docs.
    # ---------------------------------------------------------------------
        # -----------------------------------------------------------------  
    # Load pipeline (full vs. LoRA‑only)  
    # -----------------------------------------------------------------  
    if args.lora_path is None:
        pipe = DefaultDDPOStableDiffusionPipeline(args.model_id)  
    else:                       # base model + LoRA weights  
        pipe = DefaultDDPOStableDiffusionPipeline(args.base_model_id)  
        # `load_lora_weights` lives on the *inner* StableDiffusionPipeline  
        pipe.sd_pipeline.load_lora_weights(args.lora_path)  
        if args.fuse_lora:  
            pipe.sd_pipeline.fuse_lora()  
  
    # Memory optimised placement (fp16 on GPU)  
    pipe.vae.to(device, dtype)  
    pipe.text_encoder.to(device, dtype)  
    pipe.unet.to(device, dtype)  
    pipe.set_progress_bar_config(disable=True)

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Generation loop
    # ---------------------------------------------------------------------
    for prompt in tqdm(prompts, desc="Generating"):
        if args.negative_prompt:
            images = pipe(prompt, num_images_per_prompt=5, negative_prompt=args.negative_prompt).images
        else:
            images = pipe(prompt, num_images_per_prompt=5).images
        p_dir = run_dir / slugify(prompt)
        p_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(p_dir / f"{i}.png")

    print(f"✅ Done. Images saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
