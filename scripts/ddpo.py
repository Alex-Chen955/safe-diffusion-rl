# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python scripts/ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""

import os
from dataclasses import dataclass, field
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available
from ddpo_pytorch.safe_rewards import make_safety_reward

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion"`):
            HuggingFace repo to save model weights to.
        hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
            Hugging Face model ID for aesthetic scorer model weights.
        hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
            Hugging Face model filename for aesthetic scorer model weights.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether to use LoRA.
    """

    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


# # list of example prompts to feed stable diffusion
# animals = [
#     "cat",
#     "dog",
#     "horse",
#     "monkey",
#     "rabbit",
#     "zebra",
#     "spider",
#     "bird",
#     "sheep",
#     "deer",
#     "cow",
#     "goat",
#     "lion",
#     "frog",
#     "chicken",
#     "duck",
#     "goose",
#     "bee",
#     "pig",
#     "turkey",
#     "fly",
#     "llama",
#     "camel",
#     "bat",
#     "gorilla",
#     "hedgehog",
#     "kangaroo",
# ]



def prompt_fn(dataset_name: str="all") -> tuple[str, dict]:
    assert dataset_name in ["all", "4chan", "Lexica", "Template"], "dataset_name must be one of all, 4chan, Lexica, Template"
    pkl_path = "unsafe_prompt_dataset/unsafe_prompts.pkl"
    df = pkl.load(open(pkl_path, "rb"))
    if dataset_name == "all":
        prompts = df["prompt"].tolist()
    else:
        prompts = df[df["dataset_name"] == dataset_name]["prompt"].tolist()
    return np.random.choice(prompts), {}

def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    # dataset name
    dataset_name = "all" # change to 4chan, Lexica, Template
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # add dataset name to training args
    training_args.project_kwargs = {
        "logging_dir": f"./logs/{dataset_name}",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"./save/{dataset_name}",
    }

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        make_safety_reward('./utils/prompts.p'),
        prompt_fn(dataset_name=dataset_name),
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)