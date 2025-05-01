# Fine-tuning diffusion models for safe text to image generation through reinforcement leanring

## Introduction and Motivation

Text-to-image (T2I) diffusion models such as *Stable Diffusion* turn natural-language prompts into photorealistic images, yet the same models can also produce disallowed content—sexual, violent, or hateful scenes—with little friction.

Mitigation strategies to date fall into two broad categories:

1. **Detection-based approaches.** A safety checker inspects each generated image and replaces any that are flagged as unsafe with a blank frame. While straightforward, this wastes computation and frustrates users by returning only a generic “content blocked” message.
2. **Guidance-based approaches.** Techniques such as negative prompting attempt to suppress risky concepts before inference. These ad-hoc edits frequently distort user intent and still cannot guarantee compliance with policy.

Both defenses are easy to bypass and can be disabled when model weights are openly released. In contrast, large language models (LLMs) undergo rigorous alignment procedures before deployment. To bridge this gap, we propose a third approach: **model-level alignment**. Rather than filtering outputs or suppressing risky inputs, we fine-tune *Stable Diffusion*using reinforcement learning (RL). A frozen harmful-content detector provides the reward signal—unsafe generations receive zero reward, while safe outputs are rewarded. By internalizing safety constraints during training, the model eliminates the need for brittle inference-time heuristics and produces policy-compliant images with minimal impact on user experience.