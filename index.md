# Fine-tuning diffusion models for safe text to image generation through reinforcement leanring

## Introduction and Motivation

Text-to-image (T2I) diffusion models such as Stable Diffusion turn natural-language prompts into photorealistic images, yet the same models can also produce disallowed content—sexual, violent, or hateful scenes—with little friction.

Mitigation strategies to date fall into two broad categories:

1. **Detection-based approaches.** A safety checker inspects each generated image and replaces any that are flagged as unsafe with a blank frame. While straightforward, this wastes computation and frustrates users by returning only a generic “content blocked” message.
2. **Guidance-based approaches.** Techniques such as negative prompting attempt to suppress risky concepts before inference. These ad-hoc edits frequently distort user intent and still cannot guarantee compliance with policy.

Both defenses are easy to bypass and can be disabled when model weights are openly released. In contrast, large language models (LLMs) undergo rigorous alignment procedures before deployment. To bridge this gap, we propose a third approach: model-level alignment. Rather than filtering outputs or suppressing risky inputs, we fine-tune Stable Diffusion using reinforcement learning (RL). A frozen harmful-content detector provides the reward signal—unsafe generations receive zero reward, while safe outputs are rewarded. By internalizing safety constraints during training, we aim to make our model produce policy-compliant images with minimal impact on user experience.

## Methodology

![Figure 1. RL fine-tuning pipeline with LoRA for safe image generation](./assets/pipeline_final.png)

**Figure 1.** RL fine-tuning pipeline with LoRA for safe image generation.

The overall pipeline is illustrated in Figure 1. Given a set of toxic prompts, a diffusion model generates images, which are then evaluated by pre-trained safety classifiers such as NudeNet and Q16. These classifiers assign a reward signal based on the safety of the generated content. The diffusion model is subsequently fine-tuned using reinforcement learning (RL) with Low-Rank Adaptation (LoRA) to internalize safety constraints and minimize harmful generations.

Our method builds upon Denoising Diffusion Policy Optimization (DDPO), which models the denoising process as a multi-step Markov Decision Process (MDP). In this formulation, each state corresponds to a tuple $(c, t, x_t)$, the action is the denoised sample $x_{t-1}$, and the reward is only assigned at the final step based on the generated image $x_0$.

To optimize the diffusion model, we use the following DDPO loss:

<p align="center">
  <img src="./assets/Math_Formula.png" alt="DDPO Loss Function" width="600"/>
</p>
