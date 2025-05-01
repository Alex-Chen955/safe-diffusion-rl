# Fine-tuning diffusion models for safe text to image generation through reinforcement leanring

## Introduction and Motivation

Text-to-image (T2I) diffusion models such as Stable Diffusion turn natural-language prompts into photorealistic images, yet the same models can also produce disallowed content—sexual, violent, or hateful scenes—with little friction.

Mitigation strategies to date fall into two broad categories:

1. **Detection-based approaches.** A safety checker inspects each generated image and replaces any that are flagged as unsafe with a blank frame. While straightforward, this wastes computation and frustrates users by returning only a generic “content blocked” message.
2. **Guidance-based approaches.** Techniques such as negative prompting attempt to suppress risky concepts before inference. These ad-hoc edits frequently distort user intent and still cannot guarantee compliance with policy.

Both defenses are easy to bypass and can be disabled when model weights are openly released. In contrast, large language models (LLMs) undergo rigorous alignment procedures before deployment. To bridge this gap, we propose a third approach: model-level alignment. Rather than filtering outputs or suppressing risky inputs, we fine-tune Stable Diffusion using reinforcement learning (RL). A frozen harmful-content detector provides the reward signal—unsafe generations receive zero reward, while safe outputs are rewarded. By internalizing safety constraints during training, we aim to make our model produce policy-compliant images with minimal impact on user experience.

## Methodology
<!-- ![Figure 1. RL fine-tuning pipeline with LoRA for safe image generation](./assets/pipeline_final.png)
**Figure 1.** RL fine-tuning pipeline with LoRA for safe image generation. -->

<p align="center">
  <img src="./assets/pipeline_final.png" alt="RL Fine-tuning Pipeline" width="800"/>
  <br>
  <strong>Figure 1.</strong> RL fine-tuning pipeline with LoRA for safe image generation.
</p>


The overall pipeline is illustrated in Figure 1. Given a set of toxic prompts, a diffusion model generates images, which are then evaluated by pre-trained safety classifiers such as NudeNet and Q16. These classifiers assign a reward signal based on the safety of the generated content. The diffusion model is subsequently fine-tuned using reinforcement learning (RL) with Low-Rank Adaptation (LoRA) to internalize safety constraints and minimize harmful generations.

Our method builds upon Denoising Diffusion Policy Optimization (DDPO), which models the denoising process as a multi-step Markov Decision Process (MDP). In this formulation, each state corresponds to a tuple $(c, t, x_t)$, the action is the denoised sample $x_{t-1}$, and the reward is only assigned at the final step based on the generated image $x_0$.

To optimize the diffusion model, we use the following DDPO loss:

<p align="center">
  <img src="./assets/Math_Formula.png" alt="DDPO Loss Function" width="600"/>
</p>

## Experimental Results

We evaluate our method using a dataset composed of four prompt sources: Lexica (harmful), Template (harmful), 4chan (harmful), and COCO (harmless). Experiments are conducted on each individual harmful dataset, their combination, and the full dataset including both harmful and harmless prompts. We use a 90%/10% train-test split. The total dataset size is [placeholder].

### Reward Curves

<!-- <p align="center"> <img src="./assets/reward_curve.png" alt="Reward Curves" width="600"/> <br> <strong>Figure X.</strong> Training reward curves over iterations. </p> -->

The figure above shows the reward curves during training. Although the reward updates are somewhat volatile, the overall trend shows improvement across all settings, indicating that the policy effectively learns from the reward signal.

---

### Testing Metrics

We assess safety alignment on the **Inappropriate Image Prompts (I2P)** benchmark, which contains toxic prompts across seven risk categories. Images are generated using three methods for comparison:

1. Our fine-tuned diffusion model,
2. Stable Diffusion v1.5 with the safety checker disabled,
3. Stable Diffusion v1.5 using negative prompts (baseline).

To quantify safety, we use the **Inappropriate Probability (IP)** metric:

IP=NflaggedNtotal×100%,

IP=NtotalNflagged×100%,

where NflaggedNflagged is the number of outputs detected as harmful by either **Q16** or **NudeNet** classifiers.

The results are shown in the following table:

<p align="center"> <img src="./assets/testing_table.png" alt="Testing Metrics Table" width="600"/> <br> <strong>Table X.</strong> Inappropriate Probability (IP) across different generation methods. </p>

Our method consistently achieves the lowest IP across all categories, demonstrating improved safety alignment compared to baseline methods.

---

### Qualitative Results

We provide qualitative examples to highlight the effectiveness of our method:

<p align="center"> <img src="./assets/qualitative_demo_improved.png" alt="Qualitative Results" width="1000"/> <br> <strong>Figure X.</strong> Visual comparison between baseline and fine-tuned model outputs. </p>

Our model successfully removes harmful visual elements while preserving the intended atmosphere of the image. For example, although the scene remains emotionally tense, explicit or offensive content is effectively suppressed.

---

## Conclusion

*To be completed.*