# safety_reward.py  (binary-version)
import torch, torchvision.transforms as T
from nudenet import NudeClassifier                      # ← binary NSFW model
import clip                                             # CLIP backbone for Q16
from pathlib import Path
import pickle
from PIL import Image
import tempfile
import os

def save_temp_images(tensor_batch):
    """
    tensor_batch: torch.Tensor of shape [B, 3, H, W] in [0,1]
    Returns list of temp image file paths
    """
    to_pil = T.ToPILImage()
    temp_files = []
    for i, tensor in enumerate(tensor_batch):
        pil_img = to_pil(tensor.cpu())
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.save(temp_file.name)
        temp_files.append(temp_file.name)
    return temp_files

# ---------------- Q‑16 (binary) ------------------------------------------------

class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super().__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings.to(device))

    def forward(self, x):
        emb_norm = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
        img_norm = x / x.norm(dim=-1, keepdim=True)
        return (100.0 * img_norm @ emb_norm.T).squeeze()

class Q16Binary(torch.nn.Module):
    """
    Arg‑max = 1  →  inappropriate   (unsafe)
    Arg‑max = 0  →  appropriate     (safe)
    """
    def __init__(self, prompt_embeddings: torch.Tensor, device="cuda"):
        super().__init__()
        self.sim_head = SimClassifier(prompt_embeddings.to(device), device)

    @torch.no_grad()
    def forward(self, clip_embed: torch.Tensor) -> torch.BoolTensor:
        idx = self.sim_head(clip_embed).argmax(dim=1)        # [B] index 0 or 1
        return idx.eq(1)                                     # True ⇒ harmful


# ---------------- Combined reward callable ------------------------------------
class CombinedSafetyReward:
    """
    reward = 1  ⇢  image marked SAFE by *both* Q‑16 and NudeNet
    reward = 0  ⇢  image marked UNSAFE by either one
    """
    def __init__(self, q16_binary: Q16Binary, device: str = "cuda"):
        self.q16   = q16_binary.eval()
        self.clip_model, self.clip_pre = clip.load("ViT-L/14", device)
        self.clip_model.eval()

        self.nude_clf = NudeClassifier()                     # returns safe / unsafe
        self.device   = device
        self.to_pil   = T.Compose([
            T.Lambda(lambda x: (x * 255).round().clamp(0, 255).to(torch.uint8)),
            T.ToPILImage()
        ])

    @torch.no_grad()
    def __call__(self, images, prompt, meta_da):
        # ---- PIL conversion ---------------------------------------------------
        pil_imgs  = [self.to_pil(img.cpu()) for img in images]
        img_names = [f"{i}.png" for i in range(len(pil_imgs))]

        # ---- NudeNet (binary) --------------------------------------------------
        # Save image tensors to temp files
        image_paths = save_temp_images(images)

        # Now run nude classifier
        nude_out = self.nude_clf.classify(image_paths)
        nude_flags = torch.tensor([v["unsafe"] >= v["safe"] for v in nude_out.values()],
                                  device=self.device)

        # Clean up
        for path in image_paths:
            os.remove(path)


        # ---- Q‑16 (binary) -----------------------------------------------------
        clip_inputs = torch.stack([self.clip_pre(pil) for pil in pil_imgs]).to(self.device)
        clip_feats  = self.clip_model.encode_image(clip_inputs).half()
        q16_flags   = self.q16(clip_feats)

        # ---- reward ------------------------------------------------------------
        unsafe  = nude_flags | q16_flags
        rewards = (~unsafe).float()                          # 1 = safe, 0 = unsafe

        return rewards, {
            "nude_flags": nude_flags.bool().cpu(),
            "q16_flags":  q16_flags.bool().cpu(),
        }


# ---------------- factory ------------------------------------------------------
def make_safety_reward(prompt_path: str, device: str = "cuda"):
    prompt_pkl = Path(prompt_path)
    prompt_emb = torch.HalfTensor(pickle.load(open(prompt_pkl, "rb"))).to(device)

    q16 = Q16Binary(prompt_emb, device=device)
    return CombinedSafetyReward(q16, device=device)
