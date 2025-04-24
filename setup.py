from setuptools import setup, find_packages

setup(
    name="ddpo-pytorch",
    version="0.0.1",
    packages=["ddpo_pytorch"],
    python_requires=">=3.10",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]==0.27.2",
        "accelerate==0.27.2",
        "wandb==0.19.9",
        "torchvision",
        "torch==2.6.0",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.39.3",
        "trl==0.8.6",
        "numpy==1.26.4",
        "huggingface-hub==0.22.2",
        "ftfy",
        "regex",
        "tqdm",
        "peft==0.10.0",
    ],
)
