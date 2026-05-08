from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("xxx") #clip-vit-large-patch14
        self.processor = CLIPProcessor.from_pretrained("xxx") #clip-vit-large-patch14
        self.mlp = MLP()
        state_dict = torch.load(
            "./ckpt/sac+logos+ava1-l14-linearMSE.pth"
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

def main():
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    scorer = AestheticScorer(dtype=dtype).to(device)
    example_image_path = "xxx.png"
    example_image = Image.open(example_image_path)
    scores = scorer(example_image)


if __name__ == "__main__":
    main()