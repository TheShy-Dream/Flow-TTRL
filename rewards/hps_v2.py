import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model, get_tokenizer
import os
from PIL import Image
from torchvision import transforms


class HPSScorer(nn.Module):
    def __init__(
            self,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype else torch.float32

        base_model_path = "xxx" #CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
        hps_checkpoint_path = "xxx" #ckpt/HPS_v2.1_compressed.pt

        print(f"Loading HPSv2 base model...")

        self.model = create_model(
            "ViT-H-14",
            pretrained=base_model_path,
            precision='fp32' if self.dtype == torch.float32 else 'fp16',
            device=self.device
        )

        if os.path.exists(hps_checkpoint_path):
            print(f"Loading HPSv2 weights from: {hps_checkpoint_path}")
            checkpoint = torch.load(hps_checkpoint_path, map_location=self.device)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: HPS checkpoint not found at {hps_checkpoint_path}")

        self.tokenizer = get_tokenizer("ViT-H-14")

        self.model.eval()
        self.model.requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device))

    def preprocess(self, images):
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)

        images = (images - self.mean) / self.std
        return images

    def __call__(self,  images,prompts):

        if isinstance(images, list):
            images = torch.stack(images)


        images = images.to(self.device, dtype=self.dtype)

        if isinstance(prompts, str):
            prompts = [prompts]

        if len(prompts) == 1 and images.shape[0] > 1:
            prompts = prompts * images.shape[0]

        if len(prompts) != images.shape[0]:
            raise ValueError(f"Batch size mismatch: {len(prompts)} prompts vs {images.shape[0]} images")

        # Tokenize
        text_inputs = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            processed_images = self.preprocess(images)
            image_features = self.model.encode_image(processed_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            scores = (image_features * text_features).sum(dim=1)

        return scores



if __name__ == "__main__":
    scorer = HPSScorer(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)

    image_paths = [
        "../demo.png",
    ]

    if not os.path.exists(os.path.dirname(image_paths[0])):
        os.makedirs(os.path.dirname(image_paths[0]), exist_ok=True)
    if not os.path.exists(image_paths[0]):
        Image.new('RGB', (512, 512), color='red').save(image_paths[0])

    transform = transforms.Compose([
        transforms.ToTensor(),  

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])

    prompts = [
        'a cat holding a sign that says hello world',
    ]

    try:
        scores = scorer(images,prompts)
        print(f"--------------------------------")
        print(f"Input Image Shape: {images.shape}")
        print(f"HPSv2 Score: {scores.item()}")
        print(f"--------------------------------")
    except Exception as e:
        print(f"Error during scoring: {e}")