from PIL import Image
import io
import numpy as np
import torch
import torchvision.transforms.functional as TF
from collections import defaultdict


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        device = images.device
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return torch.tensor(sizes, device=device), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score():
    from rewards.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def image_similarity_score(device):
    from rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8) / 255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn


def pickscore_score(device):
    from rewards.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def hps_score(device):
    from rewards.hps_v2 import HPSScorer
    scorer = HPSScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, list):
            import torchvision.transforms.functional as TF
            images = torch.stack([TF.to_tensor(img) for img in images]).to(device)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def imagereward_score(device):
    from rewards.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def ocr_score(device):
    from rewards.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        device=images.device
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        scores = scores.to(device)
        # change tensor to list
        return scores, {}

    return _fn


def white_loss_fn(device=None, inference_dtype=None):
    def _fn(images, prompts, metadata):
        rewards = images.mean(dim=(1, 2, 3))
        return rewards, {} 

    return _fn


def black_loss_fn(device=None, inference_dtype=None):
    def _fn(images, prompts, metadata):
        rewards = -images.mean(dim=(1, 2, 3))
        return rewards, {}

    return _fn


def contrast_loss_fn(device=None, inference_dtype=None):
    def _fn(images, prompts, metadata):
        # images.view(B, -1).var(dim=1) -> (B,)
        rewards = images.view(images.shape[0], -1).var(dim=1)
        return rewards, {}

    return _fn


"""
def video_ocr_score(device):
    from ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn
"""


def multi_score(device, score_dict):
    score_functions = {
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "jpeg_incompressibility": jpeg_incompressibility,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "hps": hps_score,
        "black_loss": black_loss_fn,
        "white_loss": white_loss_fn,
        "contrast_loss": contrast_loss_fn,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[
            score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images,
                                                                                                             prompts,
                                                                                                             metadata,
                                                                                                             only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn


class MultiScorer:
    def __init__(self, device, score_dict):
        self.device = device
        self.weights = score_dict.copy()
        self.score_functions = {
            "imagereward": imagereward_score,
            "pickscore": pickscore_score,
            "aesthetic": aesthetic_score,
            "jpeg_compressibility": jpeg_compressibility,
            "jpeg_incompressibility": jpeg_incompressibility,
            "clipscore": clip_score,
            "image_similarity": image_similarity_score,
            "hps": hps_score,
            "ocr": ocr_score,
            "black_loss": black_loss_fn,
            "white_loss": white_loss_fn,
            "contrast_loss": contrast_loss_fn,
        }

        self.score_fns = {}
        for score_name, weight in self.weights.items():
            if score_name not in self.score_functions:
                raise ValueError(f"Unknown scorer: {score_name}")

            scorer_creator = self.score_functions[score_name]

            if 'device' in scorer_creator.__code__.co_varnames:
                self.score_fns[score_name] = scorer_creator(device)
            else:
                self.score_fns[score_name] = scorer_creator()

    def _ensure_tensor(self, images):

        if isinstance(images, torch.Tensor):
            return images.to(self.device)

        processed_list = []
        if not isinstance(images, (list, tuple)):
            images = [images]

        for img in images:
            if isinstance(img, np.ndarray):
                # Numpy (H, W, C) -> Tensor (C, H, W)
                img_tensor = TF.to_tensor(img)
            elif isinstance(img, Image.Image):
                # PIL -> Tensor (C, H, W)
                img_tensor = TF.to_tensor(img)
            else:
                img_tensor = img 
            processed_list.append(img_tensor)

        return torch.stack(processed_list).to(self.device)

    def update_weights(self, new_score_dict):
        uninitialized_scorers = set(new_score_dict.keys()) - set(self.score_fns.keys())
        if uninitialized_scorers:
            print(
                f"Warning: Scorers {uninitialized_scorers} were not initialized and will be ignored or need full re-initialization.")
            self.weights = {k: v for k, v in new_score_dict.items() if k in self.score_fns}
        else:
            self.weights = new_score_dict.copy()
    def __call__(self, images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = None 
        score_details = {}
        images = self._ensure_tensor(images)
        if ref_images is not None:
            ref_images = self._ensure_tensor(ref_images)


        for score_name, weight in self.weights.items():

            if score_name not in self.score_fns:
                continue 

            scorer_fn = self.score_fns[score_name]


            if score_name == "geneval":
               
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = scorer_fn(images, prompts,
                                                                                                 metadata, only_strict)

                current_scores = scores
            elif score_name == "image_similarity":
                current_scores, _ = scorer_fn(images, ref_images)
            else:

                current_scores, _ = scorer_fn(images, prompts, metadata)


            score_details[score_name] = current_scores


            weighted_scores = [weight * score for score in current_scores]

            if total_scores is None:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "../demo.png",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts = [
        'a cat holding a sign that says hello world',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "clipscore": 1.0,
        "aesthetic": 0.5,
        "jpeg_compressibility": 0.2,
        "pickscore": 1.0,
        "jpeg_incompressibility": 0.1,
        "imagereward": 1.0,
        "hps": 1.0,
        "black_loss": 0.1,
        "white_loss": 0.1,
        "contrast_loss": 0.1
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
