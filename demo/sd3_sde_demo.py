import sys, os
from rewards.rewards import multi_score, MultiScorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, DDIMScheduler, DDIMInverseScheduler
from pipeline_SD3_with_log_prob import SD3TTRLPipeline
from transformers import BitsAndBytesConfig

model_id = "stabilityai/stable-diffusion-3.5-medium"

score_dict = {
            #"clipscore": 1.0,
            #"aesthetic": 1.0,
            # "jpeg_compressibility": 1.0,
            #"pickscore": 1.0,
            # "jpeg_incompressibility": 1.0,
             "imagereward": 1.0,
             #"hps": 1.0,
            #"black_loss": 1.0,
            #"white_loss": 1.0,
            # "contrast_loss":1.0,
            #"ocr": 3.0,
        }

scorer = MultiScorer(device="cuda", score_dict=score_dict)


# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_id,
#     subfolder="transformer",
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16
# )

# pipe = SD3TTRLPipeline.from_pretrained(
#     model_id,
#     transformer=model_nf4,
#     torch_dtype=torch.bfloat16
# )

pipe = SD3TTRLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

pipe.transformer.enable_gradient_checkpointing()
pipe.enable_attention_slicing()
#pipe.enable_model_cpu_offload()

pipe = pipe.to("cuda")

prompt = 'A small astronaut in a reflective suit drifting in microgravity, holding an LED panel that shows the text Flow-TTRL, with Earth glowing softly in the background.'
generator = torch.Generator(device="cuda").manual_seed(42)

images = pipe(
    prompt,
    negative_prompt= "ugly,low resolution,blurry image,bad composition,disfigured,oversaturated",
    num_inference_steps=40,
    guidance_scale=3.5,
    generator=generator,
    noise_range=[1.5, 0.5],
    scale_factor=500,
    group_size=6,
    beta1=0.0002,
    beta2=0.0002,
    score_dict=score_dict,
    scoring_fn=scorer,
    internal_reward_timestep=0.2,
    external_reward_timestep=0.5,
    reward_diff_threshold=0.0,
    RL_interation_num=2,
)


image = images.images[0]
image.save(f"demo/1.png")
