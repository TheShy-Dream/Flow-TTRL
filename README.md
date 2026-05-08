# Flow-TTRL: Test-Time Reinforcement Learning for Flow Matching

**Official implementation of the paper: "Flow-TTRL: Test-Time Reinforcement Learning for Flow Matching"**

**Flow-TTRL** is an inference-time optimization framework designed to align flow-matching models with complex human preferences without the need for expensive fine-tuning. By leveraging RL-guided latent search, Flow-TTRL achieves highly competitive results on benchmarks like **GenEval** and **T2I-CompBench**, attaining performance comparable to proprietary models and established RL-based fine-tuning methods while consistently bolstering image fidelity and text-alignment.

## 📖 Introduction

![Introduction](assets/Introduction.png)

## 🔧 Method

![Architecture](assets/architecture.png)

## ⚙️ Requirements

Flow-TTRL is tested on Linux with NVIDIA GPUs. While **A100/H100** are recommended for optimal inference speed, the framework is compatible with consumer-grade GPUs (e.g., **RTX 3090/4090**).

### Hardware & Memory Optimization

- **Recommended:** 40GB+ VRAM (for standard FP16 inference).
- **24GB GPU Support:** For GPUs with 24GB VRAM, we strongly recommend enabling **8-bit quantization** or **bitsandbytes** to prevent Out-of-Memory (OOM) errors during the iterative DiT forward passes and reward scoring.

### Dependencies Installation

```
# Create a virtual environment
conda create -n flow-ttrl python=3.10
conda activate flow-ttrl

# Install core dependencies
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.29.post2
pip install -r requirements.txt
```

### Reward Models

To use the full potential of Flow-TTRL, please ensure the corresponding reward model checkpoints are accessible:
- HPS v2
- ImageReward
- CLIP-score
- AES
- PickScore
- PaddleOCR

Due to the double-blind review policy, all local paths to model weights in the code have been replaced with placeholders (e.g., "xxx").

To run the demos or training scripts, you must manually update these placeholders in the following files with your local directory paths.

## 🚀 Quick Start

We provide two primary demo scripts to showcase **Flow-TTRL** across different flow-matching backbones: **FLUX.1-dev** and **Stable Diffusion 3.5 (SD3.5)**. Both scripts are located in the `demo/` directory.

### 1. Running the Demos

Before running, ensure your environment is activated and you have the necessary reward model checkpoints.

- **For FLUX.1-dev:**

  Bash

  ```
  python demo/flux_sde_demo.py
  ```

- **For Stable Diffusion 3.5:**

  Bash

  ```
  python demo/sd3_sde_demo.py
  ```

### 2. Key Variables to Modify

To adapt the generation to your own prompts or to perform test-time calibration, you only need to modify a few key variables within these scripts:

#### 📝 Prompt & Rewards

- `prompt`: The text description you want to generate.
- `score_dict`: A dictionary to enable/disable specific reward models and set their weights (e.g., `{"imagereward": 1.0, "hps": 0.5}`).

#### 🛠️ Optimization through Parameter Adjustment

To achieve better results for specific prompts or reward objectives, users can adjust the core inference-time parameters mentioned in the paper—such as `scale_factor`, `RL_interation_num`, `beta`, and `noise_range`. These variables allow for the precise calibration of the reward-guided optimization process at test-time, enabling a better balance between prompt alignment and image fidelity without any model retraining.

## 📊 Qualitative Results

![GenEval Results](assets/qualitization_result_geneval.png)

![T2I-CompBench Results](assets/qualitization_result_t2ibench.png)

![Complex Prompts](assets/qualitization_result_complex.png)

![Participant Prompts](assets/qualitization_result_partiprompts.png)

![DrawBench Results](assets/qualitization_result_drawbench.png)

![Pick-a-Pic Results](assets/qualitization_result_pickapic.png)
