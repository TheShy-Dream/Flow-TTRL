# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output  # v(x_t,t)
    sample = sample  # x_t
    if prev_sample is not None:
        prev_sample = prev_sample  # x_{t-1}

    self.sigmas = self.sigmas.to(dtype=model_output.dtype)
    """
    scheduler.index_for_timestep(1)
    49
    scheduler.index_for_timestep(1000)
    0
    """
    step_index = [self.index_for_timestep(t) for t in timestep] 
    prev_step_index = [step + 1 for step in step_index] 
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1))) 
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))  
    sigma_max = self.sigmas[1].item() 
    dt = sigma_prev - sigma 

    if sde_type == 'sde':
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max,
                                                        sigma))) * noise_level 

        # our sde
        prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt) + model_output * (
                    1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt 

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        log_prob = (
                -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == 'cps':
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
            sigma_prev ** 2 - std_dev_t ** 2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1 * dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def ode_step_with_log_prob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    return_sqrt_dt: Optional[bool] = False,
):
    """
    ODE version of the flow-matching step. Deterministic propagation of the sample without noise.

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float` or `torch.FloatTensor`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        prev_sample (`torch.FloatTensor`, *optional*):
            The previous sample, if already available.
        generator (`torch.Generator`, *optional*):
            Random number generator (not used in ODE, kept for interface compatibility).
        return_sqrt_dt (`bool`, *optional*):
            Whether to return the square root of the timestep difference.
    Returns:
        prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
    """
    sample = sample  # x_t
    if prev_sample is not None:
        prev_sample = prev_sample  # x_{t-1}

    self.sigmas = self.sigmas.to(dtype=model_output.dtype)

    # get step indices
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]

    # get sigmas for current and previous steps
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))  # sigma_t
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))  # sigma_{t-1}
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    # ODE deterministic update (sde_type='sde')
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))  # for logging / interface

    prev_sample_mean = self.step(model_output, timestep, sample, return_dict=False)

    prev_sample = prev_sample_mean  # deterministic ODE, no noise
    log_prob = torch.zeros_like(prev_sample_mean)  # deterministic, log_prob=0

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    sqrt_dt = torch.sqrt(-1 * dt)

    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


