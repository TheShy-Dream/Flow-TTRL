import time
from copy import deepcopy
import os
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from typing import Union, Optional, List, Dict, Any, Callable
from diffusers.utils import replace_example_docstring, is_torch_xla_available, logging

from diffusion_patch.rl_step import RLSampler
from diffusion_patch.sde_step import sde_step_with_logprob
from rewards.rewards import MultiScorer

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class FluxTTRLPipeline(FluxPipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt: Union[str, List[str]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            true_cfg_scale: float = 1.0,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 3.5,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            negative_ip_adapter_image: Optional[PipelineImageInput] = None,
            negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            run_standard_sd: bool = False,
            noise_range: List[float] = [0.5, 0.8],
            group_size: int = 2,
            beta1: float = 0.1,
            beta2: float = 0.1,
            rationorm: bool = True,
            clip_range: float = 1e-4,
            adv_clip_max: float = 5.0,
            scale_factor: float = 1.0,
            internal_reward_timestep: float = 0.2,
            external_reward_timestep: float = 0.5,
            score_dict: Optional[Dict] = None,
            scoring_fn: MultiScorer = None,
            reward_diff_threshold: float = 0.01,
            RL_interation_num: int = 3,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        step_sizes = scale_factor * np.sqrt(scale_range)

        self.scheduler1 = deepcopy(self.scheduler)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
                negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
                negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        RL_step=[*range(int(self.num_timesteps*external_reward_timestep))]

        # 6. Prepare image embeddings

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if not run_standard_sd:
                reference_latents = latents.detach().clone()
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                noise_level = noise_range[0] + (noise_range[1] - noise_range[0]) * (i / self.num_timesteps * external_reward_timestep)
                if not run_standard_sd and i in RL_step and self.num_timesteps * external_reward_timestep > i:
                    interation_num = RL_interation_num
                    latents, reference_latents = self.rl_update_step(
                        latents=latents,
                        reference_latents=reference_latents,
                        t=t,
                        i=i,
                        rl_interation_num=interation_num,
                        timesteps=timesteps,
                        noise_level=noise_level,
                        generator=generator,
                        guidance=guidance,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        do_true_cfg=do_true_cfg,
                        negative_image_embeds=negative_image_embeds if do_true_cfg else None,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_true_cfg else None,
                        negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
                        negative_text_ids=negative_text_ids if do_true_cfg else None,
                        true_cfg_scale=true_cfg_scale,
                        group_size=group_size,
                        beta1=beta1,
                        beta2=beta2,
                        rationorm=rationorm,
                        clip_range=clip_range,
                        adv_clip_max=adv_clip_max,
                        internal_reward_timestep=internal_reward_timestep,
                        step_size=step_sizes[i],
                        score_dict=score_dict,
                        prompts=prompt,
                        width=width,
                        height=height,
                        scoring_fn=scoring_fn,
                        reward_diff_threshold=reward_diff_threshold,
                    )

                if not run_standard_sd:
                    latents = self.latent_evolution(
                        latents=latents,
                        t=t,
                        i=i,
                        guidance=guidance,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        do_true_cfg=do_true_cfg,
                        negative_image_embeds=negative_image_embeds if do_true_cfg else None,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_true_cfg else None,
                        negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
                        negative_text_ids=negative_text_ids if do_true_cfg else None,
                        true_cfg_scale=true_cfg_scale,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        external_scheduler=None,
                        noise_level=None,
                    )
                    if RL_step[-1] > i:
                        reference_latents = self.latent_evolution(
                            latents=reference_latents,
                            t=t,
                            i=i,
                            guidance=guidance,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            text_ids=text_ids,
                            latent_image_ids=latent_image_ids,
                            do_true_cfg=do_true_cfg,
                            negative_image_embeds=negative_image_embeds if do_true_cfg else None,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_true_cfg else None,
                            negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
                            negative_text_ids=negative_text_ids if do_true_cfg else None,
                            true_cfg_scale=true_cfg_scale,
                            callback_on_step_end=callback_on_step_end,
                            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                            external_scheduler=self.scheduler1,
                            noise_level=None,
                        )

                if run_standard_sd:
                    latents = self.latent_evolution(
                        latents=latents,
                        t=t,
                        i=i,
                        guidance=guidance,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        do_true_cfg=do_true_cfg,
                        negative_image_embeds=negative_image_embeds if do_true_cfg else None,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_true_cfg else None,
                        negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
                        negative_text_ids=negative_text_ids if do_true_cfg else None,
                        true_cfg_scale=true_cfg_scale,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        external_scheduler=None,
                        noise_level=None,#noise_level if i in RL_step else 
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    @staticmethod
    def update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float, type: str = "grpo", noise_level: float = 0.8,
            sqrt_dt: float = 1.0
    ) -> torch.Tensor:
        if type == "grpo":
            step_size = step_size * noise_level * sqrt_dt * 250
        else:
            step_size = step_size * noise_level * sqrt_dt * 5
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=False
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def latent_evolution(
            self,
            latents,
            t,
            i,
            guidance,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            do_true_cfg,
            negative_image_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            true_cfg_scale,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            external_scheduler,
            noise_level,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        with self.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

        if do_true_cfg:
            if negative_image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

            with self.transformer.cache_context("uncond"):
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype

        if noise_level is not None:
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred,
                t.unsqueeze(0).repeat(latents.shape[0]),
                latents,
                noise_level=noise_level,
            )

        else:
            if external_scheduler is None:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            else:
                latents = external_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        return latents

    def compute_log_prob(
            self,
            sampler,
            latents,
            t,
            guidance,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            do_true_cfg,
            negative_image_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            true_cfg_scale,
            group_size,
            noise_level,
            generator,
            rationorm,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        with self.transformer.cache_context("cond"):
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        ## expand noise and latent
        latents = latents.repeat(group_size, 1, 1, 1)
        noise_pred = noise_pred.repeat(group_size, 1, 1, 1)

        # compute the log prob of next_latents given latents under the current model
        prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            sampler.get_all_key_tensors("timesteps"),
            latents,
            prev_sample=sampler.get_all_key_tensors("next_latents"),
            noise_level=noise_level,
            return_sqrt_dt=rationorm,
            generator=generator,
        )
        if rationorm:
            return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, None

    def prdp_loss_calculation(
            self,
            sampler,
            log_prob,
            prev_sample_mean,
            std_dev_t,
            sqrt_dt,
            prev_sample_mean_ref,
            beta1,
            rationorm,
            adv_clip_max,
            clip_range,
            log_probs_ref,
            reward_diff_threshold,
    ):
        import torch

        rewards = sampler.get_all_key_tensors("rewards").view(-1) 

        log_probs_current = log_prob 
        log_probs_ref = log_probs_ref

        log_ratios = log_probs_current - log_probs_ref
        log_ratios_old_modified = torch.zeros_like(log_ratios) 

        clipped_log_ratios = torch.clamp(
            log_ratios,
            log_ratios_old_modified - clip_range,
            log_ratios_old_modified + clip_range
        )

        log_ratios_mean = log_ratios
        clipped_log_ratios_mean = clipped_log_ratios

        log_ratio_diffs = log_ratios_mean.unsqueeze(1) - log_ratios_mean.unsqueeze(0)
        clipped_log_ratio_diffs = clipped_log_ratios_mean.unsqueeze(1) - clipped_log_ratios_mean.unsqueeze(0)
        reward_diffs = rewards.unsqueeze(1) - rewards.unsqueeze(0)


        kl_reward_diff = reward_diffs / beta1

        mse_loss = (log_ratio_diffs - kl_reward_diff) ** 2
        clipped_mse_loss = (clipped_log_ratio_diffs - kl_reward_diff) ** 2

        max_mse_loss = torch.maximum(mse_loss, clipped_mse_loss)

        positive_diff_mask = reward_diffs > reward_diff_threshold

        if positive_diff_mask.sum() == 0:
            loss = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        else:
            loss = (max_mse_loss * positive_diff_mask.float()).sum() / positive_diff_mask.float().sum()

        policy_loss = loss
        kl_loss = None

        return loss, policy_loss, kl_loss

    def grpo_loss_calculation(
            self,
            sampler,
            log_prob,
            prev_sample_mean,
            std_dev_t,
            sqrt_dt,
            prev_sample_mean_ref,
            beta2,
            rationorm,
            adv_clip_max,
            clip_range,
    ):
        # grpo logic
        advantages = torch.clamp(
            sampler.get_all_key_tensors("advantages"),
            -adv_clip_max,
            adv_clip_max,
        )
        if rationorm:
            sigma_t = std_dev_t.mean()
            ratio_mean_bias = (prev_sample_mean - sampler.get_all_key_tensors("prev_latents_mean")).pow(2).mean(
                dim=tuple(range(1, log_prob.ndim)))
            ratio_mean_bias = ratio_mean_bias / (2 * (sqrt_dt.mean() * sigma_t) ** 2)
            ratio = torch.exp(
                (log_prob - sampler.get_all_key_tensors("log_probs") + ratio_mean_bias) * (sqrt_dt.mean() * sigma_t))
        else:
            ratio = torch.exp(log_prob - sampler.get_all_key_tensors("log_probs"))

        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        if rationorm:
            policy_loss = policy_loss / (sqrt_dt.mean() ** 2)

        if beta2 > 0:
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (
                    2 * std_dev_t ** 2)
            kl_loss = torch.mean(kl_loss)
            loss = policy_loss + beta2 * kl_loss
        else:
            loss = policy_loss

        return loss, policy_loss, kl_loss if beta2 > 0 else None

    def rl_update_step(
            self,
            latents,
            reference_latents,
            t,
            i,
            guidance,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            do_true_cfg,
            negative_image_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            true_cfg_scale,
            timesteps,
            noise_level,
            group_size,
            rl_interation_num,
            beta1,
            beta2,
            step_size,
            generator,
            rationorm,
            adv_clip_max,
            clip_range,
            internal_reward_timestep,
            score_dict,
            prompts,
            width,
            height,
            scoring_fn,
            reward_diff_threshold,
    ):
        reference_latents = reference_latents.clone().detach()
        for index in range(rl_interation_num):
            if index == 0:
                with torch.no_grad():
                    sampler = self.rollout(
                        latents=latents.clone(),
                        t=t,
                        i=i,
                        guidance=guidance,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        do_true_cfg=do_true_cfg,
                        negative_image_embeds=negative_image_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        true_cfg_scale=true_cfg_scale,
                        timesteps=timesteps,
                        noise_level=noise_level,
                        group_size=group_size,
                        prompts=prompts,
                        width=width,
                        height=height,
                        scoring_fn=scoring_fn,
                    )

            with torch.enable_grad():
                latents = latents.clone().detach().requires_grad_(True)
                prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = self.compute_log_prob(
                    sampler=sampler,
                    latents=latents,
                    t=t,
                    guidance=guidance,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    do_true_cfg=do_true_cfg,
                    negative_image_embeds=negative_image_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    true_cfg_scale=true_cfg_scale,
                    group_size=group_size,
                    noise_level=noise_level,
                    generator=generator,
                    rationorm=rationorm,
                )

            with torch.no_grad():
                _, log_prob_ref, prev_sample_mean_ref, _, _ = self.compute_log_prob(
                    sampler=sampler,
                    latents=reference_latents,
                    t=t,
                    guidance=guidance,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    do_true_cfg=do_true_cfg,
                    negative_image_embeds=negative_image_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    true_cfg_scale=true_cfg_scale,
                    group_size=group_size,
                    noise_level=noise_level,
                    generator=generator,
                    rationorm=rationorm,
                )

            if self.num_timesteps * internal_reward_timestep > i:
                print("prdp loss calculation")
                with torch.enable_grad():
                    loss, policy_loss, kl_loss = self.prdp_loss_calculation(
                        sampler=sampler,
                        log_prob=log_prob,
                        prev_sample_mean=prev_sample_mean,
                        std_dev_t=std_dev_t,
                        sqrt_dt=sqrt_dt,
                        prev_sample_mean_ref=prev_sample_mean_ref,
                        beta1=beta1,
                        rationorm=rationorm,
                        adv_clip_max=adv_clip_max,
                        clip_range=clip_range,
                        log_probs_ref=log_prob_ref,
                        reward_diff_threshold=reward_diff_threshold,
                    )

                print(f" ---- RL Update Step {i} ----")
                print(f"Total Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}" + (
                    f", KL Loss: {kl_loss.item():.4f}" if kl_loss is not None else ""))

                print(f"DEBUG Step {i}:")
                print(f"  > Rewards: {sampler.get_all_key_tensors('rewards')}")
                print(f"  > Advantages (Raw): {sampler.get_all_key_tensors('advantages')}")
                print(
                    f"  > Log Prob Diff (Mean): {(log_prob - sampler.get_all_key_tensors('log_probs')).mean().item()}")

                # update latents
                latents = self.update_latent(latents, loss, step_size, type='prdp', noise_level=noise_level,
                                             sqrt_dt=sqrt_dt.mean() if rationorm else 1.0)

            else:
                print("grpo loss calculation")
                with torch.enable_grad():
                    loss, policy_loss, kl_loss = self.grpo_loss_calculation(
                        sampler=sampler,
                        log_prob=log_prob,
                        prev_sample_mean=prev_sample_mean,
                        std_dev_t=std_dev_t,
                        sqrt_dt=sqrt_dt,
                        prev_sample_mean_ref=prev_sample_mean_ref,
                        beta2=beta2,
                        rationorm=rationorm,
                        adv_clip_max=adv_clip_max,
                        clip_range=clip_range,
                    )

                    print(f" ---- RL Update Step {i} ----")
                    print(f"Total Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}" + (
                        f", KL Loss: {kl_loss.item():.4f}" if kl_loss is not None else ""))

                    print(f"DEBUG Step {i}:")
                    print(f"  > Rewards: {sampler.get_all_key_tensors('rewards')}")
                    print(f"  > Advantages (Raw): {sampler.get_all_key_tensors('advantages')}")
                    print(
                        f"  > Log Prob Diff (Mean): {(log_prob - sampler.get_all_key_tensors('log_probs')).mean().item()}")

                    # update latents
                    latents = self.update_latent(latents, loss, step_size, type='grpo', noise_level=noise_level,
                                                 sqrt_dt=sqrt_dt.mean() if rationorm else 1.0)
        sampler.clear()
        return latents.clone(), reference_latents

    @torch.no_grad()
    def rollout(
            self,
            latents,
            t,
            i,
            guidance,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            do_true_cfg,
            negative_image_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            true_cfg_scale,
            timesteps,
            noise_level,
            group_size,
            prompts,
            height,
            width,
            scoring_fn,
    ):
        ###Rollout
        assert timesteps[i] == t
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        latent_copy = latents.clone().detach()

        with self.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

        if do_true_cfg:
            if negative_image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

            with self.transformer.cache_context("uncond"):
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        sampler = RLSampler()
        for index in range(group_size):
            next_latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred,
                t.unsqueeze(0).repeat(latents.shape[0]),
                latent_copy,
                noise_level=noise_level,
            )
            sampler.rollout(sample_id=index, timesteps=t, latents=latent_copy, next_latents=next_latents,
                            log_probs=log_prob,
                            prev_latents_mean=prev_latents_mean, std_dev_t=std_dev_t, rewards=None)

            if i + 1 < len(timesteps):
                next_t_val = timesteps[i + 1]
            else:
                next_t_val = torch.tensor(0, device=t.device, dtype=t.dtype)
            next_t_val = next_t_val.expand(next_latents.shape[0])
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=next_latents,
                    timestep=next_t_val / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=next_latents,
                        timestep=next_t_val / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            t_next = next_t_val.view(-1, 1, 1).to(latents.dtype)
            dt = -t_next / 1000.0
            pred_x0_latents = next_latents + dt * noise_pred
            latents_unpack = self._unpack_latents(pred_x0_latents, height, width, self.vae_scale_factor)
            latents_scaled = (latents_unpack / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image_tensor = self.vae.decode(latents_scaled, return_dict=False)[0]
            image = self.image_processor.postprocess(image_tensor, output_type="pil")

            if isinstance(prompts, str):
                prompts = [prompts]

            score, _ = scoring_fn(images=image, prompts=prompts, metadata={})
            sampler.update_reward(
                sample_id=index,
                new_rewards=score['avg'][0]
            )
            del image_tensor, latents_unpack, latents_scaled, pred_x0_latents,image
            torch.cuda.empty_cache()
        sampler.compute_and_update_advantages()
        return sampler


