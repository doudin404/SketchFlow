import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from typing import Optional, Tuple, Union, List
import numpy as np

class SketchPipeline(DiffusionPipeline):
    """
    A pipeline for generating sketches, supporting both standard diffusion sampling
    and flow-based (ODE) sampling.
    """
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        shape: Tuple[int, int, int],
        mode: str = 'diffusion',
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        cond_vec: Optional[torch.Tensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
        return_n_intermediate: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            shape (Tuple[int, int, int]): The shape of the output tensor (B, L, C).
            mode (str): Sampling mode, either 'diffusion' or 'flow'.
            num_inference_steps (int): The number of denoising steps.
            generator (torch.Generator, optional): A generator to make generation deterministic.
            cond_vec (torch.Tensor, optional): Vector condition.
            cond_tokens (torch.Tensor, optional): Token condition (for cross-attention).
            return_n_intermediate (Optional[int]): If not None, returns n intermediate steps.
            show_progress_bar (bool): Whether to show a progress bar.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The final generated sample or a list of intermediate samples.
        """
        if mode not in ['diffusion', 'flow']:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'diffusion' or 'flow'.")

        # Prepare initial latent variable
        device = next(self.unet.parameters()).device
        B, L, C = shape
        latents = torch.randn(shape, device=device, generator=generator)

        # Set scheduler/sampler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Intermediate steps logic
        collect_intermediates = False
        target_indices = set()
        if return_n_intermediate is not None:
            n = int(return_n_intermediate)
            if n <= 0:
                raise ValueError("return_n_intermediate must be a positive integer or None")
            T = len(timesteps)
            indices = [T - 1] if n == 1 else [int(round(i * (T - 1) / (n - 1))) for i in range(n)]
            target_indices = set(indices)
            collect_intermediates = True
            intermediates: List[torch.Tensor] = []

        # Denoising/Sampling loop
        iterable = self.progress_bar(timesteps) if show_progress_bar else timesteps
        for i, t in enumerate(iterable):
            t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)

            # Predict the model output (noise or velocity)
            model_pred = self.unet.predict(
                latents, 
                t_batch, 
                cond_vec=cond_vec, 
                cond=cond_tokens
            )

            # Calculate x0 (denoised sample) for recording
            if mode == 'diffusion':
                step_out = self.scheduler.step(model_pred, t, latents, generator=generator)
                x0_pred = step_out.pred_original_sample
                latents = step_out.prev_sample
            elif mode == 'flow':
                # For flow matching with linear path: x_t = (1-t)x0 + t*x1, v = x1 - x0
                # Here t is normalized [0, 1]. model_pred is v_pred.
                # x0_pred = x_t - t * v_pred
                t_norm = t.float() / self.scheduler.config.num_train_timesteps
                x0_pred = latents - t_norm * model_pred
                
                dt = 1.0 / len(timesteps)
                latents = latents + model_pred * dt

            if collect_intermediates and i in target_indices:
                intermediates.append(latents.clone())

        if collect_intermediates:
            return intermediates
        
        return latents
