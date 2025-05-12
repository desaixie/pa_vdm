import random

import torch
from torch.distributions import LogisticNormal
from einops import repeat

from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,  # 1000
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()
            
    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        pa_vdm=False,
        noise_pattern="linear",
        linear_variance_scale=0.1,
        linear_shift_scale=0.3,
        latent_chunk_size=1,
        keep_x0=False,
        variable_length=False,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        # pa vdm
        self.pa_vdm = pa_vdm
        self.noise_pattern = noise_pattern
        self.linear_variance_scale = linear_variance_scale
        self.linear_shift_scale = linear_shift_scale
        if pa_vdm:
            assert not use_discrete_timesteps, "pa vdm is not supported for discrete timesteps"
        # training
        self.training_all_progressive_timesteps = None
        self.training_num_stages = None
        self.training_latent_chunk_size = latent_chunk_size
        self.training_progressive_timesteps_stages = None
        self.keep_x0 = keep_x0
        self.variable_length = variable_length

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if self.pa_vdm and self.keep_x0:  # split x. now f stays as 50
            x_0_chunk = x_start.clone()[:, :, :self.training_latent_chunk_size]  # (b, 5, c, h, w)
            x_start = x_start.clone()[:, :, self.training_latent_chunk_size:]  # (b, f=50, c, h, w)
        b, c, f, h, w = x_start.shape  # f: num_frames_no_x0
        if x_start.device.index == 0:
            print(f"x_start: {x_start.shape}")

        if t is None:
            if self.pa_vdm:
                if self.training_all_progressive_timesteps is None:  # only set the stages once during training, without target_pix_cnt or warp_t
                    self.set_training_progressive_timesteps_stages(f, self.num_sampling_steps, x_start.device)

                if self.variable_length:
                    initialization = random.choice([True, False])
                    termination = not initialization
                else:
                    initialization = termination = False

                if self.noise_pattern == "linear":
                    t = torch.linspace(0, 1, f + 1, device=x_start.device, dtype=x_start.dtype)[1:] * self.num_timesteps # (60,)
                    t = t.unsqueeze(0).repeat(b, 1)  # (5, 60)
                elif self.noise_pattern == "linear-variance":
                    t = torch.linspace(0, 1, f + 1, device=x_start.device, dtype=x_start.dtype)[1:] * self.num_timesteps # (60,)
                    var = torch.randn_like(t) * t[0] * self.linear_variance_scale
                    t += var
                    t = t.clamp(min=0, max=self.num_timesteps).unsqueeze(0).repeat(b, 1)  # (5, 60)
                elif self.noise_pattern == "linear-variance-shift":
                    t = torch.linspace(0, 1, f + 1, device=x_start.device, dtype=x_start.dtype)[
                        1:] * self.num_timesteps  # (60,)
                    var = torch.randn_like(t) * t[0] * self.linear_variance_scale
                    shift = torch.randn(1, dtype=t.dtype, device=t.device) * t[0] * self.linear_shift_scale
                    t += var + shift
                    t = t.clamp(min=0, max=self.num_timesteps).unsqueeze(0).repeat(b, 1)  # (5, 60)
                elif self.noise_pattern == "linear-shift":
                    stage_i = random.randint(0, self.training_num_stages - 1)
                    t = self.get_progressive_timesteps(stage_i, b, training=True, variable_length=self.variable_length, initialization=initialization, termination=termination, variable_num_frames_no_x0=f)  # (b, f). chunked, not warped
                    timesteps_diff = self.num_timesteps / self.num_sampling_steps  # 20.
                    shift = (repeat(torch.randn((b, 1), device=x_start.device) * timesteps_diff * self.linear_shift_scale, "b i -> b (i f)", f=f)).to(t.dtype)  # same for all latent frames in a batch
                    t = (t + shift).clamp(min=0, max=self.num_timesteps - 1)

            else:
                if self.use_discrete_timesteps:
                    t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device)
                elif self.sample_method == "uniform":
                    t = torch.rand((b,), device=x_start.device) * self.num_timesteps
                elif self.sample_method == "logit-normal":
                    t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        if self.pa_vdm and self.keep_x0:
            x_t = self.add_noise(x_start, noise, t)
            x_t = torch.cat([x_0_chunk, x_t], dim=2)  # (b, c, f=35, h, w)
            t_0_chunk = torch.zeros_like(t[:, :self.training_latent_chunk_size])
            t = torch.cat([t_0_chunk, t], dim=1)
        else:
            x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            if self.pa_vdm and self.keep_x0:
                target = x_start - noise
                target = torch.cat([x_0_chunk, target], dim=2)  # the first chunk value does not matter. it will be zeroed out next
                x0_mask = torch.cat([torch.zeros((b, self.training_latent_chunk_size), dtype=x_start.dtype, device=x_start.device), torch.ones((b, f), dtype=x_start.dtype, device=x_start.device)], dim=1)  # (b, 35)
                loss = mean_flat((velocity_pred - target).pow(2), mask=x0_mask)
            else:
                loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000], converting to t for rectified flow, from 1 to 0

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        if self.pa_vdm:  # (5, 60)
            timepoints = timepoints.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (5, 60) -> (5, 1, 60, 1, 1)
            timepoints = timepoints.repeat(1, noise.shape[1], 1, noise.shape[3], noise.shape[4])  # (5, 1, 60, 1, 1) -> (5, 4, 60, 30, 53)
        else:
            timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (5,) -> (5, 1, 1, 1, 1)
            timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])  # (5, 1, 1, 1, 1) -> (5, 4, 60, 30, 53)
        timepoints = timepoints.to(original_samples.device)

        return timepoints * original_samples + (1 - timepoints) * noise

    def set_training_progressive_timesteps_stages(self, num_frames: int, num_inference_steps: int, device: torch.device):
        """
        progressive timesteps stages (b=1, f) for each frame. assumes b=1, needs to be repeated for batch size > 1
        need to do .to(z.dtype). keeping t in float32
        """

        self.training_all_progressive_timesteps = (torch.linspace(0, 1, num_frames + 1, device=device)[1:] * self.num_timesteps).unsqueeze(0)  # skip 0. (1, 30,): [33.33, 66.66, ..., 966.66, 1000.]

        # every latent_chunk_size frames is treated as one, using the same noise level. f' = f / c
        assert num_frames % self.training_latent_chunk_size == 0, f"num_frames should be multiple of latent_chunk_size, {num_frames} % {self.training_latent_chunk_size} != 0"
        assert num_inference_steps % ( num_frames // self.training_latent_chunk_size) == 0, f"num_inference_steps should be multiple of num_frames // latent_chunk_size, {num_inference_steps} % {num_frames // self.training_latent_chunk_size} != 0"
        self.training_num_stages = num_inference_steps // (num_frames // self.training_latent_chunk_size)  # every m steps, save latent_chunk_size frames. m = t / f' = t / (f / c) = c * (t / f)
        if self.training_all_progressive_timesteps.device.index == 0:
            print(f"all_progressive_timesteps: {self.training_all_progressive_timesteps}")
            print(f"num_stages: {self.training_num_stages}")
            print(f"latent_chunk_size: {self.training_latent_chunk_size}")

        # (b, t) -> [(b, t / m) in reverse range(m)] -> [(b, f) in reverse range(m)]
        self.training_progressive_timesteps_stages = [repeat(self.training_all_progressive_timesteps[:, (self.training_num_stages-1) - s::self.training_num_stages], "b f -> b f c", c=self.training_latent_chunk_size).flatten(1, 2) for s in range(self.training_num_stages)]
        if self.training_all_progressive_timesteps.device.index == 0:
            print(f"progressive_timesteps_stages: {self.training_progressive_timesteps_stages}")

    def get_progressive_timesteps(self, stage_i, batch_size, training, variable_length=False, initialization=False, termination=False, variable_num_frames_no_x0=None):
        """
        get the timesteps for the current stage
        """
        if training:
            timesteps = self.training_progressive_timesteps_stages[stage_i].clone().repeat(batch_size, 1)  # (b, f)
        else:
            timesteps = self.progressive_timesteps_stages[stage_i].clone().repeat(batch_size, 1)  # (b, f)
        if variable_length:
            if initialization:  # get the ending timesteps, [999]x5 from [91, 192, ..., 999]x5
                timesteps = timesteps[:, -variable_num_frames_no_x0:]
            elif termination:  # get the starting timesteps, [91]x5 from [91, ..., 999]x5
                timesteps = timesteps[:, :variable_num_frames_no_x0]
            else:
                pass
        return timesteps
