import torch
from einops import repeat
from tqdm import tqdm

from bn_configs.inference.sample_base_chunk_keep0 import keep_x0
from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        latent_chunk_size=1,
        keep_x0=False,
        variable_length=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.pa_vdm = kwargs.get("pa_vdm", False)
        self.noise_pattern = kwargs.get("noise_pattern", "linear")
        self.linear_variance_scale = kwargs.get("linear_variance_scale", 0.1)
        # inference
        self.all_progressive_timesteps = None
        self.num_stages = None
        self.latent_chunk_size = latent_chunk_size
        self.progressive_timesteps_stages = None
        self.keep_x0 = keep_x0
        self.variable_length = variable_length

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            latent_chunk_size=latent_chunk_size,
            keep_x0=keep_x0,
            variable_length=variable_length,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]  # (30,): [1000., 966.66, ..., 66.66, 33.33]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]  # list of 30 numbers -> list of 30 cuda tensor(1,)
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]  # for each individual t, scale it with the ratio to the base resolution (512x512) and base num latent frames (17). [1000., 966.66, ..., 66.66, 33.33] -> [1000., 990.6, ..., 287.1, 111.1]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z

    def sample_pavdm(
            self,
            model,
            text_encoder,
            z,  # (1, 4, 15, 45, 80)
            prompts,  # (1,): ["drone view ..."]
            device,
            additional_args=None,
            mask=None,  # ((1, 15): [[0., 1., ..., 1.]])
            guidance_scale=None,
            progress=True,
            num_frames=35,
            num_ar_latent_frames=None,
    ):
        assert mask is None, "pa vdm is not supported with mask"
        # assert self.num_sampling_steps == z.shape[2], f"num_sampling_steps should be equal to the number of frames, {self.num_sampling_steps} != {z.shape[2]}"

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)  # 1
        b, c, f, h, w = z.shape  # (1, 4, 15, 45, 80)
        num_frames = num_frames  # 35
        num_frames_no_x0 = num_frames - self.latent_chunk_size if self.keep_x0 else num_frames
        if self.variable_length:
            z = z[:, :, :self.latent_chunk_size]  # (1, 4, 5, 45, 80)
        if z.device.index == 0:
            print(f"z.shape {z.shape}, num_frames: {num_frames}, num_frames_no_x0: {num_frames_no_x0}, num_sampling_steps: {self.num_sampling_steps}, latent_chunk_size: {self.latent_chunk_size}, keep_x0: {self.keep_x0}, variable_length: {self.variable_length}")
        # text encoding
        model_args = text_encoder.encode(prompts)  # {'y': tensor(1, 1, 300, 4096), 'mask': tensor(1, 300)}
        y_null = text_encoder.null(n)  # (1, 1, 300, 4096)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)  # (2, 1, 300, 4096)
        if additional_args is not None:
            model_args.update(
                additional_args)  # height, width, num_frames, ar (aspect ratio), fps, for the multi-resolution

        if self.noise_pattern in ["linear", "linear-variance", "linear-variance-shift", "linear-shift"]:  # always sample with linear
            # always reset the progressive timesteps stages and then timestep transform them
            self.set_progressive_timesteps_stages(num_frames_no_x0, self.num_sampling_steps, self.latent_chunk_size, device)
        else:
            raise NotImplementedError(f"noise pattern {self.noise_pattern} not implemented")
        if self.use_timestep_transform:
            self.progressive_timesteps_stages = [timestep_transform(self.progressive_timesteps_stages[i].clone(), additional_args, num_timesteps=self.num_timesteps) for i in range(len(self.progressive_timesteps_stages))]
        if z.device.index == 0:
            print(f'progressive_timesteps_stages: {self.progressive_timesteps_stages}')

        # pa vdm: add noise to the initial z
        noise = torch.randn_like(z)
        t_stage0 = self.get_progressive_timesteps(0, b, training=False, variable_length=self.variable_length, initialization=True, variable_num_frames_no_x0=self.latent_chunk_size)  # (b, f)
        if self.keep_x0:  # [x_0 chunk, latents[chunk1:] noised, noise chunk],
            x_0_chunk = z.clone()[:, :, :self.latent_chunk_size]  # (b, c, n, h, w)
            t_0_chunk = torch.zeros_like(t_stage0[:, :self.latent_chunk_size])
            if self.variable_length:
                new_noise_chunk = noise
                z = torch.cat([x_0_chunk, new_noise_chunk], dim=2)  # (b, c, 2n, h, w)
            else:
                later_chunks_noised = self.scheduler.add_noise(z.clone()[:, :, self.latent_chunk_size:], noise[:, :, :-self.latent_chunk_size], t_stage0[:, :-self.latent_chunk_size])  # (b, f-n, c, h, w)
                new_noise_chunk = noise[:, :, -self.latent_chunk_size:]
                z = torch.cat([x_0_chunk, later_chunks_noised, new_noise_chunk], dim=2)  # (b, c, n+f, h, w)
        else:
            z = self.scheduler.add_noise(z, noise, t_stage0)  # (b, c, f, h, w)

        ar_frames = []
        if keep_x0:
            ar_frames.append(x_0_chunk)
        if self.variable_length:
            initialization = True
            termination = False
            variable_num_frames = z.shape[2]
            variable_num_frames_no_x0 = variable_num_frames - self.latent_chunk_size if keep_x0 else variable_num_frames
            start = 0
            end = start + variable_num_frames
            num_ar_steps = num_ar_latent_frames + 20  # 25 initialization steps - 5 free conditioning latents
        else:
            initialization = False
            termination = False
            variable_num_frames = 0
            variable_num_frames_no_x0 = 0
            start = 0
            end = start + num_frames
            num_ar_steps = num_ar_latent_frames

        progress_wrap = tqdm if progress else (lambda x: x)
        for i in progress_wrap(range(num_ar_steps)):
            stage_i = i % self.num_stages
            timesteps = self.get_progressive_timesteps(stage_i, b, training=False, variable_length=self.variable_length, initialization=initialization, termination=termination, variable_num_frames_no_x0=variable_num_frames_no_x0)  # (b, f)
            if self.keep_x0:
                timesteps_with_t0 = torch.cat([t_0_chunk, timesteps], dim=1)  # (b, n+f)
            if z.device.index == 0:
                print(f"step_i {i}, stage {stage_i}, time {timesteps}, time_with_t0 {timesteps_with_t0}")

            # forward with cfg
            z_in = torch.cat([z, z], 0)
            t = torch.cat([timesteps_with_t0, timesteps_with_t0], 0) if self.keep_x0 else torch.cat([timesteps, timesteps], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # scheduler step
            if stage_i < self.num_stages - 1:
                dt = timesteps.clone() - self.get_progressive_timesteps(stage_i + 1, b, training=False, variable_length=self.variable_length, initialization=initialization, termination=termination, variable_num_frames_no_x0=variable_num_frames_no_x0)
            else:  # stage -1
                if initialization and variable_num_frames_no_x0 < num_frames_no_x0:
                    dt = timesteps.clone() - self.get_progressive_timesteps(0, b, training=False, variable_length=self.variable_length, initialization=initialization, termination=termination, variable_num_frames_no_x0=variable_num_frames_no_x0 + self.latent_chunk_size)[:, :-self.latent_chunk_size]
                elif termination and variable_num_frames_no_x0 == self.latent_chunk_size:
                    dt = timesteps.clone()
                else:
                    dt = timesteps.clone() - torch.cat([torch.zeros_like(timesteps[:, :self.latent_chunk_size]), self.get_progressive_timesteps(0, b, training=False, variable_length=self.variable_length, initialization=initialization, termination=termination, variable_num_frames_no_x0=variable_num_frames_no_x0)[:, :-self.latent_chunk_size]], dim=1)
            if z.device.index == 0:
                print(f"step_i {i}, stage {stage_i}, variable_num_frames_no_x0: {variable_num_frames_no_x0}, timesteps: {timesteps}, dt: {dt}")
            dt = (dt / self.num_timesteps).to(z.dtype)[:, None, :, None, None]  # (1, f) -> (b, 1, f, 1, 1)
            if self.keep_x0:
                z = z.clone()[:, :, self.latent_chunk_size:]
                v_pred = v_pred[:, :, self.latent_chunk_size:]
                z = z + v_pred * dt
                z = torch.cat([x_0_chunk, z], dim=2)
            else:
                z = z + v_pred * dt

            # last stage. save and append
            if stage_i == self.num_stages - 1:
                # save a latent chunk
                if self.variable_length and initialization:
                    pass
                else:
                    if self.keep_x0:
                        x_0_chunk = z.clone()[:, :, self.latent_chunk_size:2 * self.latent_chunk_size]  # (b, c, n, h, w). new x_0_chunk is the 2nd chunk
                    else:
                        x_0_chunk = z.clone()[:, :, :self.latent_chunk_size]  # (b, c, n, h, w)
                    ar_frames.append(x_0_chunk)
                    z = z.clone()[:, :, self.latent_chunk_size:]  # remove the old chunk

                # append a new latent chunk
                if self.variable_length and termination:
                    pass
                else:
                    # save one latent and append a new latent
                    noise_frame = torch.randn_like(z[:, :, 0:self.latent_chunk_size])
                    z = torch.cat([z, noise_frame], dim=2)

                # update counters
                if self.variable_length:
                    variable_num_frames = z.shape[2]
                    variable_num_frames_no_x0 = variable_num_frames - self.latent_chunk_size if keep_x0 else variable_num_frames

                    if initialization:
                        start = start
                        end += self.latent_chunk_size
                    elif termination:
                        start += self.latent_chunk_size
                        end = end
                    else:  # regular avd
                        start += self.latent_chunk_size
                        end = start + num_frames

                    # check for initialization and termination conditions at the end
                    if initialization and variable_num_frames == num_frames:
                        initialization = False
                    if (not termination) and end >= num_ar_latent_frames:
                        termination = True
                        if end > num_ar_latent_frames:
                            end = num_ar_latent_frames
                else:
                    start += self.latent_chunk_size
                    end = start + num_frames

                if z.device.index == 0:
                    print(f'start {start}, end {end}, variable_num_frames_no_x0 {variable_num_frames_no_x0}, variable_num_frames {variable_num_frames}, initialization {initialization}, termination {termination}')

                # avd done

        return torch.cat(ar_frames, dim=2)

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    def set_progressive_timesteps_stages(self, num_frames: int, num_inference_steps: int, latent_chunk_size: int, device: torch.device):
        """
        progressive timesteps stages (b=1, f) for each frame. assumes b=1, needs to be repeated for batch size > 1
        need to do .to(z.dtype). keeping t in float32
        """

        self.all_progressive_timesteps = (torch.linspace(0, 1, num_frames + 1, device=device)[1:] * self.num_timesteps).unsqueeze(0)  # skip 0. (1, 30,): [33.33,  66.66, 100.00, 133.33, 166.66,
           #                   200.00, 233.33, 266.66, 300.00, 333.33,
           #                   366.66, 400.00, 433.33, 466.66, 500.00,
           #                   533.33, 566.66, 600.00, 633.33, 666.66,
           #                   700.00, 733.33, 766.66, 800.00, 833.33,
           #                   866.66, 900.00, 933.33, 966.66, 1000.00]

        if latent_chunk_size > 0:  # overwrite the latent_chunk_size from config
            self.latent_chunk_size = latent_chunk_size  # n. every n latents will have the same timestep and be saved/added together.

        # every latent_chunk_size frames is treated as one, using the same noise level. f' = f / c
        assert num_frames % self.latent_chunk_size == 0, f"num_frames should be multiple of latent_chunk_size, {num_frames} % {self.latent_chunk_size} != 0"
        assert num_inference_steps % ( num_frames // self.latent_chunk_size) == 0, f"num_inference_steps should be multiple of num_frames // latent_chunk_size, {num_inference_steps} % {num_frames // self.latent_chunk_size} != 0"
        self.num_stages = num_inference_steps // (num_frames // self.latent_chunk_size)  # every m steps, save latent_chunk_size frames. m = t / f' = t / (f / c) = c * (t / f)
        if self.all_progressive_timesteps.device.index == 0:
            print(f"all_progressive_timesteps: {self.all_progressive_timesteps}")
            print(f"num_stages: {self.num_stages}")
            print(f"latent_chunk_size: {self.latent_chunk_size}")

        # (b, t) -> [(b, t / m) in reverse range(m)] -> [(b, f) in reverse range(m)]
        self.progressive_timesteps_stages = [repeat(self.all_progressive_timesteps[:, (self.num_stages-1) - s::self.num_stages], "b f -> b f c", c=self.latent_chunk_size).flatten(1, 2) for s in range(self.num_stages)]
        if self.all_progressive_timesteps.device.index == 0:
            print(f"progressive_timesteps_stages: {self.progressive_timesteps_stages}")
        # latent_chunk_size=5:
        # [[ 166, 166, 166, 166, 166, 333, 333, 333, 333, 333, ..., 1000, 1000, 1000, 1000, 1000],
        #  [ 133, 133, 133, 133, 133, 300, 300, 300, 300, 300, ...,  966,  966,  966,  966,  966],
        #  [ 100, 100, 100, 100, 100, 266, 266, 266, 266, 266, ...,  933,  933,  933,  933,  933],
        #  [  66,  66,  66,  66,  66, 233, 233, 233, 233, 233, ...,  900,  900,  900,  900,  900],
        #  [  33,  33,  33,  33,  33, 200, 200, 200, 200, 200, ...,  866,  866,  866,  866,  866]]


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