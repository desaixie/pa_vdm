# pa vdm
pa_vdm = True
noise_pattern = "linear"  # always use linear, no variance, for inference
# noise_pattern = "linear-variance"
linear_variance_scale = 0.1
latent_chunk_size = 5
keep_x0 = True
variable_length = True

resolution = "240p"
aspect_ratio = "9:16"  # this is 240 x 426, but only 424 is divisible by 8
num_frames = 119  # 35 latents
num_ar_frames = 1445  # 425 latents. 1440f (60s) + 5f extra to be divisible by 17
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./evaluations/pavdm"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    pa_vdm=pa_vdm,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
    pa_vdm=pa_vdm,
    noise_pattern=noise_pattern,
    linear_variance_scale=linear_variance_scale,
    latent_chunk_size=latent_chunk_size,
    keep_x0=keep_x0,
    variable_length=variable_length,
)

aes = 6.5
flow = None
