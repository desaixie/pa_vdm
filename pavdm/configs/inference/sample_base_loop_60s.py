resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51  # 15 latents
fps = 24
frame_interval = 1
save_fps = 24
loop = 42
# 17 + (51-17) * 42 = 1445
# 1445 - 1445%24 = 1440 final number of frames
# mask_strategy = ["0", "0", "0", "0", "0", "0", "0", "0"]  # assumes 8 prompts
num_prompts = 40
mask_strategy = ["0"] * num_prompts

save_dir = "./evaluations/bn"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5  # num latent frames to condition on
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
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
)

aes = 6.5
flow = None
