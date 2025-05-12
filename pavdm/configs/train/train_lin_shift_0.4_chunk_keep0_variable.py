# pa vdm
pa_vdm = True
# noise_pattern = "linear"
noise_pattern = "linear-shift"
linear_variance_scale = 0.4
linear_shift_scale = 0.4
latent_chunk_size = 5
keep_x0 = True
variable_length = True
# training
fixed_resolution = (240, 424)  # latent size (30, 53)
data_config = "path/to/data/config"
num_frames_pavdm = 119

# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

bucket_config = {  # 20s/it
    "240p": {119: (1.0, 2)},
}

# webvid
# bucket_config = {  # 20s/it
#     "144p": {1: (1.0, 475), 51: (1.0, 51), 102: (1.0, 27), 204: (1.0, 13), 408: (1.0, 6)},
#     # ---
#     "256": {1: (1.0, 297), 51: (0.5, 20), 102: (0.5, 10), 204: (0.5, 5), 408: ((0.5, 0.5), 2)},
#     "240p": {1: (1.0, 297), 51: (0.5, 20), 102: (0.5, 10), 204: (0.5, 5), 408: ((0.5, 0.4), 2)},
#     # ---
#     "360p": {1: (1.0, 141), 51: (0.5, 8), 102: (0.5, 4), 204: (0.5, 2), 408: ((0.5, 0.3), 1)},
#     "512": {1: (1.0, 141), 51: (0.5, 8), 102: (0.5, 4), 204: (0.5, 2), 408: ((0.5, 0.2), 1)},
#     # ---
#     "480p": {1: (1.0, 89), 51: (0.5, 5), 102: (0.5, 3), 204: ((0.5, 0.5), 1), 408: (0.0, None)},
#     # ---
#     "720p": {1: (0.3, 36), 51: (0.2, 2), 102: (0.1, 1), 204: (0.0, None)},
#     "1024": {1: (0.3, 36), 51: (0.1, 2), 102: (0.1, 1), 204: (0.0, None)},
#     # ---
#     "1080p": {1: (0.1, 5)},
#     # ---
#     "2048": {1: (0.05, 5)},
# }

grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,  # requires a matching torch version
    freeze_y_embedder=True,
    pa_vdm=pa_vdm,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
    num_sampling_steps=30,
    cfg_scale=7.0,
    pa_vdm=pa_vdm,
    noise_pattern=noise_pattern,
    linear_variance_scale=linear_variance_scale,
    latent_chunk_size=latent_chunk_size,
    keep_x0=keep_x0,
)

# Mask settings
# 25%
# mask_ratios = {
#     "random": 0.01,
#     "intepolate": 0.002,
#     "quarter_random": 0.002,
#     "quarter_head": 0.002,
#     "quarter_tail": 0.002,
#     "quarter_head_tail": 0.002,
#     "image_random": 0.0,
#     "image_head": 0.22,
#     "image_tail": 0.005,
#     "image_head_tail": 0.005,
# }
mask_ratios = None

# Log settings
seed = 42
outputs = ""
eval_base_dir = ""
wandb = True
epochs = 1000
log_every = 10
ckpt_every = 1000
eval_every = 10000000000

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

# eval settings
fps = 24
frame_interval = 1
multi_resolution = "STDiT2"
aes = 6.5
prompt_path = 'pavdm/test_data/all_prompts.txt'
start_index = 0
end_index = 1
num_frames = 119  # 35 latents
resolution = "240p"
aspect_ratio = "9:16"
num_ar_frames = 1445  # 85 latents. 1440f (60s) + 5f extra
loop = 1
reference_path = ["pavdm/test_data/sora_samples/clipped_16_176_320/artgallery_s0_16f_320x176.mp4",]
batch_size = 1
num_sample = 2