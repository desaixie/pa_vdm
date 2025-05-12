# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path="/home/zhaowangbo/data/csv/image-v1_1_ext_noempty_rcp_clean_info.csv",
    num_frames=1,
    frame_interval=3,
    image_size=(2048, 2048),
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="PixArt-1B/2",
    space_scale=4.0,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-1B-2.pth",
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    subfolder="vae",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = 4
lr = 2e-5
grad_clip = 1.0
