# Progressive Autoregressive Video Diffusion Models
Official code for CVPR 2025 CVEU Workshop paper, *Progressive Autoregressive Video Diffusion Models*  
Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, Yang Zhou  
Stony Brook University, Adobe Research  

![figure1 merged](https://github.com/user-attachments/assets/c3c1a7ca-86f4-4495-8a71-75604fffba26)

[arxiv](https://arxiv.org/abs/2410.08151) [website](https://desaixie.github.io/pa-vdm/)

## News
- **[05/11/2025]** The training and inference code based on Open-Sora v1.2 are released.
- **[05/11/2025]** Our testing data is released.

## Codebase
You can `diff` this codebase with [this Open-Sora commit](https://github.com/hpcaitech/Open-Sora/tree/a29424c2373339ffe60193e009702dd03da06350) to see what we modified in PA-VDM. The core implementation differences are in `opensora/schedulers/rf/__init__.py` `opensora/models/layers/blocks.py`, `scripts/inference.py`, `opensora/schedulers/rf/rectified_flow.py`, `opensora/models/stdit/stdit3.py`.

New training/inference configs:
- `latent_chunk_size`: for "Chunked Frames". 1 for disabled, 5 for Open-Sora.
- `keep_x0`: for "Overlapped Conditioning".
- `variable_length`: for "Variable Length"

## Testing Data
We release our testing set of 40 real videos and text prompts.
The text prompts are in `pavdm/test_data/all_prompts.txt`.
Each line correspond to one video, in the format `text_prompt||video_file_path.mp4`.
The videos can be downloaded from this [huggingface dataset](https://huggingface.co/datasets/desaix/pavdm_test_test). The video filenames, e.g. artgallery_s0_16f_320x176.mp4, include the staring frame number `s0` from the raw video, the length of the video clip `16f`, and the resolution `320x176`. If you want to test at different resolutions, using different number of frames as initial condition, etc., you can extract new video clips using the same starting frame number (0th frame for artgallery.webm) from the corresponding raw videos located in the `raw/` folders.

## Inference
OpenSora's naive autoregressive long video extension with mask
```bash
bash eval/sample.sh hpcai-tech/OpenSora-STDiT-v3 0 v1.2 -2h
```
PA-VDM
```bash
torchrun --standalone --nproc_per_node ${RUNAI_NUM_OF_GPUS} scripts/inference.py pavdm/configs/inference/sample_base_chunk_keep0_variable.py \
    --ckpt-path hpcai-tech/OpenSora-STDiT-v3 \
    --save-dir outputs/ --sample-name pavdm_inference \
    --prompt-path pavdm/test_data/all_prompts.txt --start-index 0 --end-index 40 \
    --aspect-ratio 9:16 \
    --batch-size 1 --num-sample 1
```

## Training
The training code and config are provided as a template and haven't been properly tested.
You also need to prepare your own training data following Open-Sora's finetuning instructions.
```bash
torchrun --nproc_per_node ${NODE_GPU_CNT} --nnodes=${NUM_NODEs} scripts/train.py pavdm/configs/train/train_lin_shift_0.4_chunk_keep0_variable.py --data-path path/to/data.csv --ckpt-path hpcai-tech/OpenSora-STDiT-v3 --exp-name ${EXP_NAME}
```
