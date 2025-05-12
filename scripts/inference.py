import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.utils import to_pil_images
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt, initialize_condition_video,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        # enable_sequence_parallelism = coordinator.world_size > 1
        enable_sequence_parallelism = False
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024), diff_rank_seed=True)

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)  # 2s -> 51 frames
    num_ar_frames = None
    if cfg.get("pa_vdm", False):
        num_ar_frames = get_num_frames(cfg.get("num_ar_frames", '16s'))

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    if cfg.get("pa_vdm", False):
        num_ar_latent_frames = vae.get_latent_size((num_ar_frames, *image_size))[0]
        print(f'num_frames: {num_frames}, num_ar_frames: {num_ar_frames}, num_ar_latent_frames: {num_ar_latent_frames}, latent_size: {latent_size}')
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)


    if is_main_process():
        save_dir = cfg.save_dir
        save_dir += '' if save_dir.endswith('/') else '/'
        os.makedirs(save_dir, exist_ok=True)
        sample_name = cfg.get("sample_name", None)
        prompt_as_path = cfg.get("prompt_as_path", False)

    global_prompts = prompts.copy()
    if is_distributed():
        # split prompts, mask_strategy, reference_path wrt. rank
        prompts = prompts[coordinator.rank::coordinator.world_size]
        mask_strategy = mask_strategy[coordinator.rank::coordinator.world_size]
        reference_path = reference_path[coordinator.rank::coordinator.world_size]

    video_results_rows = [[] for _ in range(len(global_prompts))]  # one prompt per row
    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):  # iterate over prompts with batch size increments
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                print(f"inference sampling, z shape: {z.shape}")
                if cfg.get("pa_vdm", False):
                    masks = None
                    z = initialize_condition_video(z, refs)
                    samples = scheduler.sample_pavdm(
                        model,
                        text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                        num_frames=latent_size[0],
                        num_ar_latent_frames=num_ar_latent_frames,
                    )
                    samples = vae.decode(samples.to(dtype), num_frames=num_ar_frames)
                    if z.device.index == 0:
                        print(f'after avd inference and decode, samples shape: {samples.shape}')
                    num_extra_frames = num_ar_frames % save_fps
                    samples = samples[:, :, :-num_extra_frames]
                    if z.device.index == 0:
                        print(f'after removing extra frames, samples shape: {samples.shape}')
                else:
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                    )
                    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                video_clips.append(samples)

            # gather video clips, batch_prompts, if distributed
            batch_prompts_list = [batch_prompts]
            video_clips_list = [video_clips]
            if is_distributed():
                video_clips_list = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
                batch_prompts_list = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
                dist.gather_object(video_clips, object_gather_list=video_clips_list, dst=0)
                dist.gather_object(batch_prompts, object_gather_list=batch_prompts_list, dst=0)

            # == save samples ==
            if is_main_process():
                for rank_i, (batch_prompts_rank, video_clips_rank) in enumerate(zip(batch_prompts_list, video_clips_list)):
                    for idx, batch_prompt in enumerate(batch_prompts_rank):  # iterate over batch size
                        # handle loop
                        video = [video_clips_rank[j][idx] for j in range(loop)]
                        for j in range(1, loop):
                            video[j] = video[j][:, dframe_to_frame(condition_frame_length) :]  # remove the condition frames at the beginning. condition_frame_length=5 is converted to 17
                        video = torch.cat(video, dim=1)  # cat the looped videos together at the time dimension
                        pil_images = to_pil_images(video)
                        # columns store num_sample video results of the same prompt
                        video_results_rows[i * dist.get_world_size() + rank_i * batch_size + idx].append(pil_images)

        start_idx += len(batch_prompts)

    if is_main_process():
        logger.info("Inference finished.")
        logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
