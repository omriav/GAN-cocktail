# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
from time import gmtime, strftime
import copy
import json
import pickle
from training.networks import Discriminator, Generator
from typing import List
import psutil
import PIL.Image
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import wandb

import legacy
from metrics import metric_main
from constants import *

#----------------------------------------------------------------------------

def print_log(message: str) -> None:
    time_prefix = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(f"{time_prefix}: {message}")

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(1024 // training_set.image_shape[2], 4, 8)
    gh = np.clip(1024 // training_set.image_shape[1], 4, 8)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def get_image_grid(img, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)  # type: ignore
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8) # type: ignore

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        return PIL.Image.fromarray(img[:, :, 0], 'L')

    return PIL.Image.fromarray(img, 'RGB')

def save_image_grid(img, fname, drange, grid_size):
    image_grid = get_image_grid(img=img, drange=drange, grid_size=grid_size)
    image_grid.save(fname)

def get_models(model_paths: List[str], new_n_classes: int, device):
    generators = []
    discriminators = []

    for model_path in model_paths:
        generator, discriminator = get_extended_model(model_path=model_path, 
                                                      new_n_classes=new_n_classes,
                                                      device=device)
        generators.append(generator)
        discriminators.append(discriminator)

    return generators, discriminators

def get_extended_model(model_path: str, new_n_classes: int, device):
    with dnnlib.util.open_url(model_path) as f:
        model_data = legacy.load_network_pkl(f)

        generator = model_data["G_ema"].eval().requires_grad_(False).cpu()
        discriminator = model_data["D"].eval().requires_grad_(False).cpu()

        if generator.c_dim < new_n_classes:
            generator, discriminator = extend_GAN_embeddings(input_G=generator, 
                                                             input_D=discriminator, 
                                                             new_n_classes=new_n_classes)

        return generator.to(device), discriminator.to(device)

def get_one_hot_batch(index_value, num_classes, batch_size):
    c = F.one_hot(torch.tensor(index_value), num_classes=num_classes)
    c = c.repeat(batch_size, 1)

    return c

@torch.no_grad()
def sample_merge_generators(merge_generators, batch_size, class_percentages, device):
    samples = []
    labels = []

    class_batch_sizes = np.random.multinomial(batch_size, class_percentages)

    for i, (generator, curr_batch_size) in enumerate(zip(merge_generators, class_batch_sizes)):
        if curr_batch_size != 0:
            z = torch.randn([curr_batch_size, generator.z_dim], device=device)
            # In the future we can make the "index_value" configureable, but for now we can leave it as the first class
            c = get_one_hot_batch(index_value=0, num_classes=len(class_percentages), batch_size=curr_batch_size).to(device)

            samples.append(generator(z=z, c=c))
            labels.append(get_one_hot_batch(index_value=i, num_classes=len(class_percentages), batch_size=curr_batch_size).to(device))

    samples = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)

    return samples, labels

def extend_GAN_embeddings(input_G, input_D, new_n_classes):
    input_G.cpu()
    input_D.cpu()

    output_G = Generator(z_dim=input_G.z_dim,
                         c_dim=new_n_classes,
                         w_dim=input_G.w_dim,
                         img_channels=input_G.img_channels,
                         img_resolution=input_G.img_resolution,
                         mapping_kwargs={'num_layers': 8},
                         synthesis_kwargs={'channel_base': 32768, 'channel_max': 512, 'conv_clamp': 256, 'num_fp16_res': 4})
    output_G = output_G.eval().requires_grad_(False).cpu()

    output_D = Discriminator(c_dim=new_n_classes,
                             img_resolution=input_D.img_resolution,
                             img_channels=input_D.img_channels,
                             num_fp16_res=4,
                             conv_clamp=256,
                             epilogue_kwargs={'mbstd_group_size': 4})
    output_D = output_D.eval().requires_grad_(False).cpu()

    assert input_G.c_dim < output_G.c_dim, "New number of classes must to be bigger than the old one"

    misc.copy_params_and_buffers_and_change_mapping(src_module=input_G, dst_module=output_G)
    misc.copy_params_and_buffers_and_change_mapping(src_module=input_D, dst_module=output_D)

    return output_G, output_D

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    compute_metrics_all_data = None,    # Compute the metrics on all the data.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    debug                   = None,     # If we are in debug mode
    log_wandb               = None,     # If we want to log to wandb
    job_desc                = None,     # The job description
    save_last_k_checkpoints = None,     # The number of checkpoints to save
    num_samples             = None,     # The number of samples to log
    class_percentages       = None,     # The percentages for the multinomial sampling distribution
    metrics_dataset_kwargs  = {},       # Options for the datasets that are being used by the metrics
    was_killed              = False,    # Indicator if the process was killed
    merge_mode              = None,     # Indicator if we are in "merge mode"     
    merge_model_paths       = None,     # List of paths for the source models
    root_model_path         = None,  # The path to the root model, it not specified, will use the first merge_generator
    init_merge_generator    = None,     # Indiciator to use the first merge generator as initializer
    init_merge_discriminator = None,     # Indiciator to use the first merge discriminator as initializer
    model_fisher_pkl        = None,    # The path to the Fisher information coefficients
    ewc_reference_model_index = None,   # The index for the reference model of the EWC loss
    generator_mapping_ewc_lambda    = None,    # The lambda of the EWC for the generator weights
    generator_synthesis_ewc_lambda    = None,    # The lambda of the EWC for the generator weights
    discriminator_ewc_lambda        = None,    # The lambda for the EWC for the discriminator weights
    evaluate_metrics_first_tick = None,  # Indicator to evaluate all the metrics on the first tick
    hybrid_mode = None, # Indicator for "hybrid mode" - use a model with averaging the weights
    hybrid_model_paths = None,  # The source pickles of the second fine-tuned model (from the first merge generator)
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print_log('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    if not merge_mode:
        training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    else:
        training_set_iterator = None

    if (rank == 0) and (not merge_mode):
        print_log(f'Num images: {len(training_set)}')
        print_log(f'Image shape: {training_set.image_shape}')
        print_log(f'Label shape: {training_set.label_shape}')

    # Load fisher
    if merge_mode and model_fisher_pkl is not None:
        print_log("Loading Fisher coefficients")
        fisher_coefficients = torch.load(model_fisher_pkl, map_location=device)
    else:
        fisher_coefficients = None

    # Construct networks.
    if rank == 0:
        print_log('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    if not hybrid_mode:
        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Load merge generators
    merge_generators = merge_discriminators = []
    if merge_mode:
        merge_generators, merge_discriminators = get_models(model_paths=merge_model_paths, 
                                                            new_n_classes=training_set.label_dim,
                                                            device=device)

        if root_model_path is not None:
            root_generator, root_discriminator = get_extended_model(model_path=root_model_path, 
                                                                    new_n_classes=training_set.label_dim,
                                                                    device=device)
        else:
            root_generator = merge_generators[0]
            root_discriminator = merge_discriminators[0]

        if not hybrid_mode:
            if init_merge_generator:
                misc.copy_params_and_buffers(src_module=root_generator, dst_module=G, require_all=True)
            if init_merge_discriminator:
                misc.copy_params_and_buffers(src_module=root_discriminator, dst_module=D, require_all=True)
        else:
            source_hybrid_generators, source_hybrid_discriminators = get_models(model_paths=hybrid_model_paths,
                                                                                new_n_classes=training_set.label_dim,
                                                                                device=device)
            source_hybrid_generators = [root_generator] + source_hybrid_generators
            source_hybrid_discriminators = [root_discriminator] + source_hybrid_discriminators

            G = dnnlib.util.construct_class_by_name(source_hybrid_generators=source_hybrid_generators, 
                                                    **G_kwargs, 
                                                    **common_kwargs).train().requires_grad_(False).to(device)
            D = dnnlib.util.construct_class_by_name(source_hybrid_discriminators=source_hybrid_discriminators,
                                                    **D_kwargs, 
                                                    **common_kwargs).train().requires_grad_(False).to(device)

        if root_model_path is not None:
            # Conserve memory - there is no need for the root model after the initialization
            del root_generator
            del root_discriminator

    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    resume_data = None
    if (resume_pkl is not None):
        if rank == 0:
            print_log(f'Resuming from "{resume_pkl}"')
            if init_merge_generator:
                print_log("Discarding merge generator for resume pkl")
            if init_merge_discriminator:
                print_log("Discarding merge discriminator for resume pkl")

        # All this resuming was initially only on rank 0 device, but I think it should be
        # done on all the devices. Need to check it again
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if was_killed:
        assert resume_data is not None
        if rank == 0:
            print_log("The previous session was killed, resuming also training parameters")

        wandb_log_id = resume_data[WANDB_ID_CHECKPOINT_KEY]
        optimizers_dict = {
            "G": resume_data[G_OPTIMIZER_CHECKPOINT_KEY],
            "D": resume_data[D_OPTIMIZER_CHECKPOINT_KEY]
        }
        cur_nimg = resume_data[CUR_NIMG_CHECKPOINT_KEY]
        cur_tick = resume_data[CUR_TICK_CHECKPOINT_KEY]
    else:
        wandb_log_id = None
        optimizers_dict = None
        cur_nimg = 0
        cur_tick = 0

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print_log('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print_log(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print_log('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs, 
                                              merge_generators=merge_generators, merge_discriminators=merge_discriminators,
                                              fisher_coefficients=fisher_coefficients, ewc_reference_model_index=ewc_reference_model_index,
                                              generator_mapping_ewc_lambda=generator_mapping_ewc_lambda, generator_synthesis_ewc_lambda=generator_synthesis_ewc_lambda,
                                              discriminator_ewc_lambda=discriminator_ewc_lambda, num_classes=training_set.num_classes) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            if optimizers_dict is not None:
                opt.load_state_dict(optimizers_dict[name])

            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            if optimizers_dict is not None:
                opt.load_state_dict(optimizers_dict[name])

            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = (int(math.sqrt(num_samples)), int(math.sqrt(num_samples)))
    grid_z = None
    if rank == 0:
        grid_z = torch.randn([num_samples, G.z_dim], device=device).split(batch_gpu)
        for class_idx in range(training_set.num_classes):
            class_one_hot = F.one_hot(torch.tensor(class_idx), num_classes=training_set.num_classes).to(device)
            class_one_hot = class_one_hot.repeat(batch_gpu, 1)
            images = torch.cat([G_ema(z=z, c=class_one_hot, noise_mode='const').cpu() for z in grid_z]).numpy()
            save_image_grid(images, os.path.join(run_dir, SAMPLES_DIR, f'fakes_init_{class_idx}.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print_log('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    wandb_logger = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

        if log_wandb and not debug:
            wandb_logger = wandb.init(
                project="stylegan2-ada-pytorch",
                name=job_desc,
                id=wandb_log_id,
                resume="allow",
            )
            wandb_log_id = wandb.run.id

    # Train.
    if rank == 0:
        print("\n\n###############################################################################")
        print_log(f'Training from {cur_nimg // 1000} kimg till {total_kimg} kimg...')
        if "SLURM_JOB_ID" in os.environ.keys():
            print_log(f"SLURM job ID is: {os.environ['SLURM_JOB_ID']}")
        print("###############################################################################\n\n")
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            if not merge_mode:
                phase_real_img, phase_real_c = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1)
            else:
                phase_real_img, phase_real_c = sample_merge_generators(merge_generators=merge_generators, 
                                                                       batch_size=batch_size//num_gpus, 
                                                                       class_percentages=class_percentages, 
                                                                       device=device)

            phase_real_img = phase_real_img.split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            all_gen_c = torch.multinomial(torch.tensor(class_percentages), num_samples=len(phases) * batch_size, replacement=True)
            all_gen_c = [F.one_hot(c, num_classes=training_set.label_shape[0]) for c in all_gen_c]
            all_gen_c = torch.stack(all_gen_c).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

            # all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            # all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            # all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if (rank == 0) and (cur_tick % image_snapshot_ticks == 0):
            print_log(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print_log()
                print_log('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            for class_idx in range(training_set.num_classes):
                class_one_hot = F.one_hot(torch.tensor(class_idx), num_classes=training_set.num_classes).to(device)
                class_one_hot = class_one_hot.repeat(batch_gpu, 1)
                images = torch.cat([G_ema(z=z, c=class_one_hot, noise_mode='const').cpu() for z in grid_z]).numpy()

                curr_fake_grid = get_image_grid(images, drange=[-1,1], grid_size=grid_size)
                curr_fake_grid.save(os.path.join(run_dir, SAMPLES_DIR, f'sample_class_{class_idx}_{cur_nimg//1000:06d}.png'))
                if wandb_logger is not None:
                    wandb_logger.log({f"sample_class_{class_idx}": [wandb.Image(curr_fake_grid)]},step=int(cur_nimg / 1e3))

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            # Save models
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory

            # Save training parameters
            snapshot_data[WANDB_ID_CHECKPOINT_KEY] = wandb_log_id
            snapshot_data[G_OPTIMIZER_CHECKPOINT_KEY] = phases[0].opt.state_dict()
            snapshot_data[D_OPTIMIZER_CHECKPOINT_KEY] = phases[-1].opt.state_dict()
            snapshot_data[CUR_NIMG_CHECKPOINT_KEY] = cur_nimg
            snapshot_data[CUR_TICK_CHECKPOINT_KEY] = cur_tick

            snapshot_pkl = os.path.join(run_dir, CHECKPOINTS_DIR, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                # Save the last checkpoint
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

                # Delete the old checkpoints
                if save_last_k_checkpoints != -1:
                    checkpoints_list = sorted(
                        Path(os.path.join(run_dir, CHECKPOINTS_DIR)).iterdir(),
                        key=os.path.getctime,
                    )
                    if len(checkpoints_list) > save_last_k_checkpoints:
                        os.remove(checkpoints_list[0])

        # Evaluate metrics.
        metrics_updated = False
        if (evaluate_metrics_first_tick or cur_tick != 0) and (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print_log('Evaluating metrics...')
                metrics_updated = True
            for metric in metrics:
                # Compute the metric for each class
                average_metric = 0
                for class_idx, class_name in enumerate(training_set.class_names):
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=metrics_dataset_kwargs[class_name], num_gpus=num_gpus, rank=rank,
                        device=device, evaluating_class=class_idx, num_classes=training_set.num_classes)
                    average_metric += result_dict["results"][metric]
                    result_dict["results"] = {f"{k}_{class_name}": v for k, v in result_dict["results"].items()}
                    if rank == 0:
                        print_log(f"Tick {cur_tick}: Metric {metric} of class {class_idx}")
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)

                average_metric /= len(training_set.class_names)
                stats_metrics.update({f"average_{metric}": average_metric})

                if compute_metrics_all_data:
                    # Compute the metric on all the data
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
                        device=device, evaluating_class=class_idx, num_classes=training_set.num_classes , use_data_labels=True)
                    stats_metrics.update(result_dict.results)

        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()

        if rank == 0 and wandb_logger is not None:
            global_step = int(cur_nimg / 1e3)
            wandb.log({k: v.mean for k, v in stats_dict.items()} ,step=global_step)
            if metrics_updated:
                wandb.log({f"Metrics/{k}": v for k, v in stats_metrics.items()} ,step=global_step)

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print_log('\nExiting...')

#----------------------------------------------------------------------------
