# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

from constants import CHECKPOINTS_DIR, SAMPLES_DIR
import os
import shutil
import click
import re
from glob import glob
import json
import tempfile
import torch
import dnnlib
import numpy as np

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    compute_metrics_all_data = None, # Indicator to compute the metrics on all the data
    seed       = None, # Random seed: <int>, default = 0
    debug      = None, # Debug mode, default = False
    run_id     = None, # The run ID that will be used for creating the run_dir
    job_desc   = None, # The job description
    log_wandb  = None, # Indicator for Weights and Biases logging
    save_last_k_checkpoints = None,  # The number of last K checkpoints to save
    num_samples = None, # The number of samples to log
    evaluate_metrics_first_tick = None,  # Indicator to evaluate the metrics on the first tick

    # Architecture
    num_mapping_layers = None,  # The number of mapping layers

    # Dataset.
    data_paths = None, # Training dataset (required): <path>
    num_classes = None,  # The number of training classes
    class_percentages = None, # The percentages for each one of the classes, default = uniform
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Merge options
    merge_mode = None,         # Indicator for "merge_mode"
    merge_model_paths = None,  # The path to the pickles of the source merge models
    root_model_path = None,  # The path to the root model, it not specified, will use the first merge_generator
    init_merge_generator = None,  # Indicator to use the first merge generator as initializer
    init_merge_discriminator = None, # Indicator to use the first merge discriminator as initializer
    hybrid_mode = None, # Indicator for "hybrid mode" - use a model with averaging the weights
    hybrid_model_paths = None,  # The source pickles of the second fine-tuned models (from the first merge generator)

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma      = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # EWC
    model_fisher_pkl = None,    # The path to the Fisher information coefficients
    ewc_reference_model_index = None, # The index of the reference model for the EWC loss
    generator_mapping_ewc_lambda = None,    # The lambda of the EWC for the generator mapping weights
    generator_synthesis_ewc_lambda = None,    # The lambda of the EWC for the generator synthesis weights
    discriminator_ewc_lambda = None,    # The lambda for the EWC for the discriminator weights

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if debug is None:
        debug = False
    args.debug = debug

    if run_id < 0:
        raise UserError("--run_id must be greater than 0")

    if args.debug:
        run_id = 0
    args.run_id = run_id

    args.log_wandb = log_wandb
    args.job_desc = f"{args.run_id}_{job_desc}"
    args.save_last_k_checkpoints = save_last_k_checkpoints
    args.num_samples = num_samples
    args.evaluate_metrics_first_tick = evaluate_metrics_first_tick

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 30
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics
    args.compute_metrics_all_data = compute_metrics_all_data

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.MergedImageFolderDataset', paths=data_paths, num_classes=num_classes, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    args.metrics_dataset_kwargs = {}
    for path in data_paths:
        curr_dataset_dict = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=path, use_labels=True, max_size=None, xflip=False)
        try:
            training_set = dnnlib.util.construct_class_by_name(**curr_dataset_dict) # subclass of training.dataset.Dataset
            curr_dataset_dict.resolution = training_set.resolution # be explicit about resolution
            curr_dataset_dict.use_labels = training_set.has_labels # be explicit about labels
            curr_dataset_dict.max_size = len(training_set) # be explicit about dataset size

            args.metrics_dataset_kwargs[training_set.name] = curr_dataset_dict

            del training_set # conserve memory
        except IOError as err:
            raise UserError(f'--data: {err}')
    
    if len(class_percentages) == 0 or not merge_mode:
        actual_classes = len(data_paths)
        class_percentages = list(np.full(shape=actual_classes, fill_value=1/actual_classes))

    args.class_percentages = class_percentages

    assert len(data_paths) == len(class_percentages)
    # assert sum(args.class_percentages) == 1

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Merge Mode
    # ------------------------------------
    args.merge_mode = merge_mode
    if args.merge_mode:
        assert len(merge_model_paths) == len(data_paths), "Must provide the data for each model"
    args.merge_model_paths = merge_model_paths
    args.root_model_path = root_model_path
    args.init_merge_generator = init_merge_generator
    args.init_merge_discriminator = init_merge_discriminator
    
    if hybrid_mode:
        assert hybrid_model_paths is not None, "Must provide a hybrid_model_paths"
        if root_model_path is None:
            # We assume that if there is no alternative root path, we would like to use all the "aligned models"
            assert len(hybrid_model_paths) == (len(merge_model_paths) - 1), "Wrong number of hybrid model paths"
    args.hybrid_mode = hybrid_mode
    args.hybrid_model_paths = hybrid_model_paths

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'union_gan'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        'union_gan': dict(ref_gpus=1,  kimg=5000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator' if not hybrid_mode else "training.hybrid_model.HybridGenerator", 
                                    z_dim=512, 
                                    w_dim=512, 
                                    mapping_kwargs=dnnlib.EasyDict(), 
                                    synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator' if not hybrid_mode else "training.hybrid_model.HybridDiscriminator", 
                                    block_kwargs=dnnlib.EasyDict(), 
                                    mapping_kwargs=dnnlib.EasyDict(), 
                                    epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = num_mapping_layers
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'noaug'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Elastic weights consolidation
    # -------------------------------------------------
    if (generator_mapping_ewc_lambda != 0) or (generator_synthesis_ewc_lambda != 0) or (discriminator_ewc_lambda != 0):
        assert model_fisher_pkl is not None, "Must provide fisher pickle"

    args.model_fisher_pkl = model_fisher_pkl
    args.ewc_reference_model_index = ewc_reference_model_index
    args.generator_mapping_ewc_lambda = generator_mapping_ewc_lambda
    args.generator_synthesis_ewc_lambda = generator_synthesis_ewc_lambda
    args.discriminator_ewc_lambda = discriminator_ewc_lambda

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not args.debug and not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', default="output/train", metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 30 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--compute_metrics_all_data', help='Computing the metric on all the data', type=bool, default=True)
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
@click.option('--run_id', help='The run ID', type=int, metavar='INT')
@click.option('--job_desc', help='The job description', type=str, default="")
@click.option('--debug', help='Indicator for debug-develop mode', type=bool, metavar='BOOL')
@click.option('--log_wandb', help='Indicator for logging to Weights and Biases', type=bool, default=True, metavar='BOOL')
@click.option('--save_last_k_checkpoints', help='The number of checkpoints to save', type=int, metavar='INT', default=1)
@click.option('--num_samples', help='The number image samples to log', type=int, metavar='INT', default=64)
@click.option('--evaluate_metrics_first_tick', help='Indicator to evaluate all the metrics on the first tick', type=bool, default=False, metavar='BOOL')

# Architecture
@click.option('--num_mapping_layers', help='Number of mapping layers', type=int, default=8)

# Dataset.
@click.option('--data_paths', help='Training data (directory or zip)', multiple=True, required=True)
@click.option('--num_classes', help='The number image samples to log', type=int, metavar='INT', default=2)
@click.option('--class_percentages', help='The percentages for multinomial sampling', type=float, multiple=True)
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Merge options
@click.option('--merge_mode', help='Indicator for merge mode', type=bool, default=True, metavar='BOOL')
@click.option('--merge_model_paths', help='The paths to the checkpoints of the merge models', multiple=True)
@click.option('--root_model_path', type=str, help='The path to the root model, it not specified, will use the first merge_generator')
@click.option('--init_merge_generator', help='Indicator for merge mode', type=bool, default=True, metavar='BOOL')
@click.option('--init_merge_discriminator', help='Indicator for merge mode', type=bool, default=True, metavar='BOOL')
@click.option('--hybrid_mode', help='Indicator for hybrid mode', type=bool, default=False, metavar='BOOL')
@click.option('--hybrid_model_paths', help='The paths to the hybrid models', multiple=True)

# Base config.
@click.option('--cfg', help='Base config [default: union_gan]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar', 'union_gan']))
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: noaug]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# EWC
@click.option('--model_fisher_pkl', type=str, help='The path to the Fisher information coefficients')
@click.option('--ewc_reference_model_index', type=int, help='The index of the reference model for the ewc', default=0)
@click.option('--generator_mapping_ewc_lambda', type=float, default=0, help='The lambda for the EWC for the generator mapping weights')
@click.option('--generator_synthesis_ewc_lambda', type=float, default=0, help='The lambda for the EWC for the generator synthesis weights')
@click.option('--discriminator_ewc_lambda', type=float, default=0, help='The lambda for the EWC for the discriminator weights')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')
def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    args = None
    try:
        _, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    args.run_dir = os.path.join(outdir, f'{args.run_id:05d}')
    del args.run_id  # type: ignore
    if args.debug and os.path.isdir(args.run_dir):
        print(f"Removing train directory: {args.run_dir}")
        shutil.rmtree(args.run_dir)

    run_checkpoints_path = os.path.join(args.run_dir, CHECKPOINTS_DIR)
    if os.path.exists(run_checkpoints_path):
        checkpoints_list = glob(os.path.join(run_checkpoints_path, "*"))
        if len(checkpoints_list) > 0:
            print("The previous job was killed")
            last_checkpoint = max(checkpoints_list, key=os.path.getctime)

            args.resume_pkl = last_checkpoint
            args.was_killed = True
    else:
        print("This is a new job, the previous job wasn't killed")
        args.was_killed = False

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.paths}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    if not args.was_killed:
        print('Creating output directory...')
        os.makedirs(args.run_dir)
        os.makedirs(os.path.join(args.run_dir, CHECKPOINTS_DIR))
        os.makedirs(os.path.join(args.run_dir, SAMPLES_DIR))
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')  # type: ignore
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus) # type: ignore

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
