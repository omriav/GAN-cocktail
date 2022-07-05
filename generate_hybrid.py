# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

from general_utils.tensor_utils import convert_tensor_prediction_to_numpy
import os
import re
from training.hybrid_model import HybridGenerator
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network1', 'network1_pkl', help='Network pickle filename', required=True)
@click.option('--network2', 'network2_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_hybrid_images(
    ctx: click.Context,
    network1_pkl: str,
    network2_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
):
    print('Loading networks from "%s"...' % network1_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network1_pkl) as f:
        G1 = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    with dnnlib.util.open_url(network2_pkl) as f:
        G2 = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    hybrid_generator = HybridGenerator(source_hybrid_generators=[G1, G2],
                                       z_dim=G1.z_dim,
                                       c_dim=G1.c_dim,
                                       w_dim=G1.w_dim,
                                       img_resolution=G1.img_resolution,
                                       img_channels=G1.img_channels)
    hybrid_generator.to(device)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "source"), exist_ok=True)

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, hybrid_generator.c_dim], device=device)
    if hybrid_generator.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, hybrid_generator.z_dim)).to(device)
        hybrid_img = convert_tensor_prediction_to_numpy(
                        hybrid_generator(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode))[0]
        g_1_img = convert_tensor_prediction_to_numpy(
                        G1(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode))[0]
        g_2_img = convert_tensor_prediction_to_numpy(
                        G2(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode))[0]
        final_image = np.hstack([g_1_img, g_2_img, hybrid_img])
        PIL.Image.fromarray(final_image, 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        PIL.Image.fromarray(g_1_img, 'RGB').save(f'{outdir}/source/seed{seed:04d}_A.png')
        PIL.Image.fromarray(g_2_img, 'RGB').save(f'{outdir}/source/seed{seed:04d}_B.png')
        PIL.Image.fromarray(hybrid_img, 'RGB').save(f'{outdir}/source/seed{seed:04d}_C.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_hybrid_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
