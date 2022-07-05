# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

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
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed0', type=int)
@click.option('--seed1', type=int)
@click.option('--w0_path', type=str)
@click.option('--w1_path', type=str)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--num_samples', type=int, default=10)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def interpolate_samples(
    ctx: click.Context,
    network_pkl: str,
    seed0: int,
    seed1: int,
    w0_path: str,
    w1_path: str,
    truncation_psi: float,
    num_samples: int,
    noise_mode: str,
    outdir: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    if w0_path is None:
        label0 = F.one_hot(torch.tensor(0), G.c_dim).unsqueeze(0).to(device)
        label1 = F.one_hot(torch.tensor(1), G.c_dim).unsqueeze(0).to(device)
        z0 = torch.from_numpy(np.random.RandomState(seed0).randn(1, G.z_dim)).to(device)
        z1 = torch.from_numpy(np.random.RandomState(seed1).randn(1, G.z_dim)).to(device)
        w0 = G.mapping(z0, label0)
        w1 = G.mapping(z1, label1)
    else:
        w0 = torch.tensor(np.load(w0_path)['w'], device=device)
        w1 = torch.tensor(np.load(w1_path)['w'], device=device)

    imgs = []
    for i in range(num_samples + 1):
        curr_percentage = 1 - (i / num_samples)
        curr_w = curr_percentage * w0 + (1 - curr_percentage) * w1
        # w = G.mapping.w_avg + (w - G.mapping.w_avg) * truncation_psi
        img = G.synthesis(curr_w, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        imgs.append(img)
        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/inter{i:02d}.png')
    PIL.Image.fromarray(np.hstack(imgs), 'RGB').save(f'{outdir}/final.png')

    layer_imgs = []
    w = w0.clone()
    for i in range(-1, 12):
        if i != -1:
            w[0, i] = w1[0, i]
        img = G.synthesis(w, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        layer_imgs.append(img)
        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/layer_inter{i:02d}.png')
    PIL.Image.fromarray(np.hstack(layer_imgs), 'RGB').save(f'{outdir}/final_layer.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    interpolate_samples() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
