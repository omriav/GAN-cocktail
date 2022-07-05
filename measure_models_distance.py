"""Measure the distance between 2 given models"""

import os
import re
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy
from general_utils.tensor_utils import convert_tensor_prediction_to_numpy
from training.hybrid_model import HybridGenerator


def get_distance_between_2_models(model1, model2):
    distances = {}

    model1_paremeters = dict(model1.named_parameters())
    model2_paremeters = dict(model2.named_parameters())

    for name in model1_paremeters.keys():
        param_diff = model1_paremeters[name] - model2_paremeters[name]
        
        # L2
        distance = torch.sqrt((param_diff ** 2).sum())

        # # L1
        # distance = torch.abs(param_diff).sum()

        distances[name] = distance.item()

    return distances
        
def read_generator_discriminator(path, device):
    with dnnlib.util.open_url(path) as f:
        model = legacy.load_network_pkl(f)
        G_a = model['G_ema'].to(device)
        D = model['D'].to(device)

    return G_a, D

def print_distances(model_a, model_b, model_ab):
    dist_a_b = get_distance_between_2_models(model_a, model_b)
    dist_a_ab = get_distance_between_2_models(model_a, model_ab)

    for k in dist_a_b.keys():
        v1 = dist_a_b[k]
        v2 = dist_a_ab[k]
        print(f"{k} - v1: {v1}, v2: {v2}, div: {v1 / v2}")

    print(f"a_b: {np.array([v for v in dist_a_b.values()]).mean()}")
    print(f"a_ab: {np.array([v for v in dist_a_ab.values()]).mean()}")


@click.command()
@click.pass_context
@click.option('--model_a', help='Network pickle filename', required=True)
@click.option('--model_b', help='Network pickle filename', required=True)
@click.option('--model_ab', help='Network pickle filename', required=True)
def measure_models_distance(
    ctx: click.Context,
    model_a: str,
    model_b: str,
    model_ab: str,
):
    device = torch.device('cuda')
    G_a, D_a = read_generator_discriminator(model_a, device)
    G_b, D_b = read_generator_discriminator(model_b, device)
    G_ab, D_ab = read_generator_discriminator(model_ab, device)

    print("Generator")
    print_distances(G_a, G_b, G_ab)
    # print("\nDiscriminator")
    # print_distances(D_a, D_b, D_ab)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    measure_models_distance() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
