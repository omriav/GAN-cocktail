"""Calculated the diagonal of the Fisher Information matrix for the generator and the discriminator"""

from constants import FISHER_DISCRIMINATOR_KEY, FISHER_GENERATOR_KEY
import os
from typing import Dict

import click
import dnnlib
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import legacy

def _get_zero_parameters_dict(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    param_dict = {}

    for name, param in model.named_parameters():
        param_dict[name] = torch.zeros_like(param)

    return param_dict


def get_random_labels(batch, n_labels):
    labels = (torch.rand(batch) * n_labels).long()

    return labels


@torch.no_grad()
def _update_fisher_dict(
    model: torch.nn.Module, fisher_dict: Dict[str, torch.nn.Module], number_of_batches: int
):
    for name, param in model.named_parameters():
        fisher_dict[name] += (param.grad ** 2) / number_of_batches


def get_fisher_matrix_diag(G, D, n_actual_classes, n_samples, batch, device):
    generator_fisher = _get_zero_parameters_dict(G)
    discriminator_fisher = _get_zero_parameters_dict(D)

    G.requires_grad_(True)
    D.requires_grad_(True)

    number_of_batches = n_samples // batch
    for _ in tqdm(range(number_of_batches)):
        G.zero_grad()
        D.zero_grad()

        z = torch.randn(batch, G.z_dim).to(device)
        c = get_random_labels(batch=batch, n_labels=n_actual_classes)
        c = F.one_hot(c, num_classes=G.c_dim).to(device)

        gen_logits = D(img=G(z=z, c=c), c=c)
        loss = torch.nn.functional.softplus(-gen_logits).mean() # -log(sigmoid(gen_logits)) = NLL
        loss.backward()

        _update_fisher_dict(
            model=G, fisher_dict=generator_fisher, number_of_batches=number_of_batches
        )
        _update_fisher_dict(
            model=D,
            fisher_dict=discriminator_fisher,
            number_of_batches=number_of_batches,
        )

    return generator_fisher, discriminator_fisher

@click.command()
@click.pass_context
@click.option('--model_path', type=str, help='The path to the source GAN model', required=True)
@click.option('--n_actual_classes', type=int, default=1, help='The number of classes to calculate the Fisher information over',)
@click.option('--n_samples', type=int, default=50000, help='The number of samples to calculate the Fisher information with',)
@click.option('--batch', type=int, default=32, help='The batch size',)
@click.option('--output_path', help='Where to save the output images', type=str, required=True, metavar='DIR')
def save_fisher_coefficients(
    ctx: click.Context,
    model_path: str,
    n_actual_classes: int,
    n_samples: int,
    batch: int,
    output_path: str,
):
    assert not os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading GAN from {model_path}")
    device = torch.device('cuda')

    with dnnlib.util.open_url(model_path) as f:
        model_data = legacy.load_network_pkl(f)
        G = model_data['G_ema'].eval().to(device) # type: ignore
        D = model_data['D'].eval().to(device)  # type: ignore

    print("Calculating Fisher information matrix diag")
    generator_fisher, discriminator_fisher = get_fisher_matrix_diag(
        G=G, D=D, n_actual_classes=n_actual_classes, n_samples=n_samples, batch=batch, device=device
    )

    torch.save(
        {
            FISHER_GENERATOR_KEY: generator_fisher,
            FISHER_DISCRIMINATOR_KEY: discriminator_fisher,
        },
        output_path,
    )
    print("Done")


if __name__ == "__main__":
    save_fisher_coefficients()

