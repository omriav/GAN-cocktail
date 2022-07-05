from typing import List
from constants import FISHER_DISCRIMINATOR_KEY, FISHER_GENERATOR_KEY
import torch

from training.networks import Generator, Discriminator
from torch_utils import misc

class HybridGenerator(Generator):
    def __init__(self,
                 source_hybrid_generators: List[Generator],
                 z_dim,
                 c_dim,
                 w_dim,
                 img_resolution,
                 img_channels,
                 mapping_kwargs      = {},
                 synthesis_kwargs    = {}) -> None:
        super().__init__(z_dim = z_dim,
                         c_dim = c_dim,
                         w_dim = w_dim,
                         img_resolution = img_resolution,
                         img_channels = img_channels,
                         mapping_kwargs = mapping_kwargs,
                         synthesis_kwargs = synthesis_kwargs)

        generators_parameters = []
        for generator in source_hybrid_generators:
            generators_parameters.append(misc.named_params_and_buffers_dict(generator))

        self.requires_grad_(False)
        for name, tensor in misc.named_params_and_buffers_dict(self).items():
            new_tensor = 0
            for generator_params in generators_parameters:
                new_tensor += generator_params[name] / len(source_hybrid_generators)

            tensor.copy_(new_tensor)


class HybridDiscriminator(Discriminator):
    def __init__(self,
        source_hybrid_discriminators: List[Discriminator],
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ) -> None:
        super().__init__(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, architecture=architecture,
                         channel_base=channel_base, channel_max=channel_max, num_fp16_res=num_fp16_res, conv_clamp=conv_clamp,
                         cmap_dim=cmap_dim, block_kwargs=block_kwargs, mapping_kwargs=mapping_kwargs, epilogue_kwargs=epilogue_kwargs)

        discriminators_parameters = []
        for discriminator in source_hybrid_discriminators:
            discriminators_parameters.append(misc.named_params_and_buffers_dict(discriminator))

        self.requires_grad_(False)
        for name, tensor in misc.named_params_and_buffers_dict(self).items():
            new_tensor = 0
            for discriminator_params in discriminators_parameters:
                new_tensor += discriminator_params[name] / len(source_hybrid_discriminators)

            tensor.copy_(new_tensor)
