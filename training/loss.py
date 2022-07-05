# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from sys import prefix
from constants import FISHER_DISCRIMINATOR_KEY, FISHER_GENERATOR_KEY
from typing import Dict
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

def ewc_loss(
    train_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    fisher_coeff: Dict[str, torch.nn.Module],
    weight_fisher_prefix: str = ""
):
    loss = 0

    reference_model_paremeters = dict(reference_model.named_parameters())
    train_model_paremeters = dict(train_model.named_parameters())
    for name in train_model_paremeters.keys():
        # We do not want to enforce the same embedding as the old model
        full_name = weight_fisher_prefix + name
        if full_name not in ["mapping.embed.weight", 'mapping.embed.bias']:
            param_diff = train_model_paremeters[name] - reference_model_paremeters[name]
            param_diff_squared = param_diff ** 2
            loss += (param_diff_squared * fisher_coeff[full_name]).sum()

    return loss

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, 
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, merge_generators=None, merge_discriminators=None,
                 fisher_coefficients=None, ewc_reference_model_index=None, generator_mapping_ewc_lambda=None, generator_synthesis_ewc_lambda=None,
                 discriminator_ewc_lambda=None, num_classes=None):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.merge_generators = merge_generators
        self.merge_discriminators = merge_discriminators
        self.fisher_coefficients = fisher_coefficients
        self.ewc_reference_model_index = ewc_reference_model_index
        self.generator_mapping_ewc_lambda = generator_mapping_ewc_lambda
        self.generator_synthesis_ewc_lambda = generator_synthesis_ewc_lambda
        self.discriminator_ewc_lambda = discriminator_ewc_lambda
        self.num_classes = num_classes

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def report_loss_for_each_class(self, input_classes, logits, data_type):
        class_indices = torch.argmax(input_classes, dim=1)
        for c in range(self.num_classes):  # type: ignore
            curr_logits = logits[class_indices == c]
            training_stats.report(f'Loss/scores/{data_type}_class_{c}', curr_logits)
            training_stats.report(f'Loss/signs/{data_type}_class_{c}', curr_logits.sign())

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                self.report_loss_for_each_class(input_classes=gen_c, logits=gen_logits, data_type="fake")
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain = loss_Gmain.mean()

                if self.generator_mapping_ewc_lambda != 0:
                    loss_generator_ewc_mapping = ewc_loss(train_model=self.G_mapping, 
                                                          reference_model=self.merge_generators[self.ewc_reference_model_index].mapping,
                                                          fisher_coeff=self.fisher_coefficients[FISHER_GENERATOR_KEY],
                                                          weight_fisher_prefix = "mapping.")
                    loss_generator_ewc_mapping = self.generator_mapping_ewc_lambda * loss_generator_ewc_mapping
                    training_stats.report('Loss/G/mapping_ewc', loss_generator_ewc_mapping)
                    loss_Gmain = loss_Gmain + loss_generator_ewc_mapping

                if self.generator_synthesis_ewc_lambda != 0:
                    loss_generator_ewc_synthesis = ewc_loss(train_model=self.G_synthesis, 
                                                            reference_model=self.merge_generators[self.ewc_reference_model_index].synthesis,
                                                            fisher_coeff=self.fisher_coefficients[FISHER_GENERATOR_KEY],
                                                            weight_fisher_prefix = "synthesis.")

                    loss_generator_ewc_synthesis = self.generator_synthesis_ewc_lambda * loss_generator_ewc_synthesis
                    training_stats.report('Loss/G/synthesis_ewc', loss_generator_ewc_synthesis)
                    loss_Gmain = loss_Gmain + loss_generator_ewc_synthesis

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                self.report_loss_for_each_class(input_classes=gen_c, logits=gen_logits, data_type="fake")
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen = loss_Dgen.mean()

                if self.discriminator_ewc_lambda != 0:
                    loss_discriminator_ewc = ewc_loss(train_model=self.D, 
                                                      reference_model=self.merge_discriminators[self.ewc_reference_model_index],
                                                      fisher_coeff=self.fisher_coefficients[FISHER_DISCRIMINATOR_KEY])
                    loss_discriminator_ewc = self.discriminator_ewc_lambda * loss_discriminator_ewc
                    training_stats.report('Loss/G/loss_discriminator_ewc', loss_discriminator_ewc)
                    loss_Dgen = loss_Dgen + loss_discriminator_ewc

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                self.report_loss_for_each_class(input_classes=real_c, logits=real_logits, data_type="real")

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
