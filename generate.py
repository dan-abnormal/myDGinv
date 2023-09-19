# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import io
from torchvision.utils import make_grid, save_image
from torch_utils import distributed as dist
import classifier_lib

#----------------------------------------------------------------------------
# Proposed EDM-G++ sampler.

def edm_sampler(
    boosting, time_min, time_max, vpsde, dg_weight_1st_order, dg_weight_2nd_order, discriminator,
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, eps_scaler=1.0
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    ## Settings for boosting
    S_churn_manual = 4.
    S_noise_manual = 1.000
    period = 5
    period_weight = 2
    log_ratio = torch.tensor([np.inf] * latents.shape[0], device=latents.device)
    S_churn_vec = torch.tensor([S_churn] * latents.shape[0], device=latents.device)
    S_churn_max = torch.tensor([np.sqrt(2) - 1] * latents.shape[0], device=latents.device)
    S_noise_vec = torch.tensor([S_noise] * latents.shape[0], device=latents.device)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        S_churn_vec_ = S_churn_vec.clone()
        S_noise_vec_ = S_noise_vec.clone()

        if i % period == 0:
            if boosting:
                S_churn_vec_[log_ratio < 0.] = S_churn_manual
                S_noise_vec_[log_ratio < 0.] = S_noise_manual

        # Increase noise temporarily.
        # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        gamma_vec = torch.minimum(S_churn_vec_ / num_steps, S_churn_max) if S_min <= t_cur <= S_max else torch.zeros_like(S_churn_vec_)
        t_hat = net.round_sigma(t_cur + gamma_vec * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt()[:, None, None, None] * S_noise_vec_[:, None, None,None] * randn_like(x_cur)
        #x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

        # epsilon scaling
        pred_eps = (x_hat - denoised) / t_hat[:, None, None, None]
        #print(f'using scaler: "{eps_scaler}" at Euler step')
        pred_eps = pred_eps / eps_scaler
        denoised = x_hat - pred_eps * t_hat[:, None, None, None]

        d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
        ## DG correction
        if dg_weight_1st_order != 0.:
            discriminator_guidance, log_ratio = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_hat, t_hat, net.img_resolution, time_min, time_max, class_labels, log=True)
            if boosting:
                if i % period_weight == 0:
                    discriminator_guidance[log_ratio < 0.] *= 2.
            d_cur += dg_weight_1st_order * (discriminator_guidance / t_hat[:, None, None, None])
        x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)

            # epsilon scaling
            pred_eps = (x_next - denoised) / t_next
            #print(f'using scaler: "{eps_scaler}" at correction step')
            pred_eps = pred_eps / eps_scaler
            denoised = x_next - pred_eps * t_next

            d_prime = (x_next - denoised) / t_next
            ## DG correction
            if dg_weight_2nd_order != 0.:
                discriminator_guidance = classifier_lib.get_grad_log_ratio(discriminator, vpsde, x_next, t_next, net.img_resolution, time_min, time_max, class_labels, log=False)
                d_prime += dg_weight_2nd_order * (discriminator_guidance / t_next)
            x_next = x_hat + (t_next - t_hat)[:, None, None, None] * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',     help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--eps_scaler', 'eps_scaler',help='epsilon scaler',         metavar='FLOAT',                          type=float, default=1.0, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

#---------------------------------------------------------------------------- Options for Discriminator-Guidance
## Sampling configureation
@click.option('--do_seed',                 help='Applying manual seed or not', metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--seed',                    help='Seed number',                 metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--num_samples',             help='Num samples',                 metavar='INT',                       type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--save_type',               help='png or npz',                  metavar='png|npz',                   type=click.Choice(['png', 'npz']), default='npz')
@click.option('--device',                  help='Device', metavar='STR',                                            type=str, default='cuda:0')

## DG configuration
@click.option('--dg_weight_1st_order',     help='Weight of DG for 1st prediction',       metavar='FLOAT',           type=click.FloatRange(min=0), default=2., show_default=True)
@click.option('--dg_weight_2nd_order',     help='Weight of DG for 2nd prediction',       metavar='FLOAT',           type=click.FloatRange(min=0), default=0., show_default=True)
@click.option('--time_min',                help='Minimum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=0.01, show_default=True)
@click.option('--time_max',                help='Maximum time[0,1] to apply DG', metavar='FLOAT',                   type=click.FloatRange(min=0., max=1.), default=1.0, show_default=True)
@click.option('--boosting',                help='If true, dg scale up low log ratio samples', metavar='INT',        type=click.IntRange(min=0), default=0, show_default=True)

## Discriminator checkpoint
@click.option('--pretrained_classifier_ckpt',help='Path of ADM classifier(latent extractor)',  metavar='STR',       type=str, default='/checkpoints/ADM_classifier/32x32_classifier.pt', show_default=True)
@click.option('--discriminator_ckpt',      help='Path of discriminator',  metavar='STR',                            type=str, default='/checkpoints/discriminator/cifar_uncond/discriminator_60.pt', show_default=True)

## Discriminator architecture
@click.option('--cond',                    help='Is it conditional discriminator?', metavar='INT',                  type=click.IntRange(min=0, max=1), default=0, show_default=True)

def main(boosting, time_min, time_max, dg_weight_1st_order, dg_weight_2nd_order, cond, pretrained_classifier_ckpt, discriminator_ckpt, save_type, max_batch_size, eps_scaler, do_seed, seed, num_samples, seeds, network_pkl, outdir, class_idx, device, **sampler_kwargs):
    
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    '''
    ## Set seed
    if do_seed:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    '''

    ## Load pretrained score network.
    print(f'Loading network from "{network_pkl}"...')
    if network_pkl.endswith(".pkl"):
        with open(network_pkl, 'rb') as f:
            net = pickle.load(f)['ema'].to(device)
    elif network_pkl.endswith(".pt"):
        data1 = torch.load(network_pkl, map_location=torch.device('cpu'))
        net = data1['ema'].eval().to(device)
    #data1 = torch.load(ckpt_dir, map_location=torch.device('cpu'))
    #net = data1['ema'].eval().to(device)
    #with open(network_pkl, 'rb') as f:
    #    net = pickle.load(f)['ema'].to(device)

    ## Load discriminator
    discriminator = None
    if dg_weight_1st_order != 0 or dg_weight_2nd_order != 0:
        discriminator = classifier_lib.get_discriminator(pretrained_classifier_ckpt, discriminator_ckpt,
                                                     net.label_dim and cond, net.img_resolution, device, enable_grad=True)
    print(discriminator)
    vpsde = classifier_lib.vpsde()

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    ## Loop over batches.
    #num_batches = (num_samples // (batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    #print(f'Generating {num_samples} images to "{outdir}"...')
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    os.makedirs(outdir, exist_ok=True)
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        ## Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        ## Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        images = edm_sampler(boosting, time_min, time_max, vpsde, dg_weight_1st_order, dg_weight_2nd_order, discriminator, net, latents, class_labels, randn_like=rnd.randn_like, eps_scaler=eps_scaler, **sampler_kwargs)

        ## Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if save_type == "png":
            count = 0
            for image_np in images_np:
                image_path = os.path.join(outdir, f'{i*batch_size+count:06d}.png')
                count += 1
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        elif save_type == "npz":
            r = np.random.randint(1000000)
            with tf.io.gfile.GFile(os.path.join(outdir, f"samples_{r}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                if class_labels == None:
                    np.savez_compressed(io_buffer, samples=images_np)
                else:
                    np.savez_compressed(io_buffer, samples=images_np, label=class_labels.cpu().numpy())
                fout.write(io_buffer.getvalue())

            nrow = int(np.sqrt(images_np.shape[0]))
            image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)
            with tf.io.gfile.GFile(os.path.join(outdir, f"sample_{r}.png"), "wb") as fout:
                save_image(image_grid, fout)
    
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')



#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
