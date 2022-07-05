import copy
import torch
import torch.nn.functional as F
from . import metric_utils

class LPIPSDistanceSampler(torch.nn.Module):
    def __init__(self, G, G_kwargs, crop, vgg16):
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_kwargs = G_kwargs
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

    def forward(self, c):
        # Randomize noise buffers.
        for name, buf in self.G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))

        # Generate images.
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.z_dim], device=c.device).chunk(2)
        w0, w1 = self.G.mapping(z=torch.cat([z0,z1]), c=torch.cat([c,c])).chunk(2)
        img = self.G.synthesis(ws=torch.cat([w0,w1]), noise_mode='const', force_fp32=True, **self.G_kwargs)

        # Center crop.
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c*3 : c*7, c*2 : c*6]

        # Downsample to 256x256.
        factor = self.G.img_resolution // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

        # Scale dynamic range from [-1,1] to [0,255].
        img = (img + 1) * (255 / 2)
        img = torch.clip(img, 0, 255)
        if self.G.img_channels == 1:
            img = img.repeat([1, 3, 1, 1])

        # Evaluate differential LPIPS.
        lpips_1, lpips_2 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_1 - lpips_2).square().sum()
        return dist

#----------------------------------------------------------------------------

def compute_lpips_distance(opts, num_samples, crop, batch_size, jit=False):
    vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, num_gpus=opts.num_gpus, rank=opts.rank, verbose=opts.progress.verbose)

    # Setup sampler.
    sampler = LPIPSDistanceSampler(G=opts.G, G_kwargs=opts.G_kwargs, crop=crop, vgg16=vgg16)
    sampler.eval().requires_grad_(False).to(opts.device)

    # Sampling loop.
    dist = 0
    progress = opts.progress.sub(tag='LPIPS distance sampling', num_items=num_samples)
    for batch_start in range(0, num_samples):
        progress.update(batch_start)
        c = F.one_hot(torch.tensor(opts.evaluating_class), num_classes=opts.num_classes).pin_memory().to(opts.device)
        c = c.repeat(batch_size, 1)

        dist += sampler(c).cpu().numpy()
    progress.update(num_samples)

    dist = dist / num_samples
    return float(dist)

#----------------------------------------------------------------------------
