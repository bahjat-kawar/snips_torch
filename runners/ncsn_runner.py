import numpy as np
import glob
import tqdm

import torch.nn.functional as F
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from datasets import get_dataset, data_transform, inverse_data_transform
from models import general_anneal_Langevin_dynamics
from models import get_sigmas
from models.ema import EMAHelper
from filter_builder import get_custom_kernel

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample_general(self, score, samples, init_samples, sigma_0, sigmas, num_variations = 8, deg = 'sr4'):
        ## show stochastic variation ##
        stochastic_variations = torch.zeros((4 + num_variations) * self.config.sampling.batch_size, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

        clean = samples.view(samples.shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size)
        sample = inverse_data_transform(self.config, clean)
        stochastic_variations[0 : self.config.sampling.batch_size,:,:,:] = sample
        
        img_dim = self.config.data.image_size ** 2

        ## get degradation matrix ##
        H = 0
        if deg[:2] == 'cs':
            ## random with set singular values ##
            compress_by = int(deg[2:])
            Vt = torch.rand(img_dim, img_dim).to(self.config.device)
            Vt, _ = torch.qr(Vt, some=False)
            U = torch.rand(img_dim // compress_by, img_dim // compress_by).to(self.config.device)
            U, _ = torch.qr(U, some=False)
            S = torch.hstack((torch.eye(img_dim // compress_by), torch.zeros(img_dim // compress_by, (compress_by-1) * img_dim // compress_by))).to(self.config.device)
            H = torch.matmul(U, torch.matmul(S, Vt))
        elif deg == 'inp':
            ## crop ##
            H = torch.eye(img_dim).to(self.config.device)
            H = H[:-(self.config.data.image_size*20), :]
        elif deg == 'deblur_uni':
            ## blur ##
            H = torch.from_numpy(get_custom_kernel(type="uniform", dim = self.config.data.image_size)).type(torch.FloatTensor).to(self.config.device)
        elif deg == 'deblur_gauss':
            ## blur ##
            H = torch.from_numpy(get_custom_kernel(type="gauss", dim = self.config.data.image_size)).type(torch.FloatTensor).to(self.config.device)
        elif deg[:2] == 'sr':
            ## downscale - super resolution ##
            blur_by = int(deg[2:])
            H = torch.zeros((img_dim // (blur_by**2), img_dim)).to(self.config.device)
            for i in range(self.config.data.image_size // blur_by):
                for j in range(self.config.data.image_size // blur_by):
                    for i_inc in range(blur_by):
                        for j_inc in range(blur_by):
                            H[i * self.config.data.image_size // blur_by + j, (blur_by*i + i_inc) * self.config.data.image_size + (blur_by*j + j_inc)] = (1/blur_by**2)
        else:
            print("ERROR: degradation type not supported")
            quit()

        ## set up input for the problem ##
        y_0 = torch.matmul(H, samples.view(samples.shape[0] * self.config.data.channels,
                                      img_dim, 1)).view(samples.shape[0], self.config.data.channels, H.shape[0])
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
        torch.save(y_0, os.path.join(self.args.image_folder, "y_0.pt"))

        H_t = H.transpose(0,1)
        H_cross = torch.matmul(H_t, torch.inverse(torch.matmul(H, H_t)))
        pinv_y_0 = torch.matmul(H_cross, y_0.view(samples.shape[0] * self.config.data.channels,
                                      H.shape[0], 1))
        if deg == 'deblur_uni' or deg == 'deblur_gauss': pinv_y_0 = y_0
        sample = inverse_data_transform(self.config, pinv_y_0.view(samples.shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size))
        stochastic_variations[1 * self.config.sampling.batch_size : 2 * self.config.sampling.batch_size,:,:,:] = sample

        ## apply SNIPS ##
        for i in range(num_variations):
            all_samples = general_anneal_Langevin_dynamics(H, y_0, init_samples, score, sigmas,
                                           self.config.sampling.n_steps_each,
                                           self.config.sampling.step_lr, verbose=True,
                                           final_only=self.config.sampling.final_only,
                                           denoise=self.config.sampling.denoise, c_begin=0, sigma_0 = sigma_0)

            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size).to(self.config.device)
            stochastic_variations[(self.config.sampling.batch_size) * (i+2) : (self.config.sampling.batch_size) * (i+3),:,:,:] = inverse_data_transform(self.config, sample)

        ## calculate mean and std ##
        runs = stochastic_variations[(self.config.sampling.batch_size) * (2) : (self.config.sampling.batch_size) * (2+num_variations),:,:,:]
        runs = runs.view(-1, self.config.sampling.batch_size, self.config.data.channels,
                          self.config.data.image_size,
                          self.config.data.image_size)

        stochastic_variations[(self.config.sampling.batch_size) * (-2) : (self.config.sampling.batch_size) * (-1),:,:,:] = torch.mean(runs, dim=0)
        stochastic_variations[(self.config.sampling.batch_size) * (-1) : ,:,:,:] = torch.std(runs, dim=0)

        torch.save(stochastic_variations, os.path.join(self.args.image_folder, "results.pt"))

        image_grid = make_grid(stochastic_variations, self.config.sampling.batch_size)
        save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))

        ## report PSNRs ##
        clean = stochastic_variations[0 * self.config.sampling.batch_size : 1 * self.config.sampling.batch_size,:,:,:]

        for i in range(num_variations):
            general = stochastic_variations[(2+i) * self.config.sampling.batch_size : (3+i) * self.config.sampling.batch_size,:,:,:]
            mse = torch.mean((general - clean) ** 2)
            instance_mse = ((general - clean) ** 2).view(general.shape[0], -1).mean(1)
            psnr = torch.mean(10 * torch.log10(1/instance_mse))
            print("MSE/PSNR of the general #%d: %f, %f" % (i, mse, psnr))

        mean = stochastic_variations[(2+num_variations) * self.config.sampling.batch_size : (3+num_variations) * self.config.sampling.batch_size,:,:,:]
        mse = torch.mean((mean - clean) ** 2)
        instance_mse = ((mean - clean) ** 2).view(mean.shape[0], -1).mean(1)
        psnr = torch.mean(10 * torch.log10(1/instance_mse))
        print("MSE/PSNR of the mean: %f, %f" % (mse, psnr))
    

    def sample(self):
        score, states = 0, 0
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        sigma_0 = self.args.sigma_0

        dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=4)

        score.eval()

        data_iter = iter(dataloader)
        samples, _ = next(data_iter)
        samples = samples.to(self.config.device)
        samples = data_transform(self.config, samples)
        init_samples = torch.rand_like(samples)

        self.sample_general(score, samples, init_samples, sigma_0, sigmas, num_variations=self.args.num_variations, deg=self.args.degradation)
