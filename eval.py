# %%
import torch
from torch import nn
import matplotlib.pyplot as plt
# from pipelines import pipeline
import numpy as np
from utils import *
from tqdm import trange, tqdm
from loaders import DatasetLoader
import os, random, string
from omegaconf import OmegaConf
from utils import sample_and_prepare_batches
from pipelines import pipeline_single_freq_batch
from ray_gen import RayGenerator
from argparse import ArgumentParser
from ray_searching import ray_searching

torch.manual_seed(0)
np.random.seed(0)
# Hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def pipeline_eval(
    cfg,
    fc: torch.Tensor, 
    model: nn.Module,
    fine_model: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    encode,
    encode_viewdirs,
    synthesizer,
):
    model.eval()
    fine_model.eval()
    n_rays_lst = [rays_d.shape[0]]
    # Sampling configurations
    kwargs_sample_stratified = {
        'n_samples': cfg.sampling.n_samples,
        'perturb': cfg.sampling.perturb,
        'inverse_depth': cfg.sampling.inverse_depth
    }
    
    kwargs_sample_hierarchical = {
        "perturb": cfg.sampling.perturb_hierarchical,
        "n_new_samples": cfg.sampling.n_samples_hierarchical
    }
    
    (
        batches, 
        batches_viewdirs, 
        query_points, 
        z_vals_stratified
    )= sample_and_prepare_batches(rays_o, rays_d, cfg.sampling.near, cfg.sampling.far,
                                sampling_method="Stratified",
                                encoding_fn=encode,
                                viewdirs_encoding_fn=encode_viewdirs,
                                chunksize=cfg.training.chunksize,
                                kwargs=kwargs_sample_stratified)
    # Train coarse model
    syn_cfr, depth, amp, weights_coarse = pipeline_single_freq_batch(model,
                                fc,
                                batches,
                                batches_viewdirs,
                                query_points,
                                z_vals_stratified,
                                synthesizer,
                                n_rays_lst)
    
    # Sampling the rays with hierachical sampling
    kwargs_sample_hierarchical["z_vals"] = z_vals_stratified
    kwargs_sample_hierarchical["weights"] = weights_coarse

    # Create samples based on weights for fine model    
    (
        batches_fine,
        batches_viewdirs_fine,
        query_points_fine,
        z_vals_combined
    ) = sample_and_prepare_batches(rays_o, rays_d, cfg.sampling.near, cfg.sampling.far,
                                sampling_method="Hierarchical",
                                encoding_fn=encode,
                                viewdirs_encoding_fn=encode_viewdirs,
                                chunksize=cfg.training.chunksize,
                                kwargs=kwargs_sample_hierarchical)
    
    # Train coarse model
    syn_cfr_fine, depth_fine, amp_fine, weights_fine = pipeline_single_freq_batch(fine_model,
                                fc,
                                batches_fine,
                                batches_viewdirs_fine,
                                query_points_fine,
                                z_vals_combined,
                                synthesizer,
                                n_rays_lst)
    
    return syn_cfr, syn_cfr_fine




# %%
def main(cfg, dataset_fname, ckpt = None):
    # Set seed for reproducability
    ###################### Load Datasets #######################
    loader = DatasetLoader(dataset_fname)
    loader.split_train_val(ratio = cfg.training.train_split)
    ############################################################
    
    # Initialize models
    (model, 
     fine_model, 
     encode, 
     encode_viewdirs,
     _, 
     _,
     synthesizer
    ) = init_models(cfg, device, ckpt)

    # Loss function
    loss_fn = lambda pred, true: torch.sum(torch.abs(pred-true)**2)/torch.sum(torch.abs(true)**2) 
    snr = lambda loss: -10. * torch.log10(loss)

    fc = torch.tensor(loader.get_freq()).to(device)
    syn_cfr_lst = []
    syn_cfr_fine_lst = []
    target_cfr_lst = []
    for sta_id in tqdm(loader.valset):
        aoa = torch.tensor(loader.get_aoa(sta_id), dtype=torch.float32).to(device)
        sta_loc = torch.tensor(loader.get_loc("STA", sta_id), dtype=torch.float32).to(device)

        ray_gen = RayGenerator(sta_loc, cfg, device, aoa)
        rays_o, rays_d = ray_gen.get_rays()
        (syn_cfr, syn_cfr_fine) = pipeline_eval(
            cfg,
            fc,
            model,
            fine_model,
            rays_o,
            rays_d,
            encode,
            encode_viewdirs,
            synthesizer
        )
        target_cfr = torch.tensor(loader.get_cfr(1, sta_id)).flatten().to(device)
        syn_cfr_lst.append(syn_cfr)
        syn_cfr_fine_lst.append(syn_cfr_fine)
        target_cfr_lst.append(target_cfr)
        # print("STA ID: ", sta_id, "Coarse SNR: ", snr(loss_fn(syn_cfr, target_cfr)).item(), "Fine SNR: ", snr(loss_fn(syn_cfr_fine, target_cfr)).item())

    syn_cfr = torch.cat(syn_cfr_lst, dim=0)
    syn_cfr_fine = torch.cat(syn_cfr_fine_lst, dim=0)
    target_cfr = torch.cat(target_cfr_lst, dim=0)
    print("Total Coarse SNR: ", snr(loss_fn(syn_cfr, target_cfr)))
    print("Total Fine SNR: ", snr(loss_fn(syn_cfr_fine, target_cfr)))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, choices=['conference', 'bedroom', 'office'], default='conference')
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load('./config/default.yaml')
    if args.env == "office":
        cfg.sampling.n_samples = 256+128
        cfg.sampling.n_samples_hierarchical = 128
        cfg.sampling.far = 24
    
    if args.ckpt is None:
        raise ValueError("Please specify the checkpoint file to evaluate.")
    else:
        ckpt_fname = args.ckpt
        if not os.path.exists(ckpt_fname):
            raise ValueError("The specified checkpoint file does not exist.")

    datadir = "./data/" 
    dataset_fname = os.path.join(datadir, f"dataset_{args.env}_ch1_rt_image_fc_searched_doa.pkl")
    if not os.path.exists(dataset_fname):
        print("Fail to find dataset file in the specified path.\nRunning ray searching script to generate the dataset...")
        ray_searching(os.path.join(datadir, f"dataset_{args.env}_ch1_rt_image_fc.pkl"))

    main(cfg, dataset_fname, ckpt=ckpt_fname)

