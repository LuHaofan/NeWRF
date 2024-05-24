
import logging
from torch import nn
import torch
import numpy as np
from ray_gen import RayGenerator
import numpy as np
from utils import sample_and_prepare_batches

def pipeline_single_freq_batch(
    model: nn.Module,
    f: torch.Tensor,
    batches: torch.Tensor,
    batches_viewdirs: torch.Tensor,
    query_points: torch.Tensor,
    z_vals: torch.Tensor,
    synthesis_fn, 
    n_rays_lst
):
    logging.debug("Run pipeline_single_freq ...")
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(model(batch, viewdirs=batch_viewdirs))

    raw = torch.cat(predictions, dim=0).reshape(list(query_points.shape[:2]) + [predictions[0].shape[-1]])
    del predictions
    # ray_batches = torch.cumsum(n_rays_lst, dim=0)
    return synthesis_fn.synthesize(raw, z_vals, f, n_rays_lst)


def pipeline(
        cfg,
        sta_id,
        dsLoader,
        model, 
        fine_model, 
        encode, 
        encode_viewdirs, 
        optimizer,
        fc_lst,
        loss_fn,
        synthesizer,
        kwargs_sample_stratified,
        kwargs_sample_hierarchical,
        device,
        mode = "Train",
    ):
    # Pick a station sample from the training set / validation set
    if mode == 'Train':
        logging.info("Model in training mode")
        model.train()
        
    elif mode == 'Eval':
        logging.info("Model in evaluation mode")
        model.eval()

    # Get ground truth CFR and Station Location
    sta_loc = torch.tensor(np.array(dsLoader.get_loc_batch("STA", sta_id, normalize = False)), 
                           dtype=torch.float32).to(device)   # [batch_size, 3]

    # Get Rays for backtracing
    aoa_lst = dsLoader.get_aoa_batch(sta_id)
    rays_os = []
    rays_ds = []
    n_rays_lst = []
    for i in range(len(aoa_lst)):
        ray_gen = RayGenerator(sta_loc[i], cfg, device, torch.tensor(aoa_lst[i], dtype=torch.float32))
        rays_o, rays_d = ray_gen.get_rays()
        rays_os.append(rays_o)
        rays_ds.append(rays_d)
        assert(rays_o.shape == rays_d.shape) # [n_rays, 3]
        n_rays_lst.append(rays_o.shape[0])

    rays_o = torch.cat(rays_os, dim = 0)
    rays_d = torch.cat(rays_ds, dim = 0)
    n_rays_lst = torch.tensor(n_rays_lst, dtype=torch.int16)

    channel_idx = 0
    fc = torch.tensor(fc_lst[channel_idx], dtype=torch.float32).to(device)
    # Sampling the rays
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
    
    del batches, batches_viewdirs, query_points
    
    # Calculate Loss and backprop!
    ap_id = 1
    target_cfr = torch.tensor(np.concatenate(dsLoader.get_cfr_batch(ap_id, sta_id, fc_lst), axis=0).flatten(),
                              dtype=torch.complex64).to(device)  # [batch_size, 1]
    
    coarse_loss = loss_fn(syn_cfr, target_cfr)
    fine_loss = loss_fn(syn_cfr_fine, target_cfr)
    
    if mode == "Train":
        total_loss = 0.1*coarse_loss+0.9*fine_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    del rays_o, rays_d

    return float(coarse_loss), float(fine_loss), syn_cfr, syn_cfr_fine, weights_coarse, weights_fine

