# %%
import torch
from torch import nn
import matplotlib.pyplot as plt
from pipelines import pipeline
import numpy as np
from utils import *
from tqdm import trange
from loaders import DatasetLoader
import os, random, string
from omegaconf import OmegaConf

# %%
# Hardware
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
def main(n_iters, cfg, dataset_fname, file_dir, ckpt = None):
    # Set seed for reproducability
    ###################### Load Datasets #######################
    logging.info("Loading dataset....")
    dsLoader = DatasetLoader(dataset_fname)
    dsLoader.split_train_val(ratio = 0.8)
    ############################################################
    
    # Initialize models
    (model, 
     fine_model, 
     encode, 
     encode_viewdirs,
     optimizer, 
     scheduler,
     synthesizer,
     _
    ) = init_models(cfg, device, ckpt)

    # Loss function
    loss_fn = lambda pred, true: torch.sum(torch.abs(pred-true)**2)/torch.sum(torch.abs(true)**2) 
    # Get the list of all frequency channels
    fc_lst = dsLoader.get_channel_freqs(cfg.training.n_chs)
    
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
    
    # Training iterations
    train_psnrs = []
    val_coarse_psnrs = []
    val_fine_psnrs = []
    iternum = []
    for i in trange(n_iters):
        logging.debug(f"Iteration: {i}")
        # Run the training pipeline
        try: 
            sta_id = np.random.choice(dsLoader.trainset, cfg.training.batch_size)
            logging.info(f"sample batch: {sta_id}")
            (
                total_loss_coarse, 
                total_loss_fine,
                _, 
                _, 
                _, 
                _, 
            ) = pipeline(cfg,
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
                        mode = 'Train')

            train_psnrs.append(-10. * np.log10(total_loss_fine))
    #         Save a checkpoint at given rate
            if i % cfg.training.save_rate == 0 or i == n_iters-1:
                with torch.no_grad():
                    save_ckpt(i, model.state_dict(), fine_model.state_dict(), optimizer.state_dict(), file_dir)

            # Evaluate at given display rate.
            if i % cfg.training.display_rate == 0:
                with torch.no_grad():
                    sta_id = dsLoader.valset
                    (
                        total_loss_coarse, 
                        total_loss_fine,
                        cfr_pred_coarse, 
                        cfr_pred_fine, 
                        total_weights_coarse, 
                        total_weights_fine
                    ) = pipeline(cfg,
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
                                mode = 'Eval')
                    val_coarse_psnrs.append(-10 * np.log10(total_loss_coarse))
                    val_fine_psnrs.append(-10. * np.log10(total_loss_fine))
                    iternum.append(i)
                    if scheduler is not None:
                        scheduler.step(total_loss_fine+total_loss_coarse)
                    del cfr_pred_coarse, cfr_pred_fine, total_weights_coarse, total_weights_fine, total_loss_coarse, total_loss_fine

        except Exception: 
            save_ckpt(i, model.state_dict(), fine_model.state_dict(), optimizer.state_dict(), file_dir)  
            raise Exception

# %%
######################## Configurations ###################
cfg = OmegaConf.load('./config/default.yaml')
ofdm_cfg = OmegaConf.load('./config/ofdm.yaml')
cfg = OmegaConf.merge(cfg, ofdm_cfg)
###########################################################
torch.manual_seed(0)
np.random.seed(0)
datadir = "./data/"
dataset_fname = os.path.join(datadir, "dataset_conference_ch1_rt_image_fc.pkl")
try:
    file_dir = os.path.join(cfg.training.save_dir, ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
    os.mkdir(file_dir)
except:
    file_dir = None
# Speficify the checkpoint file name to load if retraining
#ckpt_fname = "./ckpt/3405-bird-restful/ckpt_iter_9999.pt"
ckpt_fname = None
main(cfg.training.n_iters, cfg, dataset_fname = dataset_fname, file_dir = file_dir, ckpt=ckpt_fname)


