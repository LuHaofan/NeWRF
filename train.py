# %%
import torch
from torch import nn
from pipelines import pipeline
import numpy as np
from utils import *
from tqdm import trange
from loaders import DatasetLoader
import os, random, string
from omegaconf import OmegaConf
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

# %%
# Hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.manual_seed(0)
np.random.seed(0)

# %%
def main(cfg, dataset_fname, file_dir, ckpt = None):
    ###################### Load Datasets #######################
    loader = DatasetLoader(dataset_fname)
    loader.split_train_val(ratio = cfg.training.train_split)
    ############################################################
    
    # Initialize models
    (model, 
     fine_model, 
     encode, 
     encode_viewdirs,
     optimizer, 
     scheduler,
     synthesizer
    ) = init_models(cfg, device, ckpt)

    # Loss function: NMSE loss
    loss_fn = lambda pred, true: torch.sum(torch.abs(pred-true)**2)/torch.sum(torch.abs(true)**2) 
    snr = lambda loss: -10. * np.log10(loss)

    for i in trange(cfg.training.n_iters, desc = 'Training'):
        # Run the training pipeline
        try: 
            sta_id = np.random.choice(loader.trainset, cfg.training.batch_size)
            (
                total_loss_coarse, 
                total_loss_fine,
                _, 
                _, 
                _, 
                _, 
            ) = pipeline(cfg,
                        sta_id,
                        loader,
                        model, 
                        fine_model, 
                        encode, 
                        encode_viewdirs, 
                        optimizer,
                        loss_fn,
                        synthesizer,
                        device,
                        mode = 'Train')

            # train_snrs.append(snr(total_loss_fine))
            summary_writer.add_scalar('Train SNR/Coarse', snr(total_loss_coarse), i)
            summary_writer.add_scalar('Train SNR/Fine', snr(total_loss_fine), i)
    #         Save a checkpoint at given rate
            if i % cfg.training.save_rate == 0 or i == cfg.training.n_iters-1:
                save_path = os.path.join(file_dir, "ckpt.pt") if cfg.training.overwrite else os.path.join(file_dir, f'ckpt_iter_{i}.pt')
                save_ckpt(model, fine_model, optimizer, save_path)

            # Evaluate at given display rate.
            if i % cfg.training.display_rate == 0:
                with torch.no_grad():
                    sta_id = loader.valset
                    (
                        total_loss_coarse, 
                        total_loss_fine,
                        cfr_pred_coarse, 
                        cfr_pred_fine, 
                        total_weights_coarse, 
                        total_weights_fine
                    ) = pipeline(cfg,
                                sta_id,
                                loader,
                                model, 
                                fine_model, 
                                encode, 
                                encode_viewdirs, 
                                optimizer,
                                loss_fn,
                                synthesizer,
                                device,
                                mode = 'Eval')
                    summary_writer.add_scalar('Val SNR/Coarse', snr(total_loss_coarse), i)
                    summary_writer.add_scalar('Val SNR/Fine', snr(total_loss_fine), i)
                    if scheduler is not None:
                        scheduler.step(total_loss_fine+total_loss_coarse)
                    del cfr_pred_coarse, cfr_pred_fine, total_weights_coarse, total_weights_fine, total_loss_coarse, total_loss_fine

        except Exception: 
            save_path = os.path.join(file_dir, "ckpt.pt") if cfg.training.overwrite else os.path.join(file_dir, f'ckpt_iter_{i}.pt')
            save_ckpt(model, fine_model, optimizer, save_path)  
            raise Exception

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, choices=['conference', 'bedroom', 'office'], default='conference')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--retrain', type=bool, default=False)
    args = parser.parse_args()
    cfg = OmegaConf.load('./config/default.yaml')
    if args.env == "office":
        cfg.sampling.n_samples = 256+128
        cfg.sampling.n_samples_hierarchical = 128
        cfg.sampling.far = 24
        cfg.training.n_iters = 300000

    datadir = "./data/"
    dataset_fname = os.path.join(datadir, f"dataset_{args.env}_ch1_rt_image_fc.pkl")
    if not os.path.exists(cfg.training.save_dir):
        os.mkdir(cfg.training.save_dir)
    
    file_dir = os.path.join(cfg.training.save_dir, ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
    os.mkdir(file_dir)
    print("Checkpoint will be saved at: ", file_dir)
    summary_writer = SummaryWriter(file_dir)

    if args.retrain:
        try:
            ckpt_fname = args.ckpt
        except:
            raise "Please specify the checkpoint file name to retrain"
    else:
        ckpt_fname = None

    main(cfg, dataset_fname = dataset_fname, file_dir = file_dir, ckpt=ckpt_fname)
