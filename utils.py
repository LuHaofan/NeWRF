import torch
from typing import Optional, Tuple, List, Union, Callable
import logging
from samplers import sample_stratified, sample_hierarchical
import os
from models import * 
from synthesizers import Synthesizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from encoders import PositionalEncoder

def get_chunks(
    inputs: torch.Tensor,
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(
    points: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


def prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify viewdirs to prepare for NeRF model.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def sample_and_prepare_batches(
    rays_o: torch.Tensor, # [n_theta, n_phi, 3]
    rays_d: torch.Tensor, # [n_theta, n_phi, 3]
    near: float,
    far: float,
    sampling_method: str,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    viewdirs_encoding_fn: Optional[Callable[[
        torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**15,
    kwargs: dict = None,
):
    # Sample query points along each ray.
    logging.info("Sampling query points along each ray...")
    if sampling_method == "Stratified":
        query_points, z_vals = sample_stratified(
            rays_o, rays_d, near, far, **kwargs)
    elif sampling_method == "Hierarchical":
        query_points, z_vals, rays_o, rays_d = sample_hierarchical(
            rays_o, rays_d, **kwargs)
    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                    viewdirs_encoding_fn,
                                                    chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    return batches, batches_viewdirs, query_points, z_vals

def create_model(d_input: int = 3,  
                n_layers: int = 8,
                d_filter: int = 256,
                skip: Tuple[int] = (4,),
                d_viewdirs: Optional[int] = None):
    return NeWRF(
                    d_input,
                    n_layers,
                    d_filter,
                    skip,
                    d_viewdirs
                )

def init_models(cfg, device, ckpt = None):
    r"""
    Initialize models, encoders, and optimizer for NeRF training.
    """
    if ckpt is None:
        logging.info("Initializing models")
    else:
        logging.info(f"Loading models from checkpoint:{ckpt}")
        d = torch.load(ckpt)
        
    # Encoders
    encoder = PositionalEncoder(cfg.encoder.d_input, cfg.encoder.n_freqs, log_space=cfg.encoder.log_space)
    def encode(x): return encoder(x)

    # View direction encoders
    if cfg.encoder.use_viewdirs:
        encoder_viewdirs = PositionalEncoder(cfg.encoder.d_input, cfg.encoder.n_freqs_views,
                                             log_space=cfg.encoder.log_space)

        def encode_viewdirs(x): return encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None
    
    # Models
    model = create_model(encoder.d_output,
                        n_layers=cfg.models.n_layers,
                        d_filter=cfg.models.d_filter,
                        skip=cfg.models.skip,
                        d_viewdirs=d_viewdirs)
    if ckpt is not None:
        model.load_state_dict(d['coarse_model_state_dict'])

    model.to(device)
    model_params = list(model.parameters())
    if cfg.models.use_fine_model:
        fine_model = create_model(d_input=encoder.d_output,
                                n_layers=cfg.models.n_layers_fine,
                                d_filter=cfg.models.d_filter_fine,
                                skip=cfg.models.skip,
                                d_viewdirs=d_viewdirs)
        if ckpt is not None:
            fine_model.load_state_dict(d['fine_model_state_dict'])
        
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
        
    else:
        fine_model = model

    # Synthesizer
    # Set the synthesizer for the corresponding model
    synthesizer = Synthesizer()
    # Optimizer
    if cfg.optimizer.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(model_params, lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
    if ckpt is not None:
        optimizer.load_state_dict(d['optimizer_state_dict'])
    logging.debug(optimizer)
    if cfg.optimizer.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 
                                      patience = cfg.optimizer.scheduler_patience, 
                                      factor=cfg.optimizer.scheduler_factor, 
                                      min_lr = cfg.optimizer.min_lr, verbose = True)
    else:
        scheduler = None

    return model, fine_model, encode, encode_viewdirs, optimizer, scheduler, synthesizer

def save_ckpt(coarse_model, fine_model, optimizer, save_path):
    torch.save({
        'coarse_model_state_dict': coarse_model.state_dict(),
        'fine_model_state_dict':fine_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
