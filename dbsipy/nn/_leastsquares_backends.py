import torch
import torch.nn as nn
from typing import Type, List, Tuple

from operator import add
from functools import reduce
from itertools import chain
import numpy as np 
import os
import time
import logging

from torch.nn.modules import Module
from dbsipy.core import utils
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model
from dbsipy.nn.tensor_parametizations import coorientedDiffusionTensorModel as coOSDTM
from dbsipy.nn.tensor_parametizations import isotropicDiffusionModel as IDM

from joblib import Parallel, delayed
from scipy import optimize

MIN_POSITIVE_SIGNAL = 1.0e-6

# Loss functions for regularization
  
class lfn:
    def __init__(self, loss_fn: str, diffusion_model_class, alpha = 1e-6) -> None:
        
        self.loss_fn = loss_fn
        self.F = diffusion_model_class
        self.alpha = alpha 
        pass

    def __call__(self, Y, Yhat) -> torch.FloatTensor:
        return {'mse': mse_loss,
                'ridge': ridge, 
                'lasso': lasso,
                'tv':    tv
                }[self.loss_fn](Y, Yhat, self)


def mse_loss(Y, Yhat, L: Type[lfn]):
    return torch.nn.functional.mse_loss(Yhat, Y)

def ridge(Y, Yhat, L: Type[lfn]):
    return mse_loss(Y, Yhat, L) + L.alpha * torch.abs(L.F.get_parameters()).sum()

def lasso(Y, Yhat, L: Type[lfn]):
    return mse_loss(Y, Yhat, L) + L.alpha * ((L.F.get_parameters())**2).sum()

def tv(Y, Yhat, L: Type[lfn]):
    return mse_loss(Y, Yhat, L) + L.alpha * torch.abs(L.F.tv_kernel()).sum()


# Custom nn layer for optimization

class signal_fraction(Module):
    __constants__ = ['output_dimension', 'feature_dimension']

    output_dimension  : int
    feature_dimension : int

    def __init__(self, output_dimension, feature_dimension, device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.output_dimension  = output_dimension
        self.feature_dimension = feature_dimension 

        self.weight = nn.Parameter(
                                   torch.empty(
                                               (output_dimension, feature_dimension), 
                                               **factory_kwargs
                                               )
                                    )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 1e-3, 1e-3 + MIN_POSITIVE_SIGNAL)


# Custom nn module for PGD-NNLSQ

class least_squares_optimizer(nn.Module):
    def __init__(self, A: torch.Tensor, 
                 non_linearity: nn.Module, 
                 output_dimension: int, 
                 device = 'cpu',
                 fractionPrior = None) -> None:
        super().__init__()
       
        self.A                = A
        self.ReLu             = nn.ReLU()
        self.output_dimension = output_dimension
        self.fractionPrior    = fractionPrior

        self.x   = signal_fraction(output_dimension, A.shape[-1], device= device) 
        return
    
    @property
    def v(self) -> torch.FloatTensor:
        v = self.x.weight.clamp(min = 0, max = None) # Relu w/out in-place modification  v
        return v 

    def forward(self):
        """ if   '   ': use same design matrix for each voxel (i.e. step 1)
            elif '   ': use tensor product network (i.e. step 2)      
        """
        # Handle empty batch case (0 voxels)
        if self.output_dimension == 0:
            return torch.zeros((0, self.A.shape[-2] if self.A.dim() == 3 else self.A.shape[0]), 
                             device=self.A.device, dtype=self.v.dtype)
        
        # Check actual batch size (self.v.shape[0]) to handle shape mismatches
        if self.v.shape[0] == 0:
            return torch.zeros((0, self.A.shape[-2] if self.A.dim() == 3 else self.A.shape[0]),
                             device=self.A.device, dtype=self.v.dtype)
        
        if self.A.shape[0] != self.output_dimension: 
            # Optimized: Use @ operator instead of einsum
            return self.v @ self.A.T  # (b, j) @ (j, i).T -> (b, i)
        elif self.A.shape[0] == self.output_dimension: # Step 2 of the computation   
            # Batched matrix-vector multiplication
            return torch.bmm(self.A, self.v.unsqueeze(-1)).squeeze(-1)  # (b, i, j) @ (b, j, 1) -> (b, i)
        
    def predict(self):
        with torch.no_grad():
            return self.forward()
    
    def get_parameters(self):
        return self.v 


# NN-backed version of DBSI / DBSI-IA
class DBSIModel:
    """
    Wrap least Squares Optimizer
    """
    def __init__(self, 
                 dwi:            torch.FloatTensor, 
                 bvals:          torch.FloatTensor, 
                 bvecs:          torch.FloatTensor, 
                 directions:     torch.FloatTensor, 
                 fractions:      torch.FloatTensor, 
                 restricted:     torch.FloatTensor,
                 non_restricted: torch.FloatTensor,
                 indicies:       List[int], 
                 s0_init:        torch.FloatTensor | None,
                 DBSI_CONFIG, 
                 logging_args:   Tuple[int]) -> None:
        

        device = DBSI_CONFIG.DEVICE 
        self.DEVICE = DBSI_CONFIG.DEVICE 

        self.dwi = dwi.to(    device)
        self.bvals = bvals.to(device)
        self.bvecs = bvecs.to(device)
        
        self.STEP_2_OPTIMIZER_ARGS = DBSI_CONFIG.STEP_2_OPTIMIZER_ARGS
        self.step_2_axials = DBSI_CONFIG.step_2_axials
        self.step_2_lambdas = DBSI_CONFIG.step_2_lambdas
        self.iso_basis = DBSI_CONFIG.iso_basis

        self.directions = directions
        self.model_map = (~(self.directions == 0).all(dim = 2)).sum(dim = 1)
        self.max_fiber_number = self.model_map.max().item()

        self._job_id = logging_args[0]
        self._n_jobs = logging_args[1]    
        self.pbar = logging_args[2]

        self.LINEAR_DIMS  = DBSI_CONFIG.linear_dims
        self.SPATIAL_DIMS = DBSI_CONFIG.spatial_dims
        self.MASK         = DBSI_CONFIG.mask
        self.idx          = indicies

        if DBSI_CONFIG.four_iso:
            self.highly_restricted_inds = DBSI_CONFIG.highly_restricted_inds
        self.restricted_inds = DBSI_CONFIG.restricted_inds
        self.hindered_inds = DBSI_CONFIG.hindered_inds
        self.water_inds = DBSI_CONFIG.water_inds
        
        self.configuration = DBSI_CONFIG

        self.learnable_s0 = bool(getattr(DBSI_CONFIG, 'learnable_s0', False))
        if self.learnable_s0:
            if s0_init is not None:
                s0_init = s0_init.to(device)
                s0_init = torch.clamp(s0_init, min=1.0)
                log_s0_init = torch.log(s0_init)
            else:
                log_s0_init = torch.zeros((self.dwi.shape[0],), device=device)
            self.log_s0 = torch.nn.Parameter(log_s0_init)

        self.anisotropicFractionPriors = fractions
        self.isotropicFractionPriors   = non_restricted + restricted

        self.anisotropic_models, self.isotropic_models = self._prepare_models()        
        pass 


    def _prepare_models(self) -> List[Type[least_squares_optimizer]]:

        anisotropic_models = []
        isotropic_models   = []
        
        for num_fibers in range(1, self.max_fiber_number + 1):
            N_voxels = self.dwi[self.model_map == num_fibers].shape[0]
          
            f_aniso = torch.zeros((N_voxels, self.bvals.shape[0], num_fibers*self.step_2_axials.shape[0]), device = self.DEVICE)
            f_iso   = torch.zeros((self.bvals.shape[0], self.iso_basis.shape[0]), device = self.DEVICE)

            O = utils.HouseHolder_evec_2_eframe(self.directions[self.model_map == num_fibers][:, 0:num_fibers, :]).to(self.DEVICE)
            # Optimized: Replace nested einsum with batched matrix operations
            lambdas_diag = torch.diag_embed(self.step_2_lambdas.to(self.DEVICE))  # (models, 3, 3)
            temp = torch.matmul(O.unsqueeze(2), lambdas_diag.unsqueeze(0).unsqueeze(0))  # (b, l, m, 3, 3)
            D = torch.matmul(temp, torch.transpose(O, 2, 3).unsqueeze(2)).reshape(N_voxels, num_fibers*self.step_2_axials.shape[0], 3, 3)

            # Optimized: Replace einsum with batched operations for anisotropic signal
            bvecs_expanded = self.bvecs.unsqueeze(0).unsqueeze(0)  # (1, 1, s, 3)
            Dg = torch.matmul(D.unsqueeze(2), bvecs_expanded.unsqueeze(-1)).squeeze(-1)  # (b, m, s, 3)
            gDg = torch.matmul(bvecs_expanded.unsqueeze(-2), Dg.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (b, m, s)
            f_aniso[:, :, 0:num_fibers*self.step_2_axials.shape[0]] = torch.exp(-self.bvals.unsqueeze(0).unsqueeze(-1) * gDg.transpose(1, 2))
            # Optimized: Use outer product for isotropic signal
            f_iso[:,:] = torch.exp(-torch.outer(self.bvals, self.iso_basis.to(self.DEVICE)))[None,:,:]

            anisotropic_models.append(
                                      least_squares_optimizer(f_aniso, 
                                                              non_linearity    = nn.ReLU, 
                                                              output_dimension = N_voxels, 
                                                              device           = self.DEVICE,
                                                              )
                                        )
            
            isotropic_models.append(
                                    least_squares_optimizer(
                                                            f_iso,
                                                            non_linearity = nn.ReLU, 
                                                            output_dimension = N_voxels, 
                                                            device= self.DEVICE,
                                                            )
                                    )
        # totally isotropic voxel
        num_fibers = 0
        N_voxels = self.dwi[self.model_map == num_fibers].shape[0]
        f_iso   = torch.zeros((self.bvals.shape[0], self.iso_basis.shape[0]), device = self.DEVICE)
        # Optimized: Use outer product instead of einsum
        f_iso[:,:] = torch.exp(-torch.outer(self.bvals, self.iso_basis.to(self.DEVICE)))[None,:,:]
        isotropic_models.append(
                                least_squares_optimizer(f_iso,   
                                                        non_linearity = nn.ReLU, 
                                                        output_dimension = N_voxels, 
                                                        device= self.DEVICE
                                                        )
                                )    

        return anisotropic_models, isotropic_models
        
    def forward(self) -> torch.FloatTensor:
        def _apply_A(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # A: (B, I, J) with per-voxel design OR (I, J) shared design
            if A.dim() == 2:
                return x @ A.T
            return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

        Yhat = torch.zeros(self.dwi.shape, device=self.DEVICE)

        if not self.learnable_s0:
            for num_fibers, (anisotropic_model, isotropic_model) in enumerate(zip(self.anisotropic_models, self.isotropic_models)):
                batch_dimension = Yhat[self.model_map == num_fibers + 1].shape[0]
                if batch_dimension > 0:
                    Yhat[self.model_map == num_fibers + 1] += anisotropic_model.forward() + isotropic_model.forward()

            num_fibers = -1
            batch_dimension = Yhat[self.model_map == num_fibers + 1].shape[0]
            if batch_dimension > 0:
                Yhat[self.model_map == num_fibers + 1] += self.isotropic_models[-1].forward()

            return Yhat

        # learnable_s0 mode:
        # Fit raw signal as: S(b) = S0 * (attenuation model), where fractions are normalized.
        eps = MIN_POSITIVE_SIGNAL
        for num_fibers, (anisotropic_model, isotropic_model) in enumerate(zip(self.anisotropic_models, self.isotropic_models[:-1])):
            sel = (self.model_map == num_fibers + 1)
            if not bool(sel.any()):
                continue

            x_aniso = anisotropic_model.get_parameters()
            x_iso = isotropic_model.get_parameters()
            N = torch.cat([x_aniso, x_iso], dim=1).sum(dim=1)
            N = torch.clamp(N, min=eps)
            x_aniso = x_aniso / N[:, None]
            x_iso = x_iso / N[:, None]

            attn_hat = _apply_A(anisotropic_model.A, x_aniso) + _apply_A(isotropic_model.A, x_iso)
            s0 = torch.exp(self.log_s0[sel])
            Yhat[sel] = attn_hat * s0[:, None]

        # Totally isotropic voxels (0-fiber model is last isotropic model)
        sel_iso = (self.model_map == 0)
        if bool(sel_iso.any()):
            iso_model = self.isotropic_models[-1]
            x_iso = iso_model.get_parameters()
            N = torch.clamp(x_iso.sum(dim=1), min=eps)
            x_iso = x_iso / N[:, None]
            attn_hat = _apply_A(iso_model.A, x_iso)
            s0 = torch.exp(self.log_s0[sel_iso])
            Yhat[sel_iso] = attn_hat * s0[:, None]

        return Yhat
    
    def fit(self):   
        params = list(chain.from_iterable([model.parameters() for model in (self.anisotropic_models + self.isotropic_models)]))
        if self.learnable_s0:
            params.append(self.log_s0)
        optimizer = self.STEP_2_OPTIMIZER_ARGS['optimizer'](
            params=params,
            lr=self.STEP_2_OPTIMIZER_ARGS['lr'],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr =  self.STEP_2_OPTIMIZER_ARGS['lr'] * 10, total_steps = self.STEP_2_OPTIMIZER_ARGS['epochs'])
        
        L = torch.nn.MSELoss()
        b = self.dwi.to(self.configuration.DEVICE)
        
        # Early stopping parameters (configurable; defaults preserve current behavior)
        best_loss = float('inf')
        patience_counter = 0
        patience = int(self.STEP_2_OPTIMIZER_ARGS.get('patience', 100))
        min_delta = float(self.STEP_2_OPTIMIZER_ARGS.get('min_delta', 0.0))

        profile = (os.environ.get('DBSIPY_PROFILE_STEP2', '0') == '1')
        profile_sync = (os.environ.get('DBSIPY_PROFILE_SYNC', '0') == '1')
        do_sync = bool(profile and profile_sync and torch.cuda.is_available() and ('cuda' in str(self.DEVICE).lower()))
        if profile:
            prof = {
                'total_s': 0.0,
                'zero_grad_s': 0.0,
                'forward_s': 0.0,
                'loss_s': 0.0,
                'backward_s': 0.0,
                'opt_step_s': 0.0,
                'sched_step_s': 0.0,
                'pbar_update_s': 0.0,
            }
            t_total0 = time.perf_counter()

        # Calculate adaptive intervals for progress bar and early stopping checks
        pbar_update_interval = max(1, self.STEP_2_OPTIMIZER_ARGS['epochs'] // 100)
        early_stop_check_interval = max(1, self.STEP_2_OPTIMIZER_ARGS['epochs'] // 100)

        epochs_ran = 0
        for epoch in range(self.STEP_2_OPTIMIZER_ARGS['epochs']):

            epochs_ran = epoch + 1

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                prof['zero_grad_s'] += (time.perf_counter() - t0)
                t0 = time.perf_counter()

            Yhat = self.forward()

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                prof['forward_s'] += (time.perf_counter() - t0)
                t0 = time.perf_counter()

            loss = L(Yhat, b)

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                prof['loss_s'] += (time.perf_counter() - t0)
                t0 = time.perf_counter()

            loss.backward()

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                prof['backward_s'] += (time.perf_counter() - t0)
                t0 = time.perf_counter()

            optimizer.step()

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                prof['opt_step_s'] += (time.perf_counter() - t0)
                t0 = time.perf_counter()

            scheduler.step()

            if profile:
                if do_sync:
                    torch.cuda.synchronize()
                prof['sched_step_s'] += (time.perf_counter() - t0)

            # Batch progress bar updates (every 10 epochs or at end)
            if self.pbar is not None:
                if (epoch + 1) % pbar_update_interval == 0 or epoch == self.STEP_2_OPTIMIZER_ARGS['epochs'] - 1:
                    if profile:
                        t0 = time.perf_counter()
                    self.pbar.update(pbar_update_interval if (epoch + 1) % pbar_update_interval == 0 else (epoch + 1) % pbar_update_interval)
                    if profile:
                        prof['pbar_update_s'] += (time.perf_counter() - t0)
            
            # Early stopping (only sync GPU every 10 epochs to reduce overhead)
            if (epoch + 1) % early_stop_check_interval == 0:
                current_loss = float(loss.item())  # GPU sync here
                if current_loss < (best_loss - min_delta):
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += early_stop_check_interval
                
                if patience_counter >= patience:
                    # Fill remaining progress bar steps
                    remaining = self.STEP_2_OPTIMIZER_ARGS['epochs'] - epoch - 1
                    if remaining > 0:
                        if self.pbar is not None:
                            self.pbar.update(remaining)
                    break

        if profile:
            if do_sync:
                torch.cuda.synchronize()
            prof['total_s'] = time.perf_counter() - t_total0
            denom = max(int(epochs_ran), 1)
            logging.info(
                "Step 2 profile: epochs=%d total=%.3fs (avg=%.4fs/epoch) | "
                "fwd=%.3fs bwd=%.3fs opt=%.3fs sched=%.3fs loss=%.3fs zg=%.3fs pbar=%.3fs%s",
                denom,
                prof['total_s'],
                prof['total_s'] / denom,
                prof['forward_s'],
                prof['backward_s'],
                prof['opt_step_s'],
                prof['sched_step_s'],
                prof['loss_s'],
                prof['zero_grad_s'],
                prof['pbar_update_s'],
                " (cuda sync timings)" if do_sync else "",
            )

        return FitDBSIModel(
            self.anisotropic_models,
            self.isotropic_models,
            self.configuration,
            self.model_map,
            log_s0=self.log_s0 if self.learnable_s0 else None,
        )

class FitDBSIModel:
    def __init__(self, anisotropic_models: List[Type[least_squares_optimizer]], 
                 isotropic_models: List[Type[least_squares_optimizer]], 
                 configuration,
                 model_map,
                 log_s0: torch.Tensor | None = None) -> None:
        
        self.params             = utils.ParamStoreDict()
        self.anisotropic_models = anisotropic_models
        self.isotropic_models   = isotropic_models
        self.configuration      = configuration
        self.model_map          = model_map
        self._log_s0            = log_s0

        self.DEFAULT_ISOTROPIC_CUTS = self.configuration.DEFAULT_ISOTROPIC_CUTS
        self.DEFAULT_FIBER_CUTS     = self.configuration.DEFAULT_FIBER_CUTS

        self._models_to_parameter_maps()
        pass
 
    def _models_to_parameter_maps(self):
        N_voxels = reduce(add, ([model.output_dimension for model in self.isotropic_models]))
        self.params['isotropic_spectrum'] = torch.zeros((N_voxels,len(self.configuration.iso_basis)), device= self.configuration.DEVICE)
        for compartment, indicies in self.DEFAULT_ISOTROPIC_CUTS.items():
            self.params[f'{compartment}_fraction'] = torch.zeros(N_voxels, device= self.configuration.DEVICE)
            self.params[f'{compartment}_adc'     ] = torch.zeros(N_voxels, device= self.configuration.DEVICE)

        if self._log_s0 is not None:
            with torch.no_grad():
                s0 = torch.exp(self._log_s0.detach())
                s0 = torch.clamp(s0, min=MIN_POSITIVE_SIGNAL)
            self.params['s0_map'] = s0
            
        with torch.no_grad():
            iso_basis_cu = self.configuration.iso_basis.to(self.configuration.DEVICE)
            for num_fibers, (anisotropic_model, isotropic_model) in enumerate(zip(self.anisotropic_models, self.isotropic_models[:-1])): # The last isotropic model is the 0-fiber model!
                f_aniso, x_aniso = anisotropic_model.A, anisotropic_model.get_parameters()
                f_iso  , x_iso   = isotropic_model.A, isotropic_model.get_parameters()

                # Normalize per-voxel fractions to sum to 1

                N = torch.cat([x_aniso, x_iso], dim = 1).sum(dim = 1)
                x_aniso[~ (N == 0)] /= N[ ~ (N == 0)][:, None]
                x_iso[  ~ (N == 0)] /= N[ ~ (N == 0)][:, None]          

                for compartment, indicies in self.DEFAULT_FIBER_CUTS.items():
                    self.params['fiber_%02d_local_%s_fractions' %(num_fibers+1, compartment)] = torch.zeros(N_voxels, device=self.configuration.DEVICE)
                    self.params['fiber_%02d_local_%s_signal'    %(num_fibers+1, compartment)] = torch.zeros(N_voxels, anisotropic_model.A.shape[-2], device=self.configuration.DEVICE)

                    for j in range(num_fibers+1):
                        fiber_fractions =  ((x_aniso[:,(j)*self.configuration.step_2_axials.shape[0] : (j+1)*self.configuration.step_2_axials.shape[0]])[..., indicies]).sum(dim = 1)       
                        fiber_signal = torch.clamp(
                                                    torch.einsum(
                                                                'bij, bj -> bi', 
                                                                (f_aniso[...,(j)*self.configuration.step_2_axials.shape[0] : (j+1)*self.configuration.step_2_axials.shape[0]])[..., indicies], 
                                                                (x_aniso[...,(j)*self.configuration.step_2_axials.shape[0] : (j+1)*self.configuration.step_2_axials.shape[0]])[..., indicies]
                                                                ), 
                                                    min = MIN_POSITIVE_SIGNAL, 
                                                    max = None
                                                )
                        self.params['fiber_%02d_local_%s_fractions' %(j+1, compartment)][self.model_map == (num_fibers+1)]    = fiber_fractions
                        self.params['fiber_%02d_local_%s_signal'    %(j+1, compartment)][self.model_map    == (num_fibers+1)] = fiber_signal

                self.params['isotropic_spectrum'][self.model_map == (num_fibers + 1)] = x_iso
                for compartment, indicies in self.DEFAULT_ISOTROPIC_CUTS.items():
                    self.params[f'{compartment}_fraction'][self.model_map == (num_fibers + 1)] = x_iso[:, indicies].sum(dim = 1)
                    self.params[f'{compartment}_adc'     ][self.model_map == (num_fibers + 1)] = 1e3 * (x_iso[:, indicies] * iso_basis_cu[indicies]).sum(dim = 1) / x_iso[:, indicies].sum(dim = 1)
                
            f_iso  , x_iso   = self.isotropic_models[-1].A, self.isotropic_models[-1].get_parameters()
            # Normalize per-voxel fractions to sum to 1
            N = x_iso.sum(dim = 1)
            x_iso[  ~ (N == 0)] /= N[ ~ (N == 0)][:, None]
            num_fibers = -1 
            self.params['isotropic_spectrum'][self.model_map == (num_fibers + 1)] = x_iso
            for regime, indicies in self.DEFAULT_ISOTROPIC_CUTS.items():
                    self.params[f'{regime}_fraction'][self.model_map == (num_fibers + 1)] = x_iso[:, indicies].sum(dim = 1)
                    self.params[f'{regime}_adc'     ][self.model_map == (num_fibers + 1)] = 1e3 * (x_iso[:, indicies] * iso_basis_cu[indicies]).sum(dim = 1) / x_iso[:, indicies].sum(dim = 1)

        self.params['max_group_number'] = self.model_map.max()        
        return 