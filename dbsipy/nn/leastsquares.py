import torch
import torch.nn as nn
from typing import Any, Type, Dict
import logging
import os
import concurrent.futures

from dbsipy.core import utils
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model
import dbsipy.nn._leastsquares_backends as BE

from dbsipy.dbsi.utils import fit_results_to_parameter_maps_dbsi
from dbsipy.dbsi_ia.utils import fit_results_to_parameter_maps_dbsi_ia


import psutil


from tqdm import tqdm 

from dbsipy.core.progress import make_progress_bar

ENGINES = {
    'DBSI': BE.DBSIModel,
    'IA'  : BE.DBSIModel
    }

RECONST = {
    'DBSI': fit_results_to_parameter_maps_dbsi,
    'IA'  : fit_results_to_parameter_maps_dbsi_ia
    }


def _diagnostics_enabled(model) -> bool:
    try:
        return bool(getattr(model, 'configuration').diagnostics_enabled)
    except Exception:
        return os.environ.get('DBSIPY_DIAGNOSTICS', '0') == '1'


def _reconstruct_with_diagnostics(engine: str, fit_results, data_indicies, model) -> None:
    try:
        RECONST[engine](fit_results, data_indicies, model)
    except KeyError as e:
        if _diagnostics_enabled(model):
            missing_key = e.args[0] if e.args else '<unknown>'
            try:
                available = sorted(getattr(model, 'params', {}).keys())
            except Exception:
                available = []

            logging.error(
                "Step 2 reconstruction failed with KeyError: %s (engine=%s)",
                missing_key,
                engine,
            )
            if available:
                logging.error("Allocated parameter maps: %d", len(available))
                logging.error("Allocated map key sample: %s", available[:30])
                logging.error(
                    "Hint: this usually means map naming/allocation mismatch (e.g., expected 'fiber_0d_IA_*' but model only has DBSI maps)."
                )
        raise

"""
Single Fiber Step 1 Model
-------------------------------------------------------------------
"""

def nnlsq(A: torch.Tensor, b: torch.Tensor, optimizer_args: Dict, device = 'cpu') -> torch.Tensor:
    
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``. This is a wrapper
    for a PyTorch-implemented gradient based least squares solver with ReLU non-linearity.
    
    Now supports batching for large datasets to avoid OOM errors.

    Parameters
    ----------
    A : MultiLinearOperator
        Tensor ``A`` as shown above.
    b : torch.FloatTensor
        Right-hand side Tensor (n_voxels, n_volumes).
    
    Returns
    -------
    x : torch.FloatTensor
        Solution vector.
    """

    default_optimizer_args = {'optimizer': torch.optim.Adam,
                      'scheduler': torch.optim.lr_scheduler.OneCycleLR,
                      'lr'       : 1e-4,
                      'epochs'   : 500,
                      'loss'     : 'mse',
                      'alpha'    : 0.005,
                      'device'   : 'cpu',
                      'batch_size': None,  # Auto-determine if None
                      'patience'  : 75,    # Early stopping patience (epochs)
                      'min_delta' : 0.0    # Minimum improvement to reset patience
                      }
    
    for k in optimizer_args.keys():
        if k not in default_optimizer_args.keys():
            raise TypeError ( f"invalid optimizer argument")

    for key in set(default_optimizer_args.keys()) - set(optimizer_args.keys()):
            optimizer_args[key] = default_optimizer_args[key]
    
    # Determine batch size based on available memory
    n_voxels = b.shape[0]
    batch_size = optimizer_args['batch_size']
    
    # Only enable batching for very large datasets (>250k voxels)
    # Smaller datasets process faster in a single batch
    BATCHING_THRESHOLD = 250000
    
    if n_voxels <= BATCHING_THRESHOLD:
        # Small/medium datasets: process all at once
        return _nnlsq_single_batch(A, b, optimizer_args, device)
    else:
        # Large datasets: enable batching
        if batch_size is None:
            # Auto-determine batch size based on device
            if device == 'cuda' and torch.cuda.is_available():
                batch_size = 50000  # Conservative GPU batch size
            else:
                batch_size = 100000  # CPU batch size
        
        # Process in batches
        logging.info(f"Step 1: Processing {n_voxels} voxels in batches of {batch_size}")
        return _nnlsq_batched(A, b, optimizer_args, device, batch_size)


def _nnlsq_single_batch(A: torch.Tensor, b: torch.Tensor, optimizer_args: Dict, device: str) -> torch.Tensor:
    """Original nnlsq implementation for small datasets (single batch)."""
    
    # Clear CUDA cache before allocating if using GPU
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        nnlsq_model = BE.least_squares_optimizer(A.to(device), non_linearity = nn.ReLU(), output_dimension = b.shape[0], device = device)
        b = b.to(device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logging.error(f"CUDA OOM in Step 1. Try reducing data size or using CPU.")
            logging.error(f"Data shape: {b.shape}, Model params: {A.shape}")
            # Fall back to CPU
            logging.warning("Falling back to CPU for Step 1 fitting...")
            device = 'cpu'
            torch.cuda.empty_cache()
            nnlsq_model = BE.least_squares_optimizer(A.to(device), non_linearity = nn.ReLU(), output_dimension = b.shape[0], device = device)
            b = b.to(device)
        else:
            raise

    optimizer = optimizer_args['optimizer'](params = nnlsq_model.parameters(), 
                                            lr     = optimizer_args['lr']
                                            )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = optimizer_args['lr'] * 10, total_steps = optimizer_args['epochs'])
    
    # Pre-allocate loss function outside loop
    L = torch.nn.MSELoss()

    pbar = make_progress_bar(
        total=optimizer_args['epochs'],
        desc='Step 1',
        colour='yellow',
    )

    best_loss = float('inf')
    patience_counter = 0
    patience = int(optimizer_args.get('patience', 75))
    min_delta = float(optimizer_args.get('min_delta', 0.0))
    
    # Batch updates to reduce overhead
    pbar_update_interval = max(1, min(10, optimizer_args['epochs'] // 20))
    early_stop_check_interval = max(1, min(10, optimizer_args['epochs'] // 50))

    try:
        for epoch in (range(optimizer_args['epochs'])):
            optimizer.zero_grad(set_to_none=True)
            Yhat = nnlsq_model.forward()
            loss =  L(Yhat, b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Batch progress bar updates
            if (epoch + 1) % pbar_update_interval == 0 or epoch == optimizer_args['epochs'] - 1:
                pbar.update(pbar_update_interval if (epoch + 1) % pbar_update_interval == 0 else (epoch + 1) % pbar_update_interval)

            # Early stopping check (only sync GPU every N epochs)
            if (epoch + 1) % early_stop_check_interval == 0:
                current = float(loss.item())  # GPU sync here
                if current < (best_loss - min_delta):
                    best_loss = current
                    patience_counter = 0
                else:
                    patience_counter += early_stop_check_interval
                    if patience_counter >= patience:
                        remaining = optimizer_args['epochs'] - epoch - 1
                        if remaining > 0:
                            pbar.update(remaining)
                        break
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            pbar.close()
            logging.error(f"CUDA OOM during Step 1 optimization at epoch {epoch}")
            logging.error(f"Data shape: {b.shape}, Device: {device}")
            
            # If already on CPU, we can't recover
            if device == 'cpu':
                logging.error("Already on CPU - cannot recover from OOM. Reduce data size.")
                raise
            
            # Fall back to CPU and restart
            logging.warning("Falling back to CPU and restarting Step 1...")
            torch.cuda.empty_cache()
            device = 'cpu'
            
            # Recreate model on CPU
            nnlsq_model = BE.least_squares_optimizer(A.to(device), non_linearity = nn.ReLU(), output_dimension = b.shape[0], device = device)
            b = b.to(device)
            
            # Recreate optimizer and scheduler
            optimizer = optimizer_args['optimizer'](params = nnlsq_model.parameters(), lr = optimizer_args['lr'])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = optimizer_args['lr'] * 10, total_steps = optimizer_args['epochs'])
            
            # Restart optimization from scratch
            pbar = make_progress_bar(
                total=optimizer_args['epochs'],
                desc='Step 1 (CPU)',
                colour='yellow',
            )
            
            for epoch in (range(optimizer_args['epochs'])):
                optimizer.zero_grad(set_to_none=True)
                Yhat = nnlsq_model.forward()
                loss =  L(Yhat, b)
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(1)

                current = float(loss.item())
                if current < (best_loss - min_delta):
                    best_loss = current
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        remaining = optimizer_args['epochs'] - epoch - 1
                        if remaining > 0:
                            pbar.update(remaining)
                        break
        else:
            pbar.close()
            raise
    
    pbar.close()
    return nnlsq_model.cpu().get_parameters().detach()


def _nnlsq_batched(A: torch.Tensor, b: torch.Tensor, optimizer_args: Dict, device: str, batch_size: int) -> torch.Tensor:
    """
    Batched nnlsq implementation for large datasets.
    
    Processes voxels in batches to avoid OOM errors. Each batch is independently
    optimized, which is valid since Step 1 fits each voxel independently.
    """
    n_voxels = b.shape[0]
    n_basis = A.shape[1]
    
    # Initialize output array
    x_all = torch.zeros((n_voxels, n_basis), dtype=torch.float32)
    
    # Move basis matrix to device once
    A_device = A.to(device)

    # Reuse loss instance across batches
    L = torch.nn.MSELoss()

    # Clearing the CUDA cache is expensive; do it once up-front for large runs.
    # If we hit OOM, we'll clear again in the exception path.
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Pre-allocate loss function outside batch loop
    L = torch.nn.MSELoss()
    
    # Process in batches
    n_batches = int(torch.ceil(torch.tensor(n_voxels / batch_size)))
    
    pbar_batches = make_progress_bar(
        total=n_batches,
        desc='Step 1 (batches)',
        colour='cyan',
    )
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_voxels)
        
        # Get batch data
        b_batch = b[start_idx:end_idx]
        
        # Fit this batch (using single batch implementation without batching)
        try:
            nnlsq_model = BE.least_squares_optimizer(A_device, non_linearity=nn.ReLU(), 
                                                    output_dimension=b_batch.shape[0], device=device)
            b_batch_device = b_batch.to(device)
            
            optimizer = optimizer_args['optimizer'](params=nnlsq_model.parameters(), lr=optimizer_args['lr'])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer_args['lr'] * 10, 
                                                           total_steps=optimizer_args['epochs'])
            
            # Optimize this batch (no progress bar for sub-batches)
            for epoch in range(optimizer_args['epochs']):
                optimizer.zero_grad(set_to_none=True)
                Yhat = nnlsq_model.forward()
                loss = L(Yhat, b_batch_device)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Store results
            x_all[start_idx:end_idx] = nnlsq_model.cpu().get_parameters().detach()
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device == 'cuda':
                logging.warning(f"Batch {batch_idx+1}/{n_batches} OOM on GPU, retrying on CPU...")
                torch.cuda.empty_cache()
                
                # Retry this batch on CPU
                nnlsq_model = BE.least_squares_optimizer(A.to('cpu'), non_linearity=nn.ReLU(), 
                                                        output_dimension=b_batch.shape[0], device='cpu')
                b_batch_cpu = b_batch.to('cpu')
                
                optimizer = optimizer_args['optimizer'](params=nnlsq_model.parameters(), lr=optimizer_args['lr'])
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer_args['lr'] * 10, 
                                                               total_steps=optimizer_args['epochs'])
                
                # L already allocated above, reuse it
                
                for epoch in range(optimizer_args['epochs']):
                    optimizer.zero_grad(set_to_none=True)
                    Yhat = nnlsq_model.forward()
                    loss = L(Yhat, b_batch_cpu)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                x_all[start_idx:end_idx] = nnlsq_model.cpu().get_parameters().detach()
            else:
                pbar_batches.close()
                raise
        
        pbar_batches.update(1)
    
    pbar_batches.close()
    return x_all

"""
Step 2 Multi-Fiber Modeling 
-------------------------------------------------------------------
"""

def compute(args, **kwargs):

    Model = ENGINES[args['engine']](dwi            = args['data'], 
                                    bvals          = args['DBSIModel'].bvals,
                                    bvecs          = args['DBSIModel'].bvecs,
                                    directions     = args['directions'],
                                    fractions      = args['fractions'],
                                    restricted     = args['restricted'],
                                    non_restricted = args['non_restricted'],
                                    indicies       = args['indicies'],
                                    s0_init        = args.get('s0_init', None),
                                    DBSI_CONFIG    = args['DBSIModel'].configuration,
                                    logging_args   = (args['process_id'], args['n_jobs'], args['pbar']),
                        )
                                
    fitModel = Model.fit()
    outputs = fitModel.params
    del Model, fitModel
    return outputs

class BatchedCalls:
    def __init__(self, args) -> None:
        
        self.items   = args
        self._size   = len(self.items)
    
        pass

    def __call__(self, func) -> Any:
        return [func(args) 
                for args in self.items]


def _package_data(self) -> Type[BatchedCalls]:
    
    max_num_fibers = self.DBSIModel.configuration.max_group_number

    # Memory model for batching (very approximate):
    #  - float32 forward models: ~4 bytes/val
    #  - anisotropic: (n_bvals * (max_fibers * n_axial_models))
    #  - isotropic: (n_bvals * n_iso_basis)
    #  - plus a small constant factor
    denom = (4 * (
        self.DBSIModel.bvals.shape[0] * (max_num_fibers + self.DBSIModel.configuration.step_2_axials.shape[0])
        + self.DBSIModel.configuration.iso_basis.shape[0]
    ) + 7.0 + self.DBSIModel.bvals.shape[0])
    batch_size = int(self._max_mem_prop * self._max_nbytes // denom)
    batch_size = max(1, batch_size)

    diagnostics = bool(getattr(self.DBSIModel.configuration, 'diagnostics_enabled', False)) or (os.environ.get('DBSIPY_DIAGNOSTICS', '0') == '1')
    if diagnostics:
        try:
            logging.debug(
                f"Diagnostics: Step2 batching denom≈{float(denom):.3g}, max_nbytes≈{float(self._max_nbytes):.3g}, batch_size≈{batch_size}"
            )
            logging.debug(
                f"Diagnostics: Step2 dims n_vox={int(self.DBSIModel.configuration.linear_dims):,}, n_bvals={int(self.DBSIModel.bvals.shape[0])}, max_fibers={int(max_num_fibers)}, n_iso={int(self.DBSIModel.configuration.iso_basis.shape[0])}"
            )
        except Exception:
            pass

    step2_target = getattr(self.DBSIModel, 'dwi_step2_target', self.DBSIModel.dwi)
    self.split_data, self.data_indicies, self.n_jobs = utils.split_data(step2_target, batch_size=batch_size)
    func_args = []

    total =  1 * self.n_jobs* self.DBSIModel.configuration.STEP_2_OPTIMIZER_ARGS['epochs']

    if diagnostics:
        try:
            logging.debug(f"Diagnostics: Step2 n_jobs={int(self.n_jobs)}, epochs={int(self.DBSIModel.configuration.STEP_2_OPTIMIZER_ARGS['epochs'])}, total_pbar_steps={int(total)}")
        except Exception:
            pass
        
    pbar = make_progress_bar(
        total=total,
        desc='Step 2',
        colour='MAGENTA',
    )

    for process_id, (data, indicies) in enumerate(zip(self.split_data, self.data_indicies)):

        args = utils.ParamStoreDict()    

        args['data']           = data
        args['indicies']       = indicies        
        args['process_id']     = process_id 
        args['n_jobs']         = self.n_jobs
        args['DBSIModel']      = self.DBSIModel
        args['directions']     = self.ModelPriors['direction_priors'][indicies]
        args['fractions']      = self.ModelPriors['fractions_priors'][indicies]
        args['restricted']     = self.ModelPriors['restricted_priors'][indicies]
        args['non_restricted'] = self.ModelPriors['non_restricted_priors'][indicies]
        args['pbar']           = pbar
        args['engine']         = self.engine

        # Optional S0 initializer for learnable_s0 mode.
        try:
            s0_est = getattr(self.DBSIModel, 's0_est', None)
            args['s0_init'] = (s0_est[indicies] if s0_est is not None else None)
        except Exception:
            args['s0_init'] = None

        func_args.append(args)

    return BatchedCalls(func_args)
   
class CUDABackend:
    """GPU-accelerated backend for Step 2 multi-fiber optimization.
    
    This backend partitions voxel data into batches that fit in GPU memory, runs
    optimization on each batch using CUDA-enabled PyTorch, and reconstructs the
    full parameter maps. Uses ~20% of available GPU memory by default to prevent OOM.
    
    Attributes
    ----------
    _max_mem_prop : float
        Fraction of GPU memory to use (default: 0.2 = 20%).
    _max_nbytes : int
        Maximum bytes available for batching (calculated from GPU memory).
    DBSIModel : DiffusionBasisFunctionModel
        The DBSI model containing data, bvals/bvecs, and configuration.
    ModelPriors : dict
        Prior estimates from Step 1 (fiber directions, fractions, isotropic components).
    engine : str
        Analysis engine ('DBSI' or 'IA').
    wrapper : BatchedCalls
        Callable that executes optimization on all batches.
    
    See Also
    --------
    MultiprocessingBackend : CPU-based alternative
    """
    def __init__(self, DBSIModel, ModelPriors, engine, max_mem_prop = .2 ) -> None:

        self._max_mem_prop = max_mem_prop 
        try:
            dev_idx = int(torch.cuda.current_device())
        except Exception:
            dev_idx = 0
        self._max_nbytes   = (self._max_mem_prop * torch.cuda.get_device_properties(dev_idx).total_memory) - torch.cuda.memory_allocated(dev_idx)
        self.DBSIModel     = DBSIModel
        self.ModelPriors   = ModelPriors

        self.engine  = engine 
        self.wrapper = _package_data(self)

        pass

    def __call__(self, func) -> None:
                fit_results = self.wrapper(func)
                _reconstruct_with_diagnostics(self.engine, fit_results, self.data_indicies, self.DBSIModel)
       
class MultiprocessingBackend:
    r"""CPU-based backend for Step 2 multi-fiber optimization.
    
    This backend partitions voxel data into batches based on available system RAM,
    runs optimization on CPU using PyTorch, and reconstructs parameter maps.
    Currently single-threaded but uses ~25% of available memory by default.
    
    Attributes
    ----------
    _max_mem_prop : float
        Fraction of system RAM to use (default: 0.25 = 25%).
    _max_nbytes : int
        Maximum bytes available for batching (calculated from system memory).
    DBSIModel : DiffusionBasisFunctionModel
        The DBSI model containing data, bvals/bvecs, and configuration.
    ModelPriors : dict
        Prior estimates from Step 1 (fiber directions, fractions, isotropic components).
    engine : str
        Analysis engine ('DBSI' or 'IA').
    wrapper : BatchedCalls
        Callable that executes optimization on all batches.
    
    Notes
    -----
    Single-threaded only currently. Multi-process support planned for future releases.
    Uses psutil to monitor available RAM and prevent system swapping.
    
    See Also
    --------
    CUDABackend : GPU-accelerated alternative
    """
    def __init__(self, DBSIModel, ModelPriors, engine, max_mem_prop = .25 ) -> None:
        self._max_mem_prop = max_mem_prop 
        # If nn._memory is available, use its job/cgroup-aware estimate.
        try:
            from dbsipy.nn._memory import get_effective_available_cpu_memory_bytes

            avail = int(get_effective_available_cpu_memory_bytes())
        except Exception:
            avail = int(psutil.virtual_memory().available)
        self._max_nbytes   = int(self._max_mem_prop * avail)
        self.DBSIModel     = DBSIModel
        self.ModelPriors   = ModelPriors

        self.engine  = engine 
        self.wrapper = _package_data(self)
        pass

    def __call__(self, func) -> None:
                enable_parallel = os.environ.get('DBSIPY_CPU_PARALLEL', '0') == '1'
                if not enable_parallel:
                    fit_results = self.wrapper(func)
                    _reconstruct_with_diagnostics(self.engine, fit_results, self.data_indicies, self.DBSIModel)
                    return

                items = list(getattr(self.wrapper, 'items', []))
                if not items:
                    fit_results = []
                    _reconstruct_with_diagnostics(self.engine, fit_results, self.data_indicies, self.DBSIModel)
                    return

                # Opt-in parallel execution over batches.
                # NOTE: This uses threads to avoid pickling large torch/numpy objects.
                # If you enable this, consider also controlling torch CPU thread usage
                # externally (e.g., OMP_NUM_THREADS / MKL_NUM_THREADS) to avoid oversubscription.
                max_workers_env = os.environ.get('DBSIPY_CPU_PARALLEL_WORKERS', '').strip()
                try:
                    requested_workers = int(max_workers_env) if max_workers_env else 0
                except Exception:
                    requested_workers = 0

                default_workers = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
                max_workers = requested_workers if requested_workers > 0 else int(default_workers)
                max_workers = max(1, min(max_workers, len(items)))

                logging.info(f"Step 2 CPU parallel enabled: workers={max_workers} (set DBSIPY_CPU_PARALLEL_WORKERS to override)")

                # Disable per-epoch pbar updates inside worker threads (not thread-safe).
                for args in items:
                    try:
                        args['pbar'] = None
                    except Exception:
                        pass

                pbar_batches = make_progress_bar(
                    total=len(items),
                    desc='Step 2 (batches)',
                    colour='MAGENTA',
                )

                fit_results = [None] * len(items)
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                        future_to_idx = {ex.submit(func, args): i for i, args in enumerate(items)}
                        for fut in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[fut]
                            fit_results[idx] = fut.result()
                            pbar_batches.update(1)
                finally:
                    pbar_batches.close()

                _reconstruct_with_diagnostics(self.engine, fit_results, self.data_indicies, self.DBSIModel)

BACKENDS = {
    'cuda': CUDABackend,
    'cpu' : MultiprocessingBackend
}

class MultiFiberModel:
    def __init__(self, device: str, engine: str, **kwargs) -> None:
        self.device  = device     
        self.engine  = engine
        pass

    def __call__(self, Model, ModelPriors: Dict[str, torch.FloatTensor]) -> None:
        logging.info(f"Step 2 backend: {self.device} ({BACKENDS[self.device].__name__})")
        BACKENDS[self.device](Model, ModelPriors, self.engine)(compute)
        pass 
 




