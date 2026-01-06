import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nibabel.onetime import auto_attr
from itertools import product
from tqdm import tqdm

from dbsipy.core.progress import make_progress_bar
from dbsipy.core.signal import normalize_signal
import logging


def _robust_sym_eigh_3x3_batched(
    matrices: torch.Tensor,
    *,
    device: torch.device | str,
    base_jitter: float = 1e-12,
    max_jitter: float = 1e-4,
    out_dtype: torch.dtype = torch.float32,
    min_eigval: float = 1e-4,
    max_batch_size: int | None = None,
):
    """Robust symmetric eigendecomposition for batched 3x3 matrices.

    torch.linalg.eigh can fail to converge on ill-conditioned or nearly-degenerate
    inputs (observed on real ADAM fits). This helper:
    - enforces symmetry,
    - uses intelligent GPU batch chunking to avoid CUSOLVER limits,
    - escalates diagonal jitter,
    - falls back to per-matrix CPU torch.linalg.eigh,
    - and ultimately falls back to NumPy's eigh for worst cases.
    
    Args:
        matrices: (N, 3, 3) batch of symmetric matrices
        device: Target device for output tensors
        base_jitter: Initial diagonal regularization
        max_jitter: Maximum diagonal regularization
        out_dtype: Output dtype
        min_eigval: Minimum eigenvalue for fallback cases
        max_batch_size: Maximum batch size for GPU eigh. Auto-detected if None.
    """
    if matrices.ndim != 3 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (N,3,3) tensor, got {tuple(matrices.shape)}")

    n_matrices = matrices.shape[0]
    matrices_sym = 0.5 * (matrices + matrices.transpose(-1, -2))
    matrices64 = matrices_sym.double()

    # Determine safe batch size for GPU eigendecomposition
    if max_batch_size is None:
        if str(device).startswith('cuda'):
            # Conservative limits based on CUSOLVER constraints
            # Tesla V100/A100: 200K is safe, newer GPUs can handle more
            max_batch_size = 200000
        else:
            # CPU can handle larger batches (no CUSOLVER limit)
            max_batch_size = 1000000
    
    # Fast path: batch fits within safe size, try direct eigh
    if n_matrices <= max_batch_size:
        try:
            evals, evecs = torch.linalg.eigh(matrices64)
            return evals.to(device=device, dtype=out_dtype), evecs.to(device=device, dtype=out_dtype)
        except Exception:
            # Fall through to chunked approach
            pass
    
    # GPU chunking path: split large batches to stay within CUSOLVER limits
    if str(device).startswith('cuda') and n_matrices > max_batch_size:
        try:
            n_chunks = int(np.ceil(n_matrices / max_batch_size))
            
            # Log chunking strategy for large batches
            if n_chunks > 1:
                logging.debug(
                    f"Eigendecomposition: Processing {n_matrices:,} matrices in "
                    f"{n_chunks} GPU chunks of ={max_batch_size:,} to avoid CUSOLVER limits"
                )
            
            evals_chunks = []
            evecs_chunks = []
            
            for i in range(n_chunks):
                start_idx = i * max_batch_size
                end_idx = min(start_idx + max_batch_size, n_matrices)
                chunk = matrices64[start_idx:end_idx]
                
                try:
                    ev_chunk, evec_chunk = torch.linalg.eigh(chunk)
                    evals_chunks.append(ev_chunk)
                    evecs_chunks.append(evec_chunk)
                except Exception:
                    # If GPU chunking fails, fall back to CPU for this chunk only
                    logging.debug(f"GPU eigh failed for chunk {i+1}/{n_chunks}, using CPU fallback")
                    chunk_cpu = chunk.cpu()
                    ev_chunk, evec_chunk = torch.linalg.eigh(chunk_cpu)
                    evals_chunks.append(ev_chunk.to(device))
                    evecs_chunks.append(evec_chunk.to(device))
            
            evals = torch.cat(evals_chunks, dim=0)
            evecs = torch.cat(evecs_chunks, dim=0)
            return evals.to(dtype=out_dtype), evecs.to(dtype=out_dtype)
        except Exception:
            # If chunked GPU approach fails, fall through to CPU fallback
            logging.warning(f"GPU chunked eigendecomposition failed, falling back to CPU for {n_matrices:,} matrices")
            pass

    # Slow path: per-matrix CPU with jitter escalation, NumPy ultimate fallback.
    matrices_cpu = matrices64.detach().cpu()
    n = matrices_cpu.shape[0]
    evals_cpu = torch.empty((n, 3), dtype=torch.float64)
    evecs_cpu = torch.empty((n, 3, 3), dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)

    jitters = [base_jitter, 1e-10, 1e-8, 1e-6, 1e-5, max_jitter]
    for i in range(n):
        m = matrices_cpu[i]
        m = 0.5 * (m + m.t())

        # If the matrix is non-finite, return a safe SPD identity for this entry.
        if not torch.isfinite(m).all():
            evals_cpu[i] = float(min_eigval) * torch.ones(3, dtype=torch.float64)
            evecs_cpu[i] = eye
            continue

        # Scale jitter by matrix magnitude to keep it meaningful across datasets.
        scale = float(torch.linalg.norm(m, ord="fro").item())
        scale = max(scale, 1.0)

        success = False
        for j in jitters:
            try:
                w, v = torch.linalg.eigh(m + (j * scale) * eye)
                evals_cpu[i] = w
                evecs_cpu[i] = v
                success = True
                break
            except Exception:
                continue

        if not success:
            m_np = m.numpy()
            # NumPy eigh can also fail on pathological inputs; try SVD then identity.
            try:
                w_np, v_np = np.linalg.eigh(m_np)
                evals_cpu[i] = torch.from_numpy(w_np)
                evecs_cpu[i] = torch.from_numpy(v_np)
            except np.linalg.LinAlgError:
                try:
                    u_np, s_np, _vh_np = np.linalg.svd(m_np)
                    # For symmetric matrices, U is orthonormal; use it as a basis.
                    evals_cpu[i] = torch.from_numpy(s_np)
                    evecs_cpu[i] = torch.from_numpy(u_np)
                except np.linalg.LinAlgError:
                    evals_cpu[i] = float(min_eigval) * torch.ones(3, dtype=torch.float64)
                    evecs_cpu[i] = eye

    return evals_cpu.to(device=device, dtype=out_dtype), evecs_cpu.to(device=device, dtype=out_dtype)

# Import custom exceptions for better error handling
try:
    from dbsipy.core.fast_DBSI import OptimizationError
except ImportError:
    # Fallback if import fails
    class OptimizationError(Exception):
        """Raised when optimization fails to converge or produces invalid results."""
        pass

# Physical constraints for diffusion and kurtosis tensors
# UNITS: Diffusion in mm^2/s (multiply by 1000 for um^2/ms output)
DIFFUSION_MIN = -0.005  # mm^2/s (allow small negative for numerical stability)
DIFFUSION_MAX = 0.005   # mm^2/s = 5 um^2/ms (max physiological diffusivity)



def ols_fit(A: torch.Tensor, Y: torch.Tensor):
    r"""
    Implements batch parallelized Y = (A^{T}A)^{-1}A^{T}x via Moore-Penrose pseudo-inverse
    """ 
    # Ensure signals are valid (positive, non-zero)
    Y = torch.clamp(Y, min=1e-6)
    
    # Check for NaN or Inf in inputs
    if torch.any(torch.isnan(A)) or torch.any(torch.isinf(A)):
        raise ValueError("Design matrix A contains NaN or Inf values")
    if torch.any(torch.isnan(Y)) or torch.any(torch.isinf(Y)):
        raise ValueError("Signal Y contains NaN or Inf values")
    
    # Solve in float64 for numerical stability (some models have b^2 terms up to ~1e6).
    # On CPU, torch.linalg.lstsq is typically the most robust/accurate.
    # On CUDA, lstsq can hit unsupported cuBLAS batched TRSM on some setups;
    # use a normal-equations fallback there.
    A64 = A.double()  # (n_meas, n_params)
    log_Y64 = torch.log(Y).double()  # (n_voxels, n_measurements)

    rhs = log_Y64.T  # (n_meas, n_voxels)

    if A64.device.type == 'cpu' and rhs.device.type == 'cpu':
        sol = torch.linalg.lstsq(A64, rhs).solution  # (n_params, n_voxels)
        x = sol.T.float()
    else:
        At = A64.transpose(0, 1)  # (n_params, n_meas)
        AtA = At @ A64  # (n_params, n_params)
        AtY = At @ rhs  # (n_params, n_voxels)

        # Diagonal jitter scaled to the matrix magnitude for stability.
        n_params = AtA.shape[0]
        trace = torch.trace(AtA)
        scale = torch.clamp(trace / max(n_params, 1), min=1.0)
        jitter = (1e-12 * scale).to(dtype=AtA.dtype, device=AtA.device)
        AtA_reg = AtA + jitter * torch.eye(n_params, dtype=AtA.dtype, device=AtA.device)

        try:
            sol = torch.linalg.solve(AtA_reg, AtY)  # (n_params, n_voxels)
        except RuntimeError:
            # Rare fallback for singular/ill-conditioned cases.
            sol = torch.linalg.pinv(A64) @ rhs

        x = sol.T.float()

    y_hat_log = (A @ x.T).T  # (n_voxels, n_measurements)
    y_hat = torch.exp(y_hat_log)

    return x, y_hat

def wls_fit(A: torch.Tensor, Y: torch.Tensor):
    r"""
    Implements batch parallelied WLS via Moore-Pensrose pseudo-inverse.
    Optimized with matmul instead of einsum for better performance.

    References
    ----------
    .. [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.
    .. [2] Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, 
       Nimmo-Smith I and Dipy Contributors (2014). DIPY, a library for the analysis of diffusion MRI data. 
       Frontiers in Neuroinformatics, vol.8, no.8.
    
    """
    # Initial OLS fit (gives predicted signal used for weights)
    x_dti, y_hat_dti = ols_fit(A, Y)

    # Compute weights from OLS fit (use predicted signal as weights)
    w = torch.clamp(y_hat_dti, min=1e-6)  # (n_voxels, n_measurements)
    log_Y = torch.log(torch.clamp(Y, min=1e-6))  # (n_voxels, n_measurements)

    # IMPORTANT: do NOT materialize A_wls_batched for all voxels.
    # For large images (e.g., 177k voxels), (n_vox, n_meas, n_params) can be multi-GB,
    # and the subsequent (n_vox, n_params, n_params) normal matrices can OOM on CUDA.
    # Instead, solve in voxel-chunks using weighted normal equations:
    #   (A^T diag(w^2) A) x = A^T diag(w^2) log(Y)
    A64 = A.double()  # (n_meas, n_params)
    A_T = A64.transpose(0, 1).unsqueeze(0)  # (1, n_params, n_meas)

    n_vox = int(Y.shape[0])
    n_params = int(A64.shape[1])
    x_wls_batched = torch.empty((n_vox, n_params), device=Y.device, dtype=torch.float32)

    # Chunk sizes tuned for memory safety; CPU can go larger.
    voxel_chunk = 4096 if Y.device.type != 'cpu' else 32768

    eye = torch.eye(n_params, dtype=torch.float64, device=Y.device).unsqueeze(0)  # (1, n_params, n_params)

    for v0 in range(0, n_vox, voxel_chunk):
        v1 = min(v0 + voxel_chunk, n_vox)
        w_chunk = w[v0:v1].double()  # (chunk, n_meas)
        logY_chunk = log_Y[v0:v1].double()  # (chunk, n_meas)

        # diag(w^2) is implicit; use w2 to weight rows.
        w2 = w_chunk * w_chunk  # (chunk, n_meas)

        # WeightedA: (chunk, n_meas, n_params)
        weighted_A = A64.unsqueeze(0) * w2.unsqueeze(2)

        # A_T expands without allocating full copies
        A_T_expand = A_T.expand(weighted_A.shape[0], -1, -1)  # (chunk, n_params, n_meas)

        # Normal equations
        AtA = torch.bmm(A_T_expand, weighted_A)  # (chunk, n_params, n_params)
        weighted_Y = (w2 * logY_chunk).unsqueeze(2)  # (chunk, n_meas, 1)
        AtY = torch.bmm(A_T_expand, weighted_Y)  # (chunk, n_params, 1)

        trace = AtA.diagonal(dim1=-2, dim2=-1).sum(-1)  # (chunk,)
        scale = torch.clamp(trace / max(n_params, 1), min=1.0)
        jitter = (1e-12 * scale).view(-1, 1, 1)
        AtA_reg = AtA + jitter * eye

        try:
            sol = torch.linalg.solve(AtA_reg, AtY)  # (chunk, n_params, 1)
        except RuntimeError:
            # Retry with larger jitter; if still failing, fall back to CPU lstsq.
            jitter2 = (1e-6 * scale).view(-1, 1, 1)
            AtA_reg2 = AtA + jitter2 * eye
            try:
                sol = torch.linalg.solve(AtA_reg2, AtY)
            except RuntimeError:
                sol = torch.linalg.lstsq(AtA_reg2.cpu(), AtY.cpu()).solution.to(device=Y.device)

        x_wls_batched[v0:v1] = sol.squeeze(2).float()

    # Predicted signal in unweighted space
    y_hat_log = (A @ x_wls_batched.T).T
    y_hat = torch.exp(y_hat_log)

    return x_wls_batched, y_hat

class diffusion_tensor_model():
    """DTI model for single-shell diffusion tensor estimation.
    
    UNITS CONVENTION:
    - Input bvals: s/mm^2 (MRI standard)
    - Internal diffusion tensor: mm^2/s (SI standard)
    - Output diffusivity (AD, RD, MD): um^2/ms (1 mm^2/s = 1000 um^2/ms)
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, device = 'cpu') -> None:
        self.device = device
        
        # Convert to tensors and move to device
        if isinstance(bvals, np.ndarray):
            self.bvals = torch.from_numpy(bvals).float().to(device)
        else:
            self.bvals = bvals.to(device)
        
        if isinstance(bvecs, np.ndarray):
            self.bvecs = torch.from_numpy(bvecs).float().to(device)
        else:
            self.bvecs = bvecs.to(device)
        
        # Normalize gradient directions.
        # Avoid NaNs for b=0 rows that may have [0,0,0] b-vectors.
        norms = torch.linalg.norm(self.bvecs, ord=2, axis=1)
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        self.bvecs = self.bvecs / norms[:, None]
        self.dti_design_matrix = self._make_design_matrix()
        return

    def _make_design_matrix(self):
        """
        D.shape = (n_bvals, 7) where D[i,:] = [-Bxx -Byy, -Bzz, -Bxy, -Bxz, -Byz, -1]
        """
        bvals = self.bvals
        bvecs = self.bvecs
        D = torch.zeros((bvals.shape[0], 7), device=self.device)
        D[:,0] = bvecs[:,0]**2 * bvals
        D[:,1] = bvecs[:,1]**2 * bvals 
        D[:,2] = bvecs[:,2]**2 * bvals
        D[:,3] = bvecs[:,0]*bvecs[:,1]*2.*bvals
        D[:,4] = bvecs[:,0]*bvecs[:,2]*2.*bvals
        D[:,5] = bvecs[:,1]*bvecs[:,2]*2.*bvals
        D[:,6] = torch.ones(bvals.shape[0], device=self.device)
        return -1. * D
    
    def fit(self, Y: np.ndarray, fit_method: str = 'WLS', optimizer_args: dict | None = None):
        fit_method = str(fit_method).upper()

        if fit_method in {'WLS', 'OLS'}:
            fit_methods = {'WLS': wls_fit, 'OLS': ols_fit}
            tensor_elements, predicted_signal = fit_methods[fit_method](
                self.dti_design_matrix.to(self.device),
                Y.to(self.device),
            )
            return fit_diffusion_tensor_model(predicted_signal, tensor_elements[..., 0:-1], device=self.device)

        if fit_method != 'ADAM':
            raise ValueError(f"Unsupported DTI fit_method '{fit_method}'. Use WLS, OLS, or ADAM.")

        # ADAM fit: optimize the 7 linear parameters (6 tensor elems + (-logS0)).
        # IMPORTANT: Optimize in log-signal space (same objective family as OLS/WLS).
        # Using exp() in the loss makes the objective non-convex and can amplify noise.
        # Here we use a WLS-style quadratic objective with fixed weights from a WLS warm-start.
        default_args = {'lr': 1e-2, 'epochs': 50}
        if optimizer_args:
            for k, v in optimizer_args.items():
                default_args[k] = v

        lr = float(default_args.get('lr', 1e-2))
        epochs = int(default_args.get('epochs', 50))

        A = self.dti_design_matrix.to(self.device)
        Y_t = Y.to(self.device)
        Y_t = torch.clamp(Y_t, min=1e-6)
        log_Y = torch.log(Y_t)

        # Warm-start from WLS (log-domain).
        x0, _y_hat_wls = wls_fit(A, Y_t)
        x = x0.detach().clone().requires_grad_(True)

        # IMPORTANT: match the WLS weighting semantics used by wls_fit().
        # That routine defines weights from the OLS-predicted signal, then solves a
        # weighted least squares problem in log-space. Keep those weights fixed here
        # (convex quadratic in x), which eliminates exp-induced instability.
        _x_ols, y_hat_ols = ols_fit(A, Y_t)
        w = torch.clamp(y_hat_ols.detach(), min=1e-6)
        # Normalize weights per voxel so their mean is ~1.
        # This keeps the WLS optimum unchanged (per-voxel constant scaling) but
        # prevents extreme gradient magnitudes when S0 is large.
        w_scale = torch.mean(w, dim=1, keepdim=True)
        w = w / torch.clamp(w_scale, min=1e-6)

        opt = torch.optim.Adam([x], lr=lr)
        pbar = make_progress_bar(total=epochs, desc='DTI', colour='cyan')
        # Match the efficient progress updating scheme used elsewhere:
        # limit tqdm refresh overhead while still giving useful feedback.
        pbar_update_interval = max(1, min(10, epochs // 20))
        pbar_pending = 0
        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            y_hat_log = (A @ x.T).T
            # WLS-style log-domain loss: minimize mean( (w * (A x - logY))^2 ).
            resid = w * (y_hat_log - log_Y)
            loss = torch.mean(resid * resid)
            loss.backward()
            opt.step()

            # Clamp diffusion tensor elements to a safe physiological range.
            # Diagonal: [0, 0.005] mm^2/s; off-diagonal: [-0.005, 0.005] mm^2/s.
            with torch.no_grad():
                x[:, 0:3] = torch.clamp(x[:, 0:3], min=0.0, max=0.005)
                x[:, 3:6] = torch.clamp(x[:, 3:6], min=-0.005, max=0.005)
                # Parameterization uses (-logS0) as the final coefficient.
                # Clamp S0 to [1e-6, 1e9] (arbitrary but prevents runaway exp).
                log_s0_min = float(np.log(1e-6))
                log_s0_max = float(np.log(1e9))
                x[:, 6] = torch.clamp(x[:, 6], min=-log_s0_max, max=-log_s0_min)

            pbar_pending += 1
            if (epoch + 1) % pbar_update_interval == 0 or epoch == epochs - 1:
                pbar.update(pbar_pending)
                pbar_pending = 0
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        pbar.close()

        # Final prediction
        y_hat_log = (A @ x.T).T
        predicted_signal = torch.exp(y_hat_log).detach()
        tensor_elements = x.detach()
        return fit_diffusion_tensor_model(predicted_signal, tensor_elements[..., 0:-1], device=self.device)
 
class fit_diffusion_tensor_model:
    def __init__(self, predicted_signal: torch.tensor, tensor_elements: torch.tensor, device: str):

        self.device = device
        self.predicted_signal = predicted_signal
        
        # Clean tensor elements BEFORE constructing diffusion tensors
        # Replace NaN/Inf with zeros (will result in isotropic tensor with small eigenvalues)
        self.tensor_elements = torch.where(
            torch.isfinite(tensor_elements),
            tensor_elements,
            torch.zeros_like(tensor_elements)
        )
        
        # Clamp tensor elements to reasonable diffusivity range
        # Dxx, Dyy, Dzz should be in [0, 0.005] mm^2/s (0-5 um^2/ms)
        # Off-diagonal elements can be negative but should be bounded
        self.tensor_elements[:, :3] = torch.clamp(self.tensor_elements[:, :3], min=0.0, max=0.005)  # Diagonal
        self.tensor_elements[:, 3:] = torch.clamp(self.tensor_elements[:, 3:], min=-0.005, max=0.005)  # Off-diagonal
        
        self.DTs = self._ten_elems_to_dt()
        
        # Ensure tensors are symmetric (critical for eigh)
        self.DTs = 0.5 * (self.DTs + self.DTs.transpose(-2, -1))
        
        # Use robust eigendecomposition that handles CUDA cusolver errors
        # This handles large batch sizes through intelligent GPU chunking
        self.evals, self.evecs = _robust_sym_eigh_3x3_batched(
            self.DTs,
            device=device,
            base_jitter=1e-7,
            max_jitter=1e-4,
            out_dtype=self.DTs.dtype,
            min_eigval=0.0001,  # 0.1 um^2/ms minimum eigenvalue
            max_batch_size=None  # Auto-detect safe batch size (200K for CUDA, 1M for CPU)
        )
        
        # Apply physiological constraints: diffusivities between 0.1-3.5 um^2/ms (0.0001-0.0035 mm^2/s)
        self.evals = torch.clamp(self.evals, min=0.0001, max=0.0035)
        return   

    def _ten_elems_to_dt(self):        
        DTs = torch.zeros(self.predicted_signal.shape[0], 3,3, device=self.device)
        DT_elements = {1: self.tensor_elements[:,0], # Dxx
                       4: self.tensor_elements[:,1], # Dyy
                       9: self.tensor_elements[:,2], # Dzz
                       2: self.tensor_elements[:,3], # Dxy
                       3: self.tensor_elements[:,4], # Dxz
                       6: self.tensor_elements[:,5]  # Dzy 
                       }   

        for i,j in product(range(3), repeat=2):
            DTs[:,i,j] = DT_elements[(i+1)*(j+1)] 
        return DTs

    @auto_attr 
    def fa(self):
         return torch.sqrt(torch.tensor([0.5], device=self.device)) * torch.sqrt((self.evals[:,0]-self.evals[:,1])**2 + (self.evals[:,1]-self.evals[:,2])**2 + (self.evals[:,2]-self.evals[:,0])**2) / torch.linalg.norm(self.evals, ord = 2, dim = 1)
    
    @auto_attr
    def ad(self):
        return 1e3 * self.evals[:,-1:]

    @auto_attr
    def rd(self):
        return 1e3 * 0.5*(self.evals[:,0] + self.evals[:,1])
    
    @auto_attr
    def adc(self):
        return 1e3 * 1/3 * torch.einsum('...ii', self.DTs)

    @auto_attr
    def eigen_directions(self):
        return self.evecs[:,:,-1]
    @auto_attr
    def eigen_frame(self):
        return self.evecs

    @auto_attr
    def eval_2(self):
        return 1e3 * self.evals[:, -2]
    
    @auto_attr
    def eval_3(self):
        return 1e3 * self.evals[:, -3]
    
    @auto_attr 
    def cfa(self):
        return torch.einsum('ij, i -> ij', torch.abs(self.eigen_directions), torch.clip(self.fa, 0, 1) )

class fit_model_wrapper(fit_diffusion_tensor_model):
    def __init__(self, predicted_signal: torch.tensor, tensor_elements: torch.tensor, model_type: str):
        if model_type == 'dti':
            fit_diffusion_tensor_model.__init__(self, predicted_signal, tensor_elements)
        pass 

# NODDI implementation

class NODDIModel:
    """
    NODDI (Neurite Orientation Dispersion and Density Imaging) model with ADAM optimization.
    
    Three-compartment biophysical model:
    1. Intra-cellular (IC): Sticks with Watson dispersion (restricted)
    2. Extra-cellular (EC): Hindered diffusion with tortuosity  
    3. Isotropic (ISO): Free water (CSF)
    
    Signal model:
    S/S0 = (1 - v_iso) * [(1 - v_ic) * S_ec + v_ic * S_ic] + v_iso * S_iso
    
    Parameters
    ----------
    bvals : torch.Tensor
        B-values in s/mm^2
    bvecs : torch.Tensor  
        Gradient directions (unit vectors), shape (n_volumes, 3)
    device : str
        'cpu' or 'cuda'
    
    References
    ----------
    .. [1] Zhang, H., Schneider, T., Wheeler-Kingshott, C.A., Alexander, D.C., 2012.
           NODDI: practical in vivo neurite orientation dispersion and density imaging
           of the human brain. Neuroimage 61(4), 1000-1016.
    """
    
    def __init__(self, bvals: torch.Tensor, bvecs: torch.Tensor, device='cpu'):
        self.bvals = bvals.to(device)
        self.bvecs = bvecs.to(device)
        self.bvecs = self.bvecs / (torch.linalg.norm(self.bvecs, dim=1, keepdim=True) + 1e-10)
        self.device = device
        
        # Fixed parameters (typical literature values); may be overridden in fit().
        self.d_ic_fixed = 1.7e-3  # mm^2/s = 1.7 um^2/ms (intra-cellular)
        self.d_iso_fixed = 3.0e-3  # mm^2/s = 3.0 um^2/ms (free water)
        
        # We do NOT use the precomputed Watson LUT for directional fitting.
        # Instead we use a differentiable numerical quadrature over the Watson ODF that
        # preserves dependence on the angle between gradient and mean direction.
        self.watson_lut = None
    
    def fit(self, Y: torch.Tensor, optimizer_args: dict = None, *, signal_normalization: str = 'auto'):
        """
        Fit NODDI model using ADAM optimization.
        
        Parameters
        ----------
        Y : torch.Tensor
            DWI signal, shape (n_voxels, n_volumes)
        optimizer_args : dict, optional
            Optimizer settings:
            - 'lr': learning rate (default 0.001)
            - 'epochs': number of iterations (default 500)
            - 'batch_size': voxels per batch (default 1000)
        signal_normalization : str, optional
            Signal normalization mode used to estimate a per-voxel scale (S0) and fit
            the attenuation-like signal. One of: auto | max | b0 | minb | none.
            For NODDI, 'auto' resolves to 'max'.
        
        Returns
        -------
        FitNODDIModel
            Fitted model with parameter maps
        """
        if optimizer_args is None:
            optimizer_args = {}
        
        # Default optimizer settings
        lr = optimizer_args.get('lr', 0.001)
        epochs = optimizer_args.get('epochs', 100)
        batch_size = optimizer_args.get('batch_size', None)
        
        # Optional NODDI hyperparameters
        d_ic = float(optimizer_args.get('d_ic', self.d_ic_fixed))
        d_iso = float(optimizer_args.get('d_iso', self.d_iso_fixed))
        use_tortuosity = bool(optimizer_args.get('use_tortuosity', True))
        
        self.d_ic_fixed = d_ic
        self.d_iso_fixed = d_iso
        
        # Ensure data lives on the same device as the model tensors.
        # This prevents boolean indexing and loss computation device mismatches.
        Y = Y.to(self.device)
        n_voxels = Y.shape[0]
        n_volumes = Y.shape[1]
        
        logging.info(f"Fitting NODDI on {n_voxels} voxels with {n_volumes} volumes")
        logging.info(f"Optimizer: ADAM, lr={lr}, epochs={epochs}")
        
        # Normalize signals using DBSIpy semantics (configurable).
        # This lets NODDI respect the configured S0 estimation method (e.g., b0), while
        # keeping the NODDI default ('auto') behavior as max normalization.
        _, Y_norm, scale, _, mode_used = normalize_signal(
            Y,
            self.bvals,
            mode=str(signal_normalization or 'auto'),
            engine='NODDI',
        )
        S0_est = scale[:, None]

        if mode_used == 'none':
            # NODDI's forward model predicts attenuation (S/S0). If the user disables
            # normalization, the optimization will fit raw signal units against an
            # attenuation model and results may be meaningless.
            logging.warning(
                "NODDI: signal_normalization='none' disables attenuation normalization; "
                "fit may be unstable or not meaningful. Consider 'b0' or 'max'."
            )
        
        # Initialize from DTI (rough estimates)
        initial_params = self._initialize_from_dti(Y_norm)
        
        # Create optimizer model
        model = NODDIOptimizer(
            n_voxels=n_voxels,
            initial_params=initial_params,
            bvals=self.bvals,
            bvecs=self.bvecs,
            watson_lut=self.watson_lut,
            d_ic=self.d_ic_fixed,
            d_iso=self.d_iso_fixed,
            use_tortuosity=use_tortuosity,
            device=self.device
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Auto batch sizing: larger batches reduce overhead substantially for NODDI.
        # If CUDA OOM occurs, back off until a batch fits.
        if isinstance(batch_size, str) and batch_size.strip().lower() == 'auto':
            batch_size = None
        if batch_size is None:
            # Cap auto-batch sizing to avoid overly aggressive memory usage.
            # Users can still override via optimizer_args['batch_size'] / INI noddi_batch_size.
            max_auto_batch_size = 10_000
            candidate = min(n_voxels, max_auto_batch_size)

            gpu_budget_bytes = None
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                gpu_budget_bytes = int(4 * 1024**3)

            def _probe(bs: int) -> bool:
                try:
                    with torch.no_grad():
                        peak_bytes = None
                        if gpu_budget_bytes is not None:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        if bs >= n_voxels:
                            voxel_idx = None
                            y_target = Y_norm
                        else:
                            voxel_idx = torch.arange(bs, device=self.device)
                            y_target = Y_norm[voxel_idx]
                        s_pred = model(voxel_idx=voxel_idx)
                        _ = torch.nn.functional.mse_loss(s_pred, y_target)

                        if gpu_budget_bytes is not None:
                            peak_bytes = int(torch.cuda.max_memory_allocated())

                    if gpu_budget_bytes is not None and peak_bytes is not None:
                        return peak_bytes <= gpu_budget_bytes
                    return True
                except RuntimeError as e:
                    msg = str(e).lower()
                    if 'out of memory' in msg or 'cuda' in msg and 'memory' in msg:
                        if str(self.device).startswith('cuda'):
                            torch.cuda.empty_cache()
                        return False
                    raise

            # If on CUDA, use one probe to estimate memory/voxel and scale candidate down
            # to meet the budget, then fall back to OOM backoff.
            if gpu_budget_bytes is not None:
                try:
                    probe_bs = int(max(256, min(candidate, 2048)))
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        voxel_idx = None if probe_bs >= n_voxels else torch.arange(probe_bs, device=self.device)
                        y_target = Y_norm if voxel_idx is None else Y_norm[voxel_idx]
                        s_pred = model(voxel_idx=voxel_idx)
                        _ = torch.nn.functional.mse_loss(s_pred, y_target)
                    peak = int(torch.cuda.max_memory_allocated())
                    if peak > 0:
                        bytes_per_voxel = peak / float(probe_bs)
                        # Keep a headroom margin for optimizer state, CUDA graph workspace, etc.
                        headroom = 0.85
                        scaled = int((gpu_budget_bytes * headroom) / max(1.0, bytes_per_voxel))
                        candidate = max(1, min(candidate, scaled))
                except Exception:
                    # If probing fails, keep the existing candidate and rely on OOM backoff.
                    pass

            bs = max(1, int(candidate))
            attempts = 0
            while bs >= 1:
                attempts += 1
                if _probe(bs):
                    batch_size = bs
                    break
                bs = max(1, bs // 2)

            logging.info(f"Auto batch_size selected: {int(batch_size)} (attempts={attempts})")

        # Minibatching over voxels to keep memory bounded for Watson quadrature.
        # We perform a full pass over all voxels each epoch (shuffled), which is
        # critical for per-voxel parameters to actually update beyond initialization.
        batch_size = int(batch_size) if batch_size is not None else n_voxels
        batch_size = max(1, min(batch_size, n_voxels))
        steps_per_epoch = int(np.ceil(n_voxels / batch_size))
        total_steps = epochs * steps_per_epoch
        logging.info(f"Minibatching: batch_size={batch_size}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 5, total_steps=total_steps
        )

        # Fit with progress bar
        model.train()
        pbar = make_progress_bar(total=total_steps, desc='NODDI', colour='cyan')

        # Efficient progress updating: avoid per-step tqdm overhead.
        # Target ~200 visual updates total (at most), but never less frequent than every 10 steps.
        pbar_update_interval = max(1, min(10, max(1, total_steps // 200)))
        pbar_pending = 0

        best_loss = float('inf')
        patience_counter = 0
        patience = optimizer_args.get('patience', 20)  # Aggressive early stopping
        min_delta = optimizer_args.get('min_delta', 1e-6)  # Minimum improvement threshold

        for epoch in range(epochs):
            # Shuffle voxels each epoch to ensure coverage.
            if batch_size == n_voxels:
                epoch_indices = None
            else:
                epoch_indices = torch.randperm(n_voxels, device=self.device)

            epoch_loss_sum = 0.0
            for step in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)

                if batch_size == n_voxels:
                    voxel_idx = None
                    y_target = Y_norm
                else:
                    start = step * batch_size
                    end = min((step + 1) * batch_size, n_voxels)
                    voxel_idx = epoch_indices[start:end]
                    y_target = Y_norm[voxel_idx]

                # Forward pass
                S_pred = model(voxel_idx=voxel_idx)

                # MSE loss
                loss = torch.nn.functional.mse_loss(S_pred, y_target)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss_sum += float(loss.item())

                # Update progress bar (efficiently)
                pbar_pending += 1
                if (pbar.n + pbar_pending) % pbar_update_interval == 0 or (
                    epoch == epochs - 1 and step == steps_per_epoch - 1
                ):
                    pbar.update(pbar_pending)
                    pbar_pending = 0
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            epoch_loss = epoch_loss_sum / max(1, steps_per_epoch)

            # Early stopping (based on epoch-average loss, not a single minibatch)
            if epoch_loss < (best_loss - min_delta):
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} (patience={patience}, min_delta={min_delta})")
                break

        # Flush any remaining progress updates.
        if pbar_pending:
            pbar.update(pbar_pending)
            pbar_pending = 0

        pbar.close()
        
        # Extract final parameters
        model.eval()
        with torch.no_grad():
            final_params = {
                'v_ic': model.v_ic.squeeze(1),
                'v_iso': model.v_iso.squeeze(1),
                'od': model.od.squeeze(1),
                'v_ec': model.v_ec.squeeze(1),
                'd_ic': torch.full((n_voxels,), self.d_ic_fixed, device=self.device),
                'd_ec_par': model.d_ec_par_clamped.squeeze(1),
                'd_ec_perp': model.d_ec_perp.squeeze(1),
                'd_iso': torch.full((n_voxels,), self.d_iso_fixed, device=self.device),
                'theta': model.theta.squeeze(1),
                'phi': model.phi.squeeze(1),
                'kappa': model.kappa.squeeze(1),
                'S0': S0_est.squeeze(1)
            }
        
        logging.info(f"NODDI fitting complete. Final loss: {best_loss:.6f}")
        
        return FitNODDIModel(final_params, self.bvals, self.bvecs)
    
    def _initialize_from_dti(self, Y: torch.Tensor) -> dict:
        """
        Initialize NODDI parameters from DTI fit.
        
        Uses rough approximations:
        - NDI ~ FA
        - ODI ~ (1 - FA)
        - FISO ~ 0.05
        """
        # Quick DTI fit (OLS on low b-values)
        try:
            b_mask = self.bvals < 1500
            if b_mask.sum() < 7:
                b_mask = torch.ones_like(self.bvals, dtype=torch.bool)
            
            dti_model = diffusion_tensor_model(
                self.bvals[b_mask],
                self.bvecs[b_mask],
                device=self.device
            )
            dti_fit = dti_model.fit(Y[:, b_mask], fit_method='OLS')
            
            fa = dti_fit.fa
            adc = dti_fit.adc * 1e-3  # Convert from um^2/ms back to mm^2/s for initialization
            evecs = dti_fit.eigen_frame
            
            # Detect CSF-like voxels: low FA + high ADC
            # CSF: FA < 0.2, ADC > 2.5e-3 mm^2/s (2.5 um^2/ms)
            is_csf = (fa < 0.2) & (adc > 2.5e-3)
            
            # Initialize NDI from FA (but low in CSF)
            v_ic_init = torch.clamp(fa, 0.01, 0.9)
            v_ic_init[is_csf] = 0.01  # Minimal neurites in CSF
            
            # Initialize ODI (high dispersion in CSF and GM)
            od_init = torch.clamp(1.0 - fa, 0.01, 0.99)
            od_init[is_csf] = 0.99  # Maximum dispersion in CSF
            
            # Initialize FISO (high in CSF, low elsewhere)
            v_iso_init = torch.full_like(fa, 0.05)
            v_iso_init[is_csf] = 0.9  # High free water in CSF
            
            # Initialize EC diffusivity (constrained to physiological range)
            # Typical brain: 1.0-2.0 um^2/ms (0.001-0.002 mm^2/s)
            d_ec_par_init = torch.clamp(adc, 0.001, 0.002)
            
            # Fiber direction from principal eigenvector
            primary_evec = evecs[:, :, -1]  # Shape (n_voxels, 3)
            theta_init = torch.acos(torch.clamp(primary_evec[:, 2], -1, 1))
            phi_init = torch.atan2(primary_evec[:, 1], primary_evec[:, 0])
            
        except Exception as e:
            logging.warning(f"DTI initialization failed: {e}. Using defaults.")
            n_voxels = Y.shape[0]
            v_ic_init = torch.full((n_voxels,), 0.5, device=self.device)
            od_init = torch.full((n_voxels,), 0.2, device=self.device)
            v_iso_init = torch.full((n_voxels,), 0.05, device=self.device)
            d_ec_par_init = torch.full((n_voxels,), 1.7e-3, device=self.device)
            theta_init = torch.full((n_voxels,), np.pi/2, device=self.device)
            phi_init = torch.zeros(n_voxels, device=self.device)
        
        return {
            'v_ic': v_ic_init,
            'od': od_init,
            'v_iso': v_iso_init,
            'd_ec_par': d_ec_par_init,
            'theta': theta_init,
            'phi': phi_init
        }


class NODDIOptimizer(torch.nn.Module):
    """
    ADAM-optimizable NODDI model parameters.
    
    Uses transformed parameters for unconstrained optimization:
    - Fractions [0,1]: logit transform
    - Diffusivities: log transform
    - Angles: raw (handled by modulo in forward)
    """
    
    def __init__(self, n_voxels, initial_params, bvals, bvecs, watson_lut,
                 d_ic, d_iso, use_tortuosity: bool = True, device='cpu'):
        super().__init__()
        self.device = device
        self.bvals = bvals
        self.bvecs = bvecs
        self.watson_lut = watson_lut
        self.d_ic_fixed = d_ic
        self.d_iso_fixed = d_iso
        self.use_tortuosity = use_tortuosity

        # Differentiable numerical quadrature over Watson ODF.
        # We integrate over solid angle using a tensor-product rule:
        #   dOmega = dphi dmu, where mu = cos(theta) in [-1, 1], phi in [0, 2pi).
        # The Watson density is ~ exp(kappa * mu^2), and is normalized numerically per-voxel.
        self._init_watson_quadrature(n_mu=10, n_phi=20)
        
        # Learnable parameters (use inverse transforms for initialization)
        def logit(p):
            return torch.log(p / (1 - p + 1e-10) + 1e-10)
        
        self.v_ic_logit = torch.nn.Parameter(logit(initial_params['v_ic']).unsqueeze(1))
        self.v_iso_logit = torch.nn.Parameter(logit(initial_params['v_iso']).unsqueeze(1))
        self.od_logit = torch.nn.Parameter(logit(initial_params['od']).unsqueeze(1))
        # Extra-cellular parallel diffusivity is bounded to a physiological range via a
        # smooth (sigmoid) transform to avoid hard-clamp gradients going to zero.
        # Physiological range: 0.5-3.0 um^2/ms (0.0005-0.0030 mm^2/s).
        self._d_ec_par_min = 0.0005
        self._d_ec_par_max = 0.0030
        span = self._d_ec_par_max - self._d_ec_par_min

        def inv_sigmoid_range(d):
            p = (d - self._d_ec_par_min) / (span + 1e-12)
            p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
            return torch.log(p / (1.0 - p))

        self.d_ec_par_logit = torch.nn.Parameter(inv_sigmoid_range(initial_params['d_ec_par']).unsqueeze(1))
        self.theta = torch.nn.Parameter(initial_params['theta'].unsqueeze(1))
        self.phi = torch.nn.Parameter(initial_params['phi'].unsqueeze(1))

        if not self.use_tortuosity:
            # Optional independent d_ec_perp (bounded) when tortuosity is disabled.
            # Physiological range: 0.1-1.5 um^2/ms (0.0001-0.0015 mm^2/s).
            self._d_ec_perp_min = 0.0001
            self._d_ec_perp_max = 0.0015
            span_perp = self._d_ec_perp_max - self._d_ec_perp_min

            def inv_sigmoid_range_perp(d):
                p = (d - self._d_ec_perp_min) / (span_perp + 1e-12)
                p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
                return torch.log(p / (1.0 - p))

            d_ec_perp_init = initial_params['d_ec_par'] * (1 - initial_params['v_ic'])
            self.d_ec_perp_logit = torch.nn.Parameter(inv_sigmoid_range_perp(d_ec_perp_init).unsqueeze(1))
    
    @property
    def v_ic(self):
        """Intra-cellular volume fraction [0, 1]"""
        return torch.sigmoid(self.v_ic_logit)
    
    @property
    def v_iso(self):
        """Isotropic volume fraction [0, 1]"""
        return torch.sigmoid(self.v_iso_logit)
    
    @property
    def od(self):
        """Orientation dispersion index [0, 1]"""
        return torch.sigmoid(self.od_logit)
    
    @property
    def v_ec(self):
        """Extra-cellular volume fraction (computed)"""
        return (1 - self.v_iso) * (1 - self.v_ic)
    
    @property
    def d_ec_par_clamped(self):
        """Extra-cellular parallel diffusivity with physiological constraints.

        Smoothly constrained to 0.5-3.0 um^2/ms (0.0005-0.0030 mm^2/s).
        """
        span = self._d_ec_par_max - self._d_ec_par_min
        return self._d_ec_par_min + torch.sigmoid(self.d_ec_par_logit) * span
    
    @property
    def d_ec_perp(self):
        """Extra-cellular perpendicular diffusivity (tortuosity)"""
        if self.use_tortuosity:
            return self.d_ec_par_clamped * (1 - self.v_ic)
        span_perp = self._d_ec_perp_max - self._d_ec_perp_min
        return self._d_ec_perp_min + torch.sigmoid(self.d_ec_perp_logit) * span_perp
    
    @property
    def kappa(self):
        """Watson concentration parameter"""
        from dbsipy.noddi.watson_lut import od_to_kappa
        return od_to_kappa(self.od)
    
    @property
    def fiber_direction(self):
        """Fiber direction unit vector from angles"""
        theta_wrapped = torch.remainder(self.theta, np.pi)
        phi_wrapped = torch.remainder(self.phi, 2*np.pi)
        
        nx = torch.sin(theta_wrapped) * torch.cos(phi_wrapped)
        ny = torch.sin(theta_wrapped) * torch.sin(phi_wrapped)
        nz = torch.cos(theta_wrapped)
        
        return torch.cat([nx, ny, nz], dim=1)  # (n_voxels, 3)
    
    def forward(self, voxel_idx: torch.Tensor | None = None):
        """
        Predict NODDI signal.
        
        Returns
        -------
        S : torch.Tensor, shape (n_voxels, n_volumes)
        """
        # Compute shared Watson quadrature terms once per forward call.
        # This is a major performance win: both IC and EC compartments depend on
        # the same (g dot u)^2 grid and the same Watson weights.
        kappa = self.kappa if voxel_idx is None else self.kappa[voxel_idx]  # (n_vox, 1)
        n = self.fiber_direction if voxel_idx is None else self.fiber_direction[voxel_idx]  # (n_vox, 3)
        g_local = self._gradients_in_local_frame(n)  # (n_vox, n_vol, 3)
        cos2 = self._cos2_between_g_and_u(g_local)  # (n_vox, n_vol, n_orient)

        # Watson normalized weights: (n_vox, 1, n_orient)
        base_w = self._watson_base_w  # (n_orient,)
        mu2 = self._watson_mu2  # (n_orient,)
        w = base_w.view(1, 1, -1) * torch.exp(kappa.view(-1, 1, 1) * mu2.view(1, 1, -1))
        denom = torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-20)
        w_norm = w / denom

        b = self.bvals.view(1, -1, 1)

        # IC: dispersed stick
        stick = torch.exp(-b * float(self.d_ic_fixed) * cos2)
        S_ic = (w_norm * stick).sum(dim=-1)

        # EC: dispersed zeppelin
        d_par = self.d_ec_par_clamped if voxel_idx is None else self.d_ec_par_clamped[voxel_idx]
        d_perp = self.d_ec_perp if voxel_idx is None else self.d_ec_perp[voxel_idx]
        d_perp_v = d_perp.view(-1, 1, 1)
        d_par_v = d_par.view(-1, 1, 1)
        D_app = d_perp_v + (d_par_v - d_perp_v) * cos2
        zepp = torch.exp(-b * D_app)
        S_ec = (w_norm * zepp).sum(dim=-1)

        # ISO: free water
        S_iso = self._isotropic_signal(voxel_idx=voxel_idx)

        v_iso = self.v_iso if voxel_idx is None else self.v_iso[voxel_idx]
        v_ic = self.v_ic if voxel_idx is None else self.v_ic[voxel_idx]

        # Combine according to NODDI model
        return (1 - v_iso) * ((1 - v_ic) * S_ec + v_ic * S_ic) + v_iso * S_iso
    
    def _stick_signal_dispersed(self, *, voxel_idx: torch.Tensor | None = None):
        """
        Intra-cellular compartment: stick with Watson dispersion.
        
        For dispersed sticks, the signal is:
        S_ic(b, kappa) = exp(-b * d_ic) * W(b * d_ic, kappa)
        
        where W is the Watson integral (from LUT) that accounts for
        orientation dispersion around the mean direction.
        """
        kappa = self.kappa if voxel_idx is None else self.kappa[voxel_idx]  # (n_voxels, 1)
        n = self.fiber_direction if voxel_idx is None else self.fiber_direction[voxel_idx]  # (n_voxels, 3)

        # Compute g in the local frame of each voxel's mean direction.
        g_local = self._gradients_in_local_frame(n)  # (n_voxels, n_volumes, 3)
        cos2 = self._cos2_between_g_and_u(g_local)  # (n_voxels, n_volumes, n_orient)

        b = self.bvals.view(1, -1, 1)
        stick = torch.exp(-b * float(self.d_ic_fixed) * cos2)
        return self._watson_weighted_average(stick, kappa)
    
    def _hindered_signal(self, *, voxel_idx: torch.Tensor | None = None):
        """
        Extra-cellular compartment: cylindrically symmetric hindered diffusion.
        """
        kappa = self.kappa if voxel_idx is None else self.kappa[voxel_idx]  # (n_voxels, 1)
        n = self.fiber_direction if voxel_idx is None else self.fiber_direction[voxel_idx]  # (n_voxels, 3)

        d_par = self.d_ec_par_clamped if voxel_idx is None else self.d_ec_par_clamped[voxel_idx]
        d_perp = self.d_ec_perp if voxel_idx is None else self.d_ec_perp[voxel_idx]

        g_local = self._gradients_in_local_frame(n)  # (n_voxels, n_volumes, 3)
        cos2 = self._cos2_between_g_and_u(g_local)  # (n_voxels, n_volumes, n_orient)

        # Zeppelin signal: exp(-b * (d_perp + (d_par-d_perp)*cos^2))
        b = self.bvals.view(1, -1, 1)
        d_perp_v = d_perp.view(-1, 1, 1)
        d_par_v = d_par.view(-1, 1, 1)
        D_app = d_perp_v + (d_par_v - d_perp_v) * cos2
        zepp = torch.exp(-b * D_app)
        return self._watson_weighted_average(zepp, kappa)
    
    def _isotropic_signal(self, *, voxel_idx: torch.Tensor | None = None):
        """
        Isotropic compartment: free water.
        """
        S_iso = torch.exp(-self.bvals.unsqueeze(0) * self.d_iso_fixed)
        n_vox = self.v_iso.shape[0] if voxel_idx is None else int(voxel_idx.numel())
        return S_iso.expand(n_vox, -1)

    def _init_watson_quadrature(self, *, n_mu: int, n_phi: int) -> None:
        """Precompute quadrature points on the sphere for Watson ODF integration."""
        import numpy as _np

        mu, w_mu = _np.polynomial.legendre.leggauss(int(n_mu))  # mu in [-1, 1]
        phi = _np.linspace(0.0, 2.0 * _np.pi, int(n_phi), endpoint=False)
        w_phi = (2.0 * _np.pi) / float(n_phi)

        mu = mu.astype(_np.float64)
        w_mu = w_mu.astype(_np.float64)
        phi = phi.astype(_np.float64)

        # Tensor-product grid
        mu_grid = _np.repeat(mu[:, None], n_phi, axis=1)
        phi_grid = _np.repeat(phi[None, :], n_mu, axis=0)

        sin_theta = _np.sqrt(_np.clip(1.0 - mu_grid**2, 0.0, 1.0))
        ux = sin_theta * _np.cos(phi_grid)
        uy = sin_theta * _np.sin(phi_grid)
        uz = mu_grid

        u_local = _np.stack([ux, uy, uz], axis=-1).reshape(-1, 3)
        base_w = (_np.repeat(w_mu[:, None], n_phi, axis=1) * w_phi).reshape(-1)
        mu2 = (uz.reshape(-1) ** 2)

        u_local_t = torch.from_numpy(u_local).to(device=self.device, dtype=torch.float32)
        base_w_t = torch.from_numpy(base_w).to(device=self.device, dtype=torch.float32)
        mu2_t = torch.from_numpy(mu2).to(device=self.device, dtype=torch.float32)

        # Register as buffers so they move with .to(device) and are saved with state_dict.
        self.register_buffer("_watson_u_local", u_local_t, persistent=False)
        self.register_buffer("_watson_base_w", base_w_t, persistent=False)
        self.register_buffer("_watson_mu2", mu2_t, persistent=False)

    def _gradients_in_local_frame(self, n: torch.Tensor) -> torch.Tensor:
        """Return gradient directions expressed in each voxel's local (e1,e2,n) frame."""
        # Build a stable orthonormal basis per voxel.
        n = n / (torch.linalg.norm(n, dim=1, keepdim=True) + 1e-12)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).view(1, 3)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).view(1, 3)
        use_z = (torch.abs(n[:, 2]) < 0.9).view(-1, 1)
        a = torch.where(use_z, z_axis.expand_as(n), x_axis.expand_as(n))
        e1 = torch.linalg.cross(a, n)
        e1 = e1 / (torch.linalg.norm(e1, dim=1, keepdim=True) + 1e-12)
        e2 = torch.linalg.cross(n, e1)

        # Project each gradient into the local basis: (g dot e1, g dot e2, g dot n)
        g = self.bvecs  # (n_volumes, 3)
        g1 = torch.einsum('vj,nj->nv', g, e1)
        g2 = torch.einsum('vj,nj->nv', g, e2)
        g3 = torch.einsum('vj,nj->nv', g, n)
        return torch.stack([g1, g2, g3], dim=-1)

    def _cos2_between_g_and_u(self, g_local: torch.Tensor) -> torch.Tensor:
        """Compute (g dot u)^2 for all local quadrature orientations u."""
        # g_local: (n_vox, n_vol, 3)
        n_vox, n_vol, _ = g_local.shape
        u = self._watson_u_local  # (n_orient, 3)
        # (n_vox*n_vol, 3) @ (3, n_orient) -> (n_vox*n_vol, n_orient)
        g_flat = g_local.reshape(n_vox * n_vol, 3)
        g_dot_u = g_flat @ u.t()
        g_dot_u = g_dot_u.reshape(n_vox, n_vol, -1)
        return g_dot_u * g_dot_u

    def _watson_weighted_average(self, signal_u: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Compute orientation-averaged signal under Watson ODF.

        signal_u: (n_vox, n_vol, n_orient)
        kappa: (n_vox, 1)
        """
        base_w = self._watson_base_w  # (n_orient,)
        mu2 = self._watson_mu2  # (n_orient,)
        # Watson weights per voxel (unnormalized)
        w = base_w.view(1, 1, -1) * torch.exp(kappa.view(-1, 1, 1) * mu2.view(1, 1, -1))
        denom = torch.clamp(w.sum(dim=-1), min=1e-20)
        num = (w * signal_u).sum(dim=-1)
        return num / denom


class FitNODDIModel:
    """
    Container for NODDI fit results with lazy-evaluated derived metrics.
    
    Attributes
    ----------
    model_params : dict
        Fitted parameter values
    bvals : torch.Tensor
        B-values used for fitting
    bvecs : torch.Tensor
        Gradient directions
    """
    
    def __init__(self, model_params, bvals, bvecs):
        self.model_params = model_params
        self.bvals = bvals
        self.bvecs = bvecs
    
    @auto_attr
    def ndi(self):
        """
        Neurite Density Index (NDI).
        
        Intra-cellular volume fraction (ICVF). Higher in dense white matter.
        Defined as ICVF = (1 - FISO) * v_ic.
        Typical brain WM: 0.5-0.7
        """
        # In standard NODDI outputs, the intra-cellular volume fraction (ICVF) is:
        #   ICVF = (1 - v_iso) * v_ic
        # where v_ic is the intra-cellular fraction of the non-isotropic compartment.
        v_ic = self.model_params['v_ic']
        v_iso = self.model_params['v_iso']
        return (1 - v_iso) * v_ic
    
    @auto_attr
    def odi(self):
        """
        Orientation Dispersion Index (ODI).
        
        Fiber dispersion around mean direction.
        0 = perfectly aligned, 1 = isotropic
        Typical brain WM: 0.1-0.3
        """
        return self.model_params['od']
    
    @auto_attr
    def fiso(self):
        """
        Free water fraction.
        
        CSF/free water contamination.
        Typical brain WM: 0.0-0.1
        """
        return self.model_params['v_iso']
    
    @auto_attr
    def fec(self):
        """
        Extra-cellular fraction.
        
        Hindered diffusion compartment.
        """
        return self.model_params['v_ec']
    
    @auto_attr
    def kappa(self):
        """
        Watson concentration parameter.
        
        High kappa = low dispersion (aligned fibers)
        Low kappa = high dispersion (crossing/fanning)
        """
        return self.model_params['kappa']
    
    @auto_attr
    def fiber_direction(self):
        """
        Principal fiber orientation (unit vector).
        
        Returns
        -------
        direction : torch.Tensor, shape (n_voxels, 3)
            Unit vectors (x, y, z)
        """
        theta = self.model_params['theta']
        phi = self.model_params['phi']
        
        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)
        
        return torch.stack([nx, ny, nz], dim=-1)
    
    @auto_attr
    def d_ic(self):
        """Intra-cellular diffusivity (um^2/ms)"""
        return self.model_params['d_ic'] * 1e3
    
    @auto_attr
    def d_ec_par(self):
        """Extra-cellular parallel diffusivity (um^2/ms)"""
        return self.model_params['d_ec_par'] * 1e3
    
    @auto_attr
    def d_ec_perp(self):
        """Extra-cellular perpendicular diffusivity (um^2/ms)"""
        return self.model_params['d_ec_perp'] * 1e3
    
    @auto_attr
    def d_iso(self):
        """Isotropic diffusivity (um^2/ms)"""
        return self.model_params['d_iso'] * 1e3
    
    @auto_attr
    def fiber_direction_cfa(self):
        """
        Color-coded fiber direction weighted by NDI.
        
        Returns RGB values (0-1 range) for visualization:
        - Red = |x| component
        - Green = |y| component  
        - Blue = |z| component
        - Brightness = NDI
        
        Returns
        -------
        cfa : torch.Tensor, shape (n_voxels, 3)
            RGB color encoding of fiber orientation
        """
        # Weight by NDI/ICVF (0-1 range)
        ndi_weight = torch.clamp(self.ndi, 0, 1)
        return torch.einsum('ij, i -> ij', torch.abs(self.fiber_direction), ndi_weight)
