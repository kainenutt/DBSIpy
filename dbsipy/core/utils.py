import torch
import nibabel as nb 
import os 
from typing import List, Tuple
import numpy as np
import re
import logging


MIN_POSITIVE_SIGNAL = 1.0e-6


def legacy_fiber_save_name(pmap_name: str, max_fibers: int) -> str:
    """Convert internal fiber map keys to legacy-compatible on-disk names.

    Internal convention:
      - fiber_0d_adc, fiber_0d_fa, fiber_0d_IA_adc, ...

    Saved filenames:
      - max_fibers <= 1: fiber_adc, fiber_fa, fiber_IA_adc, ...
      - max_fibers  > 1: fiber_01_adc, fiber_01_fa, fiber_01_IA_adc, ...

    Non-fiber names are returned unchanged.
    """
    if not isinstance(pmap_name, str):
        return pmap_name

    try:
        max_fibers_i = int(max_fibers)
    except Exception:
        max_fibers_i = 1

    # Internal fiber keys are of the form: fiber_{idx}d_{suffix}
    m = re.match(r'^fiber_(\d+)d_(.+)$', pmap_name)
    if m is None:
        return pmap_name

    fiber_idx = int(m.group(1))
    suffix = m.group(2)

    if max_fibers_i <= 1:
        return f"fiber_{suffix}"
    return f"fiber_{fiber_idx + 1:02d}_{suffix}"

class ParamStoreDict:
    r"""
    Global key-value storage for DBSI-derived parameters. 

    References
    ----------
    ..[1] Eli Bingham et. al., "Pyro: Deep Universal Probabilistic Programming. Journal of Machine Learning Research", 2018.
    """
    def __init__(self):
        """
        initialize ParamStore data structures
        """
        self._param_maps = {}         
     
    def clear(self):
        """
        Clear the ParamStore
        """
        self._param_maps = {}
       

    def items(self):
        """
        Iterate over ``(name, paramter_map)`` pairs. 
        """
        for name in self._param_maps.keys():
            yield name, self._param_maps[name]

    def keys(self):
        """
        Iterate over param names.
        """
        return self._param_maps.keys()

    def values(self):
        """
        Iterate over constrained parameter values.
        """
        for (name, parameter_map) in self._param_maps.items():
            yield parameter_map

    def __len__(self):
        return len(self._param_maps.keys())

    def __contains__(self, name):
        return name in self._param_maps

    def __iter__(self):
        """
        Iterate over param names.
        """
        return iter(self._param_maps.keys())

    def __delitem__(self, name):
        """
        Remove a parameter from the param store.
        """
        self._param_maps.pop(name)
       
    def __getitem__(self, name):
        """
        Get the *constrained* value of a named parameter.
        """
        return self._param_maps[name]

    def get(self, name, default=None):
        """Dict-like getter for compatibility with standard mappings."""
        return self._param_maps.get(name, default)

    def __setitem__(self, name, value):
        """
        Set the parameter map to a value. 
        """     
        self._param_maps[name] = value


class parameter_map:
    def __init__(self, pmap_name: str, 
                 pmap_shapes: List[tuple], 
                 mask: torch.FloatTensor, 
                 device = 'cpu',
                 header = None,
                 affine = None) -> None:
        self.DEVICE = device
        self.pmap_name  = pmap_name
        self.pmap_linear_shape = pmap_shapes[0]
        self.pmap_spatial_shape = pmap_shapes[1]
        self.pmap = torch.zeros(self.pmap_linear_shape, device = self.DEVICE)
        if isinstance(mask, torch.Tensor):
            self.mask = mask.to(self.DEVICE).bool()
        else:
            self.mask = torch.from_numpy(mask).to(self.DEVICE).bool()

        self.inherited_header = header
        self.affine = affine


        pass

    def _to_spatial(self):
        temp_map = torch.zeros(self.pmap_spatial_shape, device = self.DEVICE)    
        temp_map[self.mask.to(self.DEVICE)] = torch.squeeze(self.pmap.to(self.DEVICE))
        return temp_map
    
    def _prepare_for_save(self):
        # GPU -> host transfer happens here when the backing tensor lives on CUDA.
        # Keep an aggregate timing in the pipeline timing sink (if present).
        import time

        t = self._to_spatial()

        sink = None
        try:
            sink = getattr(self, '_timings_sink', None)
        except Exception:
            sink = None

        # Only count D2H when data is actually on CUDA.
        if isinstance(t, torch.Tensor) and bool(getattr(t, 'is_cuda', False)) and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            t0 = time.perf_counter()
            t_cpu = t.to('cpu')
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            dt = float(time.perf_counter() - t0)
            try:
                if isinstance(sink, dict):
                    sink['d2h_save_s'] = float(sink.get('d2h_save_s', 0.0) or 0.0) + dt
            except Exception:
                pass
            return t_cpu.detach().numpy()

        return t.to('cpu').detach().numpy()
    
    def _save(self, path):
        os.makedirs(path, exist_ok=True)

        data = self._prepare_for_save()
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32, copy=False)

            nan_count = int(np.isnan(data).sum())
            inf_count = int(np.isinf(data).sum())
            if (nan_count > 0) or (inf_count > 0):
                msg = (
                    f"Non-finite values detected while saving '{self.pmap_name}': "
                    f"nan={nan_count}, inf={inf_count}"
                )
                if os.environ.get('DBSIPY_STRICT', '0') == '1':
                    raise RuntimeError(msg)
                # Sanitization is expected to occasionally occur for unstable fits.
                # Keep the information available for debugging without spamming users.
                logging.debug(msg + "; sanitizing to 0.0")

            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        affine = self.affine
        if affine is None:
            affine = np.eye(4)

        img = nb.nifti1.Nifti1Image(data, affine=affine)

        # Copy only safe, shape-agnostic header metadata from the source DWI.
        # Copying the full header can fail because the source is 4D (DWI) while
        # parameter maps are typically 3D or 4D-with-small-last-dim.
        if self.inherited_header is not None:
            try:
                zooms = self.inherited_header.get_zooms()
                if zooms is not None and len(zooms) >= 3:
                    img.header.set_zooms(tuple(zooms[:3]) + ((1.0,) if img.ndim == 4 else ()))
            except Exception:
                pass
            for key in ("xyzt_units", "qform_code", "sform_code", "intent_code", "intent_name", "descrip", "aux_file"):
                try:
                    img.header[key] = self.inherited_header[key]
                except Exception:
                    pass

        out_path = os.path.join(path, self.pmap_name + '.nii.gz')
        if (os.environ.get('DBSIPY_IO_DIAGNOSTICS', '0') == '1') or (os.environ.get('DBSIPY_DIAGNOSTICS', '0') == '1'):
            try:
                finite = np.isfinite(data)
                vmin = float(np.min(data[finite])) if finite.any() else float('nan')
                vmax = float(np.max(data[finite])) if finite.any() else float('nan')
                nb_nan = int(np.isnan(data).sum())
                nb_inf = int(np.isinf(data).sum())
                logging.info(
                    f"DBSI IO: saving {self.pmap_name} shape={tuple(data.shape)} dtype={data.dtype} "
                    f"range=[{vmin:.6g},{vmax:.6g}] nan={nb_nan} inf={nb_inf} -> {out_path}"
                )
            except Exception:
                pass
        try:
            nb.save(img, out_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save NIfTI map '{self.pmap_name}' to '{out_path}': {e}")

    def _gradient(self):
        image = self._to_spatial()
        ndim  = image.ndim 
        g = torch.zeros((image.ndim, ) + image.shape, dtype = image.dtype)
        slices_g = [slice(None), ] * (ndim +1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0,-1)
            slices_g[0]    = ax
            g[tuple(slices_g)] = torch.diff(image, axis = ax)
            slices_g[ax+1] = slice(None)
        g /= g.numel()              
        return g 
    
def split_data(dwi, batch_size: int) -> Tuple[List[torch.FloatTensor], List[Tuple]]:
    """Split DWI data into batches for memory-efficient processing.
    
    This function divides the DWI tensor into smaller batches to prevent out-of-memory
    errors during GPU/CPU processing. Used by both CUDA and multiprocessing backends.
    
    Parameters
    ----------
    dwi : torch.FloatTensor
        Diffusion-weighted imaging data with shape (n_voxels, n_directions).
    batch_size : int
        Maximum number of voxels per batch. Calculated based on available GPU/CPU memory.
    
    Returns
    -------
    tensor_list : List[torch.FloatTensor]
        List of batched DWI tensors, each with shape (batch_size, n_directions).
    index_list : List[List[int]]
        List of voxel indices corresponding to each batch.
    n_batches : int
        Total number of batches created.
    
    Notes
    -----
    Batch size is typically calculated as:
    ``batch_size = int(0.4 * TOTAL_MEMORY / memory_per_voxel)``
    
    See Also
    --------
    CUDABackend : Uses this function with GPU memory constraints
    MultiprocessingBackend : Uses this function with CPU memory constraints
    """

    tensor_list = []
    index_list  = []

    n_voxels = int(dwi.shape[0])
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    # Ceil division: exact number of non-empty batches.
    n_batches = max((n_voxels + batch_size - 1) // batch_size, 1)

    for ii in range(n_batches):
        start = ii * batch_size
        end = min((ii + 1) * batch_size, n_voxels)
        tensor_list.append(dwi[start:end])
        index_list.append(list(range(start, end)))

    return tensor_list, index_list, n_batches

def HouseHolder_evec_2_eframe(n, device = 'cpu'):
    r"""Compute the Frenet frame (orthonormal basis) from a normal vector using Householder reflection.
    
    Given a normal vector n (fiber direction from diffusion tensor), this function constructs
    a complete orthonormal coordinate frame (normal, tangent, binormal) using the Householder
    transformation. This is essential for rotating diffusion tensors into the fiber reference frame.
    
    Parameters
    ----------
    n : torch.FloatTensor
        Normal vector(s) with shape (..., 3). Typically the principal eigenvector from DTI.
        Must be unit vectors (|n| = 1).
    device : str, optional
        PyTorch device ('cpu' or 'cuda'). Default: 'cpu'.
    
    Returns
    -------
    frame : torch.FloatTensor
        Orthonormal frame with shape (..., 3, 3), where:
        - frame[..., :, 0] = normal (input n)
        - frame[..., :, 1] = tangent (perpendicular to n)
        - frame[..., :, 2] = binormal (perpendicular to both n and tangent)
    
    Notes
    -----
    The Householder transformation is computed as:
    
    .. math::
        H = I - 2 \frac{hh^T}{h^T h}
    
    where h is constructed to maximize numerical stability by choosing the larger
    of (n[0]+1) and (n[0]-1) for the first component.
    
    This function is used during Step 2 fitting to transform diffusion tensors from
    the laboratory frame to the fiber-aligned frame for axial/radial decomposition.
    
    References
    ----------
    .. [1] Golub & Van Loan, "Matrix Computations", 4th Ed., Algorithm 5.1.1
    
    See Also
    --------
    diffusion_tensor_model : Uses this for tensor decomposition in fiber frame
    """

    n = n.to(device)

    # Normalize n safely.
    n_norm = torch.linalg.norm(n, dim=-1, keepdim=True)
    n_norm = torch.clamp(n_norm, min=1e-12)
    normal = n / n_norm

    # Pick a reference axis not (anti-)parallel to normal.
    # If |nz| is large, use y-axis; otherwise use z-axis.
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=normal.dtype)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=normal.dtype)
    use_y = (torch.abs(normal[..., 2]) > 0.9)[..., None]
    ref = torch.where(use_y, y_axis, z_axis)

    tangent = torch.linalg.cross(ref.expand_as(normal), normal, dim=-1)
    t_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    t_norm = torch.clamp(t_norm, min=1e-12)
    tangent = tangent / t_norm

    # Right-handed binormal.
    binormal = torch.linalg.cross(normal, tangent, dim=-1)
    b_norm = torch.linalg.norm(binormal, dim=-1, keepdim=True)
    b_norm = torch.clamp(b_norm, min=1e-12)
    binormal = binormal / b_norm

    return torch.stack([normal, tangent, binormal], dim=3)
