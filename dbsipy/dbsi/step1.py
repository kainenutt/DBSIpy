from __future__ import annotations

import time
from typing import Any, Dict

import torch

from dbsipy.core import utils
from dbsipy.nn import leastsquares
from dbsipy.misc.models.Linear_Models import diffusion_tensor_model as DiffusionTensorModel
import dbsipy.misc.group_it as group_it


class DiffusionBasisFunctionModel:
    r"""Diffusion Basis Function Model.

    This model aims to decompose the dMRI signal to estimate the intra-voxel fiber
    geometry of white matter tissue.

    Parameters
    ----------
    configuration:
        DBSI hyperparameters (typically from the config.ini file).

    References
    ----------
    .. [1] A. Ramirez-Manzanares, et. al., "Diffusion Basis Functions Decomposition for Estimating White Matter Intravoxel Fiber Geometry,"
           IEEE Transactions on Medical Imaging, vol. 26, no. 8, pp. 1091-1102, Aug. 2007, doi: 10.1109/TMI.2007.900461.
    """

    def __init__(self, configuration: Any) -> None:
        self.configuration = configuration

    def _prepare_forward_model(self, bvecs: torch.Tensor, bvals: torch.Tensor) -> torch.Tensor:
        A = torch.empty(
            size=(bvecs.shape[0], self.configuration.angle_basis.shape[0] + self.configuration.iso_basis.shape[0]),
            dtype=torch.float32,
        )

        # Compute the normal, tangent, and binormal vector for each principal direction from the angle basis.
        # This set of vectors forms the eigenframe.
        O = utils.HouseHolder_evec_2_eframe(self.configuration.angle_basis.unsqueeze(dim=0))
        D = torch.einsum(
            "...nij, ...njk -> nik",
            torch.einsum("...nij, ...j -> ...nij", O, self.configuration.step_1_lambdas),
            torch.transpose(O, 2, 3),
        )

        A[:, 0 : self.configuration.angle_basis.shape[0]] = torch.exp(
            torch.einsum("n, ni, nj, mij -> nm", -bvals, bvecs, bvecs, D)
        )
        A[:, self.configuration.angle_basis.shape[0] :] = torch.exp(
            torch.einsum("i,j -> ij", -bvals, self.configuration.iso_basis)
        )
        return A

    def fit(self, bvals: torch.Tensor, bvecs: torch.Tensor, dwi: torch.Tensor) -> "DiffusionBasisFunctionModelFit":
        _ = time.time()
        A = self._prepare_forward_model(bvecs, bvals)
        x = leastsquares.nnlsq(
            A=A,
            b=dwi,
            optimizer_args=dict(self.configuration.STEP_1_OPTIMIZER_ARGS, DBSI_CONFIG=self.configuration),
            device=self.configuration.DEVICE,
        )
        return DiffusionBasisFunctionModelFit(self.configuration, A, x, bvals, bvecs)


class DiffusionBasisFunctionModelFit:
    def __init__(
        self,
        configuration: Any,
        A: torch.Tensor,
        x: torch.Tensor,
        bvals: torch.Tensor,
        bvecs: torch.Tensor,
    ) -> None:
        self.configuration = configuration
        self.A = A
        self.x = x
        self.TensorModel = DiffusionTensorModel(bvals, bvecs, device=self.configuration.DEVICE)

    @property
    def x_grouped(self) -> torch.Tensor:
        # Group signal intensity coefficients by angle
        x = self.x
        x[~(x == 0).all(dim=1)] /= x[~(x == 0).all(dim=1)].sum(dim=1)[:, None]
        xg = group_it.group(
            x,
            self.configuration.angle_basis,
            self.configuration.weight_threshold,
            self.configuration.angle_threshold,
            self.configuration.max_group_number,
        )
        # Re-normalize grouped signals
        xg[~(xg == 0).all(dim=1)] /= xg[~(xg == 0).all(dim=1)].sum(dim=1)[:, None]
        return xg

    @property
    def f_fiber(self) -> torch.Tensor:
        # Compute Fiber Fractions
        return self.x_grouped[:, 0 : self.configuration.angle_basis.shape[0]].sum(dim=1)

    @property
    def f_cell(self) -> torch.Tensor:
        return (
            self.x_grouped[:, self.configuration.angle_basis.shape[0] :][:, self.configuration.restricted_inds]
        ).sum(dim=1)

    @property
    def f_water(self) -> torch.Tensor:
        return (
            self.x_grouped[:, self.configuration.angle_basis.shape[0] :][:, self.configuration.water_inds].sum(dim=1)
            + self.x_grouped[:, self.configuration.angle_basis.shape[0] :][:, self.configuration.hindered_inds].sum(dim=1)
        )

    @property
    def fiber_signal(self) -> torch.Tensor:
        # Compute Fiber Signals
        return torch.einsum(
            "ij, bj -> bi",
            self.A[:, 0 : self.configuration.angle_basis.shape[0]],
            self.x_grouped[:, 0 : self.configuration.angle_basis.shape[0]],
        )

    @property
    def directions(self) -> torch.Tensor:
        # Perform DTI on the grouped fiber signals

        # (N_voxl x N_fiber, 3) array to store direction data. Will be reshaped to (N_voxl, N_fiber, 3), but the expanded
        # form is more convient for DTI calculations
        n = torch.zeros((self.x_grouped.shape[0], 3))

        # Get fiber signal
        fiber_signal = self.fiber_signal

        # Only compute fiber signal when the fiber represents a certain, thresholded, fraction of the signal
        msk = self.f_fiber >= self.configuration.fiber_threshold

        # Perform DTI fitting. Use fit_method = "WLS" for weighted least squares. This
        # Computation is performed on a cuda device if possible, thus data must be transfered
        # back to host.
        tenfit = self.TensorModel.fit(fiber_signal[msk], fit_method="OLS")
        n[msk] = tenfit.eigen_directions.to(self.configuration.HOST)

        # Reshape directions (N_voxel, N_fiber, 3).
        n = n.reshape(n.shape[0] // self.configuration.max_group_number, self.configuration.max_group_number, 3)
        return n

    @property
    def priors(self) -> Dict[str, torch.Tensor]:
        priors_dict: Dict[str, torch.Tensor] = {}
        priors_dict["direction_priors"] = self.directions
        priors_dict["fractions_priors"] = self.f_fiber
        priors_dict["restricted_priors"] = self.f_cell
        priors_dict["non_restricted_priors"] = self.f_water
        return priors_dict
