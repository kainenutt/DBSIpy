import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np 
from torch import Tensor 

# Default PyTorch Datatype is float32. 
MIN_POSITIVE_SIGNAL = np.finfo(np.float32).eps

from torch.nn.modules import Module
from dbsipy.core import utils

class Diffusivity(Module):
    __constants__ = ['nVoxels', 'nFibers']
    
    nVoxels : int
    nFibers : int
    weight  : Tensor

    def __init__(self, nVoxels : int, nFibers : int, D0 : float, device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.nVoxels = nVoxels
        self.nFibers = nFibers
        self.D0      = D0

        self.weight  = nn.Parameter(
                                    torch.empty(
                                                (nVoxels, nFibers, 1, 1),
                                                **factory_kwargs
                                                )
                                    )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, self.D0, self.D0 + MIN_POSITIVE_SIGNAL)

class Direction(Module):
    __constants__ = ['nVoxels', 'nFibers']

    nVoxels : int
    nFibers : int
    
    def __init__(self, nVoxels : int, nFibers : int, directionPrior : torch.FloatTensor , device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.nVoxels        = nVoxels
        self.nFibers        = nFibers
        self.directionPrior = directionPrior

        self.weight  = nn.Parameter(
                                    torch.empty(
                                                (nVoxels, nFibers, 3),
                                                **factory_kwargs
                                                )
                                    )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight[...] = self.directionPrior

class signalIntensityFraction(Module):
    __constants__ = ['nVoxels', 'nFibers', 'nModels']
    
    nVoxels : int
    nFibers : int
    nModels : int

    def __init__(self, nVoxels : int, nFibers : int, nModels : int, fractionPrior : torch.FloatTensor, device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.nVoxels        = nVoxels
        self.nFibers        = nFibers
        self.nModels        = nModels
        self.fractionPrior = fractionPrior

        self.weight  = nn.Parameter(
                                    torch.empty(
                                                (nVoxels, nFibers, nModels),
                                                **factory_kwargs
                                                )
                                    )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight[...] = self.fractionPrior


class coorientedDiffusionTensorModel(nn.Module):
    def __init__(self, 
                 models          : List[str], 
                 directionPriors : torch.FloatTensor, 
                 fractionPriors  : Dict[str, torch.FloatTensor], 
                 device = 'cpu',
                 dtype  = None
                 ):
        super().__init__()


        self.DEFAULT_Da = {'fiber': 1.0e-3,  'water' : 3.0e-3, 'flow': 10.0e-3} 
        self.DEFAULT_Dr = {'fiber': .30e-3,  'water' : 1.5e-3, 'flow': 5.0e-3} 

        self.device = device 
        self.dtype  = dtype 

        r"""

        Each diffusion tensor is parametized by 5 (plus 1) free variables
        -----------------------------------------------------------------

        Parameters
        ----------

        Da : torch.FloatTensor, Shape (Nvoxels, nFibers, Nmodels, 1)
             The axial diffusivity (in um^2 / ms ) of the diffusion tensor

        Dr : torch.FloatTensor, Shape (Nvoxels, nFibers, Nmodels, 1)
             The radial diffusivity (in um^2 / ms ) of the diffusion tensor 

        Ed  : torch.FlotTensor, Shape (Nvoxels, nFibers, 3)
             The unit normal principal (eigen) direction of the diffusion tensor

        """

        self.nVoxels = directionPriors.shape[0]
        self.nFibers = directionPriors.shape[1]
        self.nModels = len(models)
        self.modelsList = models
        self.directionPriors = directionPriors
        self.fractionsPriors = fractionPriors


        r"""
        Storage Outside of Function
        """

        self.models = nn.ModuleDict({})
        self.params = nn.ModuleDict({})
                           
        self._reset_parameters()
        self._register_params()


    def _reset_parameters(self) -> None:

        with torch.no_grad():
            for j, model in enumerate(self.modelsList):
                self.models[model] = nn.ModuleDict(
                                                   {
                                                    f'Da'  : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Da[model], device = self.device, dtype = self.dtype),
                                                    f'Dr1' : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Dr[model], device = self.device, dtype = self.dtype),
                                                    f'Dr2' : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Dr[model], device = self.device, dtype = self.dtype)
                                                    }
                                                   )
            self.models['direction'] = Direction(self.nVoxels, self.nFibers, directionPrior = self.directionPriors, device = self.device, dtype = self.dtype)
            self.models['fractions'] = signalIntensityFraction(self.nVoxels, self.nFibers, self.nModels, fractionPrior = self.fractionsPriors, device = self.device, dtype = self.dtype)

            return 
    
    def _register_params(self) -> None:

        for compartmentModel, compartmentParameters in self.models.items():
            if compartmentModel in self.modelsList:
                self.params[compartmentModel] = nn.ModuleList(
                                                            [
                                                            compartmentParameters.Da,
                                                            compartmentParameters.Dr1,
                                                            compartmentParameters.Dr2
                                                            ]
                                                            )
            else:
                self.params[compartmentModel] = compartmentParameters
        return 
        
    def _fix_param_group(self, param_groups: List[str]) -> None:
        for model in set(self.models.keys()).intersection(set(param_groups)):
            self.params.pop(model)
        return

    def _add_param_group(self, param_groups: List[str], newFractionPriors : torch.FloatTensor, newDirectionPriors : torch.FloatTensor) -> None:        

        DEFAULT_TENSOR_MODEL_ORDER = ['fiber', 'water', 'flow', 'fractions', 'direction']

        self.modelsList += param_groups
        self.nModels = len(self.modelsList)

        for model in set(self.DEFAULT_Da.keys()).intersection(set(param_groups)):
            self.models[model] = nn.ModuleDict(
                                               {
                                                 f'Da'  : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Da[model], device = self.device),
                                                 f'Dr1' : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Dr[model], device = self.device) ,
                                                 f'Dr2' : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Dr[model], device = self.device)
                                                }
                                               ) 
        self.models['direction'] = Direction(self.nVoxels, self.nFibers, directionPrior = newDirectionPriors, device = self.device)
        self.models['fractions'] = signalIntensityFraction(self.nVoxels, self.nFibers, self.nModels, fractionPrior = newFractionPriors, device=self.device)

        ordered_models = { model : self.models[model] for model in DEFAULT_TENSOR_MODEL_ORDER if model in self.models.keys()}
      
        self.models = nn.ModuleDict(ordered_models)
        self._register_params()

        return 
    def _reset_param_group(self, param_groups: List[str]):
        for model in set(self.models.keys()).intersection(set(param_groups)):
            for parameter in self.models[model].values(): parameter.reset_parameters()
        return  

    @property
    def Da(self) -> torch.FloatTensor:
        return torch.cat([model.Da.weight for key, model in self.models.items() if key in self.modelsList], dim = -2)

    @property
    def Dr1(self) -> torch.FloatTensor:
        return torch.cat([model.Dr1.weight for key, model in self.models.items() if key in self.modelsList], dim = -2)

    @property
    def Dr2(self) -> torch.FloatTensor:
        return torch.cat([model.Dr2.weight for key, model in self.models.items() if key in self.modelsList], dim = -2)
 
    @property 
    def n(self) -> torch.FloatTensor:
        nonNormalDirections = self.models.direction.weight
        with torch.no_grad():
            nonNormalDirections[ torch.logical_not(nonNormalDirections.sum(dim = 2 ) == 0) ] /= torch.linalg.norm(nonNormalDirections[torch.logical_not(nonNormalDirections.sum(dim = 2 ) == 0) ], axis = -1, ord = 2)[:, None]
        return nonNormalDirections
        
    @property 
    def Ds(self) -> torch.FloatTensor:
        Da, Dr1, Dr2, n = self.Da, self.Dr1, self.Dr1, self.n

        P = utils.HouseHolder_evec_2_eframe(n, device = self.device)
        Pt = torch.transpose(P, 2, 3)
        L = torch.cat([Da, Dr1, Dr2], dim = -1).clamp(min = MIN_POSITIVE_SIGNAL, max = None)

        # Optimized: Replace nested einsum with batched matrix multiplication
        # Original: D = P @ diag(L) @ P^T for each batch and model
        # L shape: (batch, fibers, models, 3) -> expand to (batch, fibers, models, 3, 3) diagonal
        L_diag = torch.diag_embed(L)  # (batch, fibers, models, 3, 3)
        # P @ L_diag: (batch, fibers, 3, 3) @ (batch, fibers, models, 3, 3)
        temp = torch.matmul(P.unsqueeze(2), L_diag)  # (batch, fibers, models, 3, 3)
        D = torch.matmul(temp, Pt.unsqueeze(2))  # (batch, fibers, models, 3, 3) 
    
        return D 
    
    @property 
    def v(self) -> torch.FloatTensor:
        v = self.models.fractions.weight.reshape(self.nVoxels, int(self.nFibers*self.nModels)).clamp(min = MIN_POSITIVE_SIGNAL, max = None) 
        with torch.no_grad():
            v[torch.logical_not(v.sum(dim = 1 ) == 0) ] /= torch.linalg.norm(v[torch.logical_not(v.sum(dim = 1) == 0) ], axis = -1, ord = 1)[:, None]
        return v 

    @property 
    def Ds_Dict(self) -> Dict[str, torch.FloatTensor]:
        DDict   = {}
        DTs = self.Ds
        with torch.no_grad():
            for j, model in enumerate(self.modelsList):
                DDict[model] = DTs[..., j, :, :]
            return DDict  

    def __call__(self, bvals, bvecs) -> torch.FloatTensor:

        def _Az(bvals: torch.FloatTensor, bvecs: torch.FloatTensor) -> torch.FloatTensor:
            Az = torch.einsum('s, si, sj, bnmij -> bnms', -bvals, bvecs, bvecs, self.Ds)
            return Az

        def _rho(Az: torch.FloatTensor) -> torch.FloatTensor:
            return torch.exp(Az)

        return _rho(_Az(bvals, bvecs)).reshape(self.nVoxels, int ( self.nFibers * self.nModels), bvals.shape[0])
    
    def _predict(self, bvals, bvecs):
        exp_bgDg = self.__call__(bvals, bvecs)
        v        = self.v

        v_dot_exp_bgDg = torch.einsum('bms, bm -> bs', exp_bgDg, v)
        return v_dot_exp_bgDg


class isotropicDiffusionModel:
    def __init__(self, 
                 models          : List[str], 
                 fractionPriors  : Dict[str, torch.FloatTensor], 
                 device = 'cpu',
                 dtype  = None
                 ):
        super().__init__()


        self.DEFAULT_Da = {'water': 3.0e-3,  'cell' : .30e-3} 
        
        self.device = device 
        self.dtype  = dtype 

        r"""

        Each diffusion tensor is parametized by 5 (plus 1) free variables
        -----------------------------------------------------------------

        Parameters
        ----------

        Da : torch.FloatTensor, Shape (Nvoxels, nFibers, Nmodels, 1)
             The axial diffusivity (in um^2 / ms ) of the diffusion tensor

        Dr : torch.FloatTensor, Shape (Nvoxels, nFibers, Nmodels, 1)
             The radial diffusivity (in um^2 / ms ) of the diffusion tensor 

        Ed  : torch.FlotTensor, Shape (Nvoxels, nFibers, 3)
             The unit normal principal (eigen) direction of the diffusion tensor

        """

        self.nVoxels = fractionPriors.shape[0]
        self.nModels = len(models)
        self.nFibers = 1
        self.modelsList = models
        self.fractionsPriors = fractionPriors


        r"""
        Storage Outside of Function
        """

        self.models = nn.ModuleDict({})
        self.params = nn.ModuleDict({})
                           
        self._reset_parameters()
        self._register_params()


    def _reset_parameters(self) -> None:

        with torch.no_grad():
            for j, model in enumerate(self.modelsList):
                self.models[model] = nn.ModuleDict(
                                                   {
                                                    f'Da'  : Diffusivity(self.nVoxels, self.nFibers, D0 = self.DEFAULT_Da[model], device = self.device, dtype = self.dtype),
                                                    }
                                                   )
            
            self.models['direction'] = Direction(
                                                 self.nVoxels, 
                                                 self.nModels, 
                                                 directionPrior = torch.normal(0, 1, size = (self.nVoxels, self.nModels, 3)), 
                                                 device = self.device, 
                                                 dtype = self.dtype
                                                 )
            
            self.models['fractions'] = signalIntensityFraction(
                                                                self.nVoxels, 
                                                                self.nFibers, 
                                                                self.nModels, 
                                                                fractionPrior = self.fractionsPriors, 
                                                                device = self.device, 
                                                                dtype = self.dtype
                                                                )


            return 
    
    def _register_params(self) -> None:

        for compartmentModel, compartmentParameters in self.models.items():
            if compartmentModel in self.modelsList:
                self.params[compartmentModel] = nn.ModuleList(
                                                            [
                                                            compartmentParameters.Da,
                                                            ]
                                                            )
            else:
                self.params[compartmentModel] = compartmentParameters
        return 
    
    @property
    def Da(self) -> torch.FloatTensor:
        return torch.cat([model.Da.weight for key, model in self.models.items() if key in self.modelsList], dim = -2)

    @property 
    def n(self) -> torch.FloatTensor:
        nonNormalDirections = self.models.direction.weight
        with torch.no_grad():
            nonNormalDirections[ torch.logical_not(nonNormalDirections.sum(dim = 2 ) == 0) ] /= torch.linalg.norm(nonNormalDirections[torch.logical_not(nonNormalDirections.sum(dim = 2 ) == 0) ], axis = -1, ord = 2)[:, None]
        return nonNormalDirections
        
    @property 
    def Ds(self) -> torch.FloatTensor:
        Da, n = self.Da, self.n

        P = utils.HouseHolder_evec_2_eframe(n, device = self.device)

        Pt = torch.transpose(P, 2, 3)
        L = torch.cat([Da, Da, Da], dim = -1).clamp(min = MIN_POSITIVE_SIGNAL, max = None)
        D = torch.einsum(
                           'bnmij, bmjk -> bnmik', 
                           torch.einsum('bmij, bnmj -> bnmij', P, L
                                        ), Pt
                         ) # Shape : [Batch, Nfiber (1), Nmodel (1 or 2), 3, 3]
        
        return D 
    
    @property 
    def v(self) -> torch.FloatTensor:
        v = self.models.fractions.weight.reshape(self.nVoxels, int(self.nFibers*self.nModels)).clamp(min = MIN_POSITIVE_SIGNAL, max = None) 
        with torch.no_grad():
            v[torch.logical_not(v.sum(dim = 1 ) == 0) ] /= torch.linalg.norm(v[torch.logical_not(v.sum(dim = 1) == 0) ], axis = -1, ord = 1)[:, None]
        return v 

    @property 
    def Ds_Dict(self) -> Dict[str, torch.FloatTensor]:
        DDict   = {}
        DTs = self.Ds
        with torch.no_grad():
            for j, model in enumerate(self.modelsList):
                DDict[model] = DTs[..., j, :, :]
            return DDict  

    def __call__(self, bvals, bvecs) -> torch.FloatTensor:

        def _Az(bvals: torch.FloatTensor, bvecs: torch.FloatTensor) -> torch.FloatTensor:
            Az = torch.einsum('s, si, sj, bnmij -> bnms', -bvals, bvecs, bvecs, self.Ds)
            return Az

        def _rho(Az: torch.FloatTensor) -> torch.FloatTensor:
            return torch.exp(Az)
    
        return _rho(_Az(bvals, bvecs)).reshape(self.nVoxels, int ( self.nFibers * self.nModels), bvals.shape[0])
    
    def _predict(self, bvals, bvecs):
        exp_bgDg = self.__call__(bvals, bvecs)
        v        = self.v

        v_dot_exp_bgDg = torch.einsum('bms, bm -> bs', exp_bgDg, v)
        return v_dot_exp_bgDg
