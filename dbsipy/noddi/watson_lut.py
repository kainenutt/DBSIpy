"""
Watson Distribution Lookup Table for NODDI

Provides efficient evaluation of the Watson distribution integral needed for
dispersed stick signal computation in NODDI.

The Watson distribution describes fiber orientation dispersion around a mean direction
with concentration parameter kappa:
- kappa -> infinity: Perfect alignment (no dispersion)
- kappa -> 0: Isotropic distribution

References
----------
.. [1] Zhang, H., Schneider, T., Wheeler-Kingshott, C.A., Alexander, D.C., 2012.
       NODDI: practical in vivo neurite orientation dispersion and density imaging
       of the human brain. Neuroimage 61(4), 1000-1016.
"""

import torch
import numpy as np
from scipy import special
from typing import Tuple


def watson_integral_lookup_table(
    b_values: np.ndarray = None,
    kappa_values: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute Watson distribution integrals for stick signal with dispersion.
    
    The dispersed stick signal is:
    S(b, kappa) = exp(-b * d_ic) * I(b, kappa)
    
    where I(b, kappa) is the integral over the Watson distribution.
    
    Parameters
    ----------
    b_values : ndarray, optional
        B-values to tabulate (s/mm^2). Default: 0-6000 in steps of 50
    kappa_values : ndarray, optional
        Concentration parameters. Default: 0.01-64 (log-spaced)
    
    Returns
    -------
    b_grid : ndarray, shape (n_b,)
        B-values
    kappa_grid : ndarray, shape (n_kappa,)
        Kappa values
    integral_table : ndarray, shape (n_b, n_kappa)
        Watson integral I(b, kappa)
    """
    if b_values is None:
        # Default: 0 to 6000 s/mm^2 in steps of 50
        b_values = np.linspace(0, 6000, 121)
    
    if kappa_values is None:
        # Log-spaced kappa from 0.01 (high dispersion) to 64 (low dispersion)
        kappa_values = np.logspace(-2, np.log10(64), 100)
    
    n_b = len(b_values)
    n_kappa = len(kappa_values)
    integral_table = np.zeros((n_b, n_kappa))
    
    # Compute integrals using Legendre polynomial expansion
    # Based on Kaden et al. 2016 approach
    for i, b in enumerate(b_values):
        for j, kappa in enumerate(kappa_values):
            integral_table[i, j] = _watson_integral_legendre(b, kappa)
    
    return b_values, kappa_values, integral_table


def _watson_integral_legendre(b: float, kappa: float, n_terms: int = 20) -> float:
    """
    Compute Watson integral using Legendre polynomial expansion.
    
    The stick signal with Watson dispersion can be expanded as:
    I(b, kappa) = sum_n (2n+1)/2 * P_n(0) * dawson(sqrt(n(n+1)b/kappa)) / sqrt(n(n+1)b/kappa)
    
    where P_n(0) are Legendre polynomials at 0, and dawson is the Dawson function.
    
    Parameters
    ----------
    b : float
        B-value (s/mm^2)
    kappa : float
        Watson concentration parameter
    n_terms : int
        Number of Legendre terms (default 20)
    
    Returns
    -------
    integral : float
        Watson distribution integral
    """
    if b < 1e-6:
        return 1.0  # No diffusion weighting
    
    if kappa < 1e-3:
        # Very high dispersion -> isotropic (returns 1/3 for stick in 3D)
        return 1.0 / 3.0
    
    if kappa > 100:
        # Very low dispersion -> aligned stick
        return 1.0
    
    # Legendre polynomial expansion
    integral = 0.0
    
    for n in range(n_terms):
        if n == 0:
            # P_0(0) = 1, but n(n+1) = 0, handle separately
            integral += 0.5  # Simplified contribution
            continue
        
        # Legendre polynomial at 0
        # P_n(0) = 0 for odd n
        # P_n(0) = (-1)^(n/2) * (n!!) / ((n+1)!!) for even n
        if n % 2 == 1:
            continue  # Odd terms are zero
        
        # For even n:
        P_n_0 = _legendre_at_zero(n)
        
        # Argument for Dawson function
        arg = np.sqrt(n * (n + 1) * b / kappa)
        
        if arg < 1e-6:
            dawson_term = arg / 2.0  # Taylor expansion for small arg
        else:
            dawson_term = special.dawsn(arg)
        
        # Add term
        weight = (2 * n + 1) / 2.0
        term = weight * P_n_0 * dawson_term / (arg + 1e-10)
        integral += term
    
    # Ensure physically valid range
    return np.clip(integral, 0.0, 1.0)


def _legendre_at_zero(n: int) -> float:
    """
    Evaluate Legendre polynomial P_n(0).
    
    For even n: P_n(0) = (-1)^(n/2) * (n!!) / ((n+1)!!)
    For odd n: P_n(0) = 0
    
    where n!! is the double factorial.
    """
    if n % 2 == 1:
        return 0.0
    
    # Even n
    k = n // 2
    sign = (-1) ** k
    
    # Double factorial ratio: n!! / (n+1)!!
    # n!! = n * (n-2) * (n-4) * ... * 2
    # Compute iteratively
    numerator = 1.0
    denominator = 1.0
    
    for i in range(k):
        numerator *= (n - 2 * i)
        denominator *= (n + 1 - 2 * i)
    
    return sign * numerator / denominator


def od_to_kappa(od: torch.Tensor) -> torch.Tensor:
    """
    Convert Orientation Dispersion Index (ODI) to Watson concentration kappa.
    
    Uses empirical relationship from Zhang et al. 2012:
    kappa ~= 1 / tan((pi/2) * ODI)
    
    Parameters
    ----------
    od : torch.Tensor
        Orientation dispersion index [0, 1]
        0 = perfectly aligned, 1 = isotropic
    
    Returns
    -------
    kappa : torch.Tensor
        Watson concentration parameter [0.01, 64]
    """
    epsilon = 1e-6
    # Avoid division by zero and tan(pi/2)
    od_safe = torch.clamp(od, epsilon, 1.0 - epsilon)
    
    # Empirical relationship
    kappa = 1.0 / (torch.tan((np.pi / 2.0) * od_safe) + epsilon)
    
    # Clamp to reasonable range
    return torch.clamp(kappa, 0.01, 64.0)


def kappa_to_od(kappa: torch.Tensor) -> torch.Tensor:
    """
    Convert Watson concentration kappa to Orientation Dispersion Index (ODI).
    
    Inverse of od_to_kappa using:
    ODI = (2/pi) * arctan(1/kappa)
    
    Parameters
    ----------
    kappa : torch.Tensor
        Watson concentration parameter [0.01, 64]
    
    Returns
    -------
    od : torch.Tensor
        Orientation dispersion index [0, 1]
    """
    epsilon = 1e-6
    kappa_safe = torch.clamp(kappa, epsilon, 64.0)
    
    od = (2.0 / np.pi) * torch.atan(1.0 / kappa_safe)
    
    return torch.clamp(od, 0.0, 1.0)


class WatsonLUT:
    """
    Watson distribution lookup table with GPU support and interpolation.
    
    Provides fast evaluation of dispersed stick signals using pre-computed
    integrals with bilinear interpolation.
    
    Parameters
    ----------
    device : str
        'cpu' or 'cuda'
    
    Attributes
    ----------
    b_values : torch.Tensor
        B-value grid
    kappa_values : torch.Tensor
        Kappa grid (log-spaced)
    integral_table : torch.Tensor
        Pre-computed Watson integrals, shape (n_b, n_kappa)
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Generate lookup table
        b_np, kappa_np, integral_np = watson_integral_lookup_table()
        
        # Convert to tensors
        self.b_values = torch.from_numpy(b_np).float().to(device)
        self.kappa_values = torch.from_numpy(kappa_np).float().to(device)
        self.integral_table = torch.from_numpy(integral_np).float().to(device)
        
        # For interpolation
        self.b_min = self.b_values[0]
        self.b_max = self.b_values[-1]
        self.kappa_min = self.kappa_values[0]
        self.kappa_max = self.kappa_values[-1]
        
        # Grid spacing (for linear indexing)
        self.n_b = len(self.b_values)
        self.n_kappa = len(self.kappa_values)
    
    def __call__(self, b: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Watson integral using bilinear interpolation.
        
        Parameters
        ----------
        b : torch.Tensor
            B-values, any shape
        kappa : torch.Tensor
            Concentration parameters, same shape as b
        
        Returns
        -------
        integral : torch.Tensor
            Watson integrals, same shape as inputs
        """
        # Clamp to table range
        b_clamped = torch.clamp(b, self.b_min, self.b_max)
        kappa_clamped = torch.clamp(kappa, self.kappa_min, self.kappa_max)
        
        # Convert to grid indices (fractional)
        # For b: linear interpolation
        b_idx = (b_clamped - self.b_min) / (self.b_max - self.b_min) * (self.n_b - 1)
        
        # For kappa: log-space interpolation
        log_kappa = torch.log10(kappa_clamped + 1e-10)
        log_kappa_min = torch.log10(self.kappa_min)
        log_kappa_max = torch.log10(self.kappa_max)
        kappa_idx = (log_kappa - log_kappa_min) / (log_kappa_max - log_kappa_min) * (self.n_kappa - 1)
        
        # Get integer indices and fractional parts
        b_idx0 = torch.floor(b_idx).long()
        b_idx1 = torch.clamp(b_idx0 + 1, 0, self.n_b - 1)
        b_frac = b_idx - b_idx0.float()
        
        kappa_idx0 = torch.floor(kappa_idx).long()
        kappa_idx1 = torch.clamp(kappa_idx0 + 1, 0, self.n_kappa - 1)
        kappa_frac = kappa_idx - kappa_idx0.float()
        
        # Bilinear interpolation
        # Q00, Q01, Q10, Q11 are the four corners
        Q00 = self.integral_table[b_idx0, kappa_idx0]
        Q01 = self.integral_table[b_idx0, kappa_idx1]
        Q10 = self.integral_table[b_idx1, kappa_idx0]
        Q11 = self.integral_table[b_idx1, kappa_idx1]
        
        # Interpolate in b direction
        Q0 = Q00 * (1 - b_frac) + Q10 * b_frac
        Q1 = Q01 * (1 - b_frac) + Q11 * b_frac
        
        # Interpolate in kappa direction
        result = Q0 * (1 - kappa_frac) + Q1 * kappa_frac
        
        return result


if __name__ == "__main__":
    # Test the lookup table generation
    print("Generating Watson distribution lookup table...")
    b_vals, kappa_vals, integrals = watson_integral_lookup_table()
    print(f"  B-values: {b_vals.min():.0f} to {b_vals.max():.0f} s/mm^2 ({len(b_vals)} points)")
    print(f"  Kappa: {kappa_vals.min():.4f} to {kappa_vals.max():.2f} ({len(kappa_vals)} points)")
    print(f"  Integral range: {integrals.min():.4f} to {integrals.max():.4f}")
    
    # Test OD <-> kappa conversion
    print("\nTesting OD <-> Kappa conversions:")
    test_ods = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    kappas = od_to_kappa(test_ods)
    ods_reconstructed = kappa_to_od(kappas)
    
    for od, kappa, od_recon in zip(test_ods, kappas, ods_reconstructed):
        print(f"  OD={od:.2f} -> kappa={kappa:.4f} -> OD={od_recon:.2f}")
    
    # Test LUT interpolation
    print("\nTesting WatsonLUT interpolation:")
    lut = WatsonLUT(device='cpu')
    test_b = torch.tensor([0.0, 1000.0, 2000.0, 3000.0])
    test_kappa = torch.tensor([0.1, 1.0, 10.0, 50.0])
    
    for b, k in zip(test_b, test_kappa):
        integral = lut(b.unsqueeze(0), k.unsqueeze(0)).item()
        print(f"  I(b={b:.0f}, kappa={k:.2f}) = {integral:.4f}")
    
    print("\nWatson LUT ready for NODDI implementation!")
