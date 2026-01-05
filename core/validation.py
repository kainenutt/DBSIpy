"""Validation helpers and shared exceptions."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Custom exception for configuration validation errors."""


class DataError(Exception):
    """Raised when input data (DWI, mask, bvals/bvecs) is invalid or corrupted."""


class OptimizationError(Exception):
    """Raised when optimization fails to converge or produces invalid results."""


def validate_tensor(tensor, name, allow_negative: bool = False, allow_inf: bool = False):
    """Comprehensive tensor validation with informative error messages."""

    import numpy as np
    import torch

    # Convert to torch if numpy
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    # Check for NaN
    if torch.any(torch.isnan(tensor)):
        nan_count = torch.isnan(tensor).sum().item()
        nan_pct = 100.0 * nan_count / tensor.numel()
        raise DataError(
            f"{name} contains {nan_count:,} NaN values ({nan_pct:.2f}% of total).\n"
            f"This usually indicates:\n"
            f"  - Corrupted input data (check your NIfTI files)\n"
            f"  - Division by zero during normalization\n"
            f"  - Numerical overflow in calculations\n"
            f"Action: Inspect your input data for quality issues."
        )

    # Check for Inf
    if not allow_inf and torch.any(torch.isinf(tensor)):
        inf_count = torch.isinf(tensor).sum().item()
        inf_pct = 100.0 * inf_count / tensor.numel()
        raise DataError(
            f"{name} contains {inf_count:,} Inf values ({inf_pct:.2f}% of total).\n"
            f"This usually indicates:\n"
            f"  - Numerical overflow during computation\n"
            f"  - Division by very small numbers\n"
            f"  - Extreme parameter values\n"
            f"Action: Check for unreasonable values in input data."
        )

    # Check for negative values
    if not allow_negative and torch.any(tensor < 0):
        neg_count = (tensor < 0).sum().item()
        neg_pct = 100.0 * neg_count / tensor.numel()
        neg_min = tensor[tensor < 0].min().item()
        raise DataError(
            f"{name} contains {neg_count:,} negative values ({neg_pct:.2f}% of total).\n"
            f"Minimum value: {neg_min:.6f}\n"
            f"This is unexpected for diffusion MRI data.\n"
            f"Action: Check signal normalization and preprocessing steps."
        )

    return tensor
