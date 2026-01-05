"""Neural network optimization backends for DBSI fitting.

This module provides PyTorch-based optimization engines for both Step 1 (non-negative
least squares with fixed basis) and Step 2 (multi-fiber diffusion tensor refinement).

Key Components
--------------
leastsquares.py
    - nnlsq(): Gradient-based non-negative least squares solver
    - CUDABackend: GPU-accelerated batch processing
    - MultiprocessingBackend: CPU-based batch processing
    - MultiFiberModel: Orchestrates Step 2 optimization

_leastsquares_backends.py
    - least_squares_optimizer: Step 1 model with ReLU non-linearity
    - DBSIModel: Step 2 multi-fiber diffusion + kurtosis model
    - Implements forward models for DBSI and IA engines

Memory Management
-----------------
Backends automatically partition data into batches based on available memory:
- CUDA: Uses ~20% of GPU memory
- CPU: Uses ~25% of system RAM
- Batch size calculated dynamically: memory_available / memory_per_voxel

Optimization Strategy
--------------------
- Optimizer: Adam with OneCycleLR scheduling
- Loss: MSE with optional L2 ridge regularization
- Non-linearity: ReLU (Step 1), clamping (Step 2)
- Progress tracking: tqdm with color-coded bars (yellow=Step1, magenta=Step2)

See Also
--------
src.misc.models.Linear_Models : DTI implementation
src.core.fast_DBSI : Main DBSI pipeline orchestrator
"""
