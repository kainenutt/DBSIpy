DTI
===

Conceptual model
----------------

Diffusion Tensor Imaging (DTI) models each voxel as a single 3×3 diffusion tensor
:math:`D` and predicts signal attenuation of the form:

.. math::

   S(b,\mathbf{g}) = S_0\,\exp\left(-b\,\mathbf{g}^\top D\,\mathbf{g}\right)

DTI-derived scalar maps include ADC, AD, RD, and FA (DBSIpy uses ``*_adc`` naming
rather than “MD”).

Fitting methods
----------------

DBSIpy supports multiple fitting modes (configured by ``dti_fit_method``):

- **Closed-form / linear** methods (e.g., OLS/WLS-style fits), and
- **Iterative (ADAM)** optimization for consistency with the project’s optimizer-driven
  workflows.

The iterative option is useful for testing and for controlled comparisons, but it may
converge to a small numerical neighborhood rather than the exact closed-form optimum.

Practical notes
---------------

- By default, DTI uses ``signal_normalization = none`` when set to ``auto``.
- A b-value cutoff is commonly used so DTI is fit on a lower-b subset when the
  acquisition is multi-shell.
