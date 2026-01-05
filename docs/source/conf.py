from __future__ import annotations

import os
import sys
from datetime import datetime

# Add repo root so autodoc can find dbsipy without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

project = "DBSIpy"
author = "Kainen L. Utt, Ph.D., Jacob Blum, Yong Wang, Ph.D., and the DBSIpy Development Team"
copyright = f"{datetime.now().year}, {author}"

# Keep version import lightweight; mock heavy deps to avoid RTD build failures.
autodoc_mock_imports = [
    "torch",
    "dipy",
    "nibabel",
    "numpy",
    "pandas",
    "scipy",
    "joblib",
    "tqdm",
    "psutil",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

# If we later add a local logo file, put it in _static and set html_logo.
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
