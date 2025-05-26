# scatterbrain/modeling/__init__.py
"""
The modeling subpackage for ScatterBrain.

This package contains modules for defining and fitting scattering models,
including form factors and structure factors.

Modules
-------
form_factors
    Defines various analytical form factors P(q).
structure_factors
    (Future) Defines various structure factors S(q).
"""

from .form_factors import sphere_pq
# from .fitting import fit_model # To be added later

__all__ = [
    "form_factors", # Expose the module itself
    "sphere_pq",    # Expose specific functions/classes
    # "structure_factors", # Future
    # "fit_model", # Future
]