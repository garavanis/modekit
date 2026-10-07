"""modekit: modal parameter estimation for experimental and operational modal analysis."""

import jax

# the identification needs double precision
jax.config.update("jax_enable_x64", True)
