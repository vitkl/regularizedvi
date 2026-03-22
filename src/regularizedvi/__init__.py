from importlib.metadata import version

from pyro.distributions import constraints
from pyro.distributions.transforms import SoftplusTransform
from torch.distributions import biject_to, transform_to

from regularizedvi import plt
from regularizedvi._components import (
    RegularizedDecoderSCVI,
    RegularizedEncoder,
    RegularizedFCLayers,
)
from regularizedvi._model import AmbientRegularizedSCVI
from regularizedvi._module import RegularizedVAE
from regularizedvi._multimodel import RegularizedMultimodalVI
from regularizedvi._multimodule import RegularizedMultimodalVAE


# Replace default exp positive transform with softplus (numerically more stable).
# This is applied on import so all downstream Pyro/PyTorch code benefits.
@biject_to.register(constraints.positive)
@transform_to.register(constraints.positive)
def _transform_to_positive(constraint):
    return SoftplusTransform()


__all__ = [
    "AmbientRegularizedSCVI",
    "RegularizedDecoderSCVI",
    "RegularizedEncoder",
    "RegularizedFCLayers",
    "RegularizedMultimodalVAE",
    "RegularizedMultimodalVI",
    "RegularizedVAE",
]

__version__ = version("regularizedvi")
