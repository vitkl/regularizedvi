from importlib.metadata import version

from regularizedvi._components import (
    RegularizedDecoderSCVI,
    RegularizedEncoder,
    RegularizedFCLayers,
)
from regularizedvi._model import AmbientRegularizedSCVI
from regularizedvi._module import RegularizedVAE

__all__ = [
    "AmbientRegularizedSCVI",
    "RegularizedDecoderSCVI",
    "RegularizedEncoder",
    "RegularizedFCLayers",
    "RegularizedVAE",
]

__version__ = version("regularizedvi")
