from importlib.metadata import version

from regularizedvi._components import DecoderSCVI, Encoder, FCLayers
from regularizedvi._model import SCVI
from regularizedvi._module import VAE

__all__ = ["DecoderSCVI", "Encoder", "FCLayers", "SCVI", "VAE"]

__version__ = version("regularizedvi")
