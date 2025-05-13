from .version import __version__
from .asktell import AskTellFewShotTopk
from .asktellFinetuning import AskTellFinetuning

try:
    from .asktellGPR import AskTellGPR
    from .asktellRidgeRegression import AskTellRidgeKernelRegression
except ImportError:
    print("GPR Packages not installed. Do `pip install boicl[gpr]` to install them")
from .asktellNearestNeighbor import AskTellNearestNeighbor
from .pool import Pool
from .tool import BOICLTool


__all__ = [
    "AskTellFewShotTopk",
    "AskTellFinetuning",
    "AskTellGPR",
    "AskTellRidgeKernelRegression",
    "AskTellNearestNeighbor",
    "Pool",
    "BOICLTool",
]
