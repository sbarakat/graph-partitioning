import sys

from enum import Enum

class Partitioners(Enum):
    FENNEL = 1
    SCOTCH = 2
    PATOH = 3

PREDICTION_MODEL_ALGORITHM = Partitioners.FENNEL

if PREDICTION_MODEL_ALGORITHM == Partitioners.SCOTCH:
    print("Scotche")
else:
    print("Not")
