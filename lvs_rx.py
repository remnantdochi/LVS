# rx.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Receiver:
    carrier_freq: float = 475e3

