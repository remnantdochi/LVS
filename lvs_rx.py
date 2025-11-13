"""Receiver DSP block for the LVS simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from config import RxConfig


@dataclass
class ReceiverOutputs:
    """Container for the receiver stages."""

    time: NDArray[np.float64]
    processed: NDArray[np.float64]
    metadata: Dict[str, NDArray[np.float64]]


class Receiver:
    """Placeholder DSP pipeline for the MCU-side processing."""

    def __init__(self, config: RxConfig) -> None:
        self.config = config
        self._state: Dict[str, NDArray[np.float64]] = {}

    def reset(self) -> None:
        """Reset any accumulated receiver state."""
        self._state.clear()

    def process_chunk(
        self,
        time: NDArray[np.float64],
        samples: NDArray[np.float64],
    ) -> ReceiverOutputs:
        """
        Run the DSP chain on a chunk of samples.

        For now, the receiver acts as a pass-through while collecting metadata.
        """
        processed = samples

        metadata: Dict[str, NDArray[np.float64]] = {}
        return ReceiverOutputs(time=time, processed=processed, metadata=metadata)
