from dataclasses import dataclass

@dataclass
class DiffusionAlgorithm:
    """Store all configurations for Block Diffusion."""

    # The block size
    block_size: int
