"""Every setting of the Trial 07/27 research pipeline, as one frozen object.

The research code kept these as module-level constants. Field names here are the same names
in lower case, and a test checks each value against the research code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class LegacyTrialConfig:
    """Settings of Trial 07/27. ``data_dir`` must point at the folder holding the two CSV files."""

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    stroke_file: str = "healthcare-dataset-stroke-data.csv"
    loan_file: str = "loan_approval_dataset.csv"

    seed: int = 42
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)

    # Federated learning
    num_clients: int = 5
    rounds: int = 20
    local_epochs: int = 3
    batch_size: int = 32
    mu: float = 0.01
    lr: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"stroke": 1e-4, "loan": 5e-3})
    )
    agg_weighted: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({"stroke": False, "loan": True})
    )

    # Models
    hidden: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: MappingProxyType({"stroke": (64, 32), "loan": (32, 16)})
    )
    width_scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    # Differential privacy. NOTE: epsilon here is a budget for EACH ROUND (finding G2).
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    epsilon_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 9.0, 15.0, 50.0)
    epsilon_default: float = 9.0

    # Homomorphic encryption (CKKS)
    ckks_poly_modulus_degree: int = 8192
    ckks_coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 60)
    ckks_global_scale: int = 2**40
    ckks_slots: int = 4096

    # Selective parameter encryption
    spe_fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    spe_default: float = 0.25
    mask_modes: tuple[str, ...] = ("clean", "dp", "random")
    mask_mode_default: str = "clean"

    # Splitting rows among participants
    dirichlet_alpha: float = 0.5
    min_pos_per_client: int = 12
    test_fraction: float = 0.20

    # Experiment grids
    arm_stacks: tuple[str, ...] = ("plain", "dp", "he", "dp_he", "dp_spe")
    arm_algos: tuple[str, ...] = ("fedavg", "fedprox")
    arm_datasets: tuple[str, ...] = ("stroke", "loan")
    frontier_epsilons: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 9.0, 15.0, 50.0)
    frontier_fractions: tuple[float, ...] = (0.0, 0.25, 1.0)
    frontier_algo: str = "fedprox"
    frontier_dataset: str = "stroke"
    lossless_epsilons: tuple[float, ...] = (0.5, 2.0, 9.0, 50.0)
    lossless_fractions: tuple[float, ...] = (0.25, 1.0)
    sens_epsilons: tuple[float, ...] = (0.5, 2.0, 9.0, 50.0)
    sens_fractions: tuple[float, ...] = (0.1, 0.25, 0.5)
    sens_repeats: int = 64
    overhead_fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    overhead_repeats: int = 3
    overhead_p_width_scale: int = 16
    overhead_p_control_width_scale: int = 1

    # Numerical tolerance of the research code's own exactness checks
    ckks_tolerance: float = 1e-4

    @property
    def stroke_path(self) -> Path:
        return self.data_dir / self.stroke_file

    @property
    def loan_path(self) -> Path:
        return self.data_dir / self.loan_file

    def hidden_for(self, dataset: str, width_scale: int = 1) -> tuple[int, ...]:
        """Hidden layer widths for ``dataset``, each multiplied by ``width_scale``."""
        if dataset not in self.hidden:
            raise ValueError(f"Unknown dataset: {dataset}")
        return tuple(width * width_scale for width in self.hidden[dataset])
