__version__ = "0.1.1"

# Expose Baseline methods
from .baseline import (
    hestonOptimization as heston_opt_baseline,
    runSimulation as run_sim_baseline,
    pricer
)

# Expose Improved methods (Smart Initialization)
from .improved import (
    hestonOptimization as heston_opt_improved,
    runSimulation as run_sim_improved
)

# Expose the parameter recovery self-test
from .reversed import main as run_recovery_test

from .heston_package import (
    HestonPricer, 
    HestonParams, 
    OptionContract, 
    HestonCalibrator
)

__all__ = [
    "heston_opt_baseline",
    "run_sim_baseline",
    "heston_opt_improved",
    "run_sim_improved",
    "pricer",
    "run_recovery_test",
    "HestonPricer",
    "HestonParams",
    "OptionContract",
    "HestonCalibrator"
]
