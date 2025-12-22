from dataclasses import dataclass
from typing import Optional, Literal
import yaml

@dataclass
class AblationConfig:
    """Configuration for ablation study experiments."""
    
    # Experiment identification
    experiment_name: str = "dadps_ablation"
    ablation_mode: Literal[
        "baseline",           # No disorder components
        "disorder_strength",  # Only disorder strength
        "noise_schedule",     # Only disorder-aware scheduling
        "regularization",     # Only disorder regularization
        "strength_schedule",  # Disorder strength + scheduling
        "strength_reg",       # Disorder strength + regularization
        "schedule_reg",       # Scheduling + regularization
        "full"               # All components (full DA-DPS)
    ] = "full"
    
    # Component toggles (auto-set by ablation_mode)
    enable_disorder_strength: bool = True
    enable_disorder_schedule: bool = True
    enable_disorder_regularization: bool = True
    
    # Disorder strength parameters
    disorder_type: str = "uniform"
    disorder_low: float = 0.8
    disorder_high: float = 1.2
    n_disorder: int = 10
    
    # Disorder-aware noise scheduling parameters
    schedule_type: Literal["linear", "cosine", "disorder_adaptive"] = "disorder_adaptive"
    schedule_scale: float = 1.0
    disorder_schedule_weight: float = 0.5  # Weighting for disorder influence on schedule
    
    # Disorder-based regularization parameters
    reg_type: Literal["none", "variance", "entropy", "consistency"] = "consistency"
    reg_weight: float = 0.1
    reg_decay: float = 0.95  # Exponential decay schedule for regularization
    
    # Training parameters
    seed: int = 42
    num_steps: int = 50
    guidance_scale: float = 1.0
    
    # Evaluation
    n_test_samples: int = 100
    metrics: list = None
    
    def __post_init__(self):
        """Set component flags based on ablation mode."""
        mode_configs = {
            "baseline": (False, False, False),
            "disorder_strength": (True, False, False),
            "noise_schedule": (False, True, False),
            "regularization": (False, False, True),
            "strength_schedule": (True, True, False),
            "strength_reg": (True, False, True),
            "schedule_reg": (False, True, True),
            "full": (True, True, True),
        }
        
        self.enable_disorder_strength, \
        self.enable_disorder_schedule, \
        self.enable_disorder_regularization = mode_configs[self.ablation_mode]
        
        if self.metrics is None:
            self.metrics = ["psnr", "ssim", "measurement_fidelity", "lpips"]
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
