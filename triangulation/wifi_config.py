class OptimizedWifiConfig:
    def __init__(self):
        # Path loss model parameters - adjusted for weaker but more stable signals
        self.path_loss_models = {
            'free_space': 2.0,  # Reduced for better accuracy with strongest signals
            'indoor_los': 2.5,  # Adjusted for -56.5 to -80 range
            'indoor_soft': 3.0,  # Optimized for -80 to -89 range
            'indoor_hard': 3.5  # Adjusted for very weak signals (<-89)
        }

        # Signal strength thresholds - based on new quartile analysis
        self.los_threshold = -80  # 75th percentile for strong signals
        self.soft_threshold = -89  # 25th percentile for moderate signals

        # AP reliability parameters - adjusted for more stable environment
        self.min_readings = 1  # Reduced due to more consistent readings
        self.max_std_dev = 9.0  # Slightly above measured std (8.76)
        self.max_signal_range = 25  # Adjusted based on new range (-96 to -56.5)

        # Position estimation parameters
        self.signal_variance_weight = 0.8  # Increased due to more consistent signals
        self.distance_weight = 0.2  # Decreased accordingly
        self.min_aps_for_estimation = 3  # Maintained for coverage
        self.confidence_normalization = 4.0  # Adjusted for narrower signal range

        # Signal quality calculation
        self.signal_strength_offset = 110  # Adjusted for weaker signal range
        self.stability_factor = 1.0  # Reduced due to lower standard deviation

        # Reference power adjustments
        self.reference_power_offset = -83  # Set to mean signal strength
        self.min_reliable_power = -89  # Set to 25th percentile

        # Grid parameters - unchanged
        self.grid_max_x = 49
        self.grid_max_y = 49

    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
