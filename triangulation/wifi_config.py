class OptimizedWifiConfig:
    def __init__(self):
        # Path loss model parameters
        self.path_loss_models = {
            'free_space': 2.5,  # Increased from 2.0 for indoor environment
            'indoor_los': 3.0,  # Adjusted for weak signals
            'indoor_soft': 3.5,  # Adjusted for typical signal range
            'indoor_hard': 4.0  # Adjusted for weakest signals
        }

        # Signal strength thresholds
        self.los_threshold = -65  # Strong signal threshold
        self.soft_threshold = -85  # Moderate signal threshold

        # AP reliability parameters
        self.min_readings = 3  # Minimum readings needed for AP
        self.max_std_dev = 12  # Maximum standard deviation allowed
        self.max_signal_range = 30  # Maximum allowed signal variation

        # Position estimation parameters
        self.signal_variance_weight = 0.6
        self.distance_weight = 0.4
        self.min_aps_for_estimation = 4
        self.confidence_normalization = 4.0

        # Signal quality calculation
        self.signal_strength_offset = 120  # Increased due to weak signals
        self.stability_factor = 1.2  # Adjusted for variation

        # Reference power adjustments
        self.reference_power_offset = -75  # Based on mean signal
        self.min_reliable_power = -90  # Based on lower quartile

        # Grid parameters
        self.grid_max_x = 49
        self.grid_max_y = 49

    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
