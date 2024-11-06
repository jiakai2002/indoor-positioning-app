class OptimizedWifiConfig:
    def __init__(self):
        # Path loss model parameters - adjusted for larger dataset
        self.path_loss_models = {
            'free_space': 2.1,  # Adjusted for strong signals (-50.5 to -80)
            'indoor_los': 2.6,  # For -80 to -85.5 range
            'indoor_soft': 3.1,  # For -85.5 to -90 range
            'indoor_hard': 3.6   # For very weak signals (<-90)
        }

        # Signal strength thresholds - based on quartile analysis
        self.los_threshold = -80    # 75th percentile for strong signals
        self.soft_threshold = -85.5  # Median for moderate signals

        # AP reliability parameters - adjusted for larger dataset
        self.min_readings = 3       # Increased due to larger dataset
        self.max_std_dev = 10.0     # Slightly above measured std (9.67)
        self.max_signal_range = 30  # Adjusted for wider range (-96 to -50.5)

        # Position estimation parameters
        self.signal_variance_weight = 0.75  # Balanced for larger dataset
        self.distance_weight = 0.25         # Complementary weight
        self.min_aps_for_estimation = 4     # Increased due to more data
        self.confidence_normalization = 6.0  # Adjusted for larger dataset

        # Signal quality calculation
        self.signal_strength_offset = 105   # Adjusted for signal range
        self.stability_factor = 1.2         # Slightly increased for larger dataset

        # Reference power adjustments
        self.reference_power_offset = -82.75 # Set to mean signal strength
        self.min_reliable_power = -90        # Set to 25th percentile

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