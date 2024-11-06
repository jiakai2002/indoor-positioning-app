import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class MaterialType(Enum):
    """Enumeration of different wall/floor materials and their attenuation factors"""
    CONCRETE = 12.0  # dB
    BRICK = 8.0
    PLASTERBOARD = 5.0
    GLASS = 2.0
    WOOD = 3.0
    METAL = 15.0

class PropagationModel(Enum):
    """Different propagation models for different environments"""
    FREE_SPACE = "free_space"
    LOG_DISTANCE = "log_distance"
    ITU = "itu"
    COST231 = "cost231"

@dataclass
class WallInfo:
    """Information about walls between AP and measurement point"""
    material: MaterialType
    thickness: float  # in meters
    count: int = 1

@dataclass
class EnvironmentalFactors:
    """Environmental factors affecting signal propagation"""
    walls: List[WallInfo]
    floor_material: MaterialType
    floor_count: int
    humidity: float  # percentage
    temperature: float  # celsius
    is_line_of_sight: bool

@dataclass
class AccessPoint:
    mac_address: str
    locations: List[Tuple[str, float]]
    ssid: str
    channel: int
    radio_type: str
    reference_power: float = -50.0
    path_loss_exponent: float = 3.0  # Typical indoor environment value

    def calculate_distance(self, rssi: float) -> float:
        """Calculate approximate distance using log-distance path loss model"""
        return 10 ** ((self.reference_power - rssi) / (10 * self.path_loss_exponent))

class EnhancedAccessPoint(AccessPoint):
    """Enhanced version of AccessPoint with advanced propagation models"""
    def __init__(self,
                 mac_address: str,
                 locations: List[Tuple[str, float]],
                 ssid: str,
                 channel: int,
                 radio_type: str,
                 reference_power: float = -50.0,
                 frequency: float = 2.4,  # GHz
                 antenna_gain: float = 2.0,  # dBi
                 default_model: PropagationModel = PropagationModel.LOG_DISTANCE):

        super().__init__(mac_address, locations, ssid, channel, radio_type, reference_power)
        self.frequency = frequency
        self.antenna_gain = antenna_gain
        self.default_model = default_model

        # Model-specific parameters
        self.model_params = {
            PropagationModel.FREE_SPACE: {
                'exp': 2.0
            },
            PropagationModel.LOG_DISTANCE: {
                'exp': 3.0,
                'sigma': 3.0  # Shadow fading standard deviation
            },
            PropagationModel.ITU: {
                'N': 28,  # Distance power loss coefficient
                'Lf': 0  # Floor penetration loss factor
            },
            PropagationModel.COST231: {
                'exp': 3.5,
                'LC': 0  # Constant loss factor
            }
        }

    def calculate_wall_attenuation(self, walls: List[WallInfo]) -> float:
        """Calculate total wall attenuation"""
        total_attenuation = 0.0
        for wall in walls:
            # Basic WAF calculation
            base_attenuation = wall.material.value

            # Adjust for thickness
            thickness_factor = wall.thickness / 0.15  # normalized to 15cm reference

            # Multiple wall effect (not linear due to multi-path)
            count_factor = np.sqrt(wall.count)  # Sub-linear scaling for multiple walls

            wall_attenuation = base_attenuation * thickness_factor * count_factor
            total_attenuation += wall_attenuation

        return total_attenuation

    def calculate_floor_attenuation(self,
                                  material: MaterialType,
                                  floor_count: int) -> float:
        """Calculate floor attenuation"""
        if floor_count == 0:
            return 0.0

        base_attenuation = material.value * 1.5  # Floors typically have more attenuation

        # Non-linear scaling for multiple floors
        # First floor has full impact, subsequent floors have diminishing effect
        floor_factor = 1.0
        total_attenuation = 0.0

        for i in range(floor_count):
            total_attenuation += base_attenuation * floor_factor
            floor_factor *= 0.7  # Each additional floor has 70% of previous floor's impact

        return total_attenuation

    def adjust_for_environmental_factors(self,
                                      rssi: float,
                                      env_factors: EnvironmentalFactors) -> float:
        """Adjust RSSI based on environmental factors"""
        # Wall attenuation
        waf = self.calculate_wall_attenuation(env_factors.walls)

        # Floor attenuation
        faf = self.calculate_floor_attenuation(
            env_factors.floor_material,
            env_factors.floor_count
        )

        # Temperature and humidity effects
        # Empirical adjustment based on ITU-R P.676-12
        temp_factor = 0.1 * (env_factors.temperature - 20)  # Reference temp: 20°C
        humidity_factor = 0.05 * (env_factors.humidity - 50)  # Reference humidity: 50%

        # Line of sight bonus
        los_factor = 5.0 if env_factors.is_line_of_sight else 0.0

        adjusted_rssi = rssi - waf - faf - temp_factor - humidity_factor + los_factor
        return adjusted_rssi

    def calculate_distance_free_space(self, rssi: float) -> float:
        """Free space path loss model"""
        params = self.model_params[PropagationModel.FREE_SPACE]
        wavelength = 299792458 / (self.frequency * 1e9)  # c/f

        # Free space path loss equation
        pl = self.reference_power - rssi
        distance = wavelength / (4 * np.pi) * 10 ** (pl / (10 * params['exp']))
        return distance

    def calculate_distance_log_distance(self,
                                     rssi: float,
                                     env_factors: Optional[EnvironmentalFactors] = None) -> float:
        """Log-distance path loss model with shadow fading"""
        params = self.model_params[PropagationModel.LOG_DISTANCE]

        if env_factors:
            rssi = self.adjust_for_environmental_factors(rssi, env_factors)

        # Add random shadow fading
        shadow_fading = np.random.normal(0, params['sigma'])
        adjusted_rssi = rssi + shadow_fading

        # Log-distance calculation
        pl = self.reference_power - adjusted_rssi
        distance = 10 ** (pl / (10 * params['exp']))
        return distance

    def calculate_distance_itu(self,
                             rssi: float,
                             env_factors: Optional[EnvironmentalFactors] = None) -> float:
        """ITU indoor propagation model"""
        params = self.model_params[PropagationModel.ITU]

        if env_factors:
            rssi = self.adjust_for_environmental_factors(rssi, env_factors)
            params['Lf'] = env_factors.floor_count * 15  # 15 dB loss per floor (typical)

        # ITU model calculation
        pl = self.reference_power - rssi
        distance = 10 ** ((pl - 20 * np.log10(self.frequency) - params['Lf'] - 28) / params['N'])
        return distance

    def calculate_distance_cost231(self,
                                 rssi: float,
                                 env_factors: Optional[EnvironmentalFactors] = None) -> float:
        """COST231 multi-wall model"""
        params = self.model_params[PropagationModel.COST231]

        if env_factors:
            rssi = self.adjust_for_environmental_factors(rssi, env_factors)
            # Calculate constant loss factor based on walls
            params['LC'] = sum(wall.material.value for wall in env_factors.walls)

        # COST231 calculation
        pl = self.reference_power - rssi - params['LC']
        distance = 10 ** (pl / (10 * params['exp']))
        return distance

    def calculate_distance(self,
                         rssi: float,
                         model: Optional[PropagationModel] = None,
                         env_factors: Optional[EnvironmentalFactors] = None) -> float:
        """Calculate distance using specified or default propagation model"""
        if model is None:
            model = self.default_model

        distance_calculators = {
            PropagationModel.FREE_SPACE: self.calculate_distance_free_space,
            PropagationModel.LOG_DISTANCE: self.calculate_distance_log_distance,
            PropagationModel.ITU: self.calculate_distance_itu,
            PropagationModel.COST231: self.calculate_distance_cost231
        }

        calculator = distance_calculators.get(model)
        if calculator is None:
            raise ValueError(f"Unsupported propagation model: {model}")

        if model == PropagationModel.FREE_SPACE:
            return calculator(rssi)
        else:
            return calculator(rssi, env_factors)

class MaterialType(Enum):
    """Enumeration of different wall/floor materials and their attenuation factors"""
    CONCRETE = 12.0  # dB
    BRICK = 8.0
    PLASTERBOARD = 5.0
    GLASS = 2.0
    WOOD = 3.0
    METAL = 15.0

@dataclass
class WallInfo:
    """Information about walls between locations"""
    material: MaterialType
    thickness: float  # in meters
    count: int = 1

class EnhancedWiFiPositioning:
    def __init__(self, grid_size: int = 50):
        self.ap_dict: Dict[str, dict] = {}  # Changed to store AP info directly
        self.grid_size = grid_size
        self.location_coords = {
            'FToilet': (26, 13),
            'GSR2-1': (25, 5),
            'GSR2-3/2': (16, 2),
            'GSR2-4': (16, 6),
            'PrintingRoom': (28, 9),
            'Stairs1': (31, 9),
            'Stair2': (11, 8),
            'LW2.1b': (30, 20),
            'Lift2': (23, 21),
            'Lift1': (20, 21),
            'MToilet': (18, 11),
            'Walkway': (15, 21),
            'CommonArea': (13, 35),
            'Stairs3': (12, 38),
            'SR2-4b': (9, 40),
            'SR2-4a': (9, 35),
            'SR2-3b': (11, 30),
            'GSR2-6': (11, 27),
            'SR2-3a': (11, 25),
            'SR2-2a': (11, 15),
            'SR2-2b': (11, 20),
            'SR2-1b': (11, 10),
            'SR2-1a': (10, 6),
            'Stairs2': (12, 8),
            'LW2.1a': (28, 9),
        }
        self.scaler = MinMaxScaler()
        self.wall_map = self._initialize_wall_map()
        self.fingerprints = {}
        self._initialize_grid()

    def _initialize_wall_map(self) -> Dict[Tuple[str, str], List[WallInfo]]:
        wall_map = {}

        # Study Room Walls to Walkway
        for room in ['SR2-1a', 'SR2-1b', 'SR2-2a', 'SR2-2b', 'SR2-3a', 'SR2-3b', 'SR2-4a', 'SR2-4b']:
            wall_map[(room, 'Walkway')] = [
                WallInfo(MaterialType.GLASS, 0.10, 1),
                WallInfo(MaterialType.PLASTERBOARD, 0.15, 1)
            ]

        # Adjacent Study Room Walls
        for room1, room2 in [('SR2-1a', 'SR2-1b'), ('SR2-2a', 'SR2-2b'),
                            ('SR2-3a', 'SR2-3b'), ('SR2-4a', 'SR2-4b')]:
            wall_map[(room1, room2)] = [
                WallInfo(MaterialType.PLASTERBOARD, 0.15, 1)
            ]

        # Group Study Room Walls to Walkway
        for room in ['GSR2-1', 'GSR2-3/2', 'GSR2-4', 'GSR2-6']:
            wall_map[(room, 'Walkway')] = [
                WallInfo(MaterialType.GLASS, 0.10, 1),
                WallInfo(MaterialType.PLASTERBOARD, 0.15, 1)
            ]

        # Toilet Area Walls
        for toilet, adjacents in {
            'FToilet': ['Walkway', 'PrintingRoom', 'Stairs1'],
            'MToilet': ['Walkway', 'GSR2-1', 'PrintingRoom']
        }.items():
            for adj in adjacents:
                wall_map[(toilet, adj)] = [
                    WallInfo(MaterialType.CONCRETE, 0.20, 1),
                    WallInfo(MaterialType.METAL, 0.05, 1)
                ]

        # Common Area to Walkway
        wall_map[('CommonArea', 'Walkway')] = [
            WallInfo(MaterialType.GLASS, 0.10, 2),
            WallInfo(MaterialType.PLASTERBOARD, 0.15, 1)
        ]

        # Lift Walls to Walkway
        for lift in ['Lift1', 'Lift2']:
            wall_map[(lift, 'Walkway')] = [
                WallInfo(MaterialType.CONCRETE, 0.25, 1),
                WallInfo(MaterialType.METAL, 0.10, 1)
            ]

        # Stair Walls to Walkway
        for stairs in ['Stairs1', 'Stairs2', 'Stairs3']:
            wall_map[(stairs, 'Walkway')] = [
                WallInfo(MaterialType.CONCRETE, 0.25, 1)
            ]

        # Printing Room to LW2.1a
        wall_map[('PrintingRoom', 'LW2.1a')] = [
            WallInfo(MaterialType.PLASTERBOARD, 0.15, 1),
            WallInfo(MaterialType.METAL, 0.05, 1)
        ]

        # Make wall map symmetric
        symmetric_map = {}
        for (loc1, loc2), walls in wall_map.items():
            symmetric_map[(loc2, loc1)] = walls
        wall_map.update(symmetric_map)

        return wall_map


    def calculate_wall_attenuation(self, from_location: str, to_location: str) -> float:
        """Calculate total wall attenuation between two locations"""
        location_pair = tuple(sorted([from_location, to_location]))
        walls = self.wall_map.get(location_pair, [])

        total_attenuation = 0.0
        for wall in walls:
            base_attenuation = wall.material.value
            thickness_factor = wall.thickness / 0.15

            for i in range(wall.count):
                wall_factor = 0.7 ** i
                total_attenuation += base_attenuation * thickness_factor * wall_factor

        return total_attenuation

    def _initialize_grid(self):
        """Initialize the signal strength grid for heatmap generation"""
        self.grid_x = np.linspace(0, self.grid_size, self.grid_size)
        self.grid_y = np.linspace(0, self.grid_size, self.grid_size)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)

    def process_wifi_data(self, data: pd.DataFrame):
        """Process WiFi scan data and build fingerprint database"""
        logging.info("Processing WiFi scan data...")

        # Group by BSSID and location, calculate median RSSI
        grouped_data = data.groupby(['bssid', 'location'])['dbm'].median().reset_index()

        # Filter out weak signals and unreliable APs
        ap_counts = grouped_data.groupby('bssid').size()
        reliable_aps = ap_counts[ap_counts >= 5].index

        filtered_data = grouped_data[grouped_data['bssid'].isin(reliable_aps)]

        # Build fingerprint database
        self.fingerprints = {}
        for location in data['location'].unique():
            fingerprint = {}
            location_data = filtered_data[filtered_data['location'] == location]

            for _, row in location_data.iterrows():
                fingerprint[row['bssid']] = {
                    'rssi': row['dbm'],
                    'distance': self._estimate_distance(row['dbm'])
                }

            self.fingerprints[location] = fingerprint

        logging.info(f"Processed {len(reliable_aps)} reliable access points")

    def _estimate_distance(self, rssi: float, reference_power: float = -50.0) -> float:
        """Estimate distance using log-distance path loss model"""
        if rssi >= reference_power:
            return 1.0  # Minimum distance if signal is stronger than reference

        # Path loss exponent for indoor environment
        n = 3.0

        # Log-distance path loss formula
        distance = 10 ** ((reference_power - rssi) / (10 * n))
        return distance

    def _calculate_similarity(self, current_readings: Dict[str, float],
                            fingerprint: Dict[str, Dict],
                            from_location: str,
                            to_location: str) -> float:
        """Calculate similarity score with wall attenuation consideration"""
        common_aps = set(current_readings.keys()) & set(fingerprint.keys())
        if not common_aps:
            return float('-inf')

        wall_attenuation = self.calculate_wall_attenuation(from_location, to_location)

        rssi_diffs = []
        distance_diffs = []

        for ap_id in common_aps:
            current_rssi = current_readings[ap_id]
            fp_data = fingerprint[ap_id]

            # Adjust RSSI for wall attenuation
            adjusted_current_rssi = current_rssi + wall_attenuation

            rssi_diff = abs(adjusted_current_rssi - fp_data['rssi'])
            rssi_diffs.append(rssi_diff)

            current_distance = self._estimate_distance(adjusted_current_rssi)
            distance_diff = abs(current_distance - fp_data['distance'])
            distance_diffs.append(distance_diff)

        rssi_score = np.mean(rssi_diffs) / 100.0
        distance_score = np.mean(distance_diffs) / 10.0

        combined_score = 0.7 * rssi_score + 0.3 * distance_score

        # Consider AP coverage
        ap_ratio = len(common_aps) / max(len(current_readings), len(fingerprint))
        final_score = combined_score * (1 + (1 - ap_ratio))

        return -final_score

    def estimate_location(self, current_readings: pd.DataFrame, k: int = 3) -> Tuple[str, Tuple[float, float]]:
        """Estimate location using enhanced algorithm with wall attenuation"""
        current_rssi = {row['bssid']: row['dbm'] for _, row in current_readings.iterrows()}

        if not self.fingerprints:
            raise ValueError("Fingerprint database is empty. Run process_wifi_data first.")

        similarities = []
        reference_location = list(self.fingerprints.keys())[0]

        for location, fingerprint in self.fingerprints.items():
            similarity = self._calculate_similarity(
                current_rssi,
                fingerprint,
                reference_location,
                location
            )
            similarities.append((location, similarity))

        if not similarities:
            return "Unknown", (0.0, 0.0)

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for location, similarity in top_k:
            weight = np.exp(similarity * 10)
            coords = self.location_coords[location]

            weighted_x += coords[0] * weight
            weighted_y += coords[1] * weight
            total_weight += weight

        if total_weight == 0:
            return top_k[0][0], self.location_coords[top_k[0][0]]

        estimated_coords = (weighted_x / total_weight, weighted_y / total_weight)
        return top_k[0][0], estimated_coords

    def generate_heatmap(self, current_readings: pd.DataFrame) -> np.ndarray:
        """Generate signal strength heatmap with wall attenuation consideration"""
        points = []
        values = []

        for location, fingerprint in self.fingerprints.items():
            if fingerprint:
                avg_rssi = np.mean([data['rssi'] for data in fingerprint.values()])
                points.append(self.location_coords[location])
                values.append(avg_rssi)

        if not points:
            return np.full((self.grid_size, self.grid_size), -100)

        points = np.array(points)
        values = np.array(values)

        grid_z = griddata(points, values, (self.X, self.Y), method='cubic', fill_value=-100)
        return grid_z

    def estimate_location_nearest_neighbor(self, current_readings: pd.DataFrame) -> str:
        """Estimate location using the strongest signal strength approach"""
        current_rssi = {row['bssid']: row['dbm'] for _, row in current_readings.iterrows()}
        max_similarity = float('-inf')
        estimated_location = None

        for location, fingerprint in self.fingerprints.items():
            common_aps = set(current_rssi.keys()) & set(fingerprint.keys())
            if not common_aps:
                continue

            similarity = sum(current_rssi[ap] for ap in common_aps)
            if similarity > max_similarity:
                max_similarity = similarity
                estimated_location = location

        return estimated_location

    def visualize_results(self, results_df: pd.DataFrame, current_readings: pd.DataFrame,
                         floor_plan_path: str = 'data/raw/floorplan.png'):
        """Visualize results with heatmap overlay"""
        try:
            floor_plan = mpimg.imread(floor_plan_path)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Plot 1: Results on floor plan
            ax1.imshow(floor_plan, extent=[0, self.grid_size, 0, self.grid_size])

            # Plot actual positions
            actual_x = results_df['Actual X']
            actual_y = results_df['Actual Y']
            ax1.scatter(actual_x, actual_y, c='blue', marker='o', s=100,
                       label='Actual Positions', alpha=0.7)

            # Plot estimated positions
            estimated_x = results_df['Estimated X']
            estimated_y = results_df['Estimated Y']
            ax1.scatter(estimated_x, estimated_y, c='red', marker='x', s=100,
                       label='Estimated Positions', alpha=0.7)

            # Draw connection lines
            for _, row in results_df.iterrows():
                ax1.plot([row['Actual X'], row['Estimated X']],
                        [row['Actual Y'], row['Estimated Y']],
                        'g-', alpha=0.3)

            ax1.set_title('Positioning Results')
            ax1.legend()

            # Plot 2: Signal Strength Heatmap
            heatmap_data = self.generate_heatmap(current_readings)
            im = ax2.imshow(heatmap_data, extent=[0, self.grid_size, 0, self.grid_size],
                            cmap='YlOrRd', alpha=0.7)
            plt.colorbar(im, ax=ax2, label='Signal Strength (dBm)')

            # Overlay AP locations
            for location, coords in self.location_coords.items():
                ax2.plot(coords[0], coords[1], 'ko', markersize=5)
                ax2.annotate(location, (coords[0], coords[1]), fontsize=8, alpha=0.7)

            ax2.set_title('Signal Strength Heatmap')

            plt.tight_layout()
            plt.show()

            # Print error statistics
            errors = results_df['Error Distance'].dropna()
            if len(errors) > 0:
                print("\nError Statistics:")
                print(f"Mean Error Distance: {errors.mean():.2f} units")
                print(f"Median Error Distance: {errors.median():.2f} units")
                print(f"Maximum Error Distance: {errors.max():.2f} units")
                print(f"Standard Deviation: {errors.std():.2f} units")

                # Calculate percentile errors
                percentiles = [25, 50, 75, 90]
                for p in percentiles:
                    print(f"{p}th Percentile Error: {np.percentile(errors, p):.2f} units")

        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")

def plot_error_cdf(results_df):
    """Plot CDF of error distances"""
    # Sorting error distances
    error_distances = results_df['Error Distance'].dropna().sort_values().values

    # Creating the CDF values
    cdf_y = np.arange(1, len(error_distances)+1) / len(error_distances)

    # Plotting the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(error_distances, cdf_y, marker='.', linestyle='none')
    plt.xlabel('Error Distance (units)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Error Distances')
    plt.grid()
    plt.show()

    # Display summary statistics for interpretation
    print("\nCDF Summary for Error Distances:")
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        print(f"{p}th Percentile: {np.percentile(error_distances, p):.2f} units")

def plot_location_accuracy_heatmap(results_df, positioning, floor_plan_path='data/raw/floorplan.png'):
    """Generate a location-based heatmap overlaying the floor plan to show error distances."""
    # Load the floor plan image
    floor_plan = mpimg.imread(floor_plan_path)

    # Calculate mean error by location
    error_by_location = results_df.groupby('Actual Location')['Error Distance'].mean()

    # Prepare data for plotting
    x_coords = [positioning.location_coords[loc][0] for loc in error_by_location.index]
    y_coords = [positioning.location_coords[loc][1] for loc in error_by_location.index]
    errors = error_by_location.values

    # Plot setup
    plt.figure(figsize=(12, 10))
    plt.imshow(floor_plan, extent=[0, positioning.grid_size, 0, positioning.grid_size])

    # Create scatter plot overlay for error distances
    scatter = plt.scatter(x_coords, y_coords, c=errors, s=100, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Mean Error Distance (units)')

    # Labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Location-Based Error Heatmap Over Floor Plan')
    plt.grid(False)

    plt.show()


def main():
    try:
        # Load and process data
        data = pd.read_csv('data/processed/processed_rain_dataset.csv')

        # Analyze location distribution
        location_counts = data['location'].value_counts()
        print("\nSamples per location:")
        print(location_counts)

        # Filter out locations with too few samples
        min_samples = 5  # Minimum number of samples required per location
        valid_locations = location_counts[location_counts >= min_samples].index

        if len(valid_locations) == 0:
            raise ValueError("No locations have sufficient samples for training")

        print(f"\nUsing {len(valid_locations)} locations with {min_samples}+ samples each")
        filtered_data = data[data['location'].isin(valid_locations)]

        # keep only if contains SMU or eduroam
        filtered_data = filtered_data[filtered_data['ssid'].notna() & filtered_data['ssid'].str.contains('SMU|eduroam')]

        # location counts for filtered data
        location_counts = filtered_data['location'].value_counts()
        print("\nSamples per location after filtering:")
        print(location_counts)

        # Try stratified split first, fall back to random if necessary
        try:
            train_data, test_data = train_test_split(
                filtered_data,
                test_size=0.2,
                stratify=filtered_data['location'],
                random_state=69
            )
            print("Using stratified split")
        except ValueError as e:
            print("Falling back to random split due to insufficient samples")
            train_data, test_data = train_test_split(
                filtered_data,
                test_size=0.2,
                random_state=69
            )

        # Initialize and train positioning system
        positioning = EnhancedWiFiPositioning()
        positioning.process_wifi_data(train_data)

        # Test and collect results
        results = []
        for location, group in test_data.groupby('location'):
            # Ensure we have the location coordinates
            if location not in positioning.location_coords:
                print(f"Warning: No coordinates for location {location}, skipping")
                continue

            estimated_location, estimated_coords = positioning.estimate_location(group)
            actual_coords = positioning.location_coords[location]

            error_distance = np.sqrt(
                (actual_coords[0] - estimated_coords[0])**2 +
                (actual_coords[1] - estimated_coords[1])**2
            )

            # Baseline estimation
            baseline_location = positioning.estimate_location_nearest_neighbor(group)
            baseline_coords = positioning.location_coords.get(baseline_location, (0.0, 0.0))
            baseline_error = np.sqrt(
                (actual_coords[0] - baseline_coords[0])**2 +
                (actual_coords[1] - baseline_coords[1])**2
            )

            results.append({
                'Actual Location': location,
                'Estimated Location': estimated_location,
                'Actual X': actual_coords[0],
                'Actual Y': actual_coords[1],
                'Estimated X': estimated_coords[0],
                'Estimated Y': estimated_coords[1],
                'Error Distance': error_distance,
                'Baseline Location': baseline_location,
                'Baseline Error Distance': baseline_error
            })

            print(f"Location: {location:15} | Estimated: {estimated_location:15} | "
                  f"Error: {error_distance:.2f} units")

        # Create results DataFrame and visualize
        if results:
            results_df = pd.DataFrame(results)

            # Use a random group for heatmap generation if the last group is not available
            heatmap_group = group if 'group' in locals() else test_data.groupby('location').first()

            positioning.visualize_results(results_df, heatmap_group)

            # Additional analysis
            error_by_location = results_df.groupby('Actual Location')['Error Distance'].agg(
                ['mean', 'std', 'count']).sort_values('mean')
            print("\nError Analysis by Location:")
            print(error_by_location)

            plot_error_cdf(results_df)
            plot_location_accuracy_heatmap(results_df, positioning)

            # Print overall statistics
            print("\nOverall Statistics:")
            print(f"Total Samples: {len(results_df)}")
            print(f"Mean Error: {results_df['Error Distance'].mean():.2f} units")
            print(f"Median Error: {results_df['Error Distance'].median():.2f} units")
            print(f"Standard Deviation: {results_df['Error Distance'].std():.2f} units")
            print(f"Accuracy (within 1 unit): {100 * (results_df['Error Distance'] <= 1).mean():.2f}%")
            print(f"Accuracy (within 2 units): {100 * (results_df['Error Distance'] <= 2).mean():.2f}%")
            print(f"Accuracy (within 5 units): {100 * (results_df['Error Distance'] <= 5).mean():.2f}%")
            print(f"Accuracy (within 10 units): {100 * (results_df['Error Distance'] <= 10).mean():.2f}%")

            # max error
            max_error = results_df.loc[results_df['Error Distance'].idxmax()]
            print(f"\nMaximum Error: {max_error['Error Distance']:.2f} units")
            print(f"Actual Location: {max_error['Actual Location']}")
            print(f"Estimated Location: {max_error['Estimated Location']}")

        else:
            print("No valid results to display")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main()
