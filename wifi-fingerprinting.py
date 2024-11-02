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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

class WiFiPositioning:
    """
    A class to perform WiFi-based indoor positioning using fingerprinting techniques.

    Attributes:
        ap_dict (Dict[str, AccessPoint]): Dictionary of access points with BSSID as keys.
        grid_size (int): Size of the grid for heatmap generation.
        location_coords (Dict[str, Tuple[int, int]]): Coordinates of known locations.
        scaler (MinMaxScaler): Scaler for normalizing data.
        grid_x (np.ndarray): X-coordinates of the grid.
        grid_y (np.ndarray): Y-coordinates of the grid.
        X (np.ndarray): Meshgrid X-coordinates.
        Y (np.ndarray): Meshgrid Y-coordinates.
        fingerprints (Dict[str, Dict]): Fingerprint database for known locations.

    Methods:
        __init__(grid_size: int = 50):
            Initializes the WiFiPositioning instance with a specified grid size.

        _initialize_grid():
            Initializes the signal strength grid for heatmap generation.

        process_wifi_data(data: pd.DataFrame):
            Processes WiFi scan data to build a database of reliable access points.

        _estimate_path_loss(locations_rssi: List[Tuple[str, float]]) -> float:
            Estimates the path loss exponent based on RSSI measurements.

        _build_fingerprint_database():
            Builds a radio map for fingerprinting based on processed WiFi data.

        _calculate_similarity(current_readings: Dict[str, float], fingerprint: Dict) -> float:
            Calculates similarity score using both RSSI and estimated distances.

        estimate_location(current_readings: pd.DataFrame, k: int = 3) -> Tuple[str, Tuple[float, float]]:
            Estimates the location using k-nearest neighbors with weighted averaging.

        generate_heatmap(current_readings: pd.DataFrame) -> np.ndarray:
            Generates a signal strength heatmap based on current WiFi readings.

        visualize_results(results_df: pd.DataFrame, current_readings: pd.DataFrame, floor_plan_path: str = 'data/raw/floorplan.png'):
            Visualizes the positioning results and signal strength heatmap on a floor plan.
    """
    def __init__(self, grid_size: int = 50):
        self.ap_dict: Dict[str, AccessPoint] = {}
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
        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize the signal strength grid for heatmap generation"""
        self.grid_x = np.linspace(0, self.grid_size, self.grid_size)
        self.grid_y = np.linspace(0, self.grid_size, self.grid_size)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)

    def process_wifi_data(self, data: pd.DataFrame):
        logging.info("Processing WiFi scan data...")

        # Group by BSSID and location, calculate median RSSI
        grouped_data = data.groupby(['bssid', 'location'])['dbm'].median().reset_index()

        # Filter out weak signals and unreliable APs
        ap_counts = grouped_data.groupby('bssid').size()
        reliable_aps = ap_counts[ap_counts >= 5].index  # APs seen in at least 5 locations

        filtered_data = grouped_data[grouped_data['bssid'].isin(reliable_aps)]

        for bssid, group in filtered_data.groupby('bssid'):
            locations_rssi = [(row['location'], row['dbm']) for _, row in group.iterrows()]

            ap_info = data[data['bssid'] == bssid].iloc[0]
            ap = AccessPoint(
                mac_address=bssid,
                locations=locations_rssi,
                ssid=ap_info['ssid'],
                channel=ap_info['channel'],
                radio_type=ap_info['radio_type'],
                path_loss_exponent=self._estimate_path_loss(locations_rssi)
            )
            self.ap_dict[bssid] = ap

        logging.info(f"Processed {len(self.ap_dict)} reliable access points")
        self._build_fingerprint_database()

    def _estimate_path_loss(self, locations_rssi: List[Tuple[str, float]]) -> float:
        """Estimate path loss exponent based on RSSI measurements"""
        if len(locations_rssi) < 2:
            return 3.0  # Default value

        rssi_values = np.array([rssi for _, rssi in locations_rssi])
        rssi_std = np.std(rssi_values)

        # Adjust path loss based on signal stability
        if rssi_std < 5:
            return 2.5  # More stable signals suggest clearer path
        elif rssi_std < 10:
            return 3.0  # Typical indoor environment
        else:
            return 3.5  # More obstacles/interference

    def _build_fingerprint_database(self):
        """Build radio map for fingerprinting"""
        self.fingerprints = {}
        for location, coords in self.location_coords.items():
            fingerprint = {}
            for bssid, ap in self.ap_dict.items():
                rssi = next((rssi for loc, rssi in ap.locations if loc == location), None)
                if rssi is not None:
                    fingerprint[bssid] = {
                        'rssi': rssi,
                        'distance': ap.calculate_distance(rssi)
                    }
            self.fingerprints[location] = fingerprint

    def _calculate_similarity(self, current_readings: Dict[str, float],
                            fingerprint: Dict[str, Dict]) -> float:
        """Calculate improved similarity score using both RSSI and estimated distances"""
        common_aps = set(current_readings.keys()) & set(fingerprint.keys())
        if not common_aps:
            return float('-inf')

        rssi_diffs = []
        distance_diffs = []

        for ap_id in common_aps:
            current_rssi = current_readings[ap_id]
            fp_data = fingerprint[ap_id]

            # RSSI difference
            rssi_diff = abs(current_rssi - fp_data['rssi'])
            rssi_diffs.append(rssi_diff)

            # Distance difference
            current_distance = self.ap_dict[ap_id].calculate_distance(current_rssi)
            distance_diff = abs(current_distance - fp_data['distance'])
            distance_diffs.append(distance_diff)

        # Normalize differences
        rssi_score = np.mean(rssi_diffs) / 100.0  # Normalized by typical RSSI range
        distance_score = np.mean(distance_diffs) / 10.0  # Normalized by typical distance

        # Combine scores with weights
        combined_score = 0.6 * rssi_score + 0.4 * distance_score

        # Penalize for missing APs
        ap_ratio = len(common_aps) / max(len(current_readings), len(fingerprint))
        final_score = combined_score * (1 + (1 - ap_ratio))

        return -final_score  # Negative score (closer to 0 is better)

    def estimate_location(self, current_readings: pd.DataFrame, k: int = 3) -> Tuple[str, Tuple[float, float]]:
        """Estimate location using improved kNN with weighted averaging"""
        current_rssi = {row['bssid']: row['dbm'] for _, row in current_readings.iterrows()}

        similarities = []
        for location, fingerprint in self.fingerprints.items():
            similarity = self._calculate_similarity(current_rssi, fingerprint)
            similarities.append((location, similarity))

        if not similarities:
            return "Unknown", (0.0, 0.0)

        # Sort by similarity (highest/least negative first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # Calculate confidence-weighted average
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for location, similarity in top_k:
            # Convert similarity to weight using softmax-like formula
            weight = np.exp(similarity * 10)  # Amplify differences
            coords = self.location_coords[location]

            weighted_x += coords[0] * weight
            weighted_y += coords[1] * weight
            total_weight += weight

        if total_weight == 0:
            return top_k[0][0], self.location_coords[top_k[0][0]]

        estimated_coords = (weighted_x / total_weight, weighted_y / total_weight)
        return top_k[0][0], estimated_coords

    def generate_heatmap(self, current_readings: pd.DataFrame) -> np.ndarray:
        """Generate signal strength heatmap"""
        current_rssi = {row['bssid']: row['dbm'] for _, row in current_readings.iterrows()}

        # Create points for interpolation
        points = []
        values = []

        # Add known measurements
        for location, coords in self.location_coords.items():
            if location in self.fingerprints:
                avg_rssi = np.mean([data['rssi'] for data in self.fingerprints[location].values()])
                points.append(coords)
                values.append(avg_rssi)

        # Convert to numpy arrays
        points = np.array(points)
        values = np.array(values)

        # Interpolate values across the grid
        grid_z = griddata(points, values, (self.X, self.Y), method='cubic', fill_value=-100)

        return grid_z

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

# [Previous code remains the same until the main function]

def main():
    try:
        # Load and process data
        data = pd.read_csv('data/processed/merged.csv')

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
        positioning = WiFiPositioning()
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

            results.append({
                'Actual Location': location,
                'Estimated Location': estimated_location,
                'Actual X': actual_coords[0],
                'Actual Y': actual_coords[1],
                'Estimated X': estimated_coords[0],
                'Estimated Y': estimated_coords[1],
                'Error Distance': error_distance
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
