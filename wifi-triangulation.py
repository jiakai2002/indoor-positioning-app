import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.image import imread
import seaborn as sns
from pathlib import Path
import argparse
import sys

@dataclass
class AccessPoint:
    mac_address: str
    x: float
    y: float
    reference_power: float
    channel: int = 1
    ssid: str = ""

@dataclass
class RSSIMeasurement:
    ap: AccessPoint
    rssi: float
    timestamp: datetime
    people_count: int = 0
    has_los: bool = True

def classify_location(x: float, y: float) -> str:
    """
    Classify the location based on x and y coordinates.

    Returns:
        A string representing the location zone.
    """
    if 0 <= x <= 10 and 0 <= y <= 10:
        return 'Room A'
    elif 10 < x <= 20 and 0 <= y <= 10:
        return 'Room B'
    elif 0 <= x <= 10 and 10 < y <= 20:
        return 'Room C'
    elif 10 < x <= 20 and 10 < y <= 20:
        return 'Room D'
    else:
        return 'Unknown Location'

class DataLoader:
    """Modified data loader to handle your specific data format"""

    @staticmethod
    def load_access_points(filepath: str) -> List[AccessPoint]:
        """
        Load access point data from your CSV format
        Expected columns: location, ssid, network type, authentication, encryption,
                         bssid, dbm, radio_type, channel
        """
        try:
            ap_df = pd.read_csv(filepath)
            required_columns = ['bssid', 'dbm', 'ssid', 'channel']
            if not all(col.lower() in map(str.lower, ap_df.columns) for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            # Create a grid layout for APs since physical coordinates aren't provided
            num_aps = len(ap_df)
            grid_size = int(np.ceil(np.sqrt(num_aps)))
            spacing = 5.0  # 5 meters between APs

            aps = []
            for i, row in ap_df.iterrows():
                # Calculate grid position
                grid_x = (i % grid_size) * spacing
                grid_y = (i // grid_size) * spacing

                ap = AccessPoint(
                    mac_address=row['bssid'],
                    x=grid_x,
                    y=grid_y,
                    reference_power=float(row['dbm']),
                    channel=int(row['channel']) if pd.notna(row['channel']) else 1,
                    ssid=str(row['ssid'])
                )
                aps.append(ap)
            return aps
        except Exception as e:
            print(f"Error loading access points: {str(e)}")
            return []

    @staticmethod
    def load_measurements(filepath: str, ap_dict: Dict[str, AccessPoint]) -> List[RSSIMeasurement]:
        """
        Load RSSI measurements from your data format
        Uses current timestamp since temporal data isn't provided
        """
        try:
            measurements_df = pd.read_csv(filepath)
            current_time = datetime.now()

            measurements = []
            for _, row in measurements_df.iterrows():
                if row['bssid'] not in ap_dict:
                    print(f"Warning: AP {row['bssid']} not found in AP list")
                    continue

                measurement = RSSIMeasurement(
                    ap=ap_dict[row['bssid']],
                    rssi=float(row['dbm']),
                    timestamp=current_time,
                    people_count=0,  # Default since not provided
                    has_los=True     # Default since not provided
                )
                measurements.append(measurement)
            return measurements
        except Exception as e:
            print(f"Error loading measurements: {str(e)}")
            return []
class AnalysisMetrics:
    """Calculate and store various analysis metrics"""

    def __init__(self, true_positions: pd.DataFrame, estimated_positions: pd.DataFrame):
        self.true_positions = true_positions
        self.estimated_positions = estimated_positions
        self.errors = self._calculate_errors()

    def _calculate_errors(self) -> pd.DataFrame:
        """Calculate positioning errors"""
        errors = pd.DataFrame({
            'error_distance': np.sqrt(
                (self.true_positions['x'] - self.estimated_positions['x'])**2 +
                (self.true_positions['y'] - self.estimated_positions['y'])**2
            ),
            'timestamp': self.true_positions['timestamp'],
            'people_count': self.true_positions['people_count']
        })
        return errors

    def get_basic_stats(self) -> Dict[str, float]:
        """Get basic error statistics"""
        return {
            'mean_error': self.errors['error_distance'].mean(),
            'median_error': self.errors['error_distance'].median(),
            'std_error': self.errors['error_distance'].std(),
            'min_error': self.errors['error_distance'].min(),
            'max_error': self.errors['error_distance'].max(),
            'p90_error': self.errors['error_distance'].quantile(0.9),
            'p95_error': self.errors['error_distance'].quantile(0.95)
        }

    def get_temporal_analysis(self) -> pd.DataFrame:
        """Analyze error patterns over time"""
        return self.errors.set_index('timestamp').resample('1H').agg({
            'error_distance': ['mean', 'std', 'count'],
            'people_count': 'mean'
        })

    def get_crowd_impact(self) -> pd.DataFrame:
        """Analyze impact of crowd size on accuracy"""
        return self.errors.groupby('people_count').agg({
            'error_distance': ['mean', 'std', 'count']
        })

    def plot_error_distribution(self):
        """Plot error distribution"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.errors, x='error_distance', bins=30)
        plt.title('Distribution of Positioning Errors')
        plt.xlabel('Error Distance (meters)')
        plt.ylabel('Count')
        plt.show()

    def plot_temporal_heatmap(self):
        """Plot temporal heatmap of errors"""
        temporal_data = self.errors.copy()
        temporal_data['hour'] = temporal_data['timestamp'].dt.hour
        temporal_data['day'] = temporal_data['timestamp'].dt.date

        pivot_data = temporal_data.pivot_table(
            values='error_distance',
            index='day',
            columns='hour',
            aggfunc='mean'
        )

        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title('Error Heatmap by Hour and Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Date')
        plt.show()

class WiFiPositioning:
    def __init__(self, access_points: List[AccessPoint]):
        self.access_points = access_points
        self.ap_dict = {ap.mac_address: ap for ap in access_points}

    def calculate_distance(self, rssi: float, reference_power: float,
                           people_count: int = 0, has_los: bool = True) -> float:
        """Calculate distance based on RSSI and environmental factors."""
        path_loss_exponent = 2.0 if has_los else 3.5
        people_adjustment = 0.2 * people_count  # Adjust path loss based on crowd density

        adjusted_path_loss = path_loss_exponent + people_adjustment
        return 10 ** ((reference_power - rssi) / (10 * adjusted_path_loss))

    def _error_function(self, point: np.ndarray, measurements: List[RSSIMeasurement]) -> float:
        """Calculate error for optimization."""
        error = 0
        for measurement in measurements:
            ap = measurement.ap
            measured_distance = self.calculate_distance(
                measurement.rssi,
                ap.reference_power,
                measurement.people_count,
                measurement.has_los
            )
            calculated_distance = np.sqrt((point[0] - ap.x) ** 2 + (point[1] - ap.y) ** 2)
            weight = 1.0 if measurement.has_los else 0.6
            error += weight * (measured_distance - calculated_distance) ** 2
        return error

    def estimate_position(self, measurements: List[RSSIMeasurement]) -> Tuple[float, float]:
        """Estimate position using trilateration."""
        if len(measurements) < 3:
            raise ValueError("At least 3 measurements are required for triangulation")

        initial_guess = np.array([
            np.mean([ap.x for ap in self.access_points]),
            np.mean([ap.y for ap in self.access_points])
        ])

        result = minimize(
            self._error_function,
            initial_guess,
            args=(measurements,),
            method='Nelder-Mead'
        )

        return result.x[0], result.x[1]

    def visualize_positioning(self, measurements: List[RSSIMeasurement],
                            true_position: Tuple[float, float] = None,
                            room_dims: Tuple[float, float] = None):
        """Visualize the positioning result with floor plan overlay"""
        plt.figure(figsize=(12, 8))

        # Display floor plan if available
        if self.floor_plan is not None:
            plt.imshow(self.floor_plan, extent=self.floor_plan_extent, alpha=0.5)

        # Plot APs
        ap_x = [ap.x for ap in self.access_points]
        ap_y = [ap.y for ap in self.access_points]
        plt.scatter(ap_x, ap_y, c='blue', marker='^', s=100, label='Access Points')

        # Add AP labels
        for ap in self.access_points:
            plt.annotate(f"{ap.ssid}\n({ap.channel})",
                        (ap.x, ap.y),
                        xytext=(5, 5),
                        textcoords='offset points')

        # Plot estimated position
        est_x, est_y = self.estimate_position(measurements)
        plt.scatter(est_x, est_y, c='red', marker='o', s=100, label='Estimated Position')

        # Plot true position if provided
        if true_position:
            plt.scatter(true_position[0], true_position[1], c='green', marker='*',
                       s=100, label='True Position')

            # Draw error line
            plt.plot([true_position[0], est_x], [true_position[1], est_y],
                    'k--', alpha=0.5, label='Error')

            # Add error distance label
            error_distance = np.sqrt((est_x - true_position[0])**2 +
                                   (est_y - true_position[1])**2)
            plt.annotate(f'Error: {error_distance:.2f}m',
                        ((est_x + true_position[0])/2, (est_y + true_position[1])/2))

        # Draw circles representing the measured distances
        for measurement in measurements:
            ap = measurement.ap
            distance = self.calculate_distance(
                measurement.rssi,
                ap.reference_power,
                measurement.people_count,
                measurement.has_los
            )
            circle = plt.Circle((ap.x, ap.y), distance, fill=False, alpha=0.3)
            plt.gca().add_artist(circle)
            # Add RSSI value label
            plt.annotate(f'{measurement.rssi:.1f} dBm',
                        (ap.x, ap.y),
                        xytext=(5, -5),
                        textcoords='offset points')

        # Set plot limits
        if room_dims:
            plt.xlim(-1, room_dims[0] + 1)
            plt.ylim(-1, room_dims[1] + 1)

        plt.grid(True)
        plt.legend()
        plt.title('WiFi Positioning Visualization')
        plt.xlabel('X coordinate (meters)')
        plt.ylabel('Y coordinate (meters)')
        plt.axis('equal')
        plt.show()

class DataGenerator:
    """Generate test data for WiFi positioning"""

    @staticmethod
    def generate_test_dataset(
        room_width: float = 20.0,
        room_length: float = 30.0,
        measurement_grid_size: float = 2.0,
        num_aps: int = 4,
        measurements_per_point: int = 5,
        time_interval: int = 5  # minutes between measurements
    ) -> Tuple[List[AccessPoint], pd.DataFrame, pd.DataFrame]:
        """
        Generate a test dataset simulating WiFi measurements in a room
        Returns: (access_points, measurements_df, true_positions_df)
        """
        # Create APs at strategic locations
        aps = [
            AccessPoint("AP:01:02:03:04", 0.0, 0.0, -40, 1, "AP1"),
            AccessPoint("AP:01:02:03:05", room_width, 0.0, -40, 6, "AP2"),
            AccessPoint("AP:01:02:03:06", 0.0, room_length, -40, 11, "AP3"),
            AccessPoint("AP:01:02:03:07", room_width, room_length, -40, 6, "AP4")
        ][:num_aps]

        measurements_data = []
        true_positions_data = []
        base_time = datetime.now()

        # Generate grid points
        x_points = np.arange(0, room_width + measurement_grid_size, measurement_grid_size)
        y_points = np.arange(0, room_length + measurement_grid_size, measurement_grid_size)

        for x in x_points:
            for y in y_points:
                for measurement_idx in range(measurements_per_point):
                    timestamp = base_time + timedelta(minutes=measurement_idx * time_interval)
                    people_count = np.random.randint(0, 10)

                    # Record true position
                    true_positions_data.append({
                        'timestamp': timestamp,
                        'x': x,
                        'y': y,
                        'people_count': people_count
                    })

                    # Generate measurements for each AP
                    for ap in aps:
                        distance = np.sqrt((x - ap.x)**2 + (y - ap.y)**2)
                        theoretical_rssi = ap.reference_power - 20 * np.log10(max(distance, 1))

                        # Add various noise factors
                        people_noise = -0.5 * people_count
                        random_noise = np.random.normal(0, 2)
                        time_noise = np.sin(timestamp.hour * np.pi / 12) * 2

                        rssi = theoretical_rssi + people_noise + random_noise + time_noise
                        has_los = np.random.random() > (distance / (room_width + room_length))

                        measurements_data.append({
                            'timestamp': timestamp,
                            'ap_mac': ap.mac_address,
                            'rssi': rssi,
                            'people_count': people_count,
                            'has_los': has_los
                        })

        return (
            aps,
            pd.DataFrame(measurements_data),
            pd.DataFrame(true_positions_data)
        )

    @staticmethod
    def save_test_data(aps: List[AccessPoint],
                      measurements_df: pd.DataFrame,
                      output_dir: str = "./test_data"):
        """Save generated test data to CSV files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save access points
        ap_data = [{
            'mac_address': ap.mac_address,
            'x': ap.x,
            'y': ap.y,
            'reference_power': ap.reference_power,
            'channel': ap.channel,
            'ssid': ap.ssid
        } for ap in aps]
        pd.DataFrame(ap_data).to_csv(f"{output_dir}/access_points.csv", index=False)

        # Save measurements
        measurements_df.to_csv(f"{output_dir}/measurements.csv", index=False)

        print(f"Test data saved to {output_dir}/")

def run_positioning_analysis(
    positioning: WiFiPositioning,
    measurements_df: pd.DataFrame,
    ap_dict: Dict[str, AccessPoint]
) -> None:
    """Run positioning analysis and classify the location."""

    # Group measurements by timestamp
    measurements_grouped = measurements_df.groupby('timestamp')

    # Process measurements and output classified locations
    for timestamp, group in measurements_grouped:
        measurements = []
        for _, row in group.iterrows():
            measurement = RSSIMeasurement(
                ap=ap_dict[row['ap_mac']],
                rssi=row['rssi'],
                timestamp=pd.to_datetime(timestamp),
                people_count=row.get('people_count', 0),
                has_los=row.get('has_los', True)
            )
            measurements.append(measurement)

        try:
            est_x, est_y = positioning.estimate_position(measurements)
            location = classify_location(est_x, est_y)
            print(f"At {timestamp}, you are in {location}.")
        except Exception as e:
            print(f"Error processing measurements at {timestamp}: {str(e)}")

def preprocess_wifi_data(ap_data_path: str) -> tuple[List[AccessPoint], Dict[str, AccessPoint]]:
    """
    Preprocess WiFi data from your format into the required format
    Returns: (list of AccessPoints, dictionary of AccessPoints by MAC)
    """
    # Load and process access points
    data_loader = DataLoader()
    aps = data_loader.load_access_points(ap_data_path)

    # Create AP dictionary
    ap_dict = {ap.mac_address: ap for ap in aps}

    return aps, ap_dict

def main():
    try:
        # Load and process the WiFi data
        aps, ap_dict = preprocess_wifi_data('data/processed/merged.csv')

        # Initialize positioning system
        positioning = WiFiPositioning(aps)

        # Load measurements using the same CSV
        measurements = DataLoader.load_measurements('data/processed/merged.csv', ap_dict)

        # Convert measurements to DataFrame format expected by run_positioning_analysis
        measurements_df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'ap_mac': m.ap.mac_address,
            'rssi': m.rssi,
            'people_count': m.people_count,
            'has_los': m.has_los
        } for m in measurements])

        # Run the positioning analysis and classify the location
        run_positioning_analysis(
            positioning,
            measurements_df,
            ap_dict=ap_dict
        )

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
