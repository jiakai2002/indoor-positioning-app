import numpy as np
import pandas as pd
import subprocess
import platform
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

@dataclass
class AccessPoint:
    mac_address: str
    x: float
    y: float
    reference_power: float = -40.0  # Default reference power at 1 meter
    channel: int = 1
    ssid: str = ""

@dataclass
class RSSIMeasurement:
    ap: AccessPoint
    rssi: float
    timestamp: datetime
    people_count: int = 0
    has_los: bool = True

class Location:
    def __init__(self):
        # Define the location to coordinate mapping
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
            'LW2.1a': (28, 9)
        }

def classify_location(x: float, y: float, location_coords: Dict[str, Tuple[float, float]]) -> str:
    """
    Classify the location based on x and y coordinates by finding the nearest known location.
    """
    min_distance = float('inf')
    closest_location = 'Unknown Location'
    for location_name, (loc_x, loc_y) in location_coords.items():
        distance = np.sqrt((x - loc_x) ** 2 + (y - loc_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_location = location_name
    return closest_location

class DataLoader:
    @staticmethod
    def load_access_points(filepath: str, max_x: float = 50.0, max_y: float = 50.0) -> List[AccessPoint]:
        """
        Load access point data from CSV and assign positions within a grid
        """
        try:
            ap_df = pd.read_csv(filepath)
            unique_aps = ap_df.drop_duplicates(subset='bssid')

            aps = []
            num_aps = len(unique_aps)
            grid_size = int(np.ceil(np.sqrt(num_aps)))
            spacing_x = max_x / grid_size
            spacing_y = max_y / grid_size

            for idx, (_, row) in enumerate(unique_aps.iterrows()):
                # Calculate grid position
                grid_x = (idx % grid_size) * spacing_x
                grid_y = (idx // grid_size) * spacing_y

                ap = AccessPoint(
                    mac_address=row['bssid'],
                    x=grid_x,
                    y=grid_y,
                    reference_power=-40.0,  # Default reference power
                    channel=int(row['channel']) if pd.notna(row['channel']) else 1,
                    ssid=str(row['ssid'])
                )
                aps.append(ap)
            logging.info(f"Loaded {len(aps)} access points.")
            return aps
        except Exception as e:
            logging.error(f"Error loading access points: {str(e)}")
            return []

    @staticmethod
    def load_measurements(filepath: str, ap_dict: Dict[str, AccessPoint], location_coords: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Load RSSI measurements and add true positions
        """
        try:
            measurements_df = pd.read_csv(filepath)
            # Filter columns and validate
            required_columns = ['location', 'bssid', 'dbm']
            if not all(col in measurements_df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            # If there's no timestamp column, we can generate one
            if 'timestamp' not in measurements_df.columns:
                measurements_df['timestamp'] = pd.date_range(start='2021-01-01', periods=len(measurements_df), freq='T')

            # Convert timestamp to datetime
            measurements_df['timestamp'] = pd.to_datetime(measurements_df['timestamp'])

            # Add true positions based on 'location'
            measurements_df['true_x'] = measurements_df['location'].apply(
                lambda loc: location_coords.get(loc, (np.nan, np.nan))[0]
            )
            measurements_df['true_y'] = measurements_df['location'].apply(
                lambda loc: location_coords.get(loc, (np.nan, np.nan))[1]
            )

            # Drop measurements without valid true positions
            measurements_df.dropna(subset=['true_x', 'true_y'], inplace=True)

            # Ensure 'bssid' is in AP dictionary
            measurements_df = measurements_df[measurements_df['bssid'].isin(ap_dict.keys())]

            logging.info(f"Loaded {len(measurements_df)} measurements.")
            return measurements_df
        except Exception as e:
            logging.error(f"Error loading measurements: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def capture_current_wifi_data() -> pd.DataFrame:
        """
        Capture current WiFi network information using appropriate command depending on OS
        """
        try:
            networks = []
            os_type = platform.system()

            if os_type == 'Linux':
                # Run 'iwlist' command to scan networks
                result = subprocess.run(['iwlist', 'wlan0', 'scanning'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                ssid = bssid = signal = None
                for line in lines:
                    line = line.strip()
                    if 'Cell' in line:
                        bssid = line.split('Address: ')[-1]
                    elif 'ESSID:' in line:
                        ssid = line.split('ESSID:')[1].strip('"')
                    elif 'Signal level=' in line:
                        signal_line = line.split('Signal level=')[-1]
                        if 'dBm' in signal_line:
                            signal = float(signal_line.split(' dBm')[0])
                        else:
                            # Sometimes Signal level is in arbitrary units
                            signal = -100 + int(signal_line)
                        if bssid and ssid and signal is not None:
                            networks.append({
                                'bssid': bssid,
                                'ssid': ssid,
                                'dbm': signal
                            })
                            ssid = bssid = signal = None
            elif os_type == 'Darwin':
                # For macOS, use 'airport' command
                airport_path = '/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport'
                result = subprocess.run([airport_path, '-s'], capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip the header line
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        ssid = parts[0]
                        bssid = parts[1]
                        signal = int(parts[2])  # Signal strength in dBm
                        networks.append({
                            'bssid': bssid,
                            'ssid': ssid,
                            'dbm': signal
                        })
            else:
                logging.error(f"Unsupported OS for WiFi scanning: {os_type}")
                return pd.DataFrame()

            logging.info(f"Captured {len(networks)} WiFi networks.")
            return pd.DataFrame(networks)
        except Exception as e:
            logging.error(f"Error capturing WiFi data: {str(e)}")
            return pd.DataFrame()

class WiFiPositioning:
    def __init__(self, access_points: List[AccessPoint]):
        self.access_points = access_points
        self.ap_dict = {ap.mac_address: ap for ap in access_points}

    def calculate_distance(self, rssi: float, reference_power: float,
                        people_count: int = 0, has_los: bool = True) -> float:
        # More sophisticated path loss model for indoor environments
        base_path_loss_exponent = 2.0 if has_los else 4.0  # Wider range for indoor complexity

        # More nuanced environmental adjustments
        floor_factor = 1.5  # Additional loss for multi-floor environments
        wall_factor = 0.5   # Signal loss through walls
        people_factor = 0.3 * people_count  # More conservative people adjustment

        adjusted_path_loss_exponent = (
            base_path_loss_exponent +
            floor_factor +
            wall_factor +
            people_factor
        )

        # Clip extreme values to prevent unrealistic calculations
        adjusted_path_loss_exponent = np.clip(adjusted_path_loss_exponent, 2.0, 6.0)

        # More robust distance calculation
        distance = 10 ** ((reference_power - rssi) / (10 * adjusted_path_loss_exponent))

        # Add lower and upper bounds to distance
        return np.clip(distance, 1.0, 100.0)

    def _error_function(self, point: np.ndarray, measurements: List[RSSIMeasurement]) -> float:
        error = 0
        total_weight = 0

        for measurement in measurements:
            ap = measurement.ap

            # Calculate signal-based weight - stronger signals get more importance
            signal_weight = 1.0 / (abs(measurement.rssi) + 1)

            # Distance calculation
            measured_distance = self.calculate_distance(
                measurement.rssi,
                ap.reference_power,
                measurement.people_count,
                measurement.has_los
            )

            # Euclidean distance to access point
            calculated_distance = np.sqrt((point[0] - ap.x) ** 2 + (point[1] - ap.y) ** 2)

            # More sophisticated weighting
            los_weight = 1.0 if measurement.has_los else 0.5

            # Weighted squared error
            squared_error = (measured_distance - calculated_distance) ** 2
            weighted_error = squared_error * signal_weight * los_weight

            error += weighted_error
            total_weight += signal_weight * los_weight

        # Normalize error
        return error / total_weight if total_weight > 0 else float('inf')

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

def preprocess_wifi_data(ap_data_path: str, max_x: float = 50.0, max_y: float = 50.0) -> Tuple[List[AccessPoint], Dict[str, AccessPoint]]:
    """
    Preprocess WiFi data from merged CSV file, loading access points and creating a dictionary.
    """
    aps = DataLoader.load_access_points(ap_data_path, max_x, max_y)
    ap_dict = {ap.mac_address: ap for ap in aps}
    return aps, ap_dict

def calculate_positioning_accuracy(test_results_df):
    # Calculate positioning error
    test_results_df['position_error'] = np.sqrt(
        (test_results_df['estimated_x'] - test_results_df['true_x'])**2 +
        (test_results_df['estimated_y'] - test_results_df['true_y'])**2
    )

    # Location classification accuracy
    test_results_df['location_correct'] = (
        test_results_df['estimated_location'] == test_results_df['true_location']
    )

    print("Average Positioning Error:", test_results_df['position_error'].mean())
    print("Location Classification Accuracy:",
          test_results_df['location_correct'].mean() * 100, "%")

def main():
    try:
        # Initialize location coordinates
        location_instance = Location()
        location_coords = location_instance.location_coords

        # Load and process the WiFi data
        aps, ap_dict = preprocess_wifi_data('data/processed/merged.csv')

        # Initialize positioning system
        positioning = WiFiPositioning(aps)

        # Load measurements and include true positions
        measurements_df = DataLoader.load_measurements('data/processed/merged.csv', ap_dict, location_coords)
        measurements_df = measurements_df[measurements_df['ssid'] == 'WLAN-SMU'].reset_index(drop=True)

        if measurements_df.empty:
            logging.error("No measurements to process.")
            return

        # Split into training (90%) and test (10%) sets
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(measurements_df, test_size=0.1, random_state=42)

        logging.info(f"Training set size: {len(train_df)}")
        logging.info(f"Test set size: {len(test_df)}")

        # Train on 90% of data
        train_positions = []
        for location, group in train_df.groupby('location'):
            train_measurements = []
            for _, row in group.iterrows():
                measurement = RSSIMeasurement(
                    ap=ap_dict[row['bssid']],
                    rssi=float(row['dbm']),
                    timestamp=row['timestamp'],
                    people_count=0,
                    has_los=True
                )
                train_measurements.append(measurement)

            if len(train_measurements) >= 3:
                try:
                    est_x, est_y = positioning.estimate_position(train_measurements)
                    train_positions.append({
                        'location': location,
                        'est_x': est_x,
                        'est_y': est_y,
                        'true_x': group['true_x'].iloc[0],
                        'true_y': group['true_y'].iloc[0]
                    })
                    logging.info(f"Estimated position for {location}: ({est_x:.2f}, {est_y:.2f})")
                except Exception as e:
                    logging.error(f"Error processing training measurements for {location}: {str(e)}")
            else:
                logging.warning(f"Not enough measurements for training at location {location}")

        # Current location detection
        current_wifi_data = DataLoader.capture_current_wifi_data()
        current_measurements = []
        for _, row in current_wifi_data.iterrows():
            if row['bssid'] in ap_dict:
                measurement = RSSIMeasurement(
                    ap=ap_dict[row['bssid']],
                    rssi=float(row['dbm']),
                    timestamp=datetime.now(),
                    people_count=0,
                    has_los=True
                )
                current_measurements.append(measurement)

        # Estimate current position
        if len(current_measurements) >= 3:
            current_x, current_y = positioning.estimate_position(current_measurements)
            current_location = classify_location(current_x, current_y, location_coords)

            logging.info(f"Current Estimated Location: {current_location}")
            logging.info(f"Estimated Coordinates: ({current_x:.2f}, {current_y:.2f})")
        else:
            logging.error("Insufficient WiFi access points for location estimation.")

        # Validate on test set (10%)
        test_results = []
        for location, group in test_df.groupby('location'):
            test_measurements = []
            for _, row in group.iterrows():
                measurement = RSSIMeasurement(
                    ap=ap_dict[row['bssid']],
                    rssi=float(row['dbm']),
                    timestamp=row['timestamp'],
                    people_count=0,
                    has_los=True
                )
                test_measurements.append(measurement)

            if len(test_measurements) >= 3:
                try:
                    est_x, est_y = positioning.estimate_position(test_measurements)
                    estimated_location = classify_location(est_x, est_y, location_coords)
                    true_location = group['location'].iloc[0]

                    test_results.append({
                        'location': location,
                        'estimated_x': est_x,
                        'estimated_y': est_y,
                        'estimated_location': estimated_location,
                        'true_location': true_location,
                        'true_x': group['true_x'].iloc[0],
                        'true_y': group['true_y'].iloc[0]
                    })
                    logging.info(f"Test estimation for {location}: Estimated ({est_x:.2f}, {est_y:.2f}), True ({group['true_x'].iloc[0]}, {group['true_y'].iloc[0]})")
                except Exception as e:
                    logging.error(f"Error processing test measurements for {location}: {str(e)}")
            else:
                logging.warning(f"Not enough measurements for testing at location {location}")

        # Save results
        train_positions_df = pd.DataFrame(train_positions)
        test_results_df = pd.DataFrame(test_results)

        train_positions_df.to_csv('train_positions.csv', index=False)
        test_results_df.to_csv('test_results.csv', index=False)

        logging.info("\nTraining Positions saved to 'train_positions.csv'")
        logging.info("Test Results saved to 'test_results.csv'")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
