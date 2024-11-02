import argparse
from typing import List, Tuple, Optional

import pandas as pd

from triangulation.wifi_triangulation import WifiTriangulation
from wifi_scanner import WiFiScanner


class RealTimeLocalization:
    def __init__(self, training_data_path: str):
        """
        Initialize the real-time localization system.

        Args:
            training_data_path: Path to the training dataset CSV file
        """
        self.scanner = WiFiScanner()
        self.triangulator = WifiTriangulation()

        # Load and preprocess training data
        train_df = pd.read_csv(training_data_path)
        self.triangulator.preprocess_training_data(train_df)

    def get_current_measurements(self) -> List[Tuple[str, float]]:
        """Get current WiFi measurements from the scanner."""
        try:
            # Scan for WiFi networks
            scan_df = self.scanner.get_wifi_data()

            if scan_df.empty:
                return []

            # Convert scan results to the format expected by triangulator
            measurements = []
            for _, row in scan_df.iterrows():
                if pd.notna(row['bssid']) and pd.notna(row['signal_dbm']):
                    measurements.append((row['bssid'], float(row['signal_dbm'])))

            return measurements

        except Exception as e:
            print(f"Error getting measurements: {e}")
            return []

    def estimate_current_location(self) -> Tuple[Optional[float], Optional[float], float]:
        """
        Estimate the current device location.

        Returns:
            Tuple containing:
            - x coordinate (float or None)
            - y coordinate (float or None)
            - confidence score (float)
        """
        measurements = self.get_current_measurements()

        if not measurements:
            print("No valid WiFi measurements found")
            return None, None, 0.0

        return self.triangulator.estimate_location(measurements)

    def find_nearest_location(self, x: float, y: float) -> Tuple[str, float]:
        """
        Find the nearest known location to the estimated coordinates.

        Args:
            x: Estimated x coordinate
            y: Estimated y coordinate

        Returns:
            Tuple containing:
            - Name of nearest location
            - Distance to that location
        """
        min_distance = float('inf')
        nearest_location = None

        for location, coords in self.triangulator.location_coords.items():
            distance = ((x - coords[0]) ** 2 + (y - coords[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_location = location

        return nearest_location, min_distance


def main():
    parser = argparse.ArgumentParser(description='Real-time WiFi-based indoor localization')
    parser.add_argument('--training-data', type=str, default='data/processed/merged_dataset.csv',
                        help='Path to training dataset CSV file')
    parser.add_argument('--continuous', action='store_true',
                        help='Continuously monitor location')
    args = parser.parse_args()

    try:
        # Initialize localization system
        localizer = RealTimeLocalization(args.training_data)

        if args.continuous:
            print("\nStarting continuous location monitoring (Ctrl+C to stop)...")
            try:
                while True:
                    x, y, confidence = localizer.estimate_current_location()
                    if x is not None and y is not None:
                        nearest_loc, distance = localizer.find_nearest_location(x, y)
                        print(f"\nEstimated coordinates: ({x:.1f}, {y:.1f})")
                        print(f"Confidence: {confidence:.2%}")
                        print(f"Nearest known location: {nearest_loc} (distance: {distance:.1f} units)")
                    else:
                        print("\nUnable to estimate location")

                    input("\nPress Enter to scan again (Ctrl+C to exit)...")

            except KeyboardInterrupt:
                print("\nStopping location monitoring...")

        else:
            # Single location estimate
            x, y, confidence = localizer.estimate_current_location()

            if x is not None and y is not None:
                nearest_loc, distance = localizer.find_nearest_location(x, y)
                print(f"\nEstimated coordinates: ({x:.1f}, {y:.1f})")
                print(f"Confidence: {confidence:.2%}")
                print(f"Nearest known location: {nearest_loc} (distance: {distance:.1f} units)")
            else:
                print("\nUnable to estimate location")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
