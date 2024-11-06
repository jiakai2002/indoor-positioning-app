import logging
import pickle
import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


@dataclass
class AccessPoint:
    mac_address: str
    locations: List[Tuple[str, float]]
    ssid: str
    channel: int
    radio_type: str
    reference_power: float = -50.0
    path_loss_exponent: float = 3.0

    def calculate_distance(self, rssi: float) -> float:
        """Calculate approximate distance using log-distance path loss model"""
        return 10 ** ((self.reference_power - rssi) / (10 * self.path_loss_exponent))


class WiFiPositioning:
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
        self.fingerprints = {}

    def _initialize_grid(self):
        """Initialize the signal strength grid for heatmap generation"""
        self.grid_x = np.linspace(0, self.grid_size, self.grid_size)
        self.grid_y = np.linspace(0, self.grid_size, self.grid_size)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)

    def _calculate_similarity(self, current_readings: Dict[str, float],
                              fingerprint: Dict[str, Dict]) -> float:
        """Calculate similarity score using both RSSI and estimated distances"""
        common_aps = set(current_readings.keys()) & set(fingerprint.keys())
        if not common_aps:
            return float('-inf')

        rssi_diffs = []
        distance_diffs = []

        for ap_id in common_aps:
            current_rssi = current_readings[ap_id]
            fp_data = fingerprint[ap_id]

            rssi_diff = abs(current_rssi - fp_data['rssi'])
            rssi_diffs.append(rssi_diff)

            current_distance = self.ap_dict[ap_id].calculate_distance(current_rssi)
            distance_diff = abs(current_distance - fp_data['distance'])
            distance_diffs.append(distance_diff)

        rssi_score = np.mean(rssi_diffs) / 100.0
        distance_score = np.mean(distance_diffs) / 10.0
        combined_score = 0.6 * rssi_score + 0.4 * distance_score
        ap_ratio = len(common_aps) / max(len(current_readings), len(fingerprint))
        final_score = combined_score * (1 + (1 - ap_ratio))

        return -final_score

    def estimate_location(self, current_readings: pd.DataFrame, k: int = 3) -> Tuple[str, Tuple[float, float]]:
        """Estimate location using improved kNN with weighted averaging"""
        current_rssi = {row['bssid']: row['dbm'] for _, row in current_readings.iterrows()}

        similarities = []
        for location, fingerprint in self.fingerprints.items():
            similarity = self._calculate_similarity(current_rssi, fingerprint)
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


class WiFiScanner:
    def __init__(self):
        pass

    def get_wifi_data(self):
        """Execute netsh command and return the output"""
        try:
            result = subprocess.run(
                ['netsh', 'wlan', 'show', 'networks', 'mode=Bssid'],
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.SubProcessError as e:
            print(f"Error executing netsh command: {e}")
            return None

    def parse_wifi_data(self, data):
        """Parse the netsh command output into structured data"""
        if not data:
            return []

        networks = []
        current_network = None
        current_bssid = None
        skip_until_next_property = False

        lines = data.split('\n')

        for line in lines:
            orig_line = line
            line = line.strip()

            if not line:
                continue

            if line.startswith('Colocated APs'):
                skip_until_next_property = True
                continue

            if skip_until_next_property:
                if orig_line.startswith('            '):
                    continue
                else:
                    skip_until_next_property = False

            if line.startswith('SSID'):
                if current_network and current_network['ssid']:
                    if current_bssid:
                        current_network['bssids'].append(current_bssid)
                        current_bssid = None
                    networks.append(current_network)

                ssid_match = re.match(r'SSID \d+ : (.+)|SSID : (.+)', line)
                if ssid_match:
                    ssid = ssid_match.group(1) or ssid_match.group(2)
                else:
                    ssid = ''

                if ssid:
                    current_network = {
                        'ssid': ssid,
                        'bssids': [],
                        'network_type': '',
                        'authentication': '',
                        'encryption': ''
                    }
                else:
                    current_network = None

            elif current_network and not skip_until_next_property:
                if line.startswith('Network type'):
                    current_network['network_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('Authentication'):
                    current_network['authentication'] = line.split(':', 1)[1].strip()
                elif line.startswith('Encryption'):
                    current_network['encryption'] = line.split(':', 1)[1].strip()

                elif re.match(r'BSSID \d+\s*:', line):
                    if current_bssid:
                        current_network['bssids'].append(current_bssid)
                    bssid = line.split(':', 1)[1].strip()
                    current_bssid = {
                        'bssid': bssid,
                        'signal': None,
                        'radio_type': None,
                        'channel': None
                    }

                elif current_bssid:
                    if 'Signal' in line:
                        signal_match = re.search(r'(\d+)%', line)
                        if signal_match:
                            signal_percent = int(signal_match.group(1))
                            dbm = -100 + (signal_percent / 2)
                            current_bssid['signal'] = dbm
                    elif 'Radio type' in line:
                        current_bssid['radio_type'] = line.split(':', 1)[1].strip()
                    elif 'Channel' in line and 'Utilization' not in line and 'Basic' not in line:
                        try:
                            current_bssid['channel'] = int(line.split(':', 1)[1].strip())
                        except (ValueError, IndexError):
                            current_bssid['channel'] = None

        if current_network and current_network['ssid']:
            if current_bssid:
                current_network['bssids'].append(current_bssid)
            networks.append(current_network)

        return networks

    def create_dataframe(self, networks):
        """Convert parsed network data into a pandas DataFrame"""
        rows = []
        for network in networks:
            ssid = network['ssid']
            for bssid_info in network['bssids']:
                if any(keyword in ssid.lower() for keyword in ['smu', 'eduroam']):
                    row = {
                        'ssid': ssid,
                        'bssid': bssid_info['bssid'],
                        'dbm': bssid_info['signal'],
                        'radio_type': bssid_info['radio_type'],
                        'channel': bssid_info['channel']
                    }
                    rows.append(row)

        return pd.DataFrame(rows)


class LocationPredictor:
    def __init__(self, model_path='wifi_positioning_model.pkl'):
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            logging.info("Positioning model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

        self.scanner = WiFiScanner()

    def predict_location(self, scan_duration=30, interval=5):
        """Scan WiFi networks and predict location"""
        end_time = time.time() + scan_duration
        all_scans = []
        scan_count = 0

        print(f"\nScanning WiFi networks for {scan_duration} seconds...")
        try:
            while time.time() < end_time:
                scan_count += 1
                print(f"\nScan #{scan_count}")

                raw_data = self.scanner.get_wifi_data()
                networks = self.scanner.parse_wifi_data(raw_data)
                df = self.scanner.create_dataframe(networks)

                if not df.empty:
                    all_scans.append(df)
                    print(f"Found {len(df)} relevant access points")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nScanning stopped by user")

        if not all_scans:
            print("No relevant WiFi data collected")
            return None

        combined_df = pd.concat(all_scans, ignore_index=True)
        final_df = combined_df.groupby(['ssid', 'bssid', 'radio_type', 'channel'])['dbm'].median().reset_index()

        logging.info("Predicting location...")
        location_name, coordinates = self.model.estimate_location(final_df)

        return location_name, coordinates


def main():
    try:
        predictor = LocationPredictor()

        while True:
            location_name, coordinates = predictor.predict_location(scan_duration=30, interval=5)

            if location_name and coordinates:
                print("\nLocation Prediction:")
                print(f"Nearest Location: {location_name}")
                print(f"Estimated Coordinates: ({coordinates[0]:.1f}, {coordinates[1]:.1f})")
            else:
                print("\nCould not determine location")

            choice = input("\nWould you like to scan again? (y/n): ")
            if choice.lower() != 'y':
                break

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
