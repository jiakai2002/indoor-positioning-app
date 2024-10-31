import argparse
import os
import platform
import re
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class WiFiScanner:
    def __init__(self, ssid_filter: Optional[str] = None):
        self.os_type = platform.system().lower()
        if self.os_type != "windows":
            raise OSError("This script only supports Windows.")

        self.ssid_filter = ssid_filter
        self.expected_columns = [
            'timestamp', 'ssid', 'bssid', 'signal_strength', 'signal_dbm',
            'radio_type', 'band', 'channel', 'network_type',
            'authentication', 'encryption'
        ]
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_value(self, value) -> str:
        """Clean and convert value to string safely."""
        if value is None:
            return ''
        try:
            return str(value).strip()
        except:
            return str(value)

    def save_raw_output(self, output: str, filename: str = "raw_wifi_scan.txt"):
        """Save the raw output to a text file in the output directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Raw output saved to {filepath}")
        except Exception as e:
            print(f"Error saving raw output: {e}")

    def save_to_csv(self, df: pd.DataFrame, filename: str = "wifi_networks.csv"):
        """Save DataFrame to CSV in the output directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Processed data saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def parse_windows_networks(self, output: str) -> pd.DataFrame:
        """Parse Windows netsh output into a pandas DataFrame."""
        networks = []
        current_ssid = None
        current_network = {}
        current_bssid = None

        try:
            # Split output into lines and remove empty lines
            lines = [line.strip() for line in output.split('\n') if line.strip()]

            for line in lines:
                try:
                    # New SSID section
                    if line.startswith('SSID'):
                        if current_bssid:
                            networks.append(current_bssid)

                        ssid_match = re.match(r'SSID \d+ : (.*)$', line)
                        if ssid_match:
                            current_ssid = ssid_match.group(1)
                            current_network = {}
                        current_bssid = None

                    # Network type
                    elif line.startswith('    Network type'):
                        current_network['network_type'] = line.split(':', 1)[1].strip()

                    # Authentication
                    elif line.startswith('    Authentication'):
                        current_network['authentication'] = line.split(':', 1)[1].strip()

                    # Encryption
                    elif line.startswith('    Encryption'):
                        current_network['encryption'] = line.split(':', 1)[1].strip()

                    # BSSID entry
                    elif line.strip().startswith('BSSID'):
                        if current_bssid:
                            networks.append(current_bssid)
                        current_bssid = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'ssid': current_ssid,
                            'bssid': line.split(':', 1)[1].strip()
                        }
                        current_bssid.update(current_network)

                    # Signal
                    elif line.strip().startswith('Signal'):
                        signal_str = line.split(':', 1)[1].strip()
                        signal_match = re.search(r'(\d+)%', signal_str)
                        if signal_match:
                            signal_percent = int(signal_match.group(1))
                            current_bssid['signal_strength'] = signal_percent
                            current_bssid['signal_dbm'] = (signal_percent / 2) - 100

                    # Radio type
                    elif line.strip().startswith('Radio type'):
                        current_bssid['radio_type'] = line.split(':', 1)[1].strip()

                    # Band
                    elif line.strip().startswith('Band'):
                        current_bssid['band'] = line.split(':', 1)[1].strip()

                    # Channel
                    elif line.strip().startswith('Channel'):
                        current_bssid['channel'] = line.split(':', 1)[1].strip()

                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(f"Error details: {str(e)}")
                    continue

            # Add the last network
            if current_bssid:
                networks.append(current_bssid)

            # Create DataFrame
            if not networks:
                return pd.DataFrame(columns=self.expected_columns)

            df = pd.DataFrame(networks)

            # Ensure all expected columns exist, fill missing with NaN
            for col in self.expected_columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Apply SSID filter if specified
            if self.ssid_filter:
                df = df[df['ssid'].str.contains(self.ssid_filter, case=False, na=False)]

            # Reorder columns
            all_cols = self.expected_columns + [col for col in df.columns if col not in self.expected_columns]

            return df[all_cols]

        except Exception as e:
            print(f"Error parsing networks: {str(e)}")
            return pd.DataFrame(columns=self.expected_columns)

    def get_wifi_data(self) -> pd.DataFrame:
        """Get WiFi data and apply any filters."""
        try:
            cmd = "netsh wlan show networks mode=Bssid"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')

            # Save raw output
            self.save_raw_output(output)

            df = self.parse_windows_networks(output)

            # Save to CSV
            if not df.empty:
                filtered_text = f"_filtered_{self.ssid_filter}" if self.ssid_filter else ""
                self.save_to_csv(df, f"wifi_networks{filtered_text}.csv")

            return df

        except subprocess.CalledProcessError as e:
            print(f"Error running Windows WiFi scan: {e}")
            return pd.DataFrame(columns=self.expected_columns)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Scan WiFi networks with optional SSID filtering')
    parser.add_argument('--ssid', type=str, help='Filter results by SSID (case-insensitive partial match)',
                        default=None)
    args = parser.parse_args()

    try:
        scanner = WiFiScanner(ssid_filter=args.ssid)
        df = scanner.get_wifi_data()

        if not df.empty:
            print("\nFound networks:")
            display_cols = ['ssid', 'bssid', 'signal_strength', 'signal_dbm', 'channel', 'band', 'radio_type']
            # Only show columns that have data
            available_cols = [col for col in display_cols if col in df.columns]
            print(df[available_cols].to_string())
            print(f"\nTotal networks found: {len(df)}")

            if args.ssid:
                print(f"Results filtered by SSID containing: {args.ssid}")
        else:
            print("No networks found or error occurred")

    except OSError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()