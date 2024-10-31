import csv
import re
from pathlib import Path
import os

# Define location mapping dictionary
LOCATION_MAPPING = {
    'fToilet': 'FToilet',
    'gsr2-1': 'GSR2-1',
    'gsr2-2': 'GSR2-3/2',
    'gsr2-4': 'GSR2-4',
    'printingRoom': 'PrintingRoom',
    'stairs1': 'Stairs1',
    'Common_Area': 'CommonArea',
    'SR2-4b': 'SR2-4b',
    'Stairs3': 'Stairs3'
}

# List of all valid locations for validation
VALID_LOCATIONS = {
    'Stair2', 'GSR2-4', 'GSR2-3/2', 'GSR2-1', 'PrintingRoom',
    'Stairs1', 'FToilet', 'LW2.1b', 'Lift2', 'Lift1', 'MToilet',
    'Walkway', 'CommonArea', 'Stairs3', 'SR2-4b', 'SR2-4a', 'SR2-3b',
    'GSR2-6', 'SR2-3a', 'SR2-2a', 'SR2-2b', 'SR2-1b', 'SR2-1a',
    'Stairs2', 'LW2.1a'
}


def get_standardized_location(filename):
    """Convert filename to standardized location name"""
    # Remove extension and get base filename
    base_name = os.path.splitext(filename)[0]

    # Look up standardized location name
    location = LOCATION_MAPPING.get(base_name)

    if location is None:
        print(f"Warning: No mapping found for filename '{base_name}'. Using filename as location.")
        location = base_name

    if location not in VALID_LOCATIONS:
        print(f"Warning: Location '{location}' is not in the list of valid locations.")

    return location


def parse_channel_info(channel_str):
    """Parse channel information from string like '[6 (flags=0xA, 2GHz, 20MHz)]'"""
    match = re.search(r'\[(\d+)\s*\(.*?(\d+GHz),\s*(\d+MHz)', channel_str)
    if match:
        return {
            'channel_number': match.group(1),
            'band': match.group(2),
            'bandwidth': match.group(3)
        }
    return {'channel_number': '', 'band': '', 'bandwidth': ''}


def parse_security_info(rsn_str, wpa_str):
    """Parse security information from RSN and WPA strings"""
    security_info = {
        'security_type': 'none',
        'encryption': []
    }

    if rsn_str and rsn_str != '(null)':
        security_info['security_type'] = 'WPA2'
        # Extract encryption methods
        if 'ucast=' in rsn_str:
            encryption_match = re.search(r'ucast=\{(.*?)\}', rsn_str)
            if encryption_match:
                security_info['encryption'] = [e.strip() for e in encryption_match.group(1).split()]

    if wpa_str and wpa_str != '(null)':
        security_info['security_type'] = 'WPA' if security_info['security_type'] == 'none' else 'WPA/WPA2'
        if 'ucast=' in wpa_str:
            encryption_match = re.search(r'ucast=\{(.*?)\}', wpa_str)
            if encryption_match:
                security_info['encryption'].extend([e.strip() for e in encryption_match.group(1).split()])

    security_info['encryption'] = list(set(security_info['encryption']))  # Remove duplicates
    return security_info


def parse_wifi_data(input_text, location=None):
    networks = []

    # Parse summary line
    summary = {}
    summary_line = input_text.split('\n')[0]
    summary_items = summary_line.split(', ')
    for item in summary_items:
        key, value = item.split('=')
        summary[key] = value

    # Parse network entries
    network_entries = input_text.split('\n\n')[1:]  # Skip the summary line

    for entry_text in network_entries:
        if not entry_text.strip():
            continue

        try:
            network = {}

            # Add location if provided
            if location:
                network['location'] = location

            # Extract network name and SSID hex
            name_match = re.match(r"'([^']+)'\s*\(([^)]+)\)", entry_text)
            if name_match:
                network['network_name'] = name_match.group(1)
                network['ssid_hex'] = name_match.group(2)

            # Extract BSSID
            bssid_match = re.search(r'bssid=([^,]+)', entry_text)
            if bssid_match:
                network['bssid'] = bssid_match.group(1)

            # Extract and parse channel information
            channel_match = re.search(r'channel=\[(.*?)\]', entry_text)
            if channel_match:
                channel_info = parse_channel_info(channel_match.group(0))
                network.update(channel_info)

            # Extract PHY mode
            phy_match = re.search(r'phy=([^,]+)', entry_text)
            if phy_match:
                network['phy'] = phy_match.group(1).split()[0]  # Take only the first part before any parentheses

            # Extract RSSI
            rssi_match = re.search(r'rssi=(-\d+)', entry_text)
            if rssi_match:
                network['rssi'] = rssi_match.group(1)

            # Extract and parse security information
            rsn_match = re.search(r'rsn=\[(.*?)\]', entry_text) or re.search(r'rsn=\((.*?)\)', entry_text)
            wpa_match = re.search(r'wpa=\[(.*?)\]', entry_text) or re.search(r'wpa=\((.*?)\)', entry_text)

            security_info = parse_security_info(
                rsn_match.group(1) if rsn_match else None,
                wpa_match.group(1) if wpa_match else None
            )
            network.update(security_info)

            # Extract age
            age_match = re.search(r'age=(\d+ms)', entry_text)
            if age_match:
                network['age'] = age_match.group(1)

            networks.append(network)

        except Exception as e:
            print(f"Error parsing entry: {str(e)}")
            continue

    return networks


def write_to_csv(networks, output_file):
    # Define CSV columns
    columns = [
        'location', 'network_name', 'ssid_hex', 'bssid', 'channel_number',
        'band', 'bandwidth', 'phy', 'rssi', 'security_type', 'encryption',
        'age'
    ]

    # Write to file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for network in networks:
            # Convert encryption list to string
            if 'encryption' in network:
                network['encryption'] = ', '.join(network['encryption'])
            row = {col: network.get(col, '') for col in columns}
            writer.writerow(row)


def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Counter for processed files
    files_processed = 0
    all_networks = []

    # Process each file in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)

            try:
                # Get standardized location from filename
                location = get_standardized_location(filename)

                # Read input file
                with open(input_path, 'r') as file:
                    input_text = file.read()

                # Parse networks
                networks = parse_wifi_data(input_text, location)
                all_networks.extend(networks)

                files_processed += 1
                print(f"Processed {filename} -> Location: {location} -> {len(networks)} networks found")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Write all networks to a single CSV file
    if all_networks:
        output_path = os.path.join(output_dir, "mac_j.csv")
        write_to_csv(all_networks, output_path)
        print(f"\nWrote combined data to: {output_path}")

    return files_processed


def main():
    # Prompt for directory paths
    while True:
        input_dir = input("Enter the input directory path: ").strip()
        if os.path.isdir(input_dir):
            break
        print("Invalid directory path. Please try again.")

    while True:
        output_dir = input("Enter the output directory path: ").strip()
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            break
        except Exception as e:
            print(f"Error with output directory: {str(e)}")
            print("Please try again.")

    # Process files
    files_processed = process_directory(input_dir, output_dir)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Files processed: {files_processed}")
    print(f"Output files can be found in: {output_dir}")


if __name__ == "__main__":
    main()