import csv
import re
from io import StringIO
import os
from pathlib import Path

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


def parse_wifi_data(input_text, location):
    # Parse the summary line
    summary = {}
    summary_line = input_text.split('\n')[0]
    summary_items = summary_line.split(', ')
    for item in summary_items:
        key, value = item.split('=')
        summary[key] = value

    # Initialize list to store network data
    networks = []

    # Parse network entries
    network_entries = input_text.split('\n\n')[1:]  # Skip the summary line

    for entry in network_entries:
        if not entry.strip():
            continue

        network = {}

        # Add location to network data
        network['location'] = location

        # Extract network name and SSID
        entry_parts = entry.split(' - ')
        if len(entry_parts) > 0:
            # Use the full name before the dash as network_name
            network['network_name'] = entry_parts[0].strip()
            # Use the first word of the network name as SSID
            network['ssid'] = network['network_name'].split()[0]

        # Parse key-value pairs
        pairs = re.findall(r'(\w+)=([^,\n]+)(?:,|\n|$)', entry)
        for key, value in pairs:
            if key != 'ssid':  # Skip the original ssid field
                network[key] = value.strip()

        # Parse channel information
        if 'channel' in network:
            channel_info = network['channel'].split('/')
            network['band'] = channel_info[0]
            network['bandwidth'] = channel_info[1] if len(channel_info) > 1 else ''

        # Parse security information
        if 'security' in network:
            network['security_type'] = network['security']

        networks.append(network)

    return networks


def write_to_csv(networks, output_file):
    # Define CSV columns - now including location
    columns = ['location', 'network_name', 'ssid', 'bssid', 'security_type', 'band',
               'bandwidth', 'phy', 'rssi', 'bi', 'age']

    # Write directly to file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for network in networks:
            row = {col: network.get(col, '') for col in columns}
            writer.writerow(row)


def get_location_from_filename(filename):
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


def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Counter for processed files
    files_processed = 0
    all_networks = []

    # Process each file in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(input_dir, filename)

            try:
                # Get location from filename
                location = get_location_from_filename(filename)

                # Read input file
                with open(input_path, 'r') as file:
                    input_text = file.read()

                # Parse networks
                networks = parse_wifi_data(input_text, location)
                all_networks.extend(networks)

                files_processed += 1
                print(f"Processed {filename} -> Location: {location}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Write all networks to a single CSV file
    if all_networks:
        output_path = os.path.join(output_dir, "mac_s.csv")
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
            # Try to create the output directory if it doesn't exist
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