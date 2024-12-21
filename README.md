# Client Device Localization

## Introduction
This project, conducted as part of the CS204 Interconnection of Cyber-Physical Systems module focusing on developing a Wi-Fi-based Indoor Positioning System (IPS). The project leverages ubiquitous Wi-Fi access points (APs) to estimate device locations in complex indoor environments.

The system implements two primary approaches:

1. Trilateration: Adapting RSSI-based distance estimation to calculate positions despite limited AP location data, addressing challenges such as signal obstructions and multipath propagation.
2. Fingerprinting: Using a robust dataset of Wi-Fi signal characteristics, enhanced by wall attenuation factors and propagation models, to achieve accurate room-level localisation.

### Resources
- [Presentation Slides](https://www.canva.com/design/DAGVJHuiyzk/1tmhyqmmRfmP9x_EzPF_0A/view?utm_content=DAGVJHuiyzk&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h35a02c09b2)
- [Project Writeup](https://docs.google.com/document/d/12iGJxx6Ph6M9xVskmqYziKfGx7mADjZC-U3LoF9-_3U/edit?usp=sharing)

### Files and Directories

- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`output/`**: Directory where output files are saved.
- **`requirements.txt`**: Lists Python dependencies.
- **`tools/NetworkScanv3.ps1`**: PowerShell script for data collection
- **`wifi_scanner.py`**: Main Python script for scanning WiFi networks.

## Requirements

- Python 3.x
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Python Script

Run the Python script to scan WiFi networks:

```sh
python wifi_scanner.py [--ssid SSID]
```

- `--ssid`: Optional argument to filter results by SSID (case-insensitive partial match).

### Data Collection
Run the PowerShell script for network scanning:

```powershell
cd tools
powershell -File NetworkScanv3.ps1
```
