# Client Device Localization

## Introduction
This project is aimed at providing localization support for client devices using Python and PowerShell.


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