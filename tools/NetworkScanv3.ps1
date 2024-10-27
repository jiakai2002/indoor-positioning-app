# Prompt for location
$location = Read-Host "Enter the location for this scan"

# Function to parse netsh wlan show interfaces output
function Parse-WlanInterfaces {
    $output = netsh wlan show interfaces
    $interface = $null
    $interfaces = @()

    foreach ($line in $output) {
        if ($line -match '^\s*(.+?)\s*:\s*(.+)\s*$') {
            $key = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            
            if ($key -eq "Name") {
                if ($interface) {
                    $interfaces += $interface
                }
                $interface = @{Name = $value}
            } elseif ($interface) {
                $interface[$key] = $value
            }
        }
    }
    
    if ($interface) {
        $interfaces += $interface
    }
    
    return $interfaces
}

# Parse connected AP information
$connectedAPs = Parse-WlanInterfaces

# Export connected AP information to CSV
$connectedAPsCSV = foreach ($ap in $connectedAPs) {
    [PSCustomObject]@{
        Location = $location
        Name = $ap.Name
        Description = $ap.Description
        GUID = $ap.GUID
        'Physical address' = $ap.'Physical address'
        'Interface type' = $ap.'Interface type'
        State = $ap.State
        SSID = $ap.SSID
        BSSID = $ap.BSSID
        'Network type' = $ap.'Network type'
        'Radio type' = $ap.'Radio type'
        Authentication = $ap.Authentication
        Cipher = $ap.Cipher
        'Connection mode' = $ap.'Connection mode'
        Band = $ap.Band
        Channel = $ap.Channel
        'Receive rate (Mbps)' = $ap.'Receive rate (Mbps)'
        'Transmit rate (Mbps)' = $ap.'Transmit rate (Mbps)'
        Signal = $ap.Signal
        Profile = $ap.Profile
        'Hosted network status' = $ap.'Hosted network status'
    }
}

$connectedAPsCSV | Export-Csv -Path "connected_ap.csv" -NoTypeInformation -Append

# Rest of the original script for scanning all networks
$output = netsh wlan show networks mode=Bssid

# Initialize variables
$networks = @()
$currentNetwork = $null
$currentBSSID = $null

# Parse the output
foreach ($line in $output) {
    if ($line -match '^SSID \d+ : (.*)') {
        if ($currentNetwork) {
            $networks += $currentNetwork
        }
        $currentNetwork = @{
            SSID = $Matches[1].Trim()
            BSSIDs = @()
        }
    }
    elseif ($line -match '^\s*BSSID \d+\s*: (.*)') {
        if ($currentBSSID) {
            $currentNetwork.BSSIDs += $currentBSSID
        }
        $currentBSSID = @{
            BSSID = $Matches[1].Trim()
        }
    }
    elseif ($line -match '^\s*(.*?)\s*: (.*)') {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()
        if ($currentBSSID) {
            $currentBSSID[$key] = $value
        }
        elseif ($currentNetwork) {
            $currentNetwork[$key] = $value
        }
    }
}

# Add the last network and BSSID
if ($currentBSSID -and $currentNetwork) {
    $currentNetwork.BSSIDs += $currentBSSID
}
if ($currentNetwork) {
    $networks += $currentNetwork
}

# Convert to CSV
$csvData = foreach ($network in $networks) {
    foreach ($bssid in $network.BSSIDs) {
        # Extract numeric value from Signal and calculate dBm
        $signalValue = if ($bssid.Signal -match '\d+') { [int]($matches[0]) } else { 0 }
        $dBm = ($signalValue / 2) - 100

        [PSCustomObject]@{
            Location = $location
            SSID = $network.SSID
            'Network type' = $network.'Network type'
            Authentication = $network.Authentication
            Encryption = $network.Encryption
            BSSID = $bssid.BSSID
            Signal = $bssid.Signal
            'dBm' = $dBm
            'Radio type' = $bssid.'Radio type'
            Channel = $bssid.Channel
            'Basic rates (Mbps)' = $bssid.'Basic rates (Mbps)'
            'Other rates (Mbps)' = $bssid.'Other rates (Mbps)'
        }
    }
}

# File path for all networks
$csvPath = "wifi_networks.csv"

# Check if file exists
if (Test-Path $csvPath) {
    # If file exists, import existing data
    $existingData = Import-Csv $csvPath
    
    # Combine existing data with new data
    $combinedData = $existingData + $csvData
    
    # Export combined data to CSV file
    $combinedData | Export-Csv -Path $csvPath -NoTypeInformation
    
    Write-Host "Data has been appended to the existing 'wifi_networks.csv' file in the current directory."
} else {
    # If file doesn't exist, create new file with data
    $csvData | Export-Csv -Path $csvPath -NoTypeInformation
    
    Write-Host "New 'wifi_networks.csv' file has been created in the current directory."
}

Write-Host "Connected AP information has been appended to 'ConnectedAP.csv' in the current directory."