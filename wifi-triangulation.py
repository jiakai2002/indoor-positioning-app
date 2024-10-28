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

class DataLoader:
    """Handle data loading and preprocessing"""
    
    @staticmethod
    def load_access_points(filepath: str) -> List[AccessPoint]:
        """
        Load access point data from CSV
        Expected format: mac_address,x,y,reference_power,channel,ssid
        """
        try:
            ap_df = pd.read_csv(filepath)
            required_columns = ['mac_address', 'x', 'y', 'reference_power']
            if not all(col in ap_df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            aps = []
            for _, row in ap_df.iterrows():
                ap = AccessPoint(
                    mac_address=row['mac_address'],
                    x=row['x'],
                    y=row['y'],
                    reference_power=row['reference_power'],
                    channel=row.get('channel', 1),
                    ssid=row.get('ssid', '')
                )
                aps.append(ap)
            return aps
        except Exception as e:
            print(f"Error loading access points: {str(e)}")
            return []

    @staticmethod
    def load_measurements(filepath: str, ap_dict: Dict[str, AccessPoint]) -> List[RSSIMeasurement]:
        """
        Load RSSI measurements from CSV
        Expected format: timestamp,ap_mac,rssi,people_count,has_los
        """
        try:
            measurements_df = pd.read_csv(filepath)
            required_columns = ['timestamp', 'ap_mac', 'rssi']
            if not all(col in measurements_df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            measurements = []
            for _, row in measurements_df.iterrows():
                if row['ap_mac'] not in ap_dict:
                    print(f"Warning: AP {row['ap_mac']} not found in AP list")
                    continue
                
                measurement = RSSIMeasurement(
                    ap=ap_dict[row['ap_mac']],
                    rssi=row['rssi'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    people_count=row.get('people_count', 0),
                    has_los=row.get('has_los', True)
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
    def __init__(self, access_points: List[AccessPoint], floor_plan_path: Optional[str] = None):
        self.access_points = access_points
        self.ap_dict = {ap.mac_address: ap for ap in access_points}
        self.floor_plan = None
        self.floor_plan_extent = None
        
        if floor_plan_path and Path(floor_plan_path).exists():
            self.floor_plan = imread(floor_plan_path)
            # Assume the floor plan covers the entire area of AP deployment
            self.floor_plan_extent = [
                min(ap.x for ap in access_points),
                max(ap.x for ap in access_points),
                min(ap.y for ap in access_points),
                max(ap.y for ap in access_points)
            ]
        
    def calculate_distance(self, rssi: float, reference_power: float, 
                         people_count: int = 0, has_los: bool = True) -> float:
        """Enhanced distance calculation considering environmental factors"""
        path_loss_exponent = 2.0 if has_los else 3.5
        people_adjustment = 0.2 * people_count  # Adjust path loss based on crowd density
        
        adjusted_path_loss = path_loss_exponent + people_adjustment
        return 10 ** ((reference_power - rssi) / (10 * adjusted_path_loss))

    def _error_function(self, point: np.ndarray, measurements: List[RSSIMeasurement]) -> float:
        """Calculate error considering environmental factors"""
        error = 0
        for measurement in measurements:
            ap = measurement.ap
            measured_distance = self.calculate_distance(
                measurement.rssi, 
                ap.reference_power,
                measurement.people_count,
                measurement.has_los
            )
            calculated_distance = np.sqrt((point[0] - ap.x)**2 + (point[1] - ap.y)**2)
            
            # Weight the error based on line of sight
            weight = 1.0 if measurement.has_los else 0.6
            error += weight * (measured_distance - calculated_distance)**2
        return error

    def estimate_position(self, measurements: List[RSSIMeasurement]) -> Tuple[float, float]:
        """Estimate position using weighted trilateration"""
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
    true_positions_df: Optional[pd.DataFrame] = None,
    ap_dict: Dict[str, AccessPoint] = None
) -> None:
    """Run positioning analysis with either test or real data"""
    
    # Group measurements by timestamp
    measurements_grouped = measurements_df.groupby('timestamp')
    
    # Process measurements and collect results
    results = []
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
            results.append({
                'timestamp': timestamp,
                'x': est_x,
                'y': est_y,
                'num_aps': len(measurements),
                'people_count': measurements[0].people_count
            })
            
            # Print position estimate for the first few measurements
            if len(results) <= 5:
                print(f"\nPosition Estimate at {timestamp}:")
                print(f"X: {est_x:.2f} meters")
                print(f"Y: {est_y:.2f} meters")
                print(f"Number of APs used: {len(measurements)}")
                print(f"People count: {measurements[0].people_count}")
                print("-" * 50)
        except Exception as e:
            print(f"Error processing measurements at {timestamp}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    # Use provided true positions or create dummy ones
    if true_positions_df is None:
        print("No true positions provided. Using estimated positions as ground truth.")
        true_positions_df = pd.DataFrame({
            'timestamp': results_df['timestamp'],
            'x': results_df['x'],
            'y': results_df['y'],
            'people_count': results_df['people_count']
        })
    
    # Perform analysis
    analysis = AnalysisMetrics(true_positions_df, results_df)
    
    # Print basic statistics
    print("\nPositioning Analysis Results:")
    stats = analysis.get_basic_stats()
    for metric, value in stats.items():
        print(f"{metric}: {value:.2f} meters")
    
    # Plot visualizations
    analysis.plot_error_distribution()
    
    # Visualize the latest measurement
    try:
        latest_timestamp = list(measurements_grouped.groups.keys())[-1]
        latest_measurements = [
            RSSIMeasurement(
                ap=ap_dict[row['ap_mac']],
                rssi=row['rssi'],
                timestamp=pd.to_datetime(row['timestamp']),
                people_count=row.get('people_count', 0),
                has_los=row.get('has_los', True)
            )
            for _, row in measurements_df[measurements_df['timestamp'] == latest_timestamp].iterrows()
        ]
        
        latest_true_pos = None
        if true_positions_df is not None:
            matching_true_pos = true_positions_df[true_positions_df['timestamp'] == latest_timestamp]
            if not matching_true_pos.empty:
                latest_true_pos = (matching_true_pos.iloc[0]['x'], matching_true_pos.iloc[0]['y'])
        
        # Get room dimensions from APs positions
        max_x = max(ap.x for ap in positioning.access_points)
        max_y = max(ap.y for ap in positioning.access_points)
        room_dims = (max_x, max_y)
        
        print("\nVisualizing latest position estimate...")
        positioning.visualize_positioning(
            latest_measurements,
            true_position=latest_true_pos,
            room_dims=room_dims
        )
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='WiFi Positioning System')
    parser.add_argument('--mode', choices=['test', 'real'], required=True,
                       help='Run with test data or real data')
    parser.add_argument('--data-dir', default='./data',
                       help='Directory for input/output data files')
    parser.add_argument('--floor-plan', default='./floor_plan.png',
                       help='Path to floor plan image')
    parser.add_argument('--room-width', type=float, default=20.0,
                       help='Room width in meters (for test data)')
    parser.add_argument('--room-length', type=float, default=30.0,
                       help='Room length in meters (for test data)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            print("Generating and using test data...")
            # Generate test data
            aps, measurements_df, true_positions_df = DataGenerator.generate_test_dataset(
                room_width=args.room_width,
                room_length=args.room_length
            )
            
            # Optionally save test data
            DataGenerator.save_test_data(aps, measurements_df, args.data_dir)
            
        else:  # real data mode
            print("Loading real data from CSV files...")
            # Load access points and measurements from CSV
            data_loader = DataLoader()
            ap_path = Path(args.data_dir) / 'access_points.csv'
            measurements_path = Path(args.data_dir) / 'measurements.csv'
            
            if not ap_path.exists() or not measurements_path.exists():
                raise FileNotFoundError(
                    f"Required CSV files not found in {args.data_dir}"
                )
            
            aps = data_loader.load_access_points(str(ap_path))
            measurements_df = pd.read_csv(measurements_path)
            true_positions_df = None
            
            # Try to load true positions if available
            true_positions_path = Path(args.data_dir) / 'true_positions.csv'
            if true_positions_path.exists():
                true_positions_df = pd.read_csv(true_positions_path)
        
        # Initialize positioning system
        positioning = WiFiPositioning(aps, floor_plan_path=args.floor_plan)
        ap_dict = {ap.mac_address: ap for ap in aps}
        
        # Run analysis
        run_positioning_analysis(
            positioning,
            measurements_df,
            true_positions_df,
            ap_dict
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()