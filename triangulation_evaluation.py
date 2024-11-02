from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from triangulation.wifi_triangulation import WifiTriangulation


class TriangulationEvaluator:
    def __init__(self, location_coords: Dict[str, Tuple[int, int]]):
        """
        Initialize the evaluator with location coordinates

        Args:
            location_coords: Dictionary mapping location names to (x, y) coordinates
        """
        self.location_coords = location_coords
        self.results = []

    def prepare_measurements(self, row: pd.Series) -> List[Tuple[str, float]]:
        """
        Convert a row of the dataset into measurement format

        Args:
            row: Pandas Series containing WiFi measurements

        Returns:
            List of (bssid, dbm) tuples
        """
        measurements = []
        bssid = row['bssid']
        dbm = row['dbm']
        if pd.notna(dbm):
            measurements.append((bssid, float(dbm)))
        return measurements

    def group_measurements(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Group measurements by location

        Args:
            df: DataFrame containing WiFi measurements

        Returns:
            Dictionary mapping locations to lists of measurements
        """
        grouped_measurements = defaultdict(list)
        for _, row in df.iterrows():
            location = row['location']
            measurement = self.prepare_measurements(row)
            if measurement:
                grouped_measurements[location].extend(measurement)
        return grouped_measurements

    def calculate_error_metrics(self, actual_coords: Tuple[float, float],
                                predicted_coords: Tuple[float, float]) -> Dict[str, float]:
        """
        Calculate various error metrics between actual and predicted coordinates

        Args:
            actual_coords: Tuple of actual (x, y) coordinates
            predicted_coords: Tuple of predicted (x, y) coordinates

        Returns:
            Dictionary containing error metrics
        """
        # Calculate Euclidean distance in coordinate units
        euclidean_distance = np.sqrt(
            (actual_coords[0] - predicted_coords[0]) ** 2 +
            (actual_coords[1] - predicted_coords[1]) ** 2
        )

        # Calculate Manhattan distance in coordinate units
        manhattan_distance = (
                abs(actual_coords[0] - predicted_coords[0]) +
                abs(actual_coords[1] - predicted_coords[1])
        )

        # Calculate individual axis errors
        x_error = abs(actual_coords[0] - predicted_coords[0])
        y_error = abs(actual_coords[1] - predicted_coords[1])

        # Calculate error vector for direction analysis
        error_vector = (
            predicted_coords[0] - actual_coords[0],
            predicted_coords[1] - actual_coords[1]
        )

        # Calculate error angle (in degrees) to analyze directional bias
        error_angle = np.degrees(np.arctan2(error_vector[1], error_vector[0]))

        return {
            'euclidean_distance': euclidean_distance,
            'manhattan_distance': manhattan_distance,
            'x_error': x_error,
            'y_error': y_error,
            'error_angle': error_angle,
            'error_vector': error_vector
        }

    def evaluate(self, triangulator: WifiTriangulation,
                 test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the triangulation algorithm on test data

        Args:
            triangulator: Trained EnhancedWifiTriangulation instance
            test_df: Test dataset DataFrame

        Returns:
            Dictionary containing evaluation metrics
        """
        grouped_measurements = self.group_measurements(test_df)
        all_errors = []
        location_errors = defaultdict(list)
        error_vectors = []

        for location, measurements in grouped_measurements.items():
            if location not in self.location_coords:
                continue

            actual_coords = self.location_coords[location]
            pred_x, pred_y, confidence = triangulator.estimate_location(measurements)

            if pred_x is not None and pred_y is not None:
                metrics = self.calculate_error_metrics(
                    actual_coords,
                    (pred_x, pred_y)
                )
                metrics['confidence'] = confidence
                metrics['location'] = location

                all_errors.append(metrics)
                location_errors[location].append(metrics['euclidean_distance'])
                error_vectors.append(metrics['error_vector'])

                self.results.append({
                    'location': location,
                    'actual_x': actual_coords[0],
                    'actual_y': actual_coords[1],
                    'predicted_x': pred_x,
                    'predicted_y': pred_y,
                    'confidence': confidence,
                    **metrics
                })

        # Calculate overall metrics
        error_vectors = np.array(error_vectors)
        overall_metrics = {
            'mean_euclidean_error': np.mean([e['euclidean_distance'] for e in all_errors]),
            'median_euclidean_error': np.median([e['euclidean_distance'] for e in all_errors]),
            'std_euclidean_error': np.std([e['euclidean_distance'] for e in all_errors]),
            'max_euclidean_error': np.max([e['euclidean_distance'] for e in all_errors]),
            'mean_x_error': np.mean([e['x_error'] for e in all_errors]),
            'mean_y_error': np.mean([e['y_error'] for e in all_errors]),
            'rmse': np.sqrt(mean_squared_error(
                [(r['actual_x'], r['actual_y']) for r in self.results],
                [(r['predicted_x'], r['predicted_y']) for r in self.results]
            )),
            'mean_manhattan_error': np.mean([e['manhattan_distance'] for e in all_errors]),
            'mean_confidence': np.mean([e['confidence'] for e in all_errors]),
            'successful_predictions': len(all_errors),
            'total_locations': len(grouped_measurements),
            'mean_error_vector': np.mean(error_vectors, axis=0) if len(error_vectors) > 0 else (0, 0)
        }

        # Calculate error percentiles
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            overall_metrics[f'error_percentile_{p}'] = np.percentile(
                [e['euclidean_distance'] for e in all_errors], p
            )

        # Calculate per-location metrics
        location_metrics = {
            location: {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'median_error': np.median(errors),
                'max_error': np.max(errors),
                'count': len(errors)
            }
            for location, errors in location_errors.items()
        }

        overall_metrics['location_metrics'] = location_metrics
        return overall_metrics

    def plot_error_distribution(self, save_path: str = None):
        """Plot the distribution of prediction errors"""
        if not self.results:
            print("No results to plot")
            return

        errors = [r['euclidean_distance'] for r in self.results]
        plt.figure(figsize=(8, 8))
        plt.hist(errors, bins=20, edgecolor='black')
        plt.xlabel('Error (coordinate units)')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')

        # Add mean and median lines
        plt.axvline(np.mean(errors), color='red', linestyle='dashed', label=f'Mean: {np.mean(errors):.2f}')
        plt.axvline(np.median(errors), color='green', linestyle='dashed', label=f'Median: {np.median(errors):.2f}')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_predictions_map(self, save_path: str = None):
        """Plot actual vs predicted locations on a map"""
        if not self.results:
            print("No results to plot")
            return

        plt.figure(figsize=(8, 8))

        # Plot actual locations
        actual_x = [r['actual_x'] for r in self.results]
        actual_y = [r['actual_y'] for r in self.results]
        plt.scatter(actual_x, actual_y, c='blue', label='Actual', alpha=0.6)

        # Plot predicted locations
        pred_x = [r['predicted_x'] for r in self.results]
        pred_y = [r['predicted_y'] for r in self.results]
        plt.scatter(pred_x, pred_y, c='red', label='Predicted', alpha=0.6)

        # Draw lines connecting actual and predicted points
        for r in self.results:
            plt.plot([r['actual_x'], r['predicted_x']],
                     [r['actual_y'], r['predicted_y']],
                     'k-', alpha=0.2)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Actual vs Predicted Locations')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.close()


def main():
    # Load datasets
    train_path = Path(Path.cwd() / 'data' / 'processed' / 'train_dataset.csv')
    test_path = Path(Path.cwd() / 'data' / 'processed' / 'test_dataset.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Initialize triangulation system
    triangulator = WifiTriangulation()
    triangulator.preprocess_training_data(train_df)

    # Initialize evaluator
    evaluator = TriangulationEvaluator(triangulator.location_coords)

    # Run evaluation
    metrics = evaluator.evaluate(triangulator, test_df)

    # Print results
    print("\nOverall Evaluation Metrics:")
    print(f"Mean Error: {metrics['mean_euclidean_error']:.2f} units")
    print(f"Median Error: {metrics['median_euclidean_error']:.2f} units")
    print(f"RMSE: {metrics['rmse']:.2f} units")
    print(f"Standard Deviation: {metrics['std_euclidean_error']:.2f} units")
    print(f"Maximum Error: {metrics['max_euclidean_error']:.2f} units")
    print(f"Mean X-axis Error: {metrics['mean_x_error']:.2f} units")
    print(f"Mean Y-axis Error: {metrics['mean_y_error']:.2f} units")
    print(f"Mean Confidence: {metrics['mean_confidence']:.2%}")
    print(f"Successful Predictions: {metrics['successful_predictions']} / {metrics['total_locations']}")

    print("\nError Percentiles (units):")
    print(f"50th percentile: {metrics['error_percentile_50']:.2f}")
    print(f"75th percentile: {metrics['error_percentile_75']:.2f}")
    print(f"90th percentile: {metrics['error_percentile_90']:.2f}")
    print(f"95th percentile: {metrics['error_percentile_95']:.2f}")

    print("\nPer-Location Performance:")
    for location, loc_metrics in metrics['location_metrics'].items():
        print(f"\n{location}:")
        print(f"  Mean Error: {loc_metrics['mean_error']:.2f} units")
        print(f"  Median Error: {loc_metrics['median_error']:.2f} units")
        print(f"  Max Error: {loc_metrics['max_error']:.2f} units")
        print(f"  Std Error: {loc_metrics['std_error']:.2f} units")
        print(f"  Sample Count: {loc_metrics['count']}")

    # Generate plots
    evaluator.plot_error_distribution('error_distribution.png')
    evaluator.plot_predictions_map('predictions_map.png')


if __name__ == "__main__":
    main()
