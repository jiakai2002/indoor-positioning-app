from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from triangulation.optimized_wifi_triangulation import OptimizedWifiTriangulation


class OptimizedTriangulationEvaluator:
    def __init__(self, location_coords: Dict[str, Tuple[int, int]]):
        self.location_coords = location_coords
        self.results = []
        self.signal_strength_bins = np.arange(-100, -30, 10)

    def prepare_measurements(self, df: pd.DataFrame, location: str) -> List[Tuple[str, float]]:
        """Enhanced measurement preparation with validation"""
        measurements = []
        try:
            location_data = df[df['location'] == location]

            for _, row in location_data.iterrows():
                if pd.notna(row['dbm']) and pd.notna(row['bssid']):
                    measurements.append((row['bssid'], float(row['dbm'])))

            return measurements
        except Exception as e:
            print(f"Error preparing measurements for location {location}: {e}")
            return []

    def calculate_enhanced_metrics(self, actual_coords: Tuple[float, float],
                                   predicted_coords: Tuple[float, float],
                                   measurements: List[Tuple[str, float]]) -> Dict[str, float]:
        """Calculate metrics with error handling"""
        try:
            # Calculate basic distance metrics
            euclidean_distance = np.sqrt(
                (actual_coords[0] - predicted_coords[0]) ** 2 +
                (actual_coords[1] - predicted_coords[1]) ** 2
            )

            manhattan_distance = (
                    abs(actual_coords[0] - predicted_coords[0]) +
                    abs(actual_coords[1] - predicted_coords[1])
            )

            # Calculate signal strength metrics
            signal_strengths = [dbm for _, dbm in measurements]
            if signal_strengths:
                mean_signal = np.mean(signal_strengths)
                min_signal = min(signal_strengths)
                max_signal = max(signal_strengths)
                std_signal = np.std(signal_strengths) if len(signal_strengths) > 1 else 0
                strong_signals = len([s for s in signal_strengths if s > -75])
                strong_ratio = strong_signals / len(signal_strengths) if signal_strengths else 0
            else:
                mean_signal = min_signal = max_signal = std_signal = strong_ratio = 0

            return {
                'euclidean_distance': euclidean_distance,
                'manhattan_distance': manhattan_distance,
                'mean_signal_strength': mean_signal,
                'min_signal_strength': min_signal,
                'max_signal_strength': max_signal,
                'signal_strength_std': std_signal,
                'ap_count': len(measurements),
                'strong_signals_ratio': strong_ratio
            }

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'euclidean_distance': float('inf'),
                'manhattan_distance': float('inf'),
                'mean_signal_strength': 0,
                'min_signal_strength': 0,
                'max_signal_strength': 0,
                'signal_strength_std': 0,
                'ap_count': 0,
                'strong_signals_ratio': 0
            }

    def evaluate(self, triangulator, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate with comprehensive error handling"""
        try:
            if test_df.empty:
                raise ValueError("Test DataFrame is empty")

            all_errors = []
            location_errors = defaultdict(list)
            signal_strength_errors = defaultdict(list)
            successful_predictions = 0
            failed_predictions = 0

            # Process each location
            for location in self.location_coords.keys():
                try:
                    # Get measurements for this location
                    measurements = self.prepare_measurements(test_df, location)
                    if not measurements:
                        print(f"No valid measurements for location: {location}")
                        continue

                    # Get actual coordinates
                    actual_coords = self.location_coords[location]

                    # Attempt location estimation
                    pred_x, pred_y, confidence = triangulator.estimate_location(measurements)

                    if pred_x is not None and pred_y is not None:
                        # Calculate metrics
                        metrics = self.calculate_enhanced_metrics(
                            actual_coords,
                            (pred_x, pred_y),
                            measurements
                        )

                        metrics['confidence'] = confidence
                        metrics['location'] = location

                        # Store results
                        all_errors.append(metrics)
                        location_errors[location].append(metrics['euclidean_distance'])

                        # Group by signal strength
                        mean_signal = metrics['mean_signal_strength']
                        signal_bin = np.digitize(mean_signal, self.signal_strength_bins)
                        signal_strength_errors[signal_bin].append(metrics['euclidean_distance'])

                        self.results.append({
                            'location': location,
                            'actual_x': actual_coords[0],
                            'actual_y': actual_coords[1],
                            'predicted_x': pred_x,
                            'predicted_y': pred_y,
                            'confidence': confidence,
                            **metrics
                        })

                        successful_predictions += 1
                    else:
                        failed_predictions += 1
                        print(f"Failed to estimate location for: {location}")

                except Exception as e:
                    failed_predictions += 1
                    print(f"Error processing location {location}: {e}")
                    continue

            # Check if we have any successful predictions
            if not self.results:
                print("No successful predictions to evaluate")
                return self._generate_empty_metrics()

            # Calculate overall metrics
            overall_metrics = {
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'total_locations': len(self.location_coords),
                'success_rate': successful_predictions / len(self.location_coords)
            }

            # Add error metrics if we have results
            if all_errors:
                euclidean_errors = [e['euclidean_distance'] for e in all_errors]
                overall_metrics.update({
                    'mean_euclidean_error': np.mean(euclidean_errors),
                    'median_euclidean_error': np.median(euclidean_errors),
                    'std_euclidean_error': np.std(euclidean_errors),
                    'max_euclidean_error': np.max(euclidean_errors),
                    'mean_confidence': np.mean([r['confidence'] for r in self.results]),
                    'mean_ap_count': np.mean([r['ap_count'] for r in self.results])
                })

                # Calculate RMSE only if we have predictions
                if len(self.results) >= 2:
                    actual_coords = np.array([(r['actual_x'], r['actual_y']) for r in self.results])
                    pred_coords = np.array([(r['predicted_x'], r['predicted_y']) for r in self.results])
                    overall_metrics['rmse'] = np.sqrt(mean_squared_error(actual_coords, pred_coords))

            # Add per-location metrics
            location_metrics = {}
            for location, errors in location_errors.items():
                if errors:  # Only process locations with valid predictions
                    location_metrics[location] = {
                        'mean_error': np.mean(errors),
                        'median_error': np.median(errors),
                        'std_error': np.std(errors) if len(errors) > 1 else 0,
                        'max_error': np.max(errors),
                        'count': len(errors)
                    }

            overall_metrics['location_metrics'] = location_metrics
            return overall_metrics

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return self._generate_empty_metrics()

    def _generate_empty_metrics(self) -> Dict[str, float]:
        """Generate empty metrics structure when evaluation fails"""
        return {
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_locations': len(self.location_coords),
            'success_rate': 0,
            'mean_euclidean_error': float('nan'),
            'median_euclidean_error': float('nan'),
            'std_euclidean_error': float('nan'),
            'max_euclidean_error': float('nan'),
            'rmse': float('nan'),
            'mean_confidence': 0,
            'mean_ap_count': 0,
            'location_metrics': {}
        }

    def plot_enhanced_visualizations(self, save_dir: str = None):
        """Generate plots with error handling"""
        if not self.results:
            print("No results available for plotting")
            return

        try:
            save_dir = Path(save_dir) if save_dir else Path.cwd()
            save_dir.mkdir(parents=True, exist_ok=True)

            # Plot error distribution
            plt.figure(figsize=(10, 6))
            errors = [r['euclidean_distance'] for r in self.results]
            plt.hist(errors, bins=20, edgecolor='black')
            plt.xlabel('Error (units)')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Errors')
            plt.savefig(save_dir / 'error_distribution.png')
            plt.close()

            # Plot predictions map
            plt.figure(figsize=(10, 10))
            actual_x = [r['actual_x'] for r in self.results]
            actual_y = [r['actual_y'] for r in self.results]
            pred_x = [r['predicted_x'] for r in self.results]
            pred_y = [r['predicted_y'] for r in self.results]

            plt.scatter(actual_x, actual_y, c='blue', label='Actual')
            plt.scatter(pred_x, pred_y, c='red', label='Predicted')

            for r in self.results:
                plt.plot([r['actual_x'], r['predicted_x']],
                         [r['actual_y'], r['predicted_y']],
                         'k-', alpha=0.2)

            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Actual vs Predicted Locations')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / 'predictions_map.png')
            plt.close()

        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    try:
        # Load datasets
        train_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'
        test_path = Path.cwd() / 'data' / 'processed' / 'test_dataset.csv'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if train_df.empty or test_df.empty:
            raise ValueError("One or both datasets are empty")

        # Initialize systems
        triangulator = OptimizedWifiTriangulation()
        triangulator.preprocess_training_data(train_df)

        evaluator = OptimizedTriangulationEvaluator(triangulator.location_coords)

        # Run evaluation
        print("\nRunning evaluation...")
        metrics = evaluator.evaluate(triangulator, test_df)

        # Print results
        print("\nEvaluation Results:")
        print(f"Successful predictions: {metrics['successful_predictions']}")
        print(f"Failed predictions: {metrics['failed_predictions']}")
        print(f"Success rate: {metrics['success_rate']:.2%}")

        if metrics['successful_predictions'] > 0:
            print(f"\nError Metrics:")
            print(f"Mean Error: {metrics.get('mean_euclidean_error', float('nan')):.2f} units")
            print(f"Median Error: {metrics.get('median_euclidean_error', float('nan')):.2f} units")
            print(f"RMSE: {metrics.get('rmse', float('nan')):.2f} units")
            print(f"Mean AP Count: {metrics.get('mean_ap_count', 0):.1f}")

            # Generate plots
            print("\nGenerating visualization plots...")
            evaluator.plot_enhanced_visualizations('evaluation_plots')

    except Exception as e:
        print(f"\nError in main evaluation routine: {e}")


if __name__ == "__main__":
    main()
