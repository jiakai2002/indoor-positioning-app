import logging
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

    def generate_heatmap(self, current_readings: pd.DataFrame) -> np.ndarray:
        """Generate signal strength heatmap data"""
        grid = np.zeros((self.grid_size, self.grid_size))

        # Group readings by location and calculate mean signal strength
        location_strengths = current_readings.groupby('location')['dbm'].mean()

        # Create heatmap based on signal strengths
        for location, strength in location_strengths.items():
            if location in self.location_coords:
                x, y = self.location_coords[location]
                # Ensure coordinates are within grid bounds
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[int(y), int(x)] = strength

        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        smoothed_grid = gaussian_filter(grid, sigma=2)

        return smoothed_grid

    def visualize_results(self, floor_plan_path: str = 'data/raw/floorplan.png'):
        """Visualize results with heatmap overlay"""
        try:
            if not self.results:
                raise ValueError("No results available for visualization")

            # Create results DataFrame
            results_df = pd.DataFrame(self.results)

            # Convert results to the format needed for visualization
            viz_results_df = pd.DataFrame({
                'Actual X': results_df['actual_x'],
                'Actual Y': results_df['actual_y'],
                'Estimated X': results_df['predicted_x'],
                'Estimated Y': results_df['predicted_y'],
                'Error Distance': results_df['euclidean_distance']
            })

            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Plot 1: Results on coordinate system
            # Plot actual positions
            ax1.scatter(viz_results_df['Actual X'], viz_results_df['Actual Y'],
                        c='blue', marker='o', s=100, label='Actual Positions', alpha=0.7)

            # Plot estimated positions
            ax1.scatter(viz_results_df['Estimated X'], viz_results_df['Estimated Y'],
                        c='red', marker='x', s=100, label='Estimated Positions', alpha=0.7)

            # Draw connection lines
            for _, row in viz_results_df.iterrows():
                ax1.plot([row['Actual X'], row['Estimated X']],
                         [row['Actual Y'], row['Estimated Y']],
                         'g-', alpha=0.3)

            # Add location labels
            for location, (x, y) in self.location_coords.items():
                ax1.annotate(location, (x, y), fontsize=8, alpha=0.7)

            ax1.set_title('Positioning Results')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')

            # Plot 2: Error heatmap
            error_scatter = ax2.scatter(viz_results_df['Actual X'],
                                        viz_results_df['Actual Y'],
                                        c=viz_results_df['Error Distance'],
                                        cmap='YlOrRd',
                                        s=100,
                                        alpha=0.7)
            plt.colorbar(error_scatter, ax=ax2, label='Error Distance (units)')

            # Add location labels to error heatmap
            for location, (x, y) in self.location_coords.items():
                ax2.annotate(location, (x, y), fontsize=8, alpha=0.7)

            ax2.set_title('Error Heatmap')
            ax2.grid(True)
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')

            plt.tight_layout()
            plt.show()

            # Print error statistics
            errors = viz_results_df['Error Distance'].dropna()
            if len(errors) > 0:
                print("\nError Statistics:")
                print(f"Mean Error Distance: {errors.mean():.2f} units")
                print(f"Median Error Distance: {errors.median():.2f} units")
                print(f"Maximum Error Distance: {errors.max():.2f} units")
                print(f"Standard Deviation: {errors.std():.2f} units")

                # Calculate percentile errors
                percentiles = [25, 50, 75, 90]
                for p in percentiles:
                    print(f"{p}th Percentile Error: {np.percentile(errors, p):.2f} units")

        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")

    def save_results_to_csv(self, output_path: str = 'evaluation_results.csv'):
        """Save evaluation results to CSV file"""
        try:
            if not self.results:
                raise ValueError("No results available to save")

            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")


def main():
    try:
        # Load datasets
        train_path = Path.cwd() / 'data' / 'processed' / 'processed_rain_dataset.csv'
        test_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

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
            # Visualize results
            evaluator.visualize_results()

            # Save results to CSV
            evaluator.save_results_to_csv()

    except Exception as e:
        print(f"\nError in main evaluation routine: {e}")


if __name__ == "__main__":
    main()
