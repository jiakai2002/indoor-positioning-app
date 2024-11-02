from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from triangulation.wifi_config import OptimizedWifiConfig


class OptimizedWifiTriangulation:
    def __init__(self, config: Optional[OptimizedWifiConfig] = None):
        self.config = config or OptimizedWifiConfig()
        self.ap_locations: Dict[str, Tuple[float, float]] = {}
        self.ap_stats: Dict[Tuple[str, str], Dict] = {}
        self.location_coords = {
            'FToilet': (26, 13), 'GSR2-1': (25, 5), 'GSR2-3/2': (16, 2),
            'GSR2-4': (16, 6), 'PrintingRoom': (28, 9), 'Stairs1': (31, 9),
            'Stair2': (11, 8), 'LW2.1b': (30, 20), 'Lift2': (23, 21),
            'Lift1': (20, 21), 'MToilet': (18, 11), 'Walkway': (15, 21),
            'CommonArea': (13, 35), 'Stairs3': (12, 38), 'SR2-4b': (9, 40),
            'SR2-4a': (9, 35), 'SR2-3b': (11, 30), 'GSR2-6': (11, 27),
            'SR2-3a': (11, 25), 'SR2-2a': (11, 15), 'SR2-2b': (11, 20),
            'SR2-1b': (11, 10), 'SR2-1a': (10, 6), 'Stairs2': (12, 8),
            'LW2.1a': (28, 9)
        }

    def preprocess_training_data(self, train_df: pd.DataFrame) -> None:
        """Enhanced preprocessing with configuration parameters"""
        print("Starting preprocessing of training data...")

        # Group by location and BSSID to get signal statistics
        ap_stats = train_df.groupby(['location', 'bssid']).agg({
            'dbm': ['count', 'mean', 'std', 'median', 'max', 'min']
        }).reset_index()

        ap_stats.columns = ['location', 'bssid', 'count', 'mean_dbm', 'std_dbm',
                            'median_dbm', 'max_dbm', 'min_dbm']

        print(f"Initial AP readings count: {len(ap_stats)}")

        # Filter reliable APs using config parameters
        reliable_aps = ap_stats[
            (ap_stats['count'] >= self.config.min_readings) &
            (ap_stats['std_dbm'].notna()) &
            (ap_stats['std_dbm'] <= self.config.max_std_dev) &
            (ap_stats['max_dbm'] - ap_stats['min_dbm'] <= self.config.max_signal_range) &
            (ap_stats['mean_dbm'] >= self.config.min_reliable_power)
            ].copy()

        print(f"Reliable AP readings count: {len(reliable_aps)}")

        # Calculate signal quality scores using config parameters
        reliable_aps['signal_quality'] = (
                (reliable_aps['mean_dbm'] + self.config.signal_strength_offset) / 100 *
                (1 / (reliable_aps['std_dbm'] * self.config.stability_factor + 1)) *
                np.log10(reliable_aps['count'])
        )

        # Process each BSSID
        for bssid in reliable_aps['bssid'].unique():
            bssid_data = reliable_aps[reliable_aps['bssid'] == bssid]

            # Find the location with the strongest average signal for this AP
            best_location_idx = bssid_data['mean_dbm'].idxmax()
            best_location = bssid_data.loc[best_location_idx, 'location']

            # Store AP location and stats
            if best_location in self.location_coords:
                self.ap_locations[bssid] = self.location_coords[best_location]
                self.ap_stats[(best_location, bssid)] = {
                    'mean_dbm': bssid_data.loc[best_location_idx, 'mean_dbm'],
                    'std_dbm': bssid_data.loc[best_location_idx, 'std_dbm'],
                    'count': bssid_data.loc[best_location_idx, 'count'],
                    'signal_quality': bssid_data.loc[best_location_idx, 'signal_quality']
                }

        print(f"Final AP count: {len(self.ap_locations)}")
        self._print_preprocessing_summary(reliable_aps)

    def _print_preprocessing_summary(self, reliable_aps: pd.DataFrame) -> None:
        """Print summary of preprocessing results"""
        if self.ap_locations:
            ap_signals = [stats['mean_dbm'] for stats in self.ap_stats.values()]
            print(f"\nAP Signal Statistics:")
            print(f"Mean signal strength: {np.mean(ap_signals):.1f} dBm")
            print(f"Signal strength range: {np.min(ap_signals):.1f} to {np.max(ap_signals):.1f} dBm")
            print(f"APs per location (avg): {len(self.ap_locations) / len(self.location_coords):.1f}")
        else:
            print("\nWarning: No AP locations were established!")

    def _estimate_distance(self, bssid: str, dbm: float) -> Optional[float]:
        """Estimates distance using config path loss models"""
        try:
            if bssid not in self.ap_locations:
                return None

            # Get reference values
            ref_stats = None
            for (loc, ap_bssid), stats in self.ap_stats.items():
                if ap_bssid == bssid:
                    ref_stats = stats
                    break

            if ref_stats is None:
                return None

            ref_power = ref_stats['mean_dbm']
            ref_distance = 1.0

            # Select path loss exponent based on signal strength
            if dbm > self.config.los_threshold:
                path_loss = self.config.path_loss_models['free_space']
            elif dbm > self.config.soft_threshold:
                path_loss = self.config.path_loss_models['indoor_los']
            elif dbm > self.config.min_reliable_power:
                path_loss = self.config.path_loss_models['indoor_soft']
            else:
                path_loss = self.config.path_loss_models['indoor_hard']

            power_diff = ref_power - dbm
            distance = ref_distance * 10 ** (power_diff / (10 * path_loss))

            return max(0.1, min(distance, self.config.grid_max_x * 1.5))

        except Exception as e:
            print(f"Error estimating distance for AP {bssid}: {e}")
            return None

    def _calculate_position_probability(self, point: np.ndarray, measurements: List[Tuple[str, float]]) -> float:
        """Calculates position probability using config weights"""
        if len(measurements) < self.config.min_aps_for_estimation:
            return float('inf')

        total_prob = 0
        total_weight = 0

        for bssid, dbm in measurements:
            if bssid not in self.ap_locations:
                continue

            ap_coords = self.ap_locations[bssid]
            actual_distance = np.sqrt(
                (point[0] - ap_coords[0]) ** 2 +
                (point[1] - ap_coords[1]) ** 2
            )

            estimated_distance = self._estimate_distance(bssid, dbm)
            if estimated_distance is None:
                continue

            # Calculate reliability weights using config parameters
            signal_reliability = 1.0
            for loc, stats in self.ap_stats.items():
                if loc[1] == bssid:
                    signal_reliability = 1.0 / (stats['std_dbm'] * self.config.stability_factor + 1)
                    break

            distance_reliability = 1.0 / (actual_distance + 1)
            reliability = (
                    self.config.signal_variance_weight * signal_reliability +
                    self.config.distance_weight * distance_reliability
            )

            std_dev = max(1.0, estimated_distance * 0.2)
            prob = norm.pdf(actual_distance, estimated_distance, std_dev)

            total_prob += reliability * prob
            total_weight += reliability

        if total_weight == 0:
            return float('inf')

        return -(total_prob / total_weight)

    def estimate_location(self, measurements: List[Tuple[str, float]]) -> Tuple[
        Optional[float], Optional[float], float]:
        """Estimates device location using configured parameters"""
        if not self.ap_locations:
            print("Error: No AP locations available. Please run preprocess_training_data first.")
            return None, None, 0.0

        valid_measurements = [
            (bssid, dbm) for bssid, dbm in measurements
            if bssid in self.ap_locations
        ]

        if len(valid_measurements) < self.config.min_aps_for_estimation:
            print(f"Warning: Only {len(valid_measurements)} valid measurements out of {len(measurements)}")
            return None, None, 0.0

        # Calculate initial position
        weights = []
        weighted_coords = np.zeros(2)

        for bssid, dbm in valid_measurements:
            signal_weight = 1.0 / (abs(dbm) + 1)
            ap_coords = self.ap_locations[bssid]
            weights.append(signal_weight)
            weighted_coords += np.array(ap_coords) * signal_weight

        initial_guess = weighted_coords / sum(weights)

        # Optimize position
        result = minimize(
            self._calculate_position_probability,
            initial_guess,
            args=(valid_measurements,),
            method='Nelder-Mead',
            bounds=[(0, self.config.grid_max_x), (0, self.config.grid_max_y)]
        )

        if not result.success:
            return None, None, 0.0

        # Calculate confidence using config normalization
        confidence = 1.0 / (1.0 + result.fun)
        confidence *= min(1.0, len(valid_measurements) / self.config.confidence_normalization)

        return result.x[0], result.x[1], confidence
