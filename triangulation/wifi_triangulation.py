from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class WifiTriangulation:
    def __init__(self):
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

        # Enhanced parameters
        self.path_loss_models = {
            'free_space': 2.0,
            'indoor_los': 2.5,
            'indoor_soft': 3.0,
            'indoor_hard': 3.5
        }
        self.reference_distances = defaultdict(lambda: 1.0)
        self.reference_powers = defaultdict(lambda: -40.0)
        self.grid_max_x = 49
        self.grid_max_y = 49
        self.signal_variance_weight = 0.7
        self.distance_weight = 0.3
        self.min_aps_for_estimation = 3

    def preprocess_training_data(self, train_df: pd.DataFrame) -> None:
        """Preprocesses training data with advanced filtering and calibration"""
        ap_groups = train_df.groupby(['location', 'bssid'])

        ap_stats = ap_groups.agg({
            'dbm': ['count', 'mean', 'std', 'median', 'max', 'min'],
            'channel': lambda x: x.mode().iloc[0] if not x.empty else None
        }).reset_index()

        ap_stats.columns = ['location', 'bssid', 'count', 'mean_dbm', 'std_dbm',
                            'median_dbm', 'max_dbm', 'min_dbm', 'channel']

        reliable_aps = ap_stats[
            (ap_stats['count'] >= 2) &
            (ap_stats['std_dbm'].notna()) &
            (ap_stats['std_dbm'] <= 15) &
            (ap_stats['max_dbm'] - ap_stats['min_dbm'] <= 40)
            ].copy()

        reliable_aps['signal_quality'] = (
                (reliable_aps['mean_dbm'] + 100) / 100 *
                (1 / (reliable_aps['std_dbm'] + 1)) *
                np.log10(reliable_aps['count'])
        )

        for _, row in reliable_aps.iterrows():
            location = row['location']
            bssid = row['bssid']

            self.ap_stats[(location, bssid)] = {
                'mean_dbm': row['mean_dbm'],
                'median_dbm': row['median_dbm'],
                'std_dbm': row['std_dbm'],
                'count': row['count'],
                'channel': row['channel'],
                'signal_quality': row['signal_quality']
            }

            if bssid not in self.ap_locations:
                self.ap_locations[bssid] = self.location_coords[location]
            else:
                current_quality = reliable_aps[
                    (reliable_aps['bssid'] == bssid) &
                    (reliable_aps['location'] == self._get_location_for_ap(bssid))
                    ]['signal_quality'].iloc[0]

                if row['signal_quality'] > current_quality:
                    self.ap_locations[bssid] = self.location_coords[location]

        self._calibrate_path_loss_models(reliable_aps)
        print(f"Processed {len(self.ap_locations)} reliable access points")

    def _calibrate_path_loss_models(self, ap_data: pd.DataFrame) -> None:
        """Calibrates individual path loss models for each AP"""
        for bssid in ap_data['bssid'].unique():
            ap_measurements = ap_data[ap_data['bssid'] == bssid]

            if len(ap_measurements) >= 2:
                distances = []
                signal_strengths = []

                ap_location = self.ap_locations[bssid]
                for _, row in ap_measurements.iterrows():
                    loc_coords = self.location_coords[row['location']]
                    distance = np.sqrt(
                        (ap_location[0] - loc_coords[0]) ** 2 +
                        (ap_location[1] - loc_coords[1]) ** 2
                    )
                    distances.append(distance)
                    signal_strengths.append(row['mean_dbm'])

                distances = np.array(distances)
                signal_strengths = np.array(signal_strengths)

                ref_idx = np.argmax(signal_strengths)
                self.reference_powers[bssid] = signal_strengths[ref_idx]
                self.reference_distances[bssid] = distances[ref_idx]

    def _get_location_for_ap(self, bssid: str) -> Optional[str]:
        """Gets location name for an AP based on its coordinates"""
        if bssid in self.ap_locations:
            ap_coords = self.ap_locations[bssid]
            for loc, coords in self.location_coords.items():
                if coords == ap_coords:
                    return loc
        return None

    def _estimate_distance(self, bssid: str, dbm: float) -> Optional[float]:
        """Estimates distance using calibrated path loss model for specific AP"""
        try:
            ref_power = self.reference_powers[bssid]
            ref_distance = self.reference_distances[bssid]

            if dbm > -50:
                path_loss = self.path_loss_models['indoor_los']
            elif dbm > -70:
                path_loss = self.path_loss_models['indoor_soft']
            else:
                path_loss = self.path_loss_models['indoor_hard']

            power_diff = ref_power - dbm
            distance = ref_distance * 10 ** (power_diff / (10 * path_loss))

            return max(0.1, min(distance, self.grid_max_x * 1.5))

        except Exception as e:
            print(f"Error estimating distance for AP {bssid}: {e}")
            return None

    def _calculate_position_probability(self, point: np.ndarray, measurements: List[Tuple[str, float]]) -> float:
        """Calculates probability of position using enhanced probabilistic model"""
        if len(measurements) < self.min_aps_for_estimation:
            return float('inf')

        total_prob = 0
        total_weight = 0

        for bssid, dbm in measurements:
            if bssid in self.ap_locations:
                ap_coords = self.ap_locations[bssid]
                actual_distance = np.sqrt(
                    (point[0] - ap_coords[0]) ** 2 +
                    (point[1] - ap_coords[1]) ** 2
                )

                estimated_distance = self._estimate_distance(bssid, dbm)
                if estimated_distance is None:
                    continue

                signal_reliability = 1.0
                for loc, stats in self.ap_stats.items():
                    if loc[1] == bssid:
                        signal_reliability = 1.0 / (stats['std_dbm'] + 1)
                        break

                distance_reliability = 1.0 / (actual_distance + 1)
                reliability = (
                        self.signal_variance_weight * signal_reliability +
                        self.distance_weight * distance_reliability
                )

                std_dev = max(1.0, estimated_distance * 0.2)
                prob = norm.pdf(actual_distance, estimated_distance, std_dev)

                total_prob += reliability * prob
                total_weight += reliability

        if total_weight == 0:
            return float('inf')

        boundary_penalty = 0
        if (point[0] < 0 or point[0] > self.grid_max_x or
                point[1] < 0 or point[1] > self.grid_max_y):
            boundary_penalty = 1000

        return -(total_prob / total_weight) + boundary_penalty

    def estimate_location(self, measurements: List[Tuple[str, float]]) -> Tuple[
        Optional[float], Optional[float], float]:
        """
        Estimates device location using enhanced algorithm

        Args:
            measurements: List of tuples containing (bssid, dbm)

        Returns:
            Tuple containing:
            - x coordinate (float or None)
            - y coordinate (float or None)
            - confidence score (float)
        """
        if len(measurements) < self.min_aps_for_estimation:
            return None, None, 0.0

        valid_measurements = [
            (bssid, dbm) for bssid, dbm in measurements
            if bssid in self.ap_locations
        ]

        if len(valid_measurements) < self.min_aps_for_estimation:
            return None, None, 0.0

        # Calculate initial position using weighted centroid
        weights = []
        weighted_coords = np.zeros(2)

        for bssid, dbm in valid_measurements:
            signal_weight = 1.0 / (abs(dbm) + 1)
            reliability = 1.0

            for loc, stats in self.ap_stats.items():
                if loc[1] == bssid:
                    reliability = 1.0 / (stats['std_dbm'] + 1)
                    break

            weight = signal_weight * reliability
            weights.append(weight)
            weighted_coords += np.array(self.ap_locations[bssid]) * weight

        total_weight = sum(weights)
        if total_weight > 0:
            initial_guess = weighted_coords / total_weight
        else:
            coords = np.array([self.ap_locations[bssid] for bssid, _ in valid_measurements])
            initial_guess = np.mean(coords, axis=0)

        result = minimize(
            self._calculate_position_probability,
            initial_guess,
            args=(valid_measurements,),
            method='Nelder-Mead',
            bounds=[(0, self.grid_max_x), (0, self.grid_max_y)]
        )

        if not result.success:
            return None, None, 0.0

        estimated_x, estimated_y = result.x

        # Calculate confidence score
        confidence = 1.0 / (1.0 + result.fun)  # Transform error into confidence
        confidence *= min(1.0, len(valid_measurements) / 5.0)  # Reduce confidence if few APs

        return estimated_x, estimated_y, confidence


def main():
    # Example usage
    data_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'
    train_df = pd.read_csv(data_path)

    triangulator = WifiTriangulation()
    triangulator.preprocess_training_data(train_df)

    # Example measurements from PrintingRoom
    printingroom_measurements = [
        ('b0:b8:67:63:6b:92', -50.0),
        ('b4:5d:50:fd:b4:d1', -82.5),
        ('b0:b8:67:63:50:b2', -87.5),
        ('b0:b8:67:63:30:92', -82.5),
        ('b0:b8:67:63:35:42', -84.0),
        ('b0:b8:67:63:35:52', -78.0),
        ('b0:b8:67:63:4a:32', -91.0),
        ('b0:b8:67:63:4f:62', -80.0),
        ('b0:b8:67:63:4f:72', -76.0),
        ('b0:b8:67:63:50:a2', -89.0),
        ('b0:b8:67:63:50:b2', -79.0),
        ('b0:b8:67:63:6b:92', -64.0)
    ]

    # Estimate device location
    estimated_x, estimated_y, confidence = triangulator.estimate_location(printingroom_measurements)

    if estimated_x is not None:
        print(f"\nEstimated coordinates: ({estimated_x:.1f}, {estimated_y:.1f})")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("\nUnable to estimate location: No valid AP measurements found")


if __name__ == "__main__":
    main()
