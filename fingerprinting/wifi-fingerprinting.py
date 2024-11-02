import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


class EnhancedWifiFingerprinting:
    def __init__(self):
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

        self.rssi_scaler = MinMaxScaler(feature_range=(-1, 0))
        self.feature_scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200,  # Increased number of trees
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )

        self.bssid_to_index = {}
        self.fingerprints = defaultdict(list)
        self.default_rssi = -100
        self.rssi_threshold = -90
        self.ap_stats = defaultdict(dict)

    def preprocess_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess training data with location information
        """
        df = df.copy()
        df['dbm'] = df['dbm'].clip(lower=self.rssi_threshold)

        # Calculate statistics for each AP at each location
        location_ap_stats = df.groupby(['location', 'bssid'])['dbm'].agg([
            'mean', 'std', 'count'
        ]).reset_index()

        # Store AP statistics
        for _, row in location_ap_stats.iterrows():
            self.ap_stats[(row['location'], row['bssid'])] = {
                'mean': row['mean'],
                'std': row['std'],
                'count': row['count']
            }

        return df

    def preprocess_prediction_data(self, measurements: List[Tuple[str, float]]) -> pd.DataFrame:
        """
        Preprocess prediction data without location information
        """
        df = pd.DataFrame(measurements, columns=['bssid', 'dbm'])
        df['dbm'] = df['dbm'].clip(lower=self.rssi_threshold)
        return df

    def extract_features(self, measurements: pd.DataFrame) -> np.ndarray:
        """
        Extract enhanced features from measurements
        """
        # Basic RSSI vector
        fingerprint = np.full(len(self.bssid_to_index), self.default_rssi)

        # Group by BSSID and take median of signal strengths
        grouped = measurements.groupby('bssid')['dbm'].median()

        for bssid, dbm in grouped.items():
            if bssid in self.bssid_to_index:
                idx = self.bssid_to_index[bssid]
                fingerprint[idx] = dbm

        # Additional features
        n_aps = len(measurements['bssid'].unique())
        strongest_signal = measurements['dbm'].max()
        signal_range = measurements['dbm'].max() - measurements['dbm'].min()

        # Signal strength statistics
        mean_signal = measurements['dbm'].mean()
        median_signal = measurements['dbm'].median()
        std_signal = measurements['dbm'].std() if len(measurements) > 1 else 0

        # Create feature vector
        features = np.concatenate([
            fingerprint,
            [n_aps, strongest_signal, signal_range, mean_signal, median_signal, std_signal]
        ])

        return features

    def create_fingerprint_database(self, train_df: pd.DataFrame) -> None:
        """
        Create enhanced fingerprint database with additional features
        """
        print("Creating enhanced fingerprint database...")

        # Preprocess training data
        train_df = self.preprocess_training_data(train_df)

        # Get all unique BSSIDs
        all_bssids = sorted(train_df['bssid'].unique())
        self.bssid_to_index = {bssid: idx for idx, bssid in enumerate(all_bssids)}

        print(f"Found {len(all_bssids)} unique access points")

        fingerprints = []
        locations = []

        # Process each location
        for location, group in train_df.groupby('location'):
            print(f"Processing location: {location}")

            # Create multiple fingerprints with sliding window
            window_size = 5
            stride = 2

            for i in range(0, len(group) - window_size + 1, stride):
                window = group.iloc[i:i + window_size]

                if len(window) >= 3:
                    features = self.extract_features(window)
                    fingerprints.append(features)
                    locations.append(location)

                    self.fingerprints[location].append(
                        dict(zip(window['bssid'], window['dbm']))
                    )

        if not fingerprints:
            raise ValueError("No valid fingerprints could be created from the training data")

        # Convert to numpy arrays
        X = np.array(fingerprints)
        y = np.array(locations)

        print(f"Created {len(X)} fingerprints across {len(set(y))} locations")

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # Train the classifier
        self.classifier.fit(X_scaled, y)

        # Calculate feature importances
        importances = self.classifier.feature_importances_
        feature_names = (
                [f"AP_{i}" for i in range(len(all_bssids))] +
                ['n_aps', 'strongest_signal', 'signal_range', 'mean_signal', 'median_signal', 'std_signal']
        )

        print("\nTop 10 most important features:")
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx[:10]:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")

        self._save_model()

    def estimate_location(self, measurements: List[Tuple[str, float]], n_predictions: int = 5) -> Tuple[
        str, float, Tuple[float, float]]:
        """
        Estimate location with ensemble prediction
        """
        if len(measurements) < 3:
            return None, 0.0, (None, None)

        # Preprocess prediction data
        df = self.preprocess_prediction_data(measurements)

        # Extract features
        features = self.extract_features(df)
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

        # Get probability predictions
        probabilities = self.classifier.predict_proba(features_scaled)[0]

        # Get top N predictions
        top_indices = np.argsort(probabilities)[::-1][:n_predictions]
        top_locations = [self.classifier.classes_[i] for i in top_indices]
        top_probabilities = probabilities[top_indices]

        # Calculate weighted average coordinates
        weighted_x = 0
        weighted_y = 0
        total_weight = 0

        for loc, prob in zip(top_locations, top_probabilities):
            coords = self.location_coords[loc]
            weighted_x += coords[0] * prob
            weighted_y += coords[1] * prob
            total_weight += prob

        if total_weight > 0:
            final_x = weighted_x / total_weight
            final_y = weighted_y / total_weight
        else:
            return None, 0.0, (None, None)

        # Find nearest known location to the weighted position
        nearest_loc = min(
            self.location_coords.items(),
            key=lambda x: np.sqrt((x[1][0] - final_x) ** 2 + (x[1][1] - final_y) ** 2)
        )[0]

        # Calculate confidence
        confidence = top_probabilities[0] * min(1.0, len(measurements) / 5.0)

        return nearest_loc, confidence, (final_x, final_y)

    def _save_model(self) -> None:
        output_dir = Path.cwd() / 'models'
        output_dir.mkdir(exist_ok=True)

        model_data = {
            'rssi_scaler': self.rssi_scaler,
            'feature_scaler': self.feature_scaler,
            'classifier': self.classifier,
            'bssid_to_index': self.bssid_to_index,
            'fingerprints': dict(self.fingerprints),
            'default_rssi': self.default_rssi,
            'ap_stats': dict(self.ap_stats)
        }

        with open(output_dir / 'enhanced_fingerprinting_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)


def evaluate_enhanced_fingerprinting(data_path: Path, test_size: float = 0.2, random_state: int = 42):
    """
    Evaluate the enhanced fingerprinting system
    """
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Split by location
    locations = df['location'].unique()
    train_locations, test_locations = train_test_split(
        locations,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_df = df[df['location'].isin(train_locations)]
    test_df = df[df['location'].isin(test_locations)]

    print(f"Training data: {len(train_df)} measurements")
    print(f"Test data: {len(test_df)} measurements")

    # Create and train system
    system = EnhancedWifiFingerprinting()
    system.create_fingerprint_database(train_df)

    print("\nEvaluating system...")
    results = defaultdict(list)

    # Evaluate each location
    for location, group in test_df.groupby('location'):
        print(f"\nTesting location: {location}")

        # Use sliding window for testing
        window_size = 5
        stride = 2

        for i in range(0, len(group) - window_size + 1, stride):
            window = group.iloc[i:i + window_size]

            if len(window) >= 3:
                measurements = list(zip(window['bssid'], window['dbm']))
                predicted_location, confidence, coords = system.estimate_location(measurements)

                if predicted_location is None:
                    continue

                true_coords = system.location_coords[location]
                error = np.sqrt(
                    (coords[0] - true_coords[0]) ** 2 +
                    (coords[1] - true_coords[1]) ** 2
                )

                results['true_location'].append(location)
                results['predicted_location'].append(predicted_location)
                results['confidence'].append(confidence)
                results['error'].append(error)
                results['x_error'].append(abs(coords[0] - true_coords[0]))
                results['y_error'].append(abs(coords[1] - true_coords[1]))

    # Calculate and display results
    results_df = pd.DataFrame(results)

    accuracy = (results_df['true_location'] == results_df['predicted_location']).mean()
    mean_error = np.mean(results_df['error'])
    median_error = np.median(results_df['error'])
    p90_error = np.percentile(results_df['error'], 90)

    print("\nOverall Results:")
    print(f"Number of test cases: {len(results_df)}")
    print(f"Location Accuracy: {accuracy:.2%}")
    print(f"Mean Error: {mean_error:.2f} units")
    print(f"Median Error: {median_error:.2f} units")
    print(f"90th Percentile Error: {p90_error:.2f} units")

    # Location-specific analysis
    print("\nLocation-specific Results:")
    for location in results_df['true_location'].unique():
        loc_results = results_df[results_df['true_location'] == location]
        loc_accuracy = (loc_results['true_location'] == loc_results['predicted_location']).mean()
        loc_mean_error = loc_results['error'].mean()
        print(f"\n{location}:")
        print(f"  Accuracy: {loc_accuracy:.2%}")
        print(f"  Mean Error: {loc_mean_error:.2f} units")
        print(f"  Samples: {len(loc_results)}")

    # Save results
    output_dir = Path.cwd() / 'evaluation_results'
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / 'enhanced_fingerprinting_results.csv', index=False)


def main():
    data_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'
    evaluate_enhanced_fingerprinting(data_path)


if __name__ == "__main__":
    main()
