import pickle
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class WifiCoordinatePredictor:
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

        self.feature_scaler = StandardScaler()
        # Separate regressors for X and Y coordinates
        self.regressor_x = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )
        self.regressor_y = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )

        self.bssid_to_index = {}
        self.default_rssi = -100
        self.rssi_threshold = -90
        self.grid_max_x = 49
        self.grid_max_y = 49

    def train_model(self, train_df: pd.DataFrame) -> None:
        """
        Train the coordinate prediction model
        """
        print("Training coordinate prediction model...")

        # Process training data
        train_df = train_df.copy()
        train_df['dbm'] = train_df['dbm'].clip(lower=self.rssi_threshold)

        # Get all unique BSSIDs
        all_bssids = sorted(train_df['bssid'].unique())
        self.bssid_to_index = {bssid: idx for idx, bssid in enumerate(all_bssids)}

        print(f"Found {len(all_bssids)} unique access points")

        # Prepare training data
        X = []  # Features
        y_x = []  # X coordinates
        y_y = []  # Y coordinates

        # Process each location's data
        for location, group in train_df.groupby('location'):
            coords = self.location_coords[location]

            # Use sliding window to create multiple samples
            window_size = 5
            stride = 2

            for i in range(0, len(group) - window_size + 1, stride):
                window = group.iloc[i:i + window_size]

                if len(window) >= 3:
                    features = self._extract_features(window)
                    X.append(features)
                    y_x.append(coords[0])
                    y_y.append(coords[1])

        X = np.array(X)
        y_x = np.array(y_x)
        y_y = np.array(y_y)

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # Train the models
        print("Training X coordinate predictor...")
        self.regressor_x.fit(X_scaled, y_x)

        print("Training Y coordinate predictor...")
        self.regressor_y.fit(X_scaled, y_y)

        self._save_model()

    def _extract_features(self, measurements: pd.DataFrame) -> np.ndarray:
        """
        Extract features from measurements
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
        mean_signal = measurements['dbm'].mean()
        median_signal = measurements['dbm'].median()
        std_signal = measurements['dbm'].std() if len(measurements) > 1 else 0

        # Combine all features
        features = np.concatenate([
            fingerprint,
            [n_aps, strongest_signal, signal_range, mean_signal, median_signal, std_signal]
        ])

        return features

    def predict_location(self, measurements: List[Tuple[str, float]]) -> Tuple[float, float, float]:
        """
        Predict X,Y coordinates from measurements
        Returns (x, y, confidence)
        """
        if len(measurements) < 3:
            return None, None, 0.0

        # Prepare prediction data
        df = pd.DataFrame(measurements, columns=['bssid', 'dbm'])
        df['dbm'] = df['dbm'].clip(lower=self.rssi_threshold)

        # Extract features
        features = self._extract_features(df)
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

        # Predict coordinates
        pred_x = self.regressor_x.predict(features_scaled)[0]
        pred_y = self.regressor_y.predict(features_scaled)[0]

        # Bound predictions to grid
        pred_x = max(0, min(pred_x, self.grid_max_x))
        pred_y = max(0, min(pred_y, self.grid_max_y))

        # Calculate confidence based on prediction variance
        x_confidence = 1 - np.std([
            tree.predict(features_scaled)[0]
            for tree in self.regressor_x.estimators_
        ]) / self.grid_max_x

        y_confidence = 1 - np.std([
            tree.predict(features_scaled)[0]
            for tree in self.regressor_y.estimators_
        ]) / self.grid_max_y

        # Overall confidence
        confidence = (x_confidence + y_confidence) / 2 * min(1.0, len(measurements) / 5.0)

        return pred_x, pred_y, confidence

    def _save_model(self) -> None:
        output_dir = Path.cwd() / 'models'
        output_dir.mkdir(exist_ok=True)

        model_data = {
            'feature_scaler': self.feature_scaler,
            'regressor_x': self.regressor_x,
            'regressor_y': self.regressor_y,
            'bssid_to_index': self.bssid_to_index,
            'default_rssi': self.default_rssi,
            'rssi_threshold': self.rssi_threshold
        }

        with open(output_dir / 'coordinate_prediction_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.feature_scaler = model_data['feature_scaler']
        self.regressor_x = model_data['regressor_x']
        self.regressor_y = model_data['regressor_y']
        self.bssid_to_index = model_data['bssid_to_index']
        self.default_rssi = model_data['default_rssi']
        self.rssi_threshold = model_data['rssi_threshold']


# Example usage
def main():
    # Load training data
    data_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'
    train_df = pd.read_csv(data_path)

    # Create and train the model
    predictor = WifiCoordinatePredictor()
    predictor.train_model(train_df)

    # Example of how to use the trained model for prediction
    print("\nCommon Area Prediction:")

    printingroom_measurements = [
        ('18:64:72:4c:9a:60', -58.0),
        ('18:64:72:4c:9a:70', -58.0),
        ('b0:b8:67:63:30:82', -84.0),
        ('b0:b8:67:63:30:92', -84.0),
        ('b0:b8:67:63:35:42', -84.0),
        ('b0:b8:67:63:35:52', -78.0),
        ('b0:b8:67:63:4a:32', -91.0),
        ('b0:b8:67:63:4f:62', -80.0),
        ('b0:b8:67:63:4f:72', -76.0),
        ('b0:b8:67:63:50:a2', -89.0),
        ('b0:b8:67:63:50:b2', -79.0),
        ('b0:b8:67:63:6b:92', -64.0)
    ]

    x, y, confidence = predictor.predict_location(printingroom_measurements)
    if x is not None:
        print(f"Predicted coordinates: ({x:.1f}, {y:.1f})")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("Could not predict location - insufficient measurements")


# Function to predict location from measurements
def predict_coordinates(measurements: List[Tuple[str, float]], model_path: Path = None) -> Tuple[float, float, float]:
    """
    Predict coordinates from a list of BSSID and signal strength measurements

    Args:
        measurements: List of (bssid, dbm) tuples
        model_path: Path to saved model (optional)

    Returns:
        Tuple of (x_coordinate, y_coordinate, confidence)
    """
    predictor = WifiCoordinatePredictor()

    if model_path and model_path.exists():
        # Load existing model
        predictor.load_model(model_path)
    else:
        # Train new model
        data_path = Path.cwd() / 'data' / 'processed' / 'merged_dataset.csv'
        train_df = pd.read_csv(data_path)
        predictor.train_model(train_df)

    return predictor.predict_location(measurements)


if __name__ == "__main__":
    main()
