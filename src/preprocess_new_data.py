import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

os.makedirs('artifacts/data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)


def preprocess(
    input_path: str = 'data/new_data.csv',
    output_path: str = 'artifacts/data/X_new.npy',
) -> np.ndarray:
    with open('artifacts/preprocessing/feature_columns.json') as f:
        feature_cols = json.load(f)

    scaler = joblib.load('artifacts/preprocessing/scaler.pkl')

    log.info("Loading new data from %s", input_path)
    df = pd.read_csv(input_path)

    if 'DateTime' in df.columns:
        df = df.drop(columns=['DateTime'])

    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d rows with missing values", dropped)

    X = scaler.transform(df[feature_cols].values)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, X)

    log.info("Saved %d preprocessed samples to %s", len(X), output_path)
    print(f"Preprocessed {len(X)} samples -> {output_path}")
    return X


if __name__ == '__main__':
    preprocess()
