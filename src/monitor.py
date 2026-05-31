import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs('artifacts/metrics', exist_ok=True)
os.makedirs('artifacts/data', exist_ok=True)
os.makedirs('monitoring/reports', exist_ok=True)
os.makedirs('monitoring/logs', exist_ok=True)
os.makedirs('monitoring/alerts', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/monitoring.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

DRIFT_P_THRESHOLD = 0.05
TARGET_COL = 'Zone 1 Power Consumption'


def _add_time_features(df):
    dt = pd.to_datetime(df['DateTime'], dayfirst=False)
    df = df.copy()
    df['hour'] = dt.dt.hour
    df['month'] = dt.dt.month
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    return df


def detect_drift(reference: np.ndarray, current: np.ndarray, feature_names: list) -> dict:
    results = {}
    for i, name in enumerate(feature_names):
        stat, p_value = stats.ks_2samp(reference[:, i], current[:, i])
        results[name] = {
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'drift': bool(p_value < DRIFT_P_THRESHOLD),
        }
    return results


def run_monitoring(new_data_path: str = 'data/new_data.csv') -> dict:
    now = datetime.utcnow()

    with open('artifacts/preprocessing/feature_columns.json') as f:
        feature_cols = json.load(f)

    scaler = joblib.load('artifacts/preprocessing/scaler.pkl')
    rf = joblib.load('artifacts/models/random_forest.pkl')
    X_train = np.load('artifacts/data/X_train.npy')

    log.info("Loading new data from %s", new_data_path)
    new_df = _add_time_features(pd.read_csv(new_data_path))
    new_df = new_df.dropna(subset=feature_cols)

    X_new = scaler.transform(new_df[feature_cols].values)
    np.save('artifacts/data/X_new.npy', X_new)

    drift_detail = detect_drift(X_train, X_new, feature_cols)
    drifted = [f for f, r in drift_detail.items() if r['drift']]

    y_pred = rf.predict(X_new)

    report = {
        'timestamp': now.isoformat(),
        'n_samples': int(len(X_new)),
        'drift_detected': len(drifted) > 0,
        'drifted_features': drifted,
        'drift_detail': drift_detail,
    }

    if TARGET_COL in new_df.columns:
        y_true = new_df[TARGET_COL].values
        report['mae'] = float(mean_absolute_error(y_true, y_pred))
        report['r2'] = float(r2_score(y_true, y_pred))

    with open('artifacts/metrics/monitoring_metrics.json', 'w') as f:
        json.dump({k: v for k, v in report.items() if k != 'drift_detail'}, f, indent=2)

    with open('monitoring/reports/drift_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if report['drift_detected']:
        alert = {'timestamp': now.isoformat(), 'drifted_features': drifted}
        alert_path = f"monitoring/alerts/alert_{now.strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_path, 'w') as f:
            json.dump(alert, f, indent=2)
        log.warning("Drift detected in features: %s", drifted)
        print(f"ALERT: Drift detected in features: {drifted}")
    else:
        log.info("No drift detected.")
        print("No drift detected.")

    return report


if __name__ == '__main__':
    run_monitoring()
