import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('artifacts/metrics', exist_ok=True)
os.makedirs('artifacts/data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

TARGET_COL = 'Zone 1 Power Consumption'


def _add_time_features(df):
    dt = pd.to_datetime(df['DateTime'], dayfirst=False)
    df = df.copy()
    df['hour'] = dt.dt.hour
    df['month'] = dt.dt.month
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    return df


with open('artifacts/preprocessing/feature_columns.json') as f:
    feature_cols = json.load(f)

scaler = joblib.load('artifacts/preprocessing/scaler.pkl')
rf = joblib.load('artifacts/models/random_forest.pkl')
dt = joblib.load('artifacts/models/decision_tree.pkl')

log.info("Loading test data from test/test.csv")
test_df = _add_time_features(pd.read_csv('test/test.csv'))
X_test = scaler.transform(test_df[feature_cols].values)
y_test = test_df[TARGET_COL].values
np.save('artifacts/data/X_test.npy', X_test)
np.save('artifacts/data/y_test.npy', y_test)

log.info("Loading val data from val/val.csv")
val_df = _add_time_features(pd.read_csv('val/val.csv'))
X_val = scaler.transform(val_df[feature_cols].values)
y_val = val_df[TARGET_COL].values
np.save('artifacts/data/X_val.npy', X_val)
np.save('artifacts/data/y_val.npy', y_val)

metrics = {}
for name, model in [('random_forest', rf), ('decision_tree', dt)]:
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)
    metrics[name] = {
        'mae': mean_absolute_error(y_test, y_pred_test),
        'mse': mean_squared_error(y_test, y_pred_test),
        'r2': r2_score(y_test, y_pred_test),
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'val_mse': mean_squared_error(y_val, y_pred_val),
        'val_r2': r2_score(y_val, y_pred_val),
    }
    log.info("%s test MAE=%.2f R2=%.4f  val MAE=%.2f R2=%.4f",
             name, metrics[name]['mae'], metrics[name]['r2'],
             metrics[name]['val_mae'], metrics[name]['val_r2'])
    print(f"{name}: test MAE={metrics[name]['mae']:.2f}  R2={metrics[name]['r2']:.4f}"
          f"  |  val MAE={metrics[name]['val_mae']:.2f}  R2={metrics[name]['val_r2']:.4f}")

with open('artifacts/metrics/evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

log.info("Evaluation complete. Metrics saved to artifacts/metrics/evaluation_metrics.json")
