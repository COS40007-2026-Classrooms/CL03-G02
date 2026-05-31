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

with open('artifacts/preprocessing/feature_columns.json') as f:
    feature_cols = json.load(f)

scaler = joblib.load('artifacts/preprocessing/scaler.pkl')
rf = joblib.load('artifacts/models/random_forest.pkl')
dt = joblib.load('artifacts/models/decision_tree.pkl')

log.info("Loading test data from test/test.csv")
test_df = pd.read_csv('test/test.csv')

X_test = scaler.transform(test_df[feature_cols].values)
y_test = test_df[TARGET_COL].values

np.save('artifacts/data/X_test.npy', X_test)
np.save('artifacts/data/y_test.npy', y_test)

metrics = {}
for name, model in [('random_forest', rf), ('decision_tree', dt)]:
    y_pred = model.predict(X_test)
    metrics[name] = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }
    log.info("%s test MAE=%.2f R2=%.4f", name, metrics[name]['mae'], metrics[name]['r2'])
    print(f"{name}: MAE={metrics[name]['mae']:.2f}  R2={metrics[name]['r2']:.4f}")

with open('artifacts/metrics/evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

log.info("Evaluation complete. Metrics saved to artifacts/metrics/evaluation_metrics.json")
