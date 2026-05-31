import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('artifacts/models', exist_ok=True)
os.makedirs('artifacts/preprocessing', exist_ok=True)
os.makedirs('artifacts/metrics', exist_ok=True)
os.makedirs('artifacts/data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

FEATURE_COLS = [
    'Temperature', 'Humidity', 'Wind Speed',
    'general diffuse flows', 'diffuse flows',
    'hour', 'month', 'day_of_week', 'is_weekend',
]
TARGET_COL = 'Zone 1 Power Consumption'


def _add_time_features(df):
    dt = pd.to_datetime(df['DateTime'], dayfirst=False)
    df = df.copy()
    df['hour'] = dt.dt.hour
    df['month'] = dt.dt.month
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    return df


log.info("Loading training data from train/train.csv")
train_df = pd.read_csv('train/train.csv')
train_df = _add_time_features(train_df)

X_train = train_df[FEATURE_COLS].values
y_train = train_df[TARGET_COL].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, 'artifacts/preprocessing/scaler.pkl')
with open('artifacts/preprocessing/feature_columns.json', 'w') as f:
    json.dump(FEATURE_COLS, f, indent=2)

np.save('artifacts/data/X_train.npy', X_train_scaled)
np.save('artifacts/data/y_train.npy', y_train)

log.info("Training Random Forest and Decision Tree models")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train_scaled, y_train)

metrics = {}
for name, model in [('random_forest', rf), ('decision_tree', dt)]:
    y_pred = model.predict(X_train_scaled)
    metrics[name] = {
        'train_mae': mean_absolute_error(y_train, y_pred),
        'train_mse': mean_squared_error(y_train, y_pred),
        'train_r2': r2_score(y_train, y_pred),
    }
    log.info("%s train MAE=%.2f R2=%.4f", name, metrics[name]['train_mae'], metrics[name]['train_r2'])
    print(f"{name}: MAE={metrics[name]['train_mae']:.2f}  R2={metrics[name]['train_r2']:.4f}")

with open('artifacts/metrics/training_history.json', 'w') as f:
    json.dump(metrics, f, indent=2)

joblib.dump(rf, 'artifacts/models/random_forest.pkl')
joblib.dump(dt, 'artifacts/models/decision_tree.pkl')

log.info("Training complete. Models saved to artifacts/models/")
print("Models saved to artifacts/models/")
