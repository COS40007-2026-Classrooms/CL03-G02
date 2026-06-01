import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.makedirs('artifacts/lstm', exist_ok=True)
os.makedirs('artifacts/metrics', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

SEQUENCE_LEN = 20
CLASSES_NUM  = 4
TARGET_COL   = 'Zone 1 Power Consumption'
LABELS       = ['Low', 'Moderate', 'High', 'Extreme']

FEATURE_COLS = [
    'Temperature', 'Humidity', 'Wind Speed',
    'general diffuse flows', 'diffuse flows',
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend', 'Season',
    'HourSin', 'HourCos',
    'Temp_Humidity', 'Total_Solar', 'Peak_Hours', 'Night_Hours',
]


def engineer_features(df):
    df = df.copy()
    dt = pd.to_datetime(df['DateTime'], dayfirst=False)
    df['Hour']      = dt.dt.hour
    df['Month']     = dt.dt.month
    df['DayOfWeek'] = dt.dt.dayofweek
    df['IsWeekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['Season']    = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df['HourSin']   = np.sin(2 * np.pi * df['Hour'] / 24)
    df['HourCos']   = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Total_Solar']   = df['general diffuse flows'] + df['diffuse flows']
    df['Peak_Hours']    = df['Hour'].apply(lambda h: 1 if 17 <= h <= 22 else 0)
    df['Night_Hours']   = df['Hour'].apply(lambda h: 1 if h <= 6 else 0)
    return df


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)


def build_lstm(seq_len, n_features, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


log.info("Loading training data for LSTM from train/train.csv")
df = pd.read_csv('train/train.csv')
df = engineer_features(df)

le = LabelEncoder()
df['Power_Class'], bins = pd.qcut(df[TARGET_COL], q=4, labels=LABELS, retbins=True)
df['Power_Class_Num'] = le.fit_transform(df['Power_Class'])

# Save bins so evaluation can reconstruct the same classes on unseen data
joblib.dump(bins.tolist(), 'artifacts/lstm/qcut_bins.pkl')
joblib.dump(le.classes_.tolist(), 'artifacts/lstm/label_classes.pkl')

mm_scaler = MinMaxScaler()
X_mm  = mm_scaler.fit_transform(df[FEATURE_COLS])
y_arr = df['Power_Class_Num'].values

joblib.dump(mm_scaler, 'artifacts/lstm/scaler.pkl')
with open('artifacts/lstm/feature_columns.json', 'w') as f:
    json.dump(FEATURE_COLS, f, indent=2)

X_seq, y_seq = create_sequences(X_mm, y_arr, SEQUENCE_LEN)
y_cat = to_categorical(y_seq, num_classes=CLASSES_NUM)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_seq, y_cat, test_size=0.2, random_state=42)

log.info("LSTM input shape: %s  training samples: %d", X_seq.shape, X_tr.shape[0])

lstm_model = build_lstm(SEQUENCE_LEN, len(FEATURE_COLS), CLASSES_NUM)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)

history = lstm_model.fit(
    X_tr, y_tr,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

lstm_model.save('artifacts/lstm/lstm_model.keras')
log.info("LSTM model saved to artifacts/lstm/lstm_model.keras")

test_loss, test_acc = lstm_model.evaluate(X_te, y_te, verbose=0)
y_pred_cls = np.argmax(lstm_model.predict(X_te), axis=1)
y_true_cls = np.argmax(y_te, axis=1)
lstm_f1    = f1_score(y_true_cls, y_pred_cls, average='weighted')

metrics = {
    'lstm': {
        'test_accuracy': round(float(test_acc), 4),
        'test_loss':     round(float(test_loss), 4),
        'f1_weighted':   round(float(lstm_f1), 4),
        'sequence_len':  SEQUENCE_LEN,
        'epochs_trained': len(history.history['loss']),
    }
}

with open('artifacts/metrics/lstm_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

log.info("LSTM test acc=%.4f  F1=%.4f", test_acc, lstm_f1)
print(f"LSTM  Acc: {test_acc*100:.2f}%  |  F1: {lstm_f1:.4f}  |  Loss: {test_loss:.4f}")
print(classification_report(y_true_cls, y_pred_cls, target_names=LABELS))
print("LSTM training complete. Model saved.")
