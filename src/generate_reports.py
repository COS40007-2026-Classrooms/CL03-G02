import json
import os

os.makedirs('reports', exist_ok=True)

with open('artifacts/metrics/training_history.json') as f:
    train = json.load(f)

with open('artifacts/metrics/evaluation_metrics.json') as f:
    test = json.load(f)

monitoring = None
drift = None

if os.path.exists('artifacts/metrics/monitoring_metrics.json'):
    with open('artifacts/metrics/monitoring_metrics.json') as f:
        monitoring = json.load(f)

if os.path.exists('monitoring/reports/drift_report.json'):
    with open('monitoring/reports/drift_report.json') as f:
        drift = json.load(f)


def fmt(val):
    return f"{val:.4f}" if isinstance(val, float) else str(val)


with open('reports/performance_report.html', 'w') as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>Performance Report</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    table {{ border-collapse: collapse; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
    th {{ background: #eee; }}
  </style>
</head>
<body>
  <h2>Model Performance Report</h2>
  <p>Dataset: Tetuan City Power Consumption &nbsp;|&nbsp; Target: Zone 1 Power Consumption</p>

  <h3>Training Metrics</h3>
  <table>
    <tr><th>Model</th><th>MAE</th><th>MSE</th><th>R2</th></tr>
    <tr><td>Random Forest</td><td>{fmt(train['random_forest']['train_mae'])}</td><td>{fmt(train['random_forest']['train_mse'])}</td><td>{fmt(train['random_forest']['train_r2'])}</td></tr>
    <tr><td>Decision Tree</td><td>{fmt(train['decision_tree']['train_mae'])}</td><td>{fmt(train['decision_tree']['train_mse'])}</td><td>{fmt(train['decision_tree']['train_r2'])}</td></tr>
  </table>

  <h3>Test Metrics</h3>
  <table>
    <tr><th>Model</th><th>MAE</th><th>MSE</th><th>R2</th></tr>
    <tr><td>Random Forest</td><td>{fmt(test['random_forest']['mae'])}</td><td>{fmt(test['random_forest']['mse'])}</td><td>{fmt(test['random_forest']['r2'])}</td></tr>
    <tr><td>Decision Tree</td><td>{fmt(test['decision_tree']['mae'])}</td><td>{fmt(test['decision_tree']['mse'])}</td><td>{fmt(test['decision_tree']['r2'])}</td></tr>
  </table>
</body>
</html>""")


if monitoring and drift:
    drifted = drift.get('drifted_features', [])
    drift_detected = drift.get('drift_detected', False)
    drift_status = "YES" if drift_detected else "NO"
    drift_color = "red" if drift_detected else "green"
    drift_detail = drift.get('drift_detail', {})
    timestamp = monitoring.get('timestamp', 'N/A')
    n_samples = monitoring.get('n_samples', 'N/A')
    mae = monitoring.get('mae', 'N/A')
    r2 = monitoring.get('r2', 'N/A')

    drift_rows = ''
    for feature, result in drift_detail.items():
        yes_no = "Yes" if result['drift'] else "No"
        color = "red" if result['drift'] else "green"
        drift_rows += f"    <tr><td>{feature}</td><td>{fmt(result['ks_statistic'])}</td><td>{result['p_value']:.2e}</td><td style='color:{color}'>{yes_no}</td></tr>\n"

    with open('reports/monitoring_dashboard.html', 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>Monitoring Dashboard</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    table {{ border-collapse: collapse; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
    th {{ background: #eee; }}
  </style>
</head>
<body>
  <h2>Monitoring Dashboard</h2>
  <p>Timestamp: {timestamp} &nbsp;|&nbsp; Samples: {n_samples}</p>

  <h3>Drift Status</h3>
  <p style="color:{drift_color}">Drift detected: {drift_status}</p>
  <p>Drifted features: {", ".join(drifted) if drifted else "None"}</p>

  <h3>Feature Drift (KS Test)</h3>
  <table>
    <tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Drift</th></tr>
{drift_rows}  </table>

  <h3>Performance on New Data</h3>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>MAE</td><td>{fmt(mae) if isinstance(mae, float) else mae}</td></tr>
    <tr><td>R2</td><td>{fmt(r2) if isinstance(r2, float) else r2}</td></tr>
  </table>
</body>
</html>""")

print("Reports generated.")
