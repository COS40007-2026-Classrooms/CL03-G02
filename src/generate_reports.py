import json
import os

os.makedirs('reports', exist_ok=True)

with open('artifacts/metrics/training_history.json') as f:
    train = json.load(f)

with open('artifacts/metrics/evaluation_metrics.json') as f:
    test = json.load(f)

lstm_metrics = None
monitoring = None
drift = None

if os.path.exists('artifacts/metrics/lstm_metrics.json'):
    with open('artifacts/metrics/lstm_metrics.json') as f:
        lstm_metrics = json.load(f)

if os.path.exists('artifacts/metrics/monitoring_metrics.json'):
    with open('artifacts/metrics/monitoring_metrics.json') as f:
        monitoring = json.load(f)

if os.path.exists('monitoring/reports/drift_report.json'):
    with open('monitoring/reports/drift_report.json') as f:
        drift = json.load(f)


def fmt(val):
    return f"{val:.4f}" if isinstance(val, float) else str(val)


# ── Pre-extract values for chart data ─────────────────────
rf_train_mae = round(train['random_forest']['train_mae'], 2)
rf_test_mae  = round(test['random_forest']['mae'], 2)
rf_val_mae   = round(test['random_forest']['val_mae'], 2)
dt_train_mae = round(train['decision_tree']['train_mae'], 2)
dt_test_mae  = round(test['decision_tree']['mae'], 2)
dt_val_mae   = round(test['decision_tree']['val_mae'], 2)

rf_train_r2 = round(train['random_forest']['train_r2'], 4)
rf_test_r2  = round(test['random_forest']['r2'], 4)
rf_val_r2   = round(test['random_forest']['val_r2'], 4)
dt_train_r2 = round(train['decision_tree']['train_r2'], 4)
dt_test_r2  = round(test['decision_tree']['r2'], 4)
dt_val_r2   = round(test['decision_tree']['val_r2'], 4)

r2_min = round(min(rf_train_r2, rf_test_r2, rf_val_r2, dt_train_r2, dt_test_r2, dt_val_r2) - 0.05, 2)

# ── Performance Report ─────────────────────────────────────
with open('reports/performance_report.html', 'w') as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>Performance Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: sans-serif; padding: 20px; max-width: 1100px; margin: auto; }}
    h2 {{ color: #333; }}
    h3 {{ color: #555; margin-top: 30px; }}
    table {{ border-collapse: collapse; margin-bottom: 20px; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
    th {{ background: #eee; }}
    .charts {{ display: flex; gap: 30px; margin: 20px 0 30px; }}
    .chart-box {{ flex: 1; }}
  </style>
</head>
<body>
  <h2>Model Performance Report</h2>
  <p>Dataset: Tetuan City Power Consumption &nbsp;|&nbsp; Target: Zone 1 Power Consumption</p>

  <div class="charts">
    <div class="chart-box">
      <h3>MAE by Split</h3>
      <canvas id="maeChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>R&sup2; by Split</h3>
      <canvas id="r2Chart"></canvas>
    </div>
  </div>

  <h3>Training Metrics</h3>
  <table>
    <tr><th>Model</th><th>MAE</th><th>MSE</th><th>R&sup2;</th></tr>
    <tr><td>Random Forest</td><td>{fmt(train['random_forest']['train_mae'])}</td><td>{fmt(train['random_forest']['train_mse'])}</td><td>{fmt(train['random_forest']['train_r2'])}</td></tr>
    <tr><td>Decision Tree</td><td>{fmt(train['decision_tree']['train_mae'])}</td><td>{fmt(train['decision_tree']['train_mse'])}</td><td>{fmt(train['decision_tree']['train_r2'])}</td></tr>
  </table>

  <h3>Test &amp; Validation Metrics</h3>
  <table>
    <tr><th>Model</th><th>Test MAE</th><th>Test R&sup2;</th><th>Val MAE</th><th>Val R&sup2;</th></tr>
    <tr><td>Random Forest</td><td>{fmt(test['random_forest']['mae'])}</td><td>{fmt(test['random_forest']['r2'])}</td><td>{fmt(test['random_forest']['val_mae'])}</td><td>{fmt(test['random_forest']['val_r2'])}</td></tr>
    <tr><td>Decision Tree</td><td>{fmt(test['decision_tree']['mae'])}</td><td>{fmt(test['decision_tree']['r2'])}</td><td>{fmt(test['decision_tree']['val_mae'])}</td><td>{fmt(test['decision_tree']['val_r2'])}</td></tr>
  </table>

  {'<h3>Deep Learning — LSTM Classifier</h3><table><tr><th>Metric</th><th>Value</th></tr>' +
   f'<tr><td>Test Accuracy</td><td>{fmt(lstm_metrics["lstm"]["test_accuracy"])}</td></tr>' +
   f'<tr><td>Test Loss</td><td>{fmt(lstm_metrics["lstm"]["test_loss"])}</td></tr>' +
   f'<tr><td>F1 Score (Weighted)</td><td>{fmt(lstm_metrics["lstm"]["f1_weighted"])}</td></tr>' +
   f'<tr><td>Sequence Length</td><td>{lstm_metrics["lstm"]["sequence_len"]} steps ({lstm_metrics["lstm"]["sequence_len"] * 10} min)</td></tr>' +
   f'<tr><td>Epochs Trained</td><td>{lstm_metrics["lstm"]["epochs_trained"]}</td></tr>' +
   '</table>' if lstm_metrics else ''}

  <script>
    new Chart(document.getElementById('maeChart'), {{
      type: 'bar',
      data: {{
        labels: ['Train', 'Test', 'Val'],
        datasets: [
          {{
            label: 'Random Forest',
            data: [{rf_train_mae}, {rf_test_mae}, {rf_val_mae}],
            backgroundColor: 'rgba(54, 162, 235, 0.7)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }},
          {{
            label: 'Decision Tree',
            data: [{dt_train_mae}, {dt_test_mae}, {dt_val_mae}],
            backgroundColor: 'rgba(255, 99, 132, 0.7)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
          }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
          y: {{ beginAtZero: true, title: {{ display: true, text: 'MAE (Wh)' }} }}
        }}
      }}
    }});

    new Chart(document.getElementById('r2Chart'), {{
      type: 'bar',
      data: {{
        labels: ['Train', 'Test', 'Val'],
        datasets: [
          {{
            label: 'Random Forest',
            data: [{rf_train_r2}, {rf_test_r2}, {rf_val_r2}],
            backgroundColor: 'rgba(54, 162, 235, 0.7)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }},
          {{
            label: 'Decision Tree',
            data: [{dt_train_r2}, {dt_test_r2}, {dt_val_r2}],
            backgroundColor: 'rgba(255, 99, 132, 0.7)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
          }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
          y: {{ min: {r2_min}, max: 1, title: {{ display: true, text: 'R²' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>""")


# ── Monitoring Dashboard ───────────────────────────────────
if monitoring and drift:
    drifted        = drift.get('drifted_features', [])
    drift_detected = drift.get('drift_detected', False)
    drift_status   = "YES" if drift_detected else "NO"
    drift_color    = "red" if drift_detected else "green"
    drift_detail   = drift.get('drift_detail', {})
    timestamp      = monitoring.get('timestamp', 'N/A')
    n_samples      = monitoring.get('n_samples', 'N/A')
    mae            = monitoring.get('mae', 'N/A')
    r2             = monitoring.get('r2', 'N/A')

    drift_rows = ''
    for feature, result in drift_detail.items():
        yes_no = "Yes" if result['drift'] else "No"
        color  = "red" if result['drift'] else "green"
        drift_rows += f"    <tr><td>{feature}</td><td>{fmt(result['ks_statistic'])}</td><td>{result['p_value']:.2e}</td><td style='color:{color}'>{yes_no}</td></tr>\n"

    ks_labels = list(drift_detail.keys())
    ks_values = [round(drift_detail[f]['ks_statistic'], 4) for f in ks_labels]
    ks_colors = [
        'rgba(255, 99, 132, 0.7)' if drift_detail[f]['drift'] else 'rgba(75, 192, 192, 0.7)'
        for f in ks_labels
    ]
    ks_borders = [
        'rgba(255, 99, 132, 1)' if drift_detail[f]['drift'] else 'rgba(75, 192, 192, 1)'
        for f in ks_labels
    ]

    ks_labels_json  = json.dumps(ks_labels)
    ks_values_json  = json.dumps(ks_values)
    ks_colors_json  = json.dumps(ks_colors)
    ks_borders_json = json.dumps(ks_borders)

    with open('reports/monitoring_dashboard.html', 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>Monitoring Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: sans-serif; padding: 20px; max-width: 1000px; margin: auto; }}
    h2 {{ color: #333; }}
    h3 {{ color: #555; margin-top: 30px; }}
    table {{ border-collapse: collapse; margin-bottom: 20px; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
    th {{ background: #eee; }}
    .status-badge {{
      display: inline-block; padding: 6px 16px; border-radius: 4px;
      font-weight: bold; color: white;
      background: {drift_color};
    }}
    .chart-box {{ margin: 20px 0 30px; }}
  </style>
</head>
<body>
  <h2>Monitoring Dashboard</h2>
  <p>Timestamp: {timestamp} &nbsp;|&nbsp; Samples: {n_samples}</p>

  <h3>Drift Status</h3>
  <p><span class="status-badge">Drift Detected: {drift_status}</span></p>
  <p>Drifted features: {", ".join(drifted) if drifted else "None"}</p>

  <div class="chart-box">
    <h3>Feature Drift — KS Statistics</h3>
    <canvas id="ksChart"></canvas>
  </div>

  <h3>Feature Drift Detail</h3>
  <table>
    <tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Drift</th></tr>
{drift_rows}  </table>

  <h3>Performance on New Data</h3>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>MAE</td><td>{fmt(mae) if isinstance(mae, float) else mae}</td></tr>
    <tr><td>R&sup2;</td><td>{fmt(r2) if isinstance(r2, float) else r2}</td></tr>
  </table>

  <script>
    new Chart(document.getElementById('ksChart'), {{
      type: 'bar',
      data: {{
        labels: {ks_labels_json},
        datasets: [{{
          label: 'KS Statistic',
          data: {ks_values_json},
          backgroundColor: {ks_colors_json},
          borderColor: {ks_borders_json},
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          annotation: {{}}
        }},
        scales: {{
          y: {{
            beginAtZero: true,
            max: 1,
            title: {{ display: true, text: 'KS Statistic' }},
            grid: {{ color: 'rgba(0,0,0,0.05)' }}
          }},
          x: {{ ticks: {{ maxRotation: 30 }} }}
        }}
      }}
    }});
  </script>
</body>
</html>""")

print("Reports generated.")
