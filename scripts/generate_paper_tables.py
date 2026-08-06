"""
Generate formatted tables for paper/report from experiment results.
"""

import json
import os


def generate_results_table(results_dir='results'):
    """Generate a formatted results table from metrics files."""
    metrics_files = {
        'MAE + I-SMOTE (UCI HAR)': 'metrics_mae.json',
    }

    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>10}")
    print("-" * 80)

    for method, filename in metrics_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                metrics = json.load(f)

            acc = metrics.get('test_accuracy', 0)
            macro_f1 = metrics.get('macro_f1', metrics.get('macro_avg', {}).get('f1-score', 0))
            weighted_f1 = metrics.get('weighted_avg', {}).get('f1-score', 0)

            print(f"{method:<30} {acc:>10.4f} {macro_f1:>10.4f} {weighted_f1:>10.4f}")
        else:
            print(f"{method:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print("=" * 80)


def generate_per_class_table(metrics_file='results/metrics_mae.json'):
    """Generate per-class performance table."""
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    with open(metrics_file) as f:
        metrics = json.load(f)

    activity_labels = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
                      'Sitting', 'Standing', 'Laying']

    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    print(f"{'Activity':<25} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 70)

    per_class = metrics.get('per_class', {})
    for key, values in per_class.items():
        name = key if not key.isdigit() else (
            activity_labels[int(key)] if int(key) < len(activity_labels) else f"Class {key}"
        )
        p = values.get('precision', 0)
        r = values.get('recall', 0)
        f1 = values.get('f1-score', values.get('f1', 0))
        print(f"{name:<25} {p:>12.4f} {r:>12.4f} {f1:>12.4f}")

    print("=" * 70)


if __name__ == '__main__':
    generate_results_table()
    generate_per_class_table()
