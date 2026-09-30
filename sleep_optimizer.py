"""
Sleep Optimizer - Building AI course project (demo)

What it does:
1. Loads a personal sleep diary (sleep_log.csv)
2. Learns with linear regression how daily habits affect the sleep score
3. Checks the model on nights it has not seen (test set)
4. Predicts tonight's sleep score and shows which single change would help most
5. Saves a bar chart of the learned effects (images/feature_impact.png)

Usage:
    python sleep_optimizer.py
    python sleep_optimizer.py --caffeine 2 --screen 2 --exercise 0 --bedtime 1.5 --alcohol 1
"""
import argparse
import csv

import numpy as np

FEATURES = ["caffeine_after_15", "screen_time_h", "exercise",
            "bedtime_after_22_h", "alcohol_drinks"]
LABELS = ["Coffee after 15:00 (per cup)", "Screen time before bed (per h)",
          "Exercise (yes)", "Bedtime (per h after 22:00)", "Alcohol (per drink)"]


def load_data(path="sleep_log.csv"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([float(r["sleep_score"]) for r in rows])
    return X, y


def fit(X, y):
    """Least-squares linear regression. Returns [coef_1 ... coef_n, intercept]."""
    X1 = np.c_[X, np.ones(len(X))]
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def predict(coef, X):
    X = np.atleast_2d(X)
    return np.c_[X, np.ones(len(X))] @ coef


def main():
    p = argparse.ArgumentParser(description="Sleep Optimizer demo")
    p.add_argument("--caffeine", type=float, default=2, help="cups of coffee after 15:00")
    p.add_argument("--screen", type=float, default=2, help="hours of screen time before bed")
    p.add_argument("--exercise", type=float, default=0, help="1 = exercised today, 0 = no")
    p.add_argument("--bedtime", type=float, default=1.5, help="planned bedtime in hours after 22:00")
    p.add_argument("--alcohol", type=float, default=1, help="alcoholic drinks tonight")
    a = p.parse_args()

    X, y = load_data()

    # 1) Honest evaluation: train on older nights, test on the most recent ones
    n_test = max(1, len(X) // 5)
    coef_train = fit(X[:-n_test], y[:-n_test])
    mae_model = np.mean(np.abs(predict(coef_train, X[-n_test:]) - y[-n_test:]))
    mae_baseline = np.mean(np.abs(y[:-n_test].mean() - y[-n_test:]))
    print(f"Nights in diary: {len(X)}  (train {len(X) - n_test} / test {n_test})")
    print(f"Mean error on test nights: model {mae_model:.2f} points, "
          f"baseline 'always average' {mae_baseline:.2f} points\n")

    # 2) Final model on all data
    coef = fit(X, y)
    print("Learned effect on sleep score (scale 1-10):")
    for label, c in zip(LABELS, coef[:-1]):
        print(f"  {label:<32} {c:+.2f}")
    print(f"  {'Baseline (all habits = 0)':<32} {coef[-1]:.2f}\n")

    # 3) Tonight + what-if analysis
    tonight = np.array([a.caffeine, a.screen, a.exercise, a.bedtime, a.alcohol])
    base = predict(coef, tonight)[0]
    print(f"Predicted sleep score for tonight: {base:.1f} / 10")

    changes = {
        "Skip one coffee after 15:00": (0, -1),
        "One hour less screen time":   (1, -1),
        "Do 30 min of exercise":       (2, +1),
        "Go to bed 30 min earlier":    (3, -0.5),
        "Skip one alcoholic drink":    (4, -1),
    }
    lower = np.array([0, 0, 0, 0, 0])
    upper = np.array([np.inf, np.inf, 1, np.inf, np.inf])
    gains = []
    for name, (idx, delta) in changes.items():
        alt = tonight.copy()
        alt[idx] = np.clip(alt[idx] + delta, lower[idx], upper[idx])
        if alt[idx] == tonight[idx]:
            continue  # change not possible (e.g. already 0 coffee)
        gains.append((predict(coef, alt)[0] - base, name))

    if gains:
        print("\nWhat would help most tonight:")
        for g, name in sorted(gains, reverse=True):
            print(f"  {name:<30} {g:+.1f} points")

    # 4) Chart for the README
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["#2e7d32" if c > 0 else "#c62828" for c in coef[:-1]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(LABELS, coef[:-1], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Effect on sleep score (points)")
        ax.set_title("What affects my sleep? (linear regression)", loc="left")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig("images/feature_impact.png", dpi=150)
        print("\nChart saved to images/feature_impact.png")
    except ImportError:
        print("\n(matplotlib not installed - chart skipped)")


if __name__ == "__main__":
    main()
