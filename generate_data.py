"""
Generates a SYNTHETIC sleep log (sleep_log.csv) for demo purposes.
No real personal data is used. Replace sleep_log.csv with your own diary later.
"""
import csv
from datetime import date, timedelta

import numpy as np

rng = np.random.default_rng(42)          # fixed seed -> reproducible data
N_NIGHTS = 60
start = date(2026, 8, 1)

rows = []
for i in range(N_NIGHTS):
    caffeine = int(rng.choice([0, 0, 1, 1, 2, 3]))          # cups after 15:00
    screen = float(rng.choice([0, 0.5, 1, 1.5, 2, 2.5, 3]))   # hours of screen time in the last 2 h before bed
    exercise = int(rng.random() < 0.45)                       # 1 = at least 30 min exercise that day
    bedtime = float(rng.choice([0, 0.5, 1, 1.5, 2, 2.5, 3]))  # hours after 22:00
    alcohol = int(rng.choice([0, 0, 0, 1, 1, 2, 3]))          # drinks that evening

    # "Hidden truth" used only to create the demo data
    score = (8.0 - 0.8 * caffeine - 0.5 * screen + 0.7 * exercise
             - 0.6 * bedtime - 0.7 * alcohol + rng.normal(0, 0.5))
    score = float(np.clip(round(score, 1), 1.0, 10.0))

    rows.append([(start + timedelta(days=i)).isoformat(),
                 caffeine, screen, exercise, bedtime, alcohol, score])

with open("sleep_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "caffeine_after_15", "screen_time_h", "exercise",
                "bedtime_after_22_h", "alcohol_drinks", "sleep_score"])
    w.writerows(rows)

print(f"Wrote {N_NIGHTS} synthetic nights to sleep_log.csv")
