<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Sleep Optimizer

Final project for the Building AI course

## Summary

Sleep Optimizer learns from a simple personal sleep diary which daily habits help or hurt your sleep. Using linear regression, it predicts tonight's sleep score and suggests the one change that would help you most. Building AI course project.

## Background

Many people sleep badly but don't know why. Generic sleep tips ("no coffee in the evening", "less screen time") apply to everyone and therefore to no one in particular. Everyone reacts differently to caffeine, exercise, alcohol or late nights.

Problems this idea addresses:

* **Poor sleep is very common:** According to the CDC, roughly a third of U.S. adults regularly sleep less than the recommended 7 hours ([CDC FastStats: Sleep in Adults](https://www.cdc.gov/sleep/data-research/facts-stats/adults-sleep-facts-and-stats.html)).
* **People underestimate their own habits:** In a study by Drake et al. (2013), caffeine taken even 6 hours before bed measurably disrupted sleep, but participants did not notice it themselves ([Journal of Clinical Sleep Medicine, doi:10.5664/jcsm.3170](https://doi.org/10.5664/jcsm.3170)).
* **Sleep trackers collect data but rarely explain it:** Wearables show *how* you slept, but seldom *why*.

**Personal motivation:** As a dual student switching between university and work, my schedule varies a lot, and so does my sleep. I wanted to find out, with data instead of gut feeling, which of my habits actually matter.

## How is it used?

1. **Every morning** the user logs yesterday's habits and rates last night's sleep from 1 to 10 (takes under a minute).
2. **After 2-3 weeks** there is enough data for the model to learn personal effects.
3. **Every evening** the user enters the plan for tonight (e.g. 2 coffees, 2 h screen time, no exercise).
4. The app predicts the sleep score and ranks which single change would help most.

Example output of the demo:

```
Predicted sleep score for tonight: 3.9 / 10

What would help most tonight:
  Skip one coffee after 15:00    +0.8 points
  Skip one alcoholic drink       +0.8 points
  Do 30 min of exercise          +0.7 points
  One hour less screen time      +0.4 points
  Go to bed 30 min earlier       +0.3 points
```

<img src="feature_impact.png" width="600" alt="Learned effect of each habit on the sleep

**Users:** students, shift workers, parents, and anyone who wants to understand their own sleep.
**Needs to consider:** logging must be quick, advice must be understandable, and health data must stay private (stored locally, not in the cloud).

## Data sources and AI methods

**Data:** The demo uses a **synthetic** sleep diary (`sleep_log.csv`, 60 nights) created by `generate_data.py`. It contains no real personal data. In real use, the data would come from the user's own diary and optionally from wearable exports.

| Feature | Description | Example |
| ------- | ----------- | ------- |
| `caffeine_after_15` | Cups of coffee after 15:00 | 2 |
| `screen_time_h` | Screen time in the last 2 h before bed | 1.5 |
| `exercise` | At least 30 min of exercise (1 = yes) | 1 |
| `bedtime_after_22_h` | Bedtime in hours after 22:00 | 0.5 |
| `alcohol_drinks` | Alcoholic drinks that evening | 0 |
| `sleep_score` | Self-rated sleep quality, 1-10 (**target**) | 7.5 |

**AI methods:**

| Method | Used for |
| ------ | -------- |
| Linear regression (least squares) | Predicting the sleep score from habits |
| Regression coefficients | Explaining the effect of each habit |
| Train/test split | Checking the model on nights it has not seen |
| What-if analysis | Finding the most helpful change for tonight |

On the synthetic test nights, the model's mean error is **0.35 points**, compared to **1.14 points** for a baseline that always predicts the average score.

**Run the demo:**

```
pip install -r requirements.txt
python generate_data.py        # creates the synthetic sleep_log.csv
python sleep_optimizer.py      # trains the model and gives advice
python sleep_optimizer.py --caffeine 0 --screen 1 --exercise 1 --bedtime 0.5 --alcohol 0
```

Core of the model:

```python
def fit(X, y):
    """Least-squares linear regression. Returns [coef_1 ... coef_n, intercept]."""
    X1 = np.c_[X, np.ones(len(X))]
    return np.linalg.lstsq(X1, y, rcond=None)[0]
```

## Challenges

* **Correlation is not causation:** The model shows patterns, not causes. If I drink coffee mainly on stressful days, the coffee may get the blame for the stress.
* **Small data:** One person produces only one data point per night. Reliable results need several weeks of consistent logging.
* **Subjective target:** A self-rated score of 1-10 is noisy and can be biased by mood.
* **Linear model is simple:** It cannot capture interactions (e.g. coffee only matters on days without exercise) or thresholds.
* **Not a medical product:** It cannot detect sleep disorders such as insomnia or sleep apnea. Persistent sleep problems belong with a doctor.
* **Privacy:** Sleep and alcohol data are sensitive health data. They should be stored locally, and real diaries should never be pushed to a public repository.
* **Wellbeing:** The app should not create pressure to "sleep perfectly". Advice is framed as optional suggestions.

## What next?

* Import data automatically from wearables (Fitbit, Garmin, Apple Health) instead of manual logging
* Add more factors: stress level, room temperature, late meals, naps
* Use non-linear models (e.g. decision trees) to capture interactions between habits
* Cluster nights into sleep types (k-means) to find recurring patterns
* Build a simple mobile app with an evening reminder
* Test the idea with real users over several weeks

Skills and help needed: mobile app development, access to wearable APIs, and input from sleep researchers to design the features and avoid misleading advice.

## Acknowledgments

* [Building AI](https://buildingai.elementsofai.com/) course by Reaktor Innovations and the University of Helsinki, which inspired the regression approach; README structure based on the course's project template
* [CDC: Sleep in Adults](https://www.cdc.gov/sleep/data-research/facts-stats/adults-sleep-facts-and-stats.html) for background statistics
* Drake C., Roehrs T., Shambroom J., Roth T. (2013): *Caffeine Effects on Sleep Taken 0, 3, or 6 Hours before Going to Bed*, Journal of Clinical Sleep Medicine 9(11), [doi:10.5664/jcsm.3170](https://doi.org/10.5664/jcsm.3170)
* [NumPy](https://numpy.org/) (BSD license) and [Matplotlib](https://matplotlib.org/) (Matplotlib license, PSF-based)
* All code, data and images in this repository are my own work, published under the MIT License
