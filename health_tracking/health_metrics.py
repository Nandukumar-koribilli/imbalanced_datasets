"""
Health Tracking System — Health Metrics Engine
================================================
Pure-logic functions that compute health-related metrics from
activity predictions and raw sensor data.  No UI code here.
"""

import numpy as np
from config import (
    MET_VALUES,
    HR_PROFILES,
    ACTIVITY_HR_ZONE,
    STEP_CADENCE,
    SLEEP_DEEP_THRESHOLD,
    SLEEP_LIGHT_THRESHOLD,
    FALL_IMPACT_G,
    FALL_FREEFALL_G,
    FALL_POST_LAYING_SEC,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Calorie / Energy Expenditure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin–St Jeor Basal Metabolic Rate (kcal/day)."""
    if gender.lower() == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calories_burned(activity: str, duration_min: float, weight_kg: float) -> float:
    """
    Calories burned for *activity* over *duration_min* minutes.
    Uses the standard MET approximation: 1 MET ≈ 1 kcal/kg/hour,
    so the result already INCLUDES resting energy expenditure.
    """
    met = MET_VALUES.get(activity, 1.3)
    return met * weight_kg * (duration_min / 60.0)


def compute_daily_calories(activity_timeline: list[dict], weight_kg: float) -> float:
    """
    activity_timeline: list of {"activity": str, "duration_min": float}
    Returns total kcal over the timeline (resting expenditure included,
    since MET values are absolute multiples of resting metabolic rate).
    """
    return sum(
        calories_burned(seg["activity"], seg["duration_min"], weight_kg)
        for seg in activity_timeline
    )


def active_calories(activity_timeline: list[dict], weight_kg: float) -> float:
    """
    Calories burned ABOVE the 1-MET resting baseline — i.e. the energy
    attributable to movement rather than basal metabolism.
    """
    total_min = sum(seg["duration_min"] for seg in activity_timeline)
    resting_baseline = 1.0 * weight_kg * (total_min / 60.0)
    return max(0.0, compute_daily_calories(activity_timeline, weight_kg) - resting_baseline)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step Counting & Cadence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def count_steps(acc_magnitude: np.ndarray, sampling_rate: float = 50.0,
                min_peak_height: float = 0.15, min_peak_distance: int = 15) -> dict:
    """
    Count steps by detecting peaks in accelerometer magnitude.
    Returns {"steps", "cadence_spm", "duration_sec", "peak_indices"}.
    """
    from scipy.signal import find_peaks

    # Mean-subtract to centre around zero
    signal = acc_magnitude - np.mean(acc_magnitude)

    peaks, _ = find_peaks(signal, height=min_peak_height, distance=min_peak_distance)
    n_steps = len(peaks)
    duration_sec = len(acc_magnitude) / sampling_rate
    cadence = (n_steps / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    return {
        "steps": n_steps,
        "cadence_spm": round(cadence, 1),
        "duration_sec": round(duration_sec, 2),
        "peak_indices": peaks,
    }


def estimate_steps_from_timeline(activity_timeline: list[dict]) -> int:
    """
    Estimate total daily steps from an activity timeline using typical
    cadences (steps/min) per ambulatory activity. Non-ambulatory
    activities contribute zero steps.
    """
    return int(round(sum(
        seg["duration_min"] * STEP_CADENCE.get(seg["activity"], 0)
        for seg in activity_timeline
    )))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sedentary Behaviour
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sedentary_ratio(activity_timeline: list[dict]) -> dict:
    """
    Break the day into sleep / sedentary / light / active minutes using
    the standard MET cut-points (sedentary ≤ 1.5 MET while awake,
    light 1.6–2.9 MET, moderate-to-vigorous ≥ 3.0 MET).
    "Laying" is treated as sleep and excluded from sedentary time.
    `ratio` = fraction of AWAKE time spent in moderate-to-vigorous activity.
    """
    sleep = sedentary = light = active = 0.0
    for seg in activity_timeline:
        met = MET_VALUES.get(seg["activity"], 1.3)
        dur = seg["duration_min"]
        if seg["activity"] == "Laying":
            sleep += dur
        elif met <= 1.5:
            sedentary += dur
        elif met < 3.0:
            light += dur
        else:
            active += dur
    awake = active + light + sedentary
    return {
        "active_min": round(active, 1),
        "light_min": round(light, 1),
        "sedentary_min": round(sedentary, 1),
        "sleep_min": round(sleep, 1),
        "ratio": round(active / awake, 2) if awake > 0 else 0.0,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sleep Quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_sleep_stage(acc_std: float) -> str:
    """Classify a single sleep window based on accelerometer std-dev."""
    if acc_std <= SLEEP_DEEP_THRESHOLD:
        return "Deep"
    elif acc_std <= SLEEP_LIGHT_THRESHOLD:
        return "Light"
    else:
        return "REM"


def compute_sleep_score(stages: list[str]) -> int:
    """
    A simple 0-100 sleep quality score.
    Ideal: ~55 % Light, ~25 % Deep, ~20 % REM.
    """
    if not stages:
        return 0
    total = len(stages)
    deep_pct  = stages.count("Deep")  / total
    light_pct = stages.count("Light") / total
    rem_pct   = stages.count("REM")   / total

    # Penalise deviation from ideal
    deep_score  = max(0, 1 - abs(deep_pct  - 0.25) / 0.25) * 30
    light_score = max(0, 1 - abs(light_pct - 0.55) / 0.55) * 40
    rem_score   = max(0, 1 - abs(rem_pct   - 0.20) / 0.20) * 30

    return int(deep_score + light_score + rem_score)


def simulate_sleep_night(n_windows: int = 84, window_min: int = 5, seed: int = 101) -> dict:
    """
    Simulate one night of accelerometer micro-movement (std-dev per
    *window_min*-minute window) following a realistic sleep-cycle pattern,
    then classify each window with `classify_sleep_stage` — so the stages
    shown downstream are genuinely produced by the classification algorithm.
    Returns {"stds", "stages", "score", "hours"}.
    """
    rng = np.random.default_rng(seed)
    eps = 1e-4  # keep samples strictly inside each stage's band
    cycle_pattern = (
        ["Deep"] * 6 + ["Light"] * 4 + ["REM"] * 3 +
        ["Light"] * 5 + ["Deep"] * 5 + ["Light"] * 4 + ["REM"] * 4 +
        ["Light"] * 6 + ["Deep"] * 4 + ["Light"] * 5 + ["REM"] * 5 +
        ["Light"] * 8 + ["Deep"] * 3 + ["Light"] * 6 + ["REM"] * 6 +
        ["Light"] * 10
    )
    stds = []
    for i in range(n_windows):
        stage = cycle_pattern[i % len(cycle_pattern)]
        if stage == "Deep":
            stds.append(rng.uniform(0.004, SLEEP_DEEP_THRESHOLD))
        elif stage == "Light":
            stds.append(rng.uniform(SLEEP_DEEP_THRESHOLD + eps, SLEEP_LIGHT_THRESHOLD))
        else:  # REM / restless
            stds.append(rng.uniform(SLEEP_LIGHT_THRESHOLD + eps, 0.18))
    stds = np.array(stds)
    stages = [classify_sleep_stage(s) for s in stds]
    return {
        "stds": stds,
        "stages": stages,
        "score": compute_sleep_score(stages),
        "hours": round(n_windows * window_min / 60.0, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fall Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_fall(acc_magnitude: np.ndarray, sampling_rate: float = 50.0) -> dict:
    """
    Threshold-based fall detection over the three canonical phases:
      1. Free-fall — acceleration drops below FALL_FREEFALL_G shortly before impact
      2. Impact    — spike exceeding FALL_IMPACT_G
      3. Immobility — low movement around 1 g in the seconds after impact
    Returns {"fall_detected", "impact_index", "impact_g", "immobile_after"}.
    """
    n = len(acc_magnitude)
    pre_window = max(1, int(0.5 * sampling_rate))  # look-back for free-fall (~0.5 s)
    for i in range(n):
        if acc_magnitude[i] >= FALL_IMPACT_G:
            window_before = acc_magnitude[max(0, i - pre_window): i]
            if len(window_before) > 0 and np.min(window_before) <= FALL_FREEFALL_G:
                # Skip ~1 s of post-impact settling, then check for stillness
                start = min(n, i + int(1.0 * sampling_rate))
                stop = min(n, start + int(FALL_POST_LAYING_SEC * sampling_rate))
                post = acc_magnitude[start:stop]
                immobile = bool(
                    len(post) >= int(1.0 * sampling_rate)
                    and np.std(post) < 0.05
                    and abs(np.mean(post) - 1.0) < 0.2
                )
                return {
                    "fall_detected": True,
                    "impact_index": int(i),
                    "impact_g": round(float(acc_magnitude[i]), 2),
                    "immobile_after": immobile,
                }
    return {"fall_detected": False, "impact_index": None, "impact_g": 0.0,
            "immobile_after": False}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Heart Rate Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_heart_rate(activity: str, duration_samples: int = 128,
                        ramp: bool = True) -> np.ndarray:
    """
    Simulate a plausible heart-rate trace for the given activity.
    Returns an array of `duration_samples` BPM values with physiological noise.
    Set ramp=False for mid-day segments so consecutive segments don't each
    restart with an artificial warm-up dip.
    """
    zone = ACTIVITY_HR_ZONE.get(activity, "rest")
    lo, hi = HR_PROFILES[zone]
    base = np.random.uniform(lo, hi)
    # Add small physiological variability
    noise = np.random.normal(0, 2.0, duration_samples)
    hr = np.full(duration_samples, base) + noise
    if ramp:
        # Smooth ramp-up over first 20 %
        r = np.linspace(0.85, 1.0, int(duration_samples * 0.2))
        hr[: len(r)] *= r
    return np.clip(hr, 40, 200)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stress / HRV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_hrv(activity: str) -> dict:
    """
    Simulate Heart-Rate Variability metrics.
    Stressed states have lower RMSSD and SDNN.
    """
    zone = ACTIVITY_HR_ZONE.get(activity, "rest")
    if zone in ("rest",):
        rmssd = np.random.uniform(35, 65)
        sdnn  = np.random.uniform(40, 80)
    elif zone in ("light", "moderate"):
        rmssd = np.random.uniform(20, 40)
        sdnn  = np.random.uniform(25, 50)
    else:
        rmssd = np.random.uniform(8, 25)
        sdnn  = np.random.uniform(10, 30)

    stress_level = "Low" if rmssd > 35 else ("Medium" if rmssd > 18 else "High")
    return {
        "rmssd": round(rmssd, 1),
        "sdnn": round(sdnn, 1),
        "stress_level": stress_level,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Simulated 24-hour Timeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_daily_timeline(allowed_activities: list = None) -> list[dict]:
    """
    Generate a realistic simulated 24-hour activity schedule.
    Segments are contiguous and cover the full 1440 minutes.

    If *allowed_activities* is provided, any activity not in that set is
    substituted with the closest available equivalent so the timeline uses
    labels that actually exist in the selected dataset.
    Returns list of {"activity", "start_hour", "duration_min"} dicts.
    """
    schedule = [
        {"activity": "Laying",            "start_hour": 0,    "duration_min": 420},  # 0:00–7:00 sleep
        {"activity": "Standing",          "start_hour": 7,    "duration_min":  15},
        {"activity": "Walking",           "start_hour": 7.25, "duration_min":  30},
        {"activity": "Sitting",           "start_hour": 7.75, "duration_min":  45},  # breakfast
        {"activity": "Walking",           "start_hour": 8.5,  "duration_min":  30},  # commute
        {"activity": "Sitting",           "start_hour": 9,    "duration_min": 120},  # work
        {"activity": "Walking Upstairs",  "start_hour": 11,   "duration_min":  10},
        {"activity": "Standing",          "start_hour": 11.17,"duration_min":  20},
        {"activity": "Sitting",           "start_hour": 11.5, "duration_min":  90},  # work
        {"activity": "Walking",           "start_hour": 13,   "duration_min":  15},  # lunch walk
        {"activity": "Sitting",           "start_hour": 13.25,"duration_min":  45},  # lunch
        {"activity": "Sitting",           "start_hour": 14,   "duration_min": 180},  # work
        {"activity": "Walking Downstairs","start_hour": 17,   "duration_min":  10},
        {"activity": "Walking",           "start_hour": 17.17,"duration_min":  30},  # commute
        {"activity": "Standing",          "start_hour": 17.67,"duration_min":  20},
        {"activity": "Walking Upstairs",  "start_hour": 18,   "duration_min":  15},  # exercise
        {"activity": "Sitting",           "start_hour": 18.25,"duration_min":  60},  # dinner
        {"activity": "Sitting",           "start_hour": 19.25,"duration_min": 120},  # relax
        {"activity": "Standing",          "start_hour": 21.25,"duration_min":  15},
        {"activity": "Laying",            "start_hour": 21.5, "duration_min": 150},  # wind-down & sleep
    ]
    if allowed_activities is not None:
        allowed = set(allowed_activities)
        substitutions = {
            "Laying":             "Sitting",
            "Walking Upstairs":   "Stairs",
            "Walking Downstairs": "Stairs",
            "Stairs":             "Walking Upstairs",
            "Sitting":            "Standing",
        }
        for seg in schedule:
            if seg["activity"] not in allowed:
                seg["activity"] = substitutions.get(seg["activity"], next(iter(allowed)))
    return schedule
