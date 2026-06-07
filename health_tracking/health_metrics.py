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
    SLEEP_DEEP_THRESHOLD,
    SLEEP_LIGHT_THRESHOLD,
    FALL_IMPACT_G,
    FALL_FREEFALL_G,
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
    """Active calories burned for *activity* over *duration_min* minutes."""
    met = MET_VALUES.get(activity, 1.3)
    return met * weight_kg * (duration_min / 60.0)


def compute_daily_calories(activity_timeline: list[dict], weight_kg: float) -> float:
    """
    activity_timeline: list of {"activity": str, "duration_min": float}
    Returns total active kcal.
    """
    return sum(
        calories_burned(seg["activity"], seg["duration_min"], weight_kg)
        for seg in activity_timeline
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step Counting & Cadence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def count_steps(acc_magnitude: np.ndarray, sampling_rate: float = 50.0,
                min_peak_height: float = 0.15, min_peak_distance: int = 15) -> dict:
    """
    Count steps by detecting peaks in accelerometer magnitude.
    Returns {"steps": int, "cadence_spm": float, "duration_sec": float}.
    """
    from scipy.signal import find_peaks

    # Mean-subtract to centre around zero
    signal = acc_magnitude - np.mean(acc_magnitude)

    peaks, _ = find_peaks(signal, height=min_peak_height, distance=min_peak_distance)
    n_steps = len(peaks)
    duration_sec = len(acc_magnitude) / sampling_rate
    cadence = (n_steps / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    return {"steps": n_steps, "cadence_spm": round(cadence, 1), "duration_sec": round(duration_sec, 2)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sedentary Behaviour
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sedentary_ratio(activity_timeline: list[dict]) -> dict:
    """
    Compute active vs sedentary minutes from timeline.
    Returns {"active_min": float, "sedentary_min": float, "ratio": float}.
    """
    active = sum(
        s["duration_min"] for s in activity_timeline
        if s["activity"] in ("Walking", "Walking Upstairs", "Walking Downstairs",
                              "Jogging", "Stairs", "Kicking", "Dribbling",
                              "Playing Catch")
    )
    sedentary = sum(
        s["duration_min"] for s in activity_timeline
        if s["activity"] in ("Sitting", "Standing", "Laying", "Typing", "Writing")
    )
    total = active + sedentary if (active + sedentary) > 0 else 1
    return {
        "active_min": round(active, 1),
        "sedentary_min": round(sedentary, 1),
        "ratio": round(active / total, 2),
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fall Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_fall(acc_magnitude: np.ndarray, sampling_rate: float = 50.0) -> dict:
    """
    Simple threshold-based fall detection.
    Looks for:  free-fall (low g) → impact (high g) pattern.
    Returns {"fall_detected": bool, "impact_index": int | None, "impact_g": float}.
    """
    for i in range(1, len(acc_magnitude) - 1):
        if acc_magnitude[i] >= FALL_IMPACT_G:
            # Check for preceding free-fall (low g in the 10 samples before)
            window_before = acc_magnitude[max(0, i - 10): i]
            if len(window_before) > 0 and np.min(window_before) <= FALL_FREEFALL_G:
                return {
                    "fall_detected": True,
                    "impact_index": int(i),
                    "impact_g": round(float(acc_magnitude[i]), 2),
                }
    return {"fall_detected": False, "impact_index": None, "impact_g": 0.0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Heart Rate Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_heart_rate(activity: str, duration_samples: int = 128) -> np.ndarray:
    """
    Simulate a plausible heart-rate trace for the given activity.
    Returns an array of `duration_samples` BPM values with physiological noise.
    """
    zone = ACTIVITY_HR_ZONE.get(activity, "rest")
    lo, hi = HR_PROFILES[zone]
    base = np.random.uniform(lo, hi)
    # Add small physiological variability
    noise = np.random.normal(0, 2.0, duration_samples)
    # Smooth ramp-up over first 20 %
    ramp = np.linspace(0.85, 1.0, int(duration_samples * 0.2))
    hr = np.full(duration_samples, base) + noise
    hr[: len(ramp)] *= ramp
    return np.clip(hr, 40, 200)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stress / HRV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_hrv(activity: str, duration_samples: int = 60) -> dict:
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

def generate_daily_timeline(label_map: dict = None) -> list[dict]:
    """
    Generate a realistic simulated 24-hour activity schedule.
    Returns list of {"activity", "start_hour", "duration_min"} dicts.
    """
    schedule = [
        {"activity": "Laying",            "start_hour": 0,   "duration_min": 420},  # 0:00–7:00 sleep
        {"activity": "Standing",          "start_hour": 7,   "duration_min":  15},
        {"activity": "Walking",           "start_hour": 7.25,"duration_min":  30},
        {"activity": "Sitting",           "start_hour": 7.75,"duration_min":  45},  # breakfast
        {"activity": "Walking",           "start_hour": 8.5, "duration_min":  20},  # commute
        {"activity": "Sitting",           "start_hour": 9,   "duration_min": 120},  # work
        {"activity": "Walking Upstairs",  "start_hour": 11,  "duration_min":  10},
        {"activity": "Standing",          "start_hour": 11.17,"duration_min": 20},
        {"activity": "Sitting",           "start_hour": 11.5,"duration_min":  90},  # work
        {"activity": "Walking",           "start_hour": 13,  "duration_min":  15},  # lunch walk
        {"activity": "Sitting",           "start_hour": 13.25,"duration_min": 45},  # lunch
        {"activity": "Sitting",           "start_hour": 14,  "duration_min": 180},  # work
        {"activity": "Walking Downstairs","start_hour": 17,  "duration_min":  10},
        {"activity": "Walking",           "start_hour": 17.17,"duration_min": 30},  # commute
        {"activity": "Standing",          "start_hour": 17.67,"duration_min": 20},
        {"activity": "Walking Upstairs",  "start_hour": 18,  "duration_min":  15},  # exercise
        {"activity": "Sitting",           "start_hour": 18.25,"duration_min": 60},  # dinner
        {"activity": "Sitting",           "start_hour": 19.25,"duration_min": 120}, # relax
        {"activity": "Standing",          "start_hour": 21.25,"duration_min": 15},
        {"activity": "Laying",            "start_hour": 21.5,"duration_min": 150},  # wind-down & sleep
    ]
    return schedule
