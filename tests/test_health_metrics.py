"""Unit tests for the health-tracking metrics engine (no torch required)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "health_tracking"))

from health_metrics import (  # noqa: E402
    compute_bmr, calories_burned, compute_daily_calories, active_calories,
    count_steps, estimate_steps_from_timeline, sedentary_ratio,
    classify_sleep_stage, compute_sleep_score, simulate_sleep_night,
    detect_fall, simulate_heart_rate, simulate_hrv, infer_activity_from_hr,
    generate_daily_timeline,
)


# ── Timeline ─────────────────────────────────────────────────────────────────

def test_timeline_covers_full_day():
    tl = generate_daily_timeline()
    assert abs(sum(s["duration_min"] for s in tl) - 1440) < 1e-6


def test_timeline_segments_contiguous():
    tl = generate_daily_timeline()
    for a, b in zip(tl, tl[1:]):
        assert abs((a["start_hour"] + a["duration_min"] / 60) - b["start_hour"]) < 0.02


def test_timeline_respects_allowed_activities():
    allowed = ["Walking", "Sitting", "Standing", "Stairs", "Jogging"]
    tl = generate_daily_timeline(allowed_activities=allowed)
    assert all(seg["activity"] in allowed for seg in tl)


# ── Calories ─────────────────────────────────────────────────────────────────

def test_bmr_mifflin_st_jeor():
    assert compute_bmr(70, 170, 25, "Male") == pytest.approx(1642.5)
    assert compute_bmr(70, 170, 25, "Female") == pytest.approx(1476.5)


def test_calories_met_formula():
    # Walking = 3.5 MET; 30 min at 70 kg -> 3.5 * 70 * 0.5 = 122.5
    assert calories_burned("Walking", 30, 70) == pytest.approx(122.5)


def test_active_calories_excludes_resting_baseline():
    tl = generate_daily_timeline()
    total = compute_daily_calories(tl, 70)
    active = active_calories(tl, 70)
    assert active == pytest.approx(total - 70 * 24)
    assert 0 < active < total


# ── Steps ────────────────────────────────────────────────────────────────────

def test_count_steps_detects_walking_frequency():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 5, 500)  # 5 s at 100 Hz
    acc = 1.0 + 0.4 * np.sin(2 * np.pi * 2.0 * t) + 0.1 * rng.standard_normal(500)
    info = count_steps(acc, sampling_rate=100, min_peak_height=0.12, min_peak_distance=35)
    assert 8 <= info["steps"] <= 12          # ~2 Hz -> ~10 steps
    assert len(info["peak_indices"]) == info["steps"]
    assert 100 <= info["cadence_spm"] <= 140


def test_estimate_steps_deterministic_and_plausible():
    tl = generate_daily_timeline()
    steps = estimate_steps_from_timeline(tl)
    assert steps == estimate_steps_from_timeline(tl)
    assert 5000 < steps < 20000


# ── Sedentary ────────────────────────────────────────────────────────────────

def test_sedentary_excludes_sleep_and_partitions_day():
    sed = sedentary_ratio(generate_daily_timeline())
    assert sed["sleep_min"] == 570  # Laying segments
    total = sed["active_min"] + sed["light_min"] + sed["sedentary_min"] + sed["sleep_min"]
    assert total == pytest.approx(1440, abs=0.5)


# ── Sleep ────────────────────────────────────────────────────────────────────

def test_sleep_stage_thresholds():
    assert classify_sleep_stage(0.01) == "Deep"
    assert classify_sleep_stage(0.05) == "Light"
    assert classify_sleep_stage(0.12) == "REM"


def test_simulated_night_is_classified_not_hardcoded():
    night = simulate_sleep_night()
    assert [classify_sleep_stage(s) for s in night["stds"]] == night["stages"]
    assert 0 <= night["score"] <= 100
    assert night["hours"] == 7.0


def test_sleep_score_edge_cases():
    assert compute_sleep_score([]) == 0
    ideal = ["Deep"] * 25 + ["Light"] * 55 + ["REM"] * 20
    assert compute_sleep_score(ideal) == 100


# ── Fall detection ───────────────────────────────────────────────────────────

def _fall_signal():
    rng = np.random.default_rng(77)
    sig = np.ones(400)
    t = np.linspace(0, 4, 400)
    sig[:100] += 0.15 * np.sin(2 * np.pi * 2 * t[:100]) + 0.05 * rng.standard_normal(100)
    sig[100:130] = np.linspace(1.0, 0.2, 30)   # free-fall
    sig[130], sig[131], sig[132] = 4.5, 3.8, 2.5  # impact
    sig[133:160] = np.linspace(1.8, 1.0, 27)   # settling
    sig[160:] = 0.98 + 0.01 * rng.standard_normal(240)  # immobile
    return sig


def test_fall_detected_with_immobility():
    res = detect_fall(_fall_signal(), sampling_rate=100)
    assert res["fall_detected"] and res["impact_index"] == 130
    assert res["immobile_after"]


def test_walking_is_not_a_fall():
    rng = np.random.default_rng(55)
    t = np.linspace(0, 4, 400)
    walk = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * t) + 0.08 * rng.standard_normal(400)
    assert not detect_fall(walk, sampling_rate=100)["fall_detected"]


# ── HR / HRV ─────────────────────────────────────────────────────────────────

def test_heart_rate_bounds_and_zones():
    hr = simulate_heart_rate("Walking", 200)
    assert hr.min() >= 40 and hr.max() <= 200
    resting = simulate_heart_rate("Sitting", 100, ramp=False)
    assert 50 < resting.mean() < 85


def test_hrv_stress_bands():
    hrv = simulate_hrv("Laying")
    assert set(hrv) == {"rmssd", "sdnn", "stress_level"}
    assert hrv["stress_level"] in ("Low", "Medium", "High")


# ── HR → activity reverse lookup ─────────────────────────────────────────────

def test_hr_lookup_resting_rate_maps_to_rest_activities():
    guess = infer_activity_from_hr(65)
    assert guess["zone"] == "rest"
    from config import ACTIVITY_HR_ZONE
    assert ACTIVITY_HR_ZONE[guess["ranked"][0][0]] == "rest"


def test_hr_lookup_vigorous_rate_maps_to_vigorous_activities():
    guess = infer_activity_from_hr(160)
    assert guess["zone"] == "vigorous"
    from config import ACTIVITY_HR_ZONE
    assert ACTIVITY_HR_ZONE[guess["ranked"][0][0]] == "vigorous"


def test_hr_lookup_respects_activity_subset():
    uci_only = ["Walking", "Walking Upstairs", "Walking Downstairs",
                "Sitting", "Standing", "Laying"]
    guess = infer_activity_from_hr(120, uci_only)
    assert all(a in uci_only for a, _ in guess["ranked"])
    assert guess["ranked"][0][1] == 1.0  # top score normalised


def test_hr_lookup_out_of_range_labels():
    assert infer_activity_from_hr(45)["zone"] == "below rest"
    assert infer_activity_from_hr(200)["zone"] == "above vigorous"
