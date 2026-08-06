"""
Health Tracking System — Configuration & Constants (UCI HAR)
============================================================
Centralised settings for activity-to-MET mapping, demographics,
sleep scoring, fall detection thresholds, and dataset label maps.
"""

# ─── Activity → MET Mapping ───────────────────────────────────────────────────
# Metabolic Equivalent of Task (MET) values from the Compendium of Physical Activities
# 1 MET ≈ 1 kcal/kg/hour (resting metabolic rate)
MET_VALUES = {
    "Walking":            3.5,
    "Walking Upstairs":   8.0,
    "Walking Downstairs": 6.0,
    "Sitting":            1.3,
    "Standing":           1.8,
    "Laying":             1.0,
}

# ─── Typical Step Cadence (steps/min) ─────────────────────────────────────────
# Used to estimate daily step counts from the activity timeline.
STEP_CADENCE = {
    "Walking":            100,
    "Walking Upstairs":    75,
    "Walking Downstairs":  85,
}

# ─── Default User Demographics ─────────────────────────────────────────────────
DEFAULT_WEIGHT_KG   = 70.0
DEFAULT_HEIGHT_CM   = 170.0
DEFAULT_AGE         = 25
DEFAULT_GENDER      = "Male"

# ─── Sleep Scoring Thresholds ──────────────────────────────────────────────────
# Standard deviation of accelerometer magnitude during a "Laying" window
SLEEP_DEEP_THRESHOLD   = 0.02   # very still → deep sleep
SLEEP_LIGHT_THRESHOLD  = 0.08   # moderate movement → light sleep

# ─── Fall Detection Thresholds ─────────────────────────────────────────────────
FALL_IMPACT_G          = 3.0    # sudden spike in g-force (normalised gravity units)
FALL_FREEFALL_G        = 0.4    # near-weightlessness before impact
FALL_POST_LAYING_SEC   = 5.0    # person remains laying after impact for this long

# ─── Heart Rate Zones (bpm) ────────────────────────────────────────────────────
HR_PROFILES = {
    "rest":     (58, 72),
    "light":    (72, 100),
    "moderate": (100, 140),
    "vigorous": (140, 175),
}

# Map activities to HR zones
ACTIVITY_HR_ZONE = {
    "Laying":             "rest",
    "Sitting":            "rest",
    "Standing":           "light",
    "Walking":            "moderate",
    "Walking Downstairs": "moderate",
    "Walking Upstairs":   "vigorous",
}

# ─── Dataset Label Map ────────────────────────────────────────────────────────
UCI_HAR_LABELS = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying",
}

# ─── Activity Categories (for colour-coding & grouping) ───────────────────────
ACTIVITY_CATEGORIES = {
    "Walking":            "locomotion",
    "Walking Upstairs":   "locomotion",
    "Walking Downstairs": "locomotion",
    "Sitting":            "sedentary",
    "Standing":           "sedentary",
    "Laying":             "sedentary",
}

CATEGORY_COLORS = {
    "locomotion":   "#22c55e",   # green
    "sedentary":    "#ef4444",   # red
}

# ─── Colour Palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg_dark":       "#0f172a",
    "bg_card":       "#1e293b",
    "bg_card_hover": "#334155",
    "accent":        "#6366f1",    # indigo
    "accent_light":  "#818cf8",
    "success":       "#22c55e",
    "warning":       "#f59e0b",
    "danger":        "#ef4444",
    "text_primary":  "#f1f5f9",
    "text_muted":    "#94a3b8",
    "border":        "#334155",
}
