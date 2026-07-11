"""Regression test: the WISDM label map must match the dataset's official key.

The WISDM letter codes are NOT alphabetical by meaning (F=typing, M=kicking...),
which caused a real bug where 13/18 activities displayed the wrong name.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "health_tracking"))

KEY_PATH = os.path.join(os.path.dirname(__file__), "..", "wisdm-dataset", "activity_key.txt")

# Official short word (from activity_key.txt) -> our display name
WORD_TO_DISPLAY = {
    "walking": "Walking", "jogging": "Jogging", "stairs": "Stairs",
    "sitting": "Sitting", "standing": "Standing", "typing": "Typing",
    "teeth": "Brushing Teeth", "soup": "Eating Soup", "chips": "Eating Chips",
    "pasta": "Eating Pasta", "drinking": "Drinking", "sandwich": "Eating Sandwich",
    "kicking": "Kicking", "catch": "Playing Catch", "dribbling": "Dribbling",
    "writing": "Writing", "clapping": "Clapping", "folding": "Folding",
}


@pytest.mark.skipif(not os.path.exists(KEY_PATH), reason="WISDM dataset not on disk")
def test_wisdm_labels_match_official_activity_key():
    from config import WISDM_LABELS

    letter_to_word = {}
    with open(KEY_PATH) as fh:
        for line in fh:
            if "=" in line:
                word, letter = (p.strip() for p in line.split("="))
                letter_to_word[letter] = word

    # sklearn's LabelEncoder sorts the letters; index i = i-th sorted letter
    for idx, letter in enumerate(sorted(letter_to_word)):
        expected = WORD_TO_DISPLAY[letter_to_word[letter]]
        assert WISDM_LABELS[idx] == expected, (
            f"index {idx} (letter {letter}): config says {WISDM_LABELS[idx]!r}, "
            f"official key says {expected!r}"
        )


def test_wisdm_labels_cover_met_and_hr_tables():
    """Every WISDM activity name must exist in the MET / HR-zone tables,
    otherwise calories and heart-rate pages silently fall back to defaults."""
    from config import WISDM_LABELS, MET_VALUES, ACTIVITY_HR_ZONE, ACTIVITY_CATEGORIES

    for name in WISDM_LABELS.values():
        assert name in MET_VALUES, f"{name} missing from MET_VALUES"
        assert name in ACTIVITY_HR_ZONE, f"{name} missing from ACTIVITY_HR_ZONE"
        assert name in ACTIVITY_CATEGORIES, f"{name} missing from ACTIVITY_CATEGORIES"
