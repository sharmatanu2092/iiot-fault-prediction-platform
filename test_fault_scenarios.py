"""
Simulate realistic fault scenarios and validate the feature extractor
produces separable feature vectors for each fault class.

Run: pytest tests/fault_injection/test_fault_scenarios.py -v
"""

import math
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../cloud/lambda"))
from feature_extractor import extract_features

random.seed(99)


def make_window(vib_fn, temp_fn, curr_fn, n=100):
    return {
        "vibration":   [vib_fn(i)  for i in range(n)],
        "temperature": [temp_fn(i) for i in range(n)],
        "current":     [curr_fn(i) for i in range(n)],
    }


def test_normal_low_vibration_rms():
    win = make_window(
        vib_fn  = lambda i: 0.12 * math.sin(2*math.pi*25*i/100) + random.gauss(0, 0.01),
        temp_fn = lambda i: 62.0 + random.gauss(0, 0.3),
        curr_fn = lambda i: 4.2  + 0.3 * math.sin(2*math.pi*50*i/100) + random.gauss(0, 0.05)
    )
    f = extract_features(win)
    vib_rms = f[0]
    crest   = f[4]
    assert vib_rms < 0.15, f"Normal vib_rms too high: {vib_rms}"
    assert crest   < 6.0,  f"Normal crest factor too high: {crest}"


def test_bearing_wear_elevated_rms_vs_normal():
    def normal_vib(i):
        return 0.12 * math.sin(2*math.pi*25*i/100) + random.gauss(0, 0.01)

    def bearing_vib(i):
        return (0.15 * math.sin(2*math.pi*25*i/100) +
                0.35 * math.sin(2*math.pi*187*i/100) +
                random.gauss(0, 0.02))

    win_n = make_window(normal_vib,  lambda i: 62.0, lambda i: 4.2)
    win_b = make_window(bearing_vib, lambda i: 64.0, lambda i: 4.4)

    rms_normal  = extract_features(win_n)[0]
    rms_bearing = extract_features(win_b)[0]
    assert rms_bearing > rms_normal * 1.5, (
        f"Bearing rms ({rms_bearing:.3f}) should be >1.5x normal ({rms_normal:.3f})"
    )


def test_overheating_detected_by_temp():
    win = make_window(
        vib_fn  = lambda i: 0.13 * math.sin(2*math.pi*25*i/100) + random.gauss(0, 0.01),
        temp_fn = lambda i: 62.0 + (i / 100.0) * 28.0,
        curr_fn = lambda i: 5.8  + random.gauss(0, 0.2)
    )
    f = extract_features(win)
    temp_mean = f[7]
    temp_rise = f[8]
    assert temp_mean > 75,  f"Overheating temp_mean too low: {temp_mean}"
    assert temp_rise > 5.0, f"Overheating temp_rise too low: {temp_rise}"


def test_electrical_fault_current_spikes():
    def curr(i):
        if i in (10, 25, 50, 72, 88):
            return 4.2 + random.uniform(3.5, 5.0)
        return 4.2 + random.gauss(0, 0.08)

    win = make_window(
        vib_fn  = lambda i: 0.13 * math.sin(2*math.pi*25*i/100),
        temp_fn = lambda i: 63.0,
        curr_fn = curr
    )
    f = extract_features(win)
    curr_p2p  = f[10]
    curr_kurt = f[11]
    assert curr_p2p  > 3.0, f"Electrical fault curr_p2p too low: {curr_p2p}"
    assert curr_kurt > 5.0, f"Electrical fault kurtosis too low: {curr_kurt}"


def test_motor_imbalance_higher_rms_than_normal():
    def imbalance_vib(i):
        return (0.5 * math.sin(2*math.pi*50*i/100) +
                0.3 * math.sin(2*math.pi*50*i/100 + 1.2) +
                random.gauss(0, 0.03))

    def normal_vib(i):
        return 0.12 * math.sin(2*math.pi*25*i/100) + random.gauss(0, 0.01)

    win_n = make_window(normal_vib,    lambda i: 62.0, lambda i: 4.2)
    win_m = make_window(imbalance_vib, lambda i: 65.0, lambda i: 5.1)

    rms_normal    = extract_features(win_n)[0]
    rms_imbalance = extract_features(win_m)[0]
    assert rms_imbalance > rms_normal * 1.5, (
        f"Imbalance rms ({rms_imbalance:.3f}) should be >1.5x normal ({rms_normal:.3f})"
    )
