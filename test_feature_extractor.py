"""
Unit tests for cloud/lambda/feature_extractor.py
Run: pytest tests/unit/test_feature_extractor.py -v
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../cloud/lambda"))

from feature_extractor import (
    _mean, _variance, _rms, _peak_to_peak,
    _kurtosis, _skewness, _crest_factor,
    _fft_dominant_freq, extract_features
)


def test_mean_uniform():
    assert abs(_mean([1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0) < 1e-9


def test_mean_empty():
    assert _mean([]) == 0.0


def test_rms_dc():
    vals = [3.0] * 100
    assert abs(_rms(vals) - 3.0) < 1e-9


def test_rms_sine():
    import math
    n   = 1000
    sig = [math.sin(2 * math.pi * 10 * i / n) for i in range(n)]
    # RMS of a pure sine is 1/sqrt(2)
    assert abs(_rms(sig) - 1.0 / math.sqrt(2)) < 0.01


def test_peak_to_peak():
    vals = [1.0, -2.0, 3.0, 0.5]
    assert abs(_peak_to_peak(vals) - 5.0) < 1e-9


def test_peak_to_peak_empty():
    assert _peak_to_peak([]) == 0.0


def test_kurtosis_gaussian():
    # Gaussian should have kurtosis close to 3
    import random
    random.seed(0)
    vals = [random.gauss(0, 1) for _ in range(5000)]
    k = _kurtosis(vals)
    assert 2.5 < k < 3.5, f"Expected ~3, got {k}"


def test_kurtosis_impulse():
    # Signal with a single spike should have much higher kurtosis than 3
    vals = [0.01] * 999 + [10.0]
    assert _kurtosis(vals) > 50


def test_skewness_symmetric():
    vals = list(range(-50, 51))
    assert abs(_skewness(vals)) < 0.05


def test_crest_factor_sine():
    import math
    n   = 1000
    sig = [math.sin(2 * math.pi * 5 * i / n) for i in range(n)]
    cf  = _crest_factor(sig)
    # crest factor of a sine = sqrt(2) ~ 1.414
    assert abs(cf - math.sqrt(2)) < 0.05


def test_fft_dominant_freq():
    n  = 200
    fs = 100
    # 10 Hz pure sine
    sig = [math.sin(2 * math.pi * 10 * i / fs) for i in range(n)]
    freq, mag = _fft_dominant_freq(sig, fs=fs)
    assert abs(freq - 10.0) < 2.0, f"Expected ~10 Hz, got {freq}"
    assert mag > 0


def test_fft_empty():
    freq, mag = _fft_dominant_freq([], fs=100)
    assert freq == 0.0 and mag == 0.0


def test_extract_features_shape():
    window = {
        "vibration":   [0.1 + 0.01 * math.sin(i) for i in range(100)],
        "temperature": [62.0 + 0.1 * i for i in range(100)],
        "current":     [4.2 + 0.05 * math.cos(i) for i in range(100)],
    }
    feats = extract_features(window)
    assert len(feats) == 12
    assert all(isinstance(f, float) for f in feats)


def test_extract_features_normal_range():
    window = {
        "vibration":   [0.12] * 100,
        "temperature": [62.0] * 100,
        "current":     [4.2]  * 100,
    }
    feats = extract_features(window)
    vib_rms = feats[0]
    assert abs(vib_rms - 0.12) < 0.01


def test_extract_features_fault_kurtosis():
    # Bearing wear: spike-heavy signal should produce kurtosis >> 3
    base  = [0.1] * 95
    spikes = [3.5, -3.2, 4.1, -3.8, 2.9]
    window = {
        "vibration":   base + spikes,
        "temperature": [62.0] * 100,
        "current":     [4.2]  * 100,
    }
    feats    = extract_features(window)
    kurtosis = feats[2]
    assert kurtosis > 5, f"Expected elevated kurtosis, got {kurtosis}"
