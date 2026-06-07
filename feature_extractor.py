import json
import math
import boto3
import os
from collections import defaultdict

ENDPOINT       = os.environ.get("SAGEMAKER_ENDPOINT", "iiot-fault-classifier-v2")
FAULT_THRESHOLD = float(os.environ.get("FAULT_THRESHOLD", "0.72"))
SNS_ARN        = os.environ.get("SNS_TOPIC_ARN", "")

# Lazy-init AWS clients so the module can be imported in unit tests
# without live credentials or a configured region.
_sm_runtime = None
_sns_client = None


def _get_sm():
    global _sm_runtime
    if _sm_runtime is None:
        _sm_runtime = boto3.client(
            "sagemaker-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
    return _sm_runtime


def _get_sns():
    global _sns_client
    if _sns_client is None:
        _sns_client = boto3.client(
            "sns",
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
    return _sns_client


# Sliding window: node_id -> {sensor: [readings]}
_windows = defaultdict(lambda: {"vibration": [], "temperature": [], "current": []})
WINDOW_SIZE = 100  # 1 second at 100 Hz


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _variance(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def _rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0


def _peak_to_peak(vals):
    return max(vals) - min(vals) if vals else 0.0


def _kurtosis(vals):
    if len(vals) < 4:
        return 0.0
    m  = _mean(vals)
    s4 = sum((v - m) ** 4 for v in vals) / len(vals)
    s2 = _variance(vals)
    return s4 / (s2 ** 2) if s2 > 1e-9 else 0.0


def _skewness(vals):
    if len(vals) < 3:
        return 0.0
    m  = _mean(vals)
    s3 = sum((v - m) ** 3 for v in vals) / len(vals)
    s2 = _variance(vals)
    return s3 / (s2 ** 1.5) if s2 > 1e-9 else 0.0


def _crest_factor(vals):
    rms = _rms(vals)
    return max(abs(v) for v in vals) / rms if rms > 1e-9 else 0.0


def _fft_dominant_freq(vals, fs=100):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = _mean(vals)
    x    = [v - mean for v in vals]
    half = n // 2
    magnitudes = []
    for k in range(1, half):
        re = sum(x[i] * math.cos(2 * math.pi * k * i / n) for i in range(n))
        im = sum(x[i] * math.sin(2 * math.pi * k * i / n) for i in range(n))
        magnitudes.append((k * fs / n, math.sqrt(re * re + im * im)))
    if not magnitudes:
        return 0.0, 0.0
    dom = max(magnitudes, key=lambda t: t[1])
    return dom[0], dom[1]


def extract_features(window):
    vib  = window["vibration"]
    temp = window["temperature"]
    curr = window["current"]

    dom_freq, dom_mag = _fft_dominant_freq(vib)

    return [
        _rms(vib),
        _peak_to_peak(vib),
        _kurtosis(vib),
        _skewness(vib),
        _crest_factor(vib),
        dom_freq,
        dom_mag,
        _mean(temp),
        max(temp) - _mean(temp),   # temp rise over window
        _rms(curr),
        _peak_to_peak(curr),
        _kurtosis(curr),
    ]


def lambda_handler(event, context):
    records = event.get("Records", [event])

    for record in records:
        body = record.get("body", record)
        if isinstance(body, str):
            body = json.loads(body)

        node_id = body.get("node_id", "unknown")
        sensor  = body.get("sensor")
        value   = body.get("value")
        ts_ms   = body.get("ts_ms", 0)

        if sensor not in ("vibration", "temperature", "current") or value is None:
            continue

        win = _windows[node_id]
        win[sensor].append(float(value))

        if len(win[sensor]) > WINDOW_SIZE:
            win[sensor].pop(0)

        if not all(len(win[s]) >= WINDOW_SIZE for s in ("vibration", "temperature", "current")):
            continue

        features = extract_features(win)
        csv_row  = ",".join(f"{f:.6f}" for f in features)

        response = _get_sm().invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="text/csv",
            Body=csv_row
        )

        result      = json.loads(response["Body"].read())
        fault_class = result.get("predicted_label", "UNKNOWN")
        confidence  = result.get("probabilities", {}).get(fault_class, 0.0)

        print(json.dumps({
            "node_id":     node_id,
            "ts_ms":       ts_ms,
            "fault_class": fault_class,
            "confidence":  confidence,
        }))

        if fault_class != "NORMAL" and confidence >= FAULT_THRESHOLD and SNS_ARN:
            _get_sns().publish(
                TopicArn=SNS_ARN,
                Subject=f"[FAULT] {node_id} {fault_class}",
                Message=json.dumps({
                    "node_id":    node_id,
                    "fault":      fault_class,
                    "confidence": round(confidence, 4),
                    "ts_ms":      ts_ms
                })
            )

        for s in ("vibration", "temperature", "current"):
            win[s].clear()

    return {"statusCode": 200}
