import json
from datetime import date
from typing import Iterable, List


def serialize_probabilities(probabilities: Iterable[float]) -> str:
    return json.dumps([float(value) for value in probabilities])


def deserialize_probabilities(raw_value: str) -> List[float]:
    if not raw_value:
        return [0.0] * 5
    try:
        data = json.loads(raw_value)
        return [float(value) for value in data]
    except (TypeError, ValueError):
        return [0.0] * 5


def ensure_date(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def today_iso() -> str:
    return date.today().isoformat()
