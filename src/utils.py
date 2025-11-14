# src/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

for d in (RESULTS_DIR, ARTIFACTS_DIR, METRICS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def dump_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)

def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)
