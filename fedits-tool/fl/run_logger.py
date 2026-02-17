# run_logger.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RunPaths:
    run_dir: str
    events_path: str
    metrics_path: str


class RunLogger:
    """Append-only run logger: JSONL events + CSV round metrics."""

    def __init__(self, base_dir: str = "outputs", run_id: Optional[str] = None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        self.paths = RunPaths(
            run_dir=run_dir,
            events_path=os.path.join(run_dir, "events.jsonl"),
            metrics_path=os.path.join(run_dir, "round_metrics.csv"),
        )

        self._metrics_header_written = os.path.exists(self.paths.metrics_path)

    def log_event(self, event_type: str, server_round: int, payload: Dict[str, Any]) -> None:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "round": server_round,
            "payload": payload,
        }
        with open(self.paths.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_round_metrics(
        self,
        server_round: int,
        selected: int,
        received: int,
        kept: int,
        dropped: int,
        agg_scalar: Optional[float],
    ) -> None:
        row = {
            "round": server_round,
            "selected": selected,
            "received": received,
            "kept": kept,
            "dropped": dropped,
            "agg_scalar": "" if agg_scalar is None else f"{agg_scalar:.8f}",
        }
        write_header = not self._metrics_header_written
        with open(self.paths.metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                self._metrics_header_written = True
            w.writerow(row)

    def write_manifest(self, extra: Optional[Dict[str, Any]] = None) -> None:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": self.paths.run_dir,
        }
        if extra:
            manifest.update(extra)
        path = os.path.join(self.paths.run_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
