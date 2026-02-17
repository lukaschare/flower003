# progress_ui.py
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


@dataclass
class RoundState:
    server_round: int = 0
    selected: List[str] = field(default_factory=list)
    received: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    agg_scalar: Optional[float] = None


class ProgressBoard:
    """A live terminal dashboard for FL rounds."""
    def __init__(self, num_rounds: int, title: str = "Flower + Orchestrator (T0)"):
        self.console = Console()
        self.num_rounds = num_rounds
        self.title = title
        self.state = RoundState()
        self._lock = threading.Lock()
        self._live: Optional[Live] = None

        self._progress = Progress(
            TextColumn("Progress"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} rounds"),
            TimeElapsedColumn(),
            expand=True,
        )
        self._task_id = self._progress.add_task("rounds", total=num_rounds)

    def __enter__(self):
        self._live = Live(self.render(), console=self.console, refresh_per_second=10, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    def _refresh(self):
        if self._live:
            self._live.update(self.render())

    def render(self):
        with self._lock:
            s = self.state

            t = Table(title=f"Round {s.server_round}/{self.num_rounds}", expand=True)
            t.add_column("Field", style="bold")
            t.add_column("Value")

            t.add_row("SELECT", ", ".join(s.selected) if s.selected else "-")
            t.add_row("RECV", ", ".join(s.received) if s.received else "-")
            t.add_row("KEEP", ", ".join(s.kept) if s.kept else "-")
            t.add_row("DROP", ", ".join(s.dropped) if s.dropped else "-")
            t.add_row("AGG_SCALAR", "-" if s.agg_scalar is None else f"{s.agg_scalar:.6f}")

            group = Group(
                Panel(self._progress, title=self.title),
                Panel(t, title="Live Status"),
            )
            return group

    # --- hooks called by Strategy ---
    def on_round_start(self, server_round: int):
        with self._lock:
            self.state = RoundState(server_round=server_round)
        self._refresh()

    def on_select(self, selected: List[str]):
        with self._lock:
            self.state.selected = list(selected)
        self._refresh()

    def on_recv(self, received: List[str]):
        with self._lock:
            self.state.received = list(received)
        self._refresh()

    def on_verdict(self, kept: List[str], dropped: List[str]):
        with self._lock:
            self.state.kept = list(kept)
            self.state.dropped = list(dropped)
        self._refresh()

    def on_agg(self, agg_scalar: Optional[float]):
        with self._lock:
            self.state.agg_scalar = agg_scalar
        self._refresh()

    def on_round_end(self):
        self._progress.advance(self._task_id, 1)
        self._refresh()

    def log(self, msg: str):
        self.console.log(msg)
