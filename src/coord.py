from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import fcntl

from .utils import atomic_write_json, ensure_dir


class Coordinator:
    """
    GRPO-only coordinator:
      - a single state.json guarded by a file lock
      - used by learner (to publish adapter path / round) and eval watcher (to read it)
    """

    def __init__(self, coord_dir: str | Path):
        self.coord_dir = Path(coord_dir)
        self.lock_path = self.coord_dir / "state.lock"
        self.state_path = self.coord_dir / "state.json"

    def init(self, state: Dict[str, Any], clean: bool = False) -> None:
        ensure_dir(self.coord_dir)

        if clean and self.coord_dir.exists():
            # Only remove coordination artifacts (state + lock). Keep other directories untouched.
            try:
                if self.state_path.exists():
                    self.state_path.unlink()
            except Exception:
                pass
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except Exception:
                pass

        if not self.state_path.exists():
            atomic_write_json(self.state_path, state)

    def _lock(self):
        ensure_dir(self.coord_dir)
        f = open(self.lock_path, "a+")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(f"Missing state: {self.state_path}")
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_state(self, **kwargs) -> Dict[str, Any]:
        lockf = self._lock()
        try:
            st = self.read_state()
            st.update(kwargs)
            atomic_write_json(self.state_path, st)
            return st
        finally:
            try:
                fcntl.flock(lockf, fcntl.LOCK_UN)
            finally:
                lockf.close()

    def set_phase(self, phase: str) -> Dict[str, Any]:
        return self.update_state(phase=phase)