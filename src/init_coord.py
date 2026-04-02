from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .coord import Coordinator
from .utils import ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    coord_dir = cfg["paths"]["coord_dir"]
    adapter_root = cfg["persistent"].get(
        "adapter_root",
        str(Path(cfg["paths"]["outputs_dir"]) / "teacher_lora_persistent"),
    )
    live_adapter_root = cfg["persistent"].get(
        "live_adapter_root",
        str(Path(cfg["paths"]["outputs_dir"]) / "teacher_lora_live"),
    )

    ensure_dir(coord_dir)
    ensure_dir(adapter_root)
    ensure_dir(live_adapter_root)

    init_adapter: Optional[str] = cfg.get("models", {}).get("teacher_init_adapter", None)
    max_rounds = int(args.rounds if args.rounds is not None else cfg.get("persistent", {}).get("rounds", 1))

    state: Dict[str, Any] = {
        "phase": "train",
        "round": 0,
        "max_rounds": max_rounds,
        "teacher_adapter": init_adapter,      # persistent round adapter (may be None)
        "teacher_ref_adapter": init_adapter,  # may be None
        "policy_root": str(adapter_root),
        "live_policy_root": str(live_adapter_root),
        "teacher_live_adapter_path": None,
        "teacher_live_adapter_name": None,
        "teacher_live_step": None,
        # eval_watch will add last_teacher_eval_round later
    }

    coord = Coordinator(coord_dir)
    coord.init(state, clean=bool(args.clean) or bool(cfg.get("persistent", {}).get("clean_coord", False)))
    print(f"Initialized GRPO coord state at {coord.state_path}")


if __name__ == "__main__":
    main()
