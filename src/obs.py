from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def make_len_row(
    *,
    stage: str,
    max_model_len: int,
    prompt_tokens: int,
    completion_tokens_actual: int | None,
    completion_tokens_cap: int,
    finish_reason: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    actual = int(completion_tokens_actual or 0)
    row = {
        "ts": time.time(),
        "stage": stage,
        "max_model_len": int(max_model_len),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens_actual": actual,
        "completion_tokens_cap": int(completion_tokens_cap),
        "total_tokens_actual": int(prompt_tokens) + actual,
        "total_tokens_cap": int(prompt_tokens) + int(completion_tokens_cap),
        "headroom_actual": int(max_model_len) - (int(prompt_tokens) + actual),
        "headroom_cap": int(max_model_len) - (int(prompt_tokens) + int(completion_tokens_cap)),
        "finish_reason": finish_reason,
        "truncated_flag": (
            (str(finish_reason).lower() in {"length", "max_tokens", "max_length"})
            if finish_reason is not None else
            (actual >= int(completion_tokens_cap))
        ),
        "overflow_risk_flag": (int(prompt_tokens) + int(completion_tokens_cap) > int(max_model_len)),
    }
    if extra:
        row.update(extra)
    return row