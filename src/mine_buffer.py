from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from .utils import ensure_dir, load_config, read_jsonl, write_jsonl


def _parse_interval(x: Any, name: str) -> Tuple[float, float]:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} must be a 2-item list/tuple like [lo, hi], got: {x!r}")
    lo = float(x[0])
    hi = float(x[1])
    if lo > hi:
        raise ValueError(f"{name} has lo > hi: {x!r}")
    return lo, hi


def _in_closed_interval(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["data_dir"])

    train_pool = {ex["id"]: ex for ex in read_jsonl(cfg["paths"]["train_pool_path"])}
    s_cache = {ex["id"]: ex for ex in read_jsonl(cfg["paths"]["student_cache_merged"])}
    t_cache = {ex["id"]: ex for ex in read_jsonl(cfg["paths"]["teacher_cache_merged"])}

    teacher_lo, teacher_hi = _parse_interval(cfg["mining"]["teacher_interval"], "mining.teacher_interval")
    student_lo, student_hi = _parse_interval(cfg["mining"]["student_interval"], "mining.student_interval")

    buf: List[Dict[str, Any]] = []
    missing = 0
    for _id, ex in train_pool.items():
        s = s_cache.get(_id)
        t = t_cache.get(_id)
        if s is None or t is None:
            missing += 1
            continue

        p_s = float(s.get("p_hat", 0.0))
        p_t = float(t.get("p_hat", 0.0))
        fails = list(s.get("failures", []))

        teacher_ok = _in_closed_interval(p_t, teacher_lo, teacher_hi)
        student_ok = _in_closed_interval(p_s, student_lo, student_hi)

        if teacher_ok and student_ok and len(fails) > 0:
            buf.append(
                {
                    "id": _id,
                    "question": ex["question"],
                    "answer": str(ex["answer"]),
                    "p_s_hat": p_s,
                    "p_t_hat": p_t,
                    "failures": fails,
                }
            )

    write_jsonl(cfg["paths"]["replay_buffer_path"], buf)
    print(
        f"Wrote replay buffer: {cfg['paths']['replay_buffer_path']} "
        f"(n={len(buf)}), missing cache={missing}, "
        f"teacher_interval=[{teacher_lo}, {teacher_hi}], "
        f"student_interval=[{student_lo}, {student_hi}]"
    )


if __name__ == "__main__":
    main()