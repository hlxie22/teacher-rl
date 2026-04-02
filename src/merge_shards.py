from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import (
    ensure_dir,
    load_config,
    read_jsonl,
    quick_sig,
    is_done,
    write_done,
    sha256_file,
)


def _collect_rank_outputs_for_shard(
    in_dir: Path, who: str, sid: int, n_shards: int
) -> Tuple[int, List[Tuple[int, Path, Path, Dict[str, Any]]]]:
    """
    Returns:
      (world_size, [(rank, jsonl_path, done_path, done_json), ...]) sorted by rank

    Enforces:
      - no old-style per-shard file exists
      - every rank file has a done marker
      - all done markers agree on world_size
      - ranks cover 0..world_size-1
      - each rank done marker has found==expected
    """
    old_style = in_dir / f"{who}_cache.shard{sid}-of-{n_shards}.jsonl"
    if old_style.exists():
        raise SystemExit(
            f"[merge_shards] Found old-style shard file {old_style}. "
            "This pipeline expects Design B rank outputs only. Delete old cache outputs."
        )

    rank_files = sorted(in_dir.glob(f"{who}_cache.shard{sid}-of-{n_shards}.rank*.jsonl"))
    if not rank_files:
        raise SystemExit(
            f"[merge_shards] Missing rank files for shard {sid}/{n_shards}: "
            f"{who}_cache.shard{sid}-of-{n_shards}.rank*.jsonl"
        )

    entries: List[Tuple[int, Path, Path, Dict[str, Any]]] = []
    world_size = None

    for rf in rank_files:
        # parse rank from filename "...rank{r}.jsonl"
        name = rf.name
        try:
            r_str = name.split(".rank", 1)[1].split(".jsonl", 1)[0]
            r = int(r_str)
        except Exception:
            raise SystemExit(f"[merge_shards] Could not parse rank from filename: {rf}")

        donep = rf.with_suffix(rf.suffix + ".done.json")
        if not donep.exists():
            raise SystemExit(f"[merge_shards] Missing done marker: {donep}")

        try:
            d = json.loads(donep.read_text(encoding="utf-8"))
        except Exception:
            raise SystemExit(f"[merge_shards] Could not read done marker JSON: {donep}")

        ws = int(d.get("world_size", -1))
        if ws <= 0:
            raise SystemExit(f"[merge_shards] Bad world_size in {donep}: {ws}")

        if world_size is None:
            world_size = ws
        elif world_size != ws:
            raise SystemExit(
                f"[merge_shards] Inconsistent world_size for shard {sid}: saw {world_size} and {ws}"
            )

        exp = int(d.get("expected", -1))
        found = int(d.get("found", -1))
        if exp < 0 or found < 0 or found != exp:
            raise SystemExit(
                f"[merge_shards] Rank output incomplete per done marker: {donep} expected={exp} found={found}"
            )

        entries.append((r, rf, donep, d))

    assert world_size is not None

    # ensure full rank coverage
    ranks = sorted({r for (r, _, _, _) in entries})
    expected_ranks = list(range(world_size))
    if ranks != expected_ranks:
        raise SystemExit(
            f"[merge_shards] Missing ranks for shard {sid}: have {ranks}, expected {expected_ranks}. "
            "Likely a failed SLURM task."
        )

    entries.sort(key=lambda x: x[0])
    return world_size, entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--what", choices=["student_cache", "teacher_cache"], required=True)
    ap.add_argument("--force", action="store_true", help="Ignore done marker and rebuild merge output.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.what == "student_cache":
        in_dir = Path(cfg["paths"]["student_cache_dir"])
        out = Path(cfg["paths"]["student_cache_merged"])
        who = "student"
        n_shards = int((cfg.get("jobs", {}) or {}).get("cache_student_jobs", 1))
    else:
        in_dir = Path(cfg["paths"]["teacher_cache_dir"])
        out = Path(cfg["paths"]["teacher_cache_merged"])
        who = "teacher"
        n_shards = int((cfg.get("jobs", {}) or {}).get("cache_teacher_jobs", 1))

    ensure_dir(out.parent)
    out_done = out.with_suffix(out.suffix + ".done.json")

    # Gather done signatures + expected totals
    shard_done_sigs: List[Dict[str, Any]] = []
    expected_total = 0

    shard_rank_layout: List[Tuple[int, List[Tuple[int, Path, Path, Dict[str, Any]]]]] = []
    for sid in range(n_shards):
        _, entries = _collect_rank_outputs_for_shard(in_dir, who=who, sid=sid, n_shards=n_shards)
        shard_rank_layout.append((sid, entries))

        for (r, rf, donep, d) in entries:
            expected_total += int(d.get("expected", 0))
            shard_done_sigs.append(quick_sig(donep))

    sig = {"config_sha256": sha256_file(args.config), "what": args.what, "rank_done": shard_done_sigs}

    if (not args.force) and is_done(out_done, sig, [out]):
        # If done marker matches, trust it.
        # (You can add expensive line-count validation here if you want.)
        print(f"[merge_shards] already done -> {out}")
        return

    # Stream-merge rank files in deterministic order: shard 0.., rank 0..
    tmp = out.with_suffix(out.suffix + ".tmp")
    n_rows = 0
    with open(tmp, "w", encoding="utf-8") as wf:
        for sid, entries in shard_rank_layout:
            for r, rf, donep, d in entries:
                for row in read_jsonl(rf):
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1

    tmp.replace(out)
    write_done(out_done, sig, extra={"rows": n_rows, "expected_total": expected_total})

    # optional sanity check: merged row count equals expected_total (fast)
    if expected_total > 0 and n_rows != expected_total:
        raise SystemExit(
            f"[merge_shards] Row-count mismatch after merge: wrote {n_rows}, expected_total {expected_total}. "
            "This suggests duplicate/missing ids or inconsistent rank partitioning."
        )

    print(f"[merge_shards] Merged {n_rows} rows -> {out} (+ {out_done})")


if __name__ == "__main__":
    main()