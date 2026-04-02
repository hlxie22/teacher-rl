# src/teacher_eval.py
from __future__ import annotations
from .gpu_pin import pin_one_gpu_per_task_early

pin_one_gpu_per_task_early()

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from vllm import SamplingParams

from .utils import (
    aime_score,
    build_teacher_solve_messages,
    build_teacher_solve_messages_numeric,
    get_role_runtime_cfg,
    load_config,
    numeric_score,
    read_jsonl,
    seed_everything,
)
from .vllm_infer import VLLMChatRunner


def _refuse_torchrun_ddp() -> None:
    ws = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if ws > 1 or ("RANK" in os.environ) or ("LOCAL_RANK" in os.environ):
        raise SystemExit(
            "[teacher_eval] ERROR: torchrun/DDP execution is no longer supported for teacher_eval.\n"
            "Use SLURM task-parallel:\n"
            "  srun --ntasks=4 --gpus-per-task=1 python -m src.teacher_eval --parallelism slurm ...\n"
        )


def _slurm_rank_world() -> tuple[int, int, int]:
    if "SLURM_PROCID" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    return rank, world, local


def _is_int_like(x: Any) -> bool:
    if isinstance(x, int):
        return True
    if isinstance(x, str):
        s = x.strip()
        return s.lstrip("-").isdigit()
    return False


def _to_int(x: Any) -> int:
    if isinstance(x, int):
        return int(x)
    return int(str(x).strip())


def _chunks(xs: List[Any], bs: int) -> Iterable[List[Any]]:
    bs = max(1, int(bs))
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def _sample(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(items)), n)
    return [items[i] for i in idxs]


def _mean_from_sum_count(s: float, c: int) -> float:
    if c <= 0:
        return float("nan")
    return float(s) / float(c)


def main() -> None:
    _refuse_torchrun_ddp()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--round", type=int, default=0, help="Used only to seed sampling deterministically.")
    ap.add_argument(
        "--parallelism",
        choices=["slurm"],
        default="slurm",
        help="New-style only: SLURM task-parallel (1 GPU per task).",
    )
    ap.add_argument("--out", default=None, help="Path to write summary JSON (written by rank0).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("training", {}).get("seed", 0)) + 777 + int(args.round))

    adapter = args.adapter
    coord_state = Path(cfg["paths"]["coord_dir"]) / "state.json"
    if adapter is None and coord_state.exists():
        try:
            adapter = json.loads(coord_state.read_text(encoding="utf-8")).get("teacher_adapter")
        except Exception:
            adapter = None

    rank, world_size, local_rank = _slurm_rank_world()

    enable_thinking = bool((cfg.get("models", {}) or {}).get("enable_thinking", True))
    runtime = get_role_runtime_cfg(cfg, "teacher")

    teacher = VLLMChatRunner(
        model_id=cfg["models"]["teacher_model_id"],
        tp_size=1,
        dtype=runtime["dtype"],
        max_model_len=runtime["max_model_len"],
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_lora=True,
        max_loras=runtime["max_loras"],
        max_lora_rank=runtime["max_lora_rank"],
        attention_backend=runtime["attention_backend"],
        chat_template_kwargs={"enable_thinking": enable_thinking},
        max_num_seqs=runtime["max_num_seqs"],
        max_num_batched_tokens=runtime["max_num_batched_tokens"],
        language_model_only=runtime["language_model_only"],
        cpu_offload_gb=runtime["cpu_offload_gb"],
        enforce_eager=runtime["enforce_eager"],
        disable_log_stats=runtime["disable_log_stats"],
    )

    ecfg = (cfg.get("eval", {}) or {}).get("teacher_eval", {}) or {}
    iid_source = str(ecfg.get("iid_source", "replay_buffer"))
    n_iid = int(ecfg.get("n_iid", 128))
    n_ood_map = dict(ecfg.get("n_ood", {}) or {})

    # IID set
    if iid_source == "train_pool":
        iid_items = list(read_jsonl(cfg["paths"]["train_pool_path"]))
    else:
        iid_items = list(read_jsonl(cfg["paths"]["replay_buffer_path"]))
    iid_items = [x for x in iid_items if x.get("answer") is not None]
    iid = _sample(iid_items, n_iid, seed=1000 + args.round)

    # OOD sets
    ood_splits: List[Tuple[str, List[Dict[str, Any]]]] = []
    if int(n_ood_map.get("aime24", 0)) > 0:
        xs = [x for x in read_jsonl(cfg["paths"]["eval_aime24_path"]) if _is_int_like(x.get("answer"))]
        ood_splits.append(("aime24", _sample(xs, int(n_ood_map["aime24"]), seed=2000 + args.round)))
    if int(n_ood_map.get("aime25", 0)) > 0:
        xs = [x for x in read_jsonl(cfg["paths"]["eval_aime25_path"]) if _is_int_like(x.get("answer"))]
        ood_splits.append(("aime25", _sample(xs, int(n_ood_map["aime25"]), seed=3000 + args.round)))

    if int(n_ood_map.get("hmmt25", 0)) > 0:
        xs = list(read_jsonl(cfg["paths"]["eval_hmmt25_path"]))
        ood_splits.append(("hmmt25", _sample(xs, int(n_ood_map["hmmt25"]), seed=4000 + args.round)))
    if int(n_ood_map.get("hmmt26", 0)) > 0:
        xs = list(read_jsonl(cfg["paths"]["eval_hmmt26_path"]))
        ood_splits.append(("hmmt26", _sample(xs, int(n_ood_map["hmmt26"]), seed=5000 + args.round)))

    if int(n_ood_map.get("amc", 0)) > 0:
        xs = [x for x in read_jsonl(cfg["paths"]["eval_amc_path"]) if _is_int_like(x.get("answer"))]
        ood_splits.append(("amc", _sample(xs, int(n_ood_map["amc"]), seed=6000 + args.round)))

    solve_temp = float(ecfg.get("temperature_solve", ecfg.get("temperature", 0.2)))
    solve_top_p = float(ecfg.get("top_p_solve", ecfg.get("top_p", 0.95)))
    solve_max = int(ecfg.get("max_new_tokens_solve", 1024))
    sp_solve = SamplingParams(max_tokens=solve_max, temperature=solve_temp, top_p=solve_top_p)

    eval_bs = int(ecfg.get("batch_size", (cfg.get("inference", {}) or {}).get("teacher_batch_size", 16)))

    def my_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if world_size <= 1:
            return items
        return [ex for i, ex in enumerate(items) if (i % world_size) == rank]

    def run_bucket_sum_count(name: str, items: List[Dict[str, Any]]) -> Tuple[float, int]:
        if not items:
            return 0.0, 0

        mine = my_items(items)
        s = 0.0
        c = 0

        # Only explicit AIME/AMC buckets should use integer extraction + aime_score.
        # IID and HMMT buckets should use numeric_score, even when the gold happens
        # to look like an integer such as "2" or "15".
        use_aime_scoring = name in ("aime24", "aime25", "amc")

        for chunk in _chunks(mine, eval_bs):
            batch_msgs: List[List[Dict[str, str]]] = []
            golds: List[Any] = []

            for ex in chunk:
                q = ex["question"]
                if use_aime_scoring:
                    golds.append(_to_int(ex["answer"]))
                    batch_msgs.append(build_teacher_solve_messages(q, cfg))
                else:
                    golds.append(str(ex["answer"]))
                    batch_msgs.append(build_teacher_solve_messages_numeric(q, cfg))

            outs = teacher.generate_from_messages(
                batch_msgs,
                sp_solve,
                lora_path=adapter,
                lora_name="teacher_eval",
            )

            for out_text, g in zip(outs, golds):
                if use_aime_scoring:
                    s += float(aime_score(out_text, int(g)))
                else:
                    s += float(numeric_score(out_text, str(g)))
                c += 1

        return float(s), int(c)

    results_base: Dict[str, Any] = {
        "round": int(args.round),
        "adapter": adapter,
        "parallelism": "slurm",
        "tp_size": 1,
        "iid_source": iid_source,
        "n_iid": len(iid),
        "ood_counts": {k: len(v) for k, v in ood_splits},
        "acc": {},
    }

    bucket_defs: List[Tuple[str, List[Dict[str, Any]]]] = [("iid", iid)] + ood_splits
    sums_counts: Dict[str, Dict[str, Any]] = {}

    for bname, items2 in bucket_defs:
        s2, c2 = run_bucket_sum_count(bname, items2)
        sums_counts[bname] = {"sum": float(s2), "count": int(c2)}

    if args.out is None:
        if world_size == 1:
            results_base["acc"] = {k: _mean_from_sum_count(v["sum"], v["count"]) for k, v in sums_counts.items()}
            print("=== Teacher Eval (solve) ===")
            print(json.dumps(results_base, indent=2))
        return

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    run_id = os.environ.get("RTT_EVAL_RUN_ID") or f"round{args.round}"
    tmp_dir = outp.parent / f".teacher_eval_tmp.{run_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_path = tmp_dir / f"partial.rank{rank}.json"
    done_path = tmp_dir / f"done.rank{rank}.json"

    part_payload = {
        "rank": rank,
        "world_size": world_size,
        "round": int(args.round),
        "adapter": adapter,
        "sums_counts": sums_counts,
        "meta": {
            "n_iid": len(iid),
            "ood_counts": {k: len(v) for k, v in ood_splits},
        },
        "timestamp": time.time(),
    }
    part_path.write_text(json.dumps(part_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    done_path.write_text(json.dumps({"ok": True, "rank": rank, "world_size": world_size, "t": time.time()}), encoding="utf-8")

    if world_size > 1 and rank != 0:
        return

    expected_done = [tmp_dir / f"done.rank{r}.json" for r in range(world_size)]
    t0 = time.time()
    timeout_s = int(os.environ.get("RTT_EVAL_TIMEOUT_S", "3600"))

    while True:
        if all(p.exists() for p in expected_done):
            break
        if time.time() - t0 > timeout_s:
            missing = [str(p) for p in expected_done if not p.exists()]
            raise SystemExit(f"[teacher_eval] Timeout waiting for ranks: missing {missing[:3]} ...")
        time.sleep(1.0)

    agg: Dict[str, Dict[str, float]] = {}
    for r in range(world_size):
        pp = tmp_dir / f"partial.rank{r}.json"
        if not pp.exists():
            raise SystemExit(f"[teacher_eval] Missing partial: {pp}")
        d = json.loads(pp.read_text(encoding="utf-8"))
        sc = d.get("sums_counts", {}) or {}
        for bname, vals in sc.items():
            s3 = float(vals.get("sum", 0.0))
            c3 = int(vals.get("count", 0))
            if bname not in agg:
                agg[bname] = {"sum": 0.0, "count": 0.0}
            agg[bname]["sum"] += s3
            agg[bname]["count"] += float(c3)

    results_base["acc"] = {
        bname: _mean_from_sum_count(float(v["sum"]), int(v["count"])) for bname, v in agg.items()
    }

    print("=== Teacher Eval (solve) ===")
    print(json.dumps(results_base, indent=2))
    outp.write_text(json.dumps(results_base, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        for p in tmp_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        tmp_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
