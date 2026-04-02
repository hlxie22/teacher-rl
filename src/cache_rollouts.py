# src/cache_rollouts.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from vllm import SamplingParams

from .gpu_pin import pin_one_gpu_per_task_early
from .utils import (
    RTT_STOP_SEQUENCE,
    atomic_write_json,
    build_student_attempt_messages,
    build_teacher_solve_messages_numeric,
    ensure_dir,
    get_role_runtime_cfg,
    load_config,
    numeric_score,
    read_jsonl,
    seed_everything,
)
from .vllm_infer import VLLMChatRunner


def _is_truncated(finish_reason: Any, completion_tokens: int, max_new_tokens: int) -> bool:
    fr = finish_reason or ""
    if isinstance(fr, str):
        frl = fr.lower()
        if frl in ("length", "max_tokens", "max_length"):
            return True
    return int(completion_tokens or 0) >= int(max_new_tokens)


def _get_job_shard(cfg: Dict[str, Any], args, who: str) -> tuple[int, int]:
    if args.shard_id is not None and args.num_shards is not None:
        return int(args.shard_id), int(args.num_shards)

    sid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    if who == "student":
        n = int(cfg.get("jobs", {}).get("cache_student_jobs", 1))
    else:
        n = int(cfg.get("jobs", {}).get("cache_teacher_jobs", 1))

    n = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", str(n)))
    return sid, n


def _safe_load_jsonl_rows(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    _id = r.get("id")
                    if _id is None:
                        continue
                    sid = str(_id)
                    if sid not in rows:
                        rows[sid] = r
        except Exception:
            continue
    return rows


def _slurm_rank_world() -> tuple[int, int, int]:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    return rank, world, local


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--who", choices=["student", "teacher"], required=True)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="fsync progress every N newly-written rows per process (smaller = safer, larger = faster).",
    )
    ap.add_argument(
        "--parallelism",
        choices=["slurm", "tp"],
        default=None,
        help="Override cfg.inference.cache_parallelism. 'slurm' = one task per GPU. 'tp' = one process uses multiple GPUs.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("training", {}).get("seed", 0)) + 17)

    who = args.who
    shard_id, num_shards = _get_job_shard(cfg, args, who)

    inf = cfg.get("inference", {})
    par = str(args.parallelism or inf.get("cache_parallelism", "slurm") or "slurm").lower()
    if par not in ("slurm", "tp"):
        raise SystemExit(f"Unsupported cache_parallelism={par}. Use 'slurm' (recommended) or 'tp'.")

    if par == "slurm":
        pin_one_gpu_per_task_early()

    if par == "tp":
        rank, world_size, local_rank = 0, 1, 0
        tp_size = max(1, int(inf.get("cache_tp_size", cfg.get("resources", {}).get("gpus_per_job", 1)) or 1))
        if not torch.cuda.is_available():
            raise RuntimeError("cache_parallelism=tp requires CUDA (vLLM multi-GPU).")
        visible = torch.cuda.device_count()
        if visible > 0 and tp_size > visible:
            raise RuntimeError(
                f"cache_tp_size={tp_size} but only {visible} CUDA device(s) visible. "
                "Check SLURM allocation / CUDA_VISIBLE_DEVICES."
            )
    else:
        rank, world_size, local_rank = _slurm_rank_world()
        tp_size = 1

    if rank == 0:
        print(
            f"[cache:{who}] parallelism={par} rank/world={rank}/{world_size} local_rank={local_rank} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
            flush=True,
        )

    train_pool = list(read_jsonl(cfg["paths"]["train_pool_path"]))

    job_idxs = [i for i in range(len(train_pool)) if (i % num_shards) == shard_id]
    rank_idxs = [i for j, i in enumerate(job_idxs) if (j % max(1, world_size)) == rank]

    cache_dir = cfg["paths"]["student_cache_dir"] if who == "student" else cfg["paths"]["teacher_cache_dir"]
    ensure_dir(cache_dir)

    out_path = Path(cache_dir) / f"{who}_cache.shard{shard_id}-of-{num_shards}.rank{rank}.jsonl"
    done_path = out_path.with_suffix(out_path.suffix + ".done.json")

    expected_ids = [str(train_pool[i]["id"]) for i in rank_idxs]
    expected_count = len(expected_ids)

    old_style = Path(cache_dir) / f"{who}_cache.shard{shard_id}-of-{num_shards}.jsonl"
    if old_style.exists():
        raise SystemExit(
            f"[cache:{who}] Found old-style shard file {old_style}. "
            "Delete old cache outputs before running Design B."
        )

    if done_path.exists() and out_path.exists():
        if rank == 0:
            print(f"[cache:{who}] rank file already done -> {out_path}", flush=True)
        return

    if out_path.exists() and not done_path.exists():
        parsed = _safe_load_jsonl_rows([out_path])
        if len(parsed) == expected_count:
            atomic_write_json(
                done_path,
                {
                    "who": who,
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "rank": rank,
                    "world_size": world_size,
                    "expected": expected_count,
                    "found": len(parsed),
                    "timestamp": time.time(),
                    "note": "marker reconstructed from existing rank shard",
                },
            )
            if rank == 0:
                print(f"[cache:{who}] reconstructed done marker -> {done_path}", flush=True)
            return
        else:
            if rank == 0:
                print(
                    f"[cache:{who}] resuming partial shard: have {len(parsed)}/{expected_count} rows in {out_path}",
                    flush=True,
                )

    sp_common = dict(
        temperature=float(cfg["caching"]["temperature"]),
        top_p=float(cfg["caching"]["top_p"]),
    )

    if who == "student":
        model_id = cfg["models"]["student_model_id"]
        K = int(cfg["caching"]["K_student"])
        max_new = int(cfg["caching"]["max_new_tokens_student"])
        batch_size = int((cfg.get("inference", {}) or {}).get("student_batch_size", 16))
    else:
        model_id = cfg["models"]["teacher_model_id"]
        K = int(cfg["caching"]["K_teacher"])
        max_new = int(cfg["caching"]["max_new_tokens_teacher_solve"])
        batch_size = int((cfg.get("inference", {}) or {}).get("teacher_batch_size", 16))

    sp = SamplingParams(max_tokens=max_new, n=K, stop=["<RTT_END>"], **sp_common)

    enable_thinking = bool((cfg.get("models", {}) or {}).get("enable_thinking", True))
    runtime = get_role_runtime_cfg(cfg, who)

    runner = VLLMChatRunner(
        model_id=model_id,
        tp_size=tp_size,
        dtype=runtime["dtype"],
        max_model_len=runtime["max_model_len"],
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_lora=False,
        attention_backend=runtime["attention_backend"],
        chat_template_kwargs={"enable_thinking": enable_thinking},
        max_num_seqs=runtime["max_num_seqs"],
        max_num_batched_tokens=runtime["max_num_batched_tokens"],
        language_model_only=runtime["language_model_only"],
        cpu_offload_gb=runtime["cpu_offload_gb"],
        enforce_eager=runtime["enforce_eager"],
        disable_log_stats=runtime["disable_log_stats"],
    )

    it = rank_idxs
    if rank == 0:
        it = tqdm(rank_idxs, desc=f"{who} cache shard {shard_id}/{num_shards} ({par}, tp={tp_size})")

    done_ids = set()
    if out_path.exists():
        existing = _safe_load_jsonl_rows([out_path])
        done_ids = set(existing.keys())

    fsync_every = max(1, int(args.checkpoint_every))
    wrote_since_fsync = 0

    t_start = time.perf_counter()
    local_prompts = 0
    local_completions = 0
    local_trunc = 0
    local_prompt_tokens = 0
    local_completion_tokens = 0

    f = open(out_path, "a", encoding="utf-8", buffering=1)
    try:
        pending: List[Dict[str, Any]] = []

        def flush_batch(batch: List[Dict[str, Any]]) -> None:
            nonlocal wrote_since_fsync
            nonlocal local_prompts, local_completions, local_trunc, local_prompt_tokens, local_completion_tokens
            if not batch:
                return

            batch_msgs: List[List[Dict[str, str]]] = []
            golds: List[str] = []
            ids: List[str] = []

            for ex in batch:
                ex_id = str(ex["id"])
                q = ex["question"]
                gold = str(ex["answer"])
                ids.append(ex_id)
                golds.append(gold)

                if who == "student":
                    batch_msgs.append(
                        build_student_attempt_messages(
                            q,
                            cfg,
                            hint=None,
                            include_stop_marker=True,
                        )
                    )
                else:
                    batch_msgs.append(build_teacher_solve_messages_numeric(q, cfg))

            metas = runner.generate_from_messages(batch_msgs, sp, return_meta=True)

            for ex_id, gold, meta in zip(ids, golds, metas):
                completions = list(meta.get("completions", []))
                local_prompts += 1
                local_prompt_tokens += int(meta.get("prompt_tokens", 0) or 0)
                local_completions += len(completions)

                texts = [c.get("text", "") for c in completions]

                for c in completions:
                    ctoks = int(c.get("completion_tokens", 0) or 0)
                    local_completion_tokens += ctoks
                    if _is_truncated(c.get("finish_reason", None), ctoks, max_new):
                        local_trunc += 1

                if who == "student":
                    succ = 0
                    failures: List[str] = []
                    for out in texts:
                        s = numeric_score(out, gold)
                        succ += s
                        if s == 0:
                            failures.append(out)
                    p_hat = succ / max(1, K)
                    row = {"id": ex_id, "p_hat": p_hat, "failures": failures, "K": K}
                else:
                    succ = 0
                    for out in texts:
                        succ += numeric_score(out, gold)
                    p_hat = succ / max(1, K)
                    row = {"id": ex_id, "p_hat": p_hat, "K": K}

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done_ids.add(ex_id)
                wrote_since_fsync += 1

                if wrote_since_fsync >= fsync_every:
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                    wrote_since_fsync = 0

        for i in it:
            ex = train_pool[i]
            ex_id = str(ex["id"])
            if ex_id in done_ids:
                continue
            pending.append(ex)
            if len(pending) >= batch_size:
                flush_batch(pending)
                pending = []

        flush_batch(pending)

    finally:
        try:
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        finally:
            f.close()

    parsed = _safe_load_jsonl_rows([out_path])
    found = len(parsed)
    if found != expected_count:
        print(
            f"[cache:{who}] ERROR: rank output incomplete: shard {shard_id}/{num_shards} "
            f"rank {rank}/{world_size} parsed={found} expected={expected_count} file={out_path}",
            flush=True,
        )
        raise SystemExit(1)

    elapsed = float(time.perf_counter() - t_start)
    trunc_rate = (local_trunc / local_completions) if local_completions > 0 else 0.0
    toks_per_s = ((local_prompt_tokens + local_completion_tokens) / elapsed) if elapsed > 0 else 0.0

    atomic_write_json(
        done_path,
        {
            "who": who,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "rank": rank,
            "world_size": world_size,
            "expected": expected_count,
            "found": found,
            "timestamp": time.time(),
            "cache_stats": {
                "K": int(K),
                "max_new_tokens": int(max_new),
                "stop_sequence": RTT_STOP_SEQUENCE,
                "prompts_total": int(local_prompts),
                "completions_total": int(local_completions),
                "truncated_total": int(local_trunc),
                "truncated_rate": float(trunc_rate),
                "prompt_tokens_total": float(local_prompt_tokens),
                "completion_tokens_total": float(local_completion_tokens),
                "elapsed_s": float(elapsed),
                "tokens_per_s": float(toks_per_s),
            },
        },
    )

    if rank == 0:
        print(f"[cache:{who}] wrote rank outputs -> {out_path} (+ {done_path})", flush=True)


if __name__ == "__main__":
    main()
