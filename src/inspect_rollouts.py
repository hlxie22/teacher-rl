from __future__ import annotations

from .gpu_pin import pin_one_gpu_per_task_early

pin_one_gpu_per_task_early()

import argparse
import gc
import json
import os
import random
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from vllm import SamplingParams

from .utils import (
    RTT_STOP_MARKER,
    RTT_STOP_SEQUENCE,
    aime_score,
    extract_final_int,
    load_config,
    read_jsonl,
    seed_everything,
    build_teacher_solve_messages,
)
from .vllm_infer import VLLMChatRunner


ENV_PREFIX = "RTT_INSPECT_"

# Requested behavior: always add this instruction to prompts.
STEP_BY_STEP_INSTRUCTION = "Think step by step."


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(ENV_PREFIX + name, default)


def _parse_bool(x: Optional[str], default: Optional[bool] = None) -> Optional[bool]:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _parse_int(x: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if x is None or str(x).strip() == "":
        return default
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _parse_float(x: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if x is None or str(x).strip() == "":
        return default
    try:
        return float(str(x).strip())
    except Exception:
        return default


def _parse_csv(x: Optional[str]) -> List[str]:
    if not x:
        return []
    return [s.strip() for s in str(x).split(",") if s.strip()]


# -------------------------
# SLURM task-parallel rank info (no torch.distributed)
# -------------------------
def _rank_world_local() -> Tuple[int, int, int]:
    """
    Prefer SLURM ranks if present; otherwise fall back to torchrun-style envs.
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        world = int(os.environ.get("SLURM_NTASKS", "1"))
        local = int(os.environ.get("SLURM_LOCALID", "0"))
        return max(0, rank), max(1, world), max(0, local)

    # fallback: torchrun-style envs (still no torch.distributed usage)
    rank = int(os.environ.get("RANK", "0") or "0")
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    local = int(os.environ.get("LOCAL_RANK", "0") or "0")
    return max(0, rank), max(1, world), max(0, local)


def _set_sane_cuda_device(local_rank: int) -> None:
    if not torch.cuda.is_available():
        return
    n = torch.cuda.device_count()
    dev = 0 if n <= 1 else int(local_rank) % n
    try:
        torch.cuda.set_device(dev)
    except Exception:
        pass


def _rank0(rank: int) -> bool:
    return int(rank) == 0


def _shard_list(xs: List[Any], rank: int, world: int) -> List[Any]:
    n = len(xs)
    if world <= 1 or n == 0:
        return xs
    per = (n + world - 1) // world
    a = rank * per
    b = min(a + per, n)
    return xs[a:b]


def _student_messages(question: str) -> List[Dict[str, str]]:
    user = (
        question
        + "\n\n"
        + STEP_BY_STEP_INSTRUCTION
        + "\n\nEnd with 'Final answer: <integer>'."
        + f"\nAfter the final answer line, output a NEW LINE containing exactly: {RTT_STOP_MARKER}\n"
    )
    return [{"role": "user", "content": user}]


def _append_stop_marker_instruction(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Ensure teacher prompts end with RTT_STOP_MARKER (STOP_MODE=marker),
    and include requested step-by-step instruction.
    """
    extra = (
        STEP_BY_STEP_INSTRUCTION
        + "\n"
        + "End with 'Final answer: <integer>'.\n"
        f"After the final answer line, output a NEW LINE containing exactly: {RTT_STOP_MARKER}\n"
    )
    out = list(msgs or [])
    out.append({"role": "user", "content": extra})
    return out


def _has_banned_substring(q: str, banned: List[str]) -> bool:
    ql = (q or "").lower()
    return any(s.lower() in ql for s in banned)


_INT_ANY_RE = re.compile(r"(-?\d+)")


def _parse_int_answer(a: Any) -> Optional[int]:
    if a is None:
        return None
    if isinstance(a, bool):
        return None
    if isinstance(a, int):
        return int(a)
    s = str(a).strip()
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        pass
    ints = _INT_ANY_RE.findall(s)
    if not ints:
        return None
    try:
        return int(ints[-1])
    except Exception:
        return None


def _get_first(ex: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in ex:
            v = ex.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
    return None


def _build_pool_from_hf(
    hf_id: str,
    *,
    split: str = "train",
    name: Optional[str] = None,
    banned: Optional[List[str]] = None,
    year_max: Optional[int] = None,
    prefer_id_fields: Sequence[str] = ("id", "ID", "uid"),
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    banned = list(banned or [])
    ds = load_dataset(hf_id, name=name, split=split)

    items: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = _get_first(
            ex,
            keys=(
                "question",
                "Question",
                "problem",
                "Problem",
                "prompt",
                "input",
                "text",
                "statement",
            ),
        )
        if q is None:
            continue
        q = str(q).strip()
        if not q:
            continue
        if banned and _has_banned_substring(q, banned):
            continue

        if year_max is not None:
            y = _get_first(ex, keys=("Year", "year", "YEAR"))
            try:
                yy = int(y) if y is not None else 0
            except Exception:
                yy = 0
            if yy and yy > int(year_max):
                continue

        a = _get_first(
            ex,
            keys=(
                "answer",
                "Answer",
                "final_answer",
                "ground_truth_answer",
                "gt_answer",
                "target",
                "label",
            ),
        )
        ai = _parse_int_answer(a)
        if ai is None:
            continue

        rid = None
        for k in prefer_id_fields:
            v = ex.get(k)
            if v is not None and str(v).strip() != "":
                rid = str(v).strip()
                break
        if rid is None:
            rid = f"{hf_id}:{split}:{i}"

        items.append({"id": rid, "question": q, "answer": ai, "source": hf_id})
    if not items:
        raise SystemExit(f"HF dataset load produced 0 usable (question,int answer) rows: {hf_id} split={split}")
    return items


def _build_train_pool_from_hf(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    banned = list((cfg.get("filtering", {}) or {}).get("banned_substrings", []) or [])
    aime_cfg = (cfg.get("datasets", {}) or {}).get("aime_train", {}) or {}
    hf_id = str(aime_cfg.get("hf_id", "")).strip()
    if not hf_id:
        raise SystemExit("configs: missing datasets.aime_train.hf_id")
    year_max = int(aime_cfg.get("year_max", 9999))

    items = _build_pool_from_hf(hf_id, split="train", banned=banned, year_max=year_max)

    amc_cfg = (cfg.get("datasets", {}) or {}).get("amc_aug", {}) or {}
    if bool(amc_cfg.get("enabled", False)):
        amc_id = str(amc_cfg.get("hf_id", "")).strip()
        if amc_id:
            items2 = _build_pool_from_hf(amc_id, split="train", banned=banned)
            for j, ex in enumerate(items2):
                ex["id"] = f"AMC-AUG-{j}"
                ex["year"] = 0
            items.extend(items2)

    if not items:
        raise SystemExit("HF fallback produced empty train_pool.")
    return items


def _load_pool(cfg: Dict[str, Any], dataset_mode: str) -> List[Dict[str, Any]]:
    dm = (dataset_mode or "train_pool").strip()
    banned = list((cfg.get("filtering", {}) or {}).get("banned_substrings", []) or [])

    if dm.lower() in ("train_pool", "local", "default"):
        p = Path(cfg["paths"]["train_pool_path"])
        if p.exists():
            try:
                xs = list(read_jsonl(p))
                out: List[Dict[str, Any]] = []
                for ex in xs:
                    q = str(ex.get("question", "") or "").strip()
                    ai = _parse_int_answer(ex.get("answer", None))
                    if not q or ai is None:
                        continue
                    if banned and _has_banned_substring(q, banned):
                        continue
                    out.append(
                        {
                            "id": str(ex.get("id", "")),
                            "question": q,
                            "answer": ai,
                            "source": str(ex.get("source", "train_pool")),
                        }
                    )
                if out:
                    return out
            except Exception:
                pass
        print(f"[inspect_rollouts] train_pool not found/empty at {p}. Falling back to HF dataset load.", flush=True)
        return _build_train_pool_from_hf(cfg)

    if dm.lower() in ("aime", "aime_train"):
        return _build_train_pool_from_hf(cfg)

    if dm.lower() in ("amc", "amc_aug"):
        amc_cfg = (cfg.get("datasets", {}) or {}).get("amc_aug", {}) or {}
        amc_id = str(amc_cfg.get("hf_id", "")).strip() or "AI-MO/aimo-validation-amc"
        split = (_env("HF_SPLIT", "train") or "train").strip()
        name = (_env("HF_NAME", None) or None)
        return _build_pool_from_hf(amc_id, split=split, name=name, banned=banned)

    if dm.lower() in ("math_level5", "math-level-5", "math5", "mathl5"):
        hf_id = (_env("HF_ID", None) or "").strip() or "AI-MO/aimo-validation-math-level-5"
        split = (_env("HF_SPLIT", "train") or "train").strip()
        name = (_env("HF_NAME", None) or None)
        return _build_pool_from_hf(hf_id, split=split, name=name, banned=banned)

    hf_id = dm
    if dm.lower().startswith("hf:"):
        hf_id = dm.split(":", 1)[1].strip()
    if "/" in hf_id:
        split = (_env("HF_SPLIT", "train") or "train").strip()
        name = (_env("HF_NAME", None) or None)
        return _build_pool_from_hf(hf_id, split=split, name=name, banned=banned)

    raise SystemExit(
        f"Unknown RTT_INSPECT_DATASET={dataset_mode!r}. "
        "Use train_pool | aime | amc | math_level5 | hf:<repo_id>."
    )


def _pick_examples(pool: List[Dict[str, Any]], n: int, seed: int, ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if ids:
        wanted = set(str(x) for x in ids)
        picked = [ex for ex in pool if str(ex.get("id")) in wanted]
        return picked[:n] if n > 0 else picked

    if n <= 0:
        return []
    if n >= len(pool):
        return list(pool)

    rng = random.Random(seed)
    idxs = rng.sample(range(len(pool)), n)
    return [pool[i] for i in idxs]


def _resolve_stop_list(stop_mode: str) -> List[str]:
    sm = (stop_mode or "marker").strip().lower()
    if sm == "sequence":
        return [RTT_STOP_SEQUENCE]
    if sm == "both":
        return [RTT_STOP_MARKER, RTT_STOP_SEQUENCE]
    return [RTT_STOP_MARKER]


def _is_truncated(
    finish_reason: Any,
    completion_tokens: int,
    max_new_tokens: int,
    *,
    prompt_tokens: Optional[int],
    max_model_len: int,
    truncate_on_context: bool,
) -> bool:
    fr = finish_reason or ""
    if isinstance(fr, str) and fr.lower() in ("length", "max_tokens", "max_length"):
        return True
    if int(completion_tokens or 0) >= int(max_new_tokens):
        return True
    if truncate_on_context and isinstance(prompt_tokens, int) and prompt_tokens >= 0:
        room = int(max_model_len) - int(prompt_tokens)
        if room > 0 and int(completion_tokens or 0) >= max(1, room - 2):
            return True
    return False


def _encode_no_special(tokenizer: Any, s: str) -> List[int]:
    try:
        return list(tokenizer.encode(s, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(s))


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


def _split_qwen3_thinking_tokens(token_ids: Optional[List[int]], think_end_ids: List[int]) -> Tuple[int, int, int]:
    if not isinstance(token_ids, list):
        return (0, 0, 0)
    total = len(token_ids)
    if total == 0 or not think_end_ids:
        return (total, 0, total)
    j = _find_last_subsequence(token_ids, think_end_ids)
    if j < 0:
        return (total, 0, total)
    split = j + len(think_end_ids)
    split = max(0, min(split, total))
    thinking = split
    answer = total - split
    return (total, thinking, answer)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _split_qwen3_thinking_text(text: str) -> Tuple[str, str]:
    if not text:
        return ("", "")
    s = str(text)
    i = s.find(_THINK_OPEN)
    j = s.rfind(_THINK_CLOSE)
    if i < 0 or j < 0 or j <= i:
        return ("", s)
    think_start = i + len(_THINK_OPEN)
    think_end = j
    thinking = s[think_start:think_end].strip()
    completion = s[j + len(_THINK_CLOSE) :].lstrip()
    return (thinking, completion)


# -------------------------
# Rendezvous helpers (filesystem, rank0 waits)
# -------------------------
def _run_id() -> str:
    rid = (_env("RUN_ID", None) or "").strip()
    if rid:
        return rid
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        return f"job{jid}"
    return f"pid{os.getpid()}"


def _tmp_dir_for(jsonl_out_base: str, cfg: Dict[str, Any]) -> Path:
    rid = _run_id()
    if jsonl_out_base:
        base = Path(jsonl_out_base).parent
    else:
        base = Path((cfg.get("paths", {}) or {}).get("outputs_dir", "outputs"))
    base.mkdir(parents=True, exist_ok=True)
    d = base / f".inspect_tmp.{rid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _wait_all_done(tmp_dir: Path, world_size: int, timeout_s: int) -> None:
    expected = [tmp_dir / f"done.rank{r}.json" for r in range(world_size)]
    t0 = time.time()
    while True:
        if all(p.exists() for p in expected):
            return
        if time.time() - t0 > float(timeout_s):
            missing = [str(p) for p in expected if not p.exists()]
            raise SystemExit(f"[inspect_rollouts] Timeout waiting for ranks. Missing: {missing[:3]} ...")
        time.sleep(0.5)


def _free_runner(runner: Optional[VLLMChatRunner]) -> None:
    if runner is None:
        return
    try:
        del runner
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    rank, world_size, local_rank = _rank_world_local()
    _set_sane_cuda_device(local_rank)

    ap = argparse.ArgumentParser(description="Inspect vLLM rollouts and truncation behavior (SLURM task-parallel, no DDP).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--who", choices=["student", "teacher", "both"], default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ids", type=str, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--jsonl-out", type=str, default=None)
    args = ap.parse_args()

    config_path = args.config or _env("CONFIG") or "configs/default.yml"
    who = (args.who or _env("WHO") or "both").strip().lower()
    if who not in ("student", "teacher", "both"):
        raise SystemExit(f"Invalid WHO={who!r}. Use student|teacher|both.")

    n_total = args.n if args.n is not None else (_parse_int(_env("N"), 5) or 5)
    seed = args.seed if args.seed is not None else (_parse_int(_env("SEED"), 0) or 0)
    ids = _parse_csv(args.ids if args.ids is not None else (_env("IDS", "") or ""))

    dataset_mode = (_env("DATASET", "train_pool") or "train_pool").strip()

    only_trunc_print = bool(_parse_bool(_env("ONLY_TRUNCATED"), False))
    full_output = bool(_parse_bool(_env("FULL_OUTPUT"), False))
    show_q_chars = _parse_int(_env("SHOW_QUESTION_CHARS"), 800) or 800
    show_out_chars = _parse_int(_env("SHOW_OUTPUT_CHARS"), 2000) or 2000

    jsonl_out_base = args.jsonl_out if args.jsonl_out is not None else (_env("JSONL_OUT") or "")
    jsonl_only_trunc = bool(_parse_bool(_env("JSONL_ONLY_TRUNCATED"), False))
    fsync_every = _parse_int(_env("FSYNC_EVERY"), 1) or 1
    merge_jsonl = bool(_parse_bool(_env("MERGE_JSONL"), True))

    stop_mode = _env("STOP_MODE", "marker") or "marker"
    truncate_on_context = bool(_parse_bool(_env("TRUNCATE_ON_CONTEXT"), True))

    cfg = load_config(config_path)
    seed_everything(int(cfg.get("training", {}).get("seed", 0)) + 12345 + int(seed))

    pool = _load_pool(cfg, dataset_mode=dataset_mode)

    picked_global = _pick_examples(pool, n=int(n_total), seed=int(seed), ids=ids if ids else None)
    picked_local = _shard_list(picked_global, rank=rank, world=world_size)

    print(
        f"[rank {rank}/{world_size}] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"torch_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"selected_global={len(picked_global)} selected_local={len(picked_local)}",
        flush=True,
    )

    inf = cfg.get("inference", {}) or {}

    env_mml = _parse_int(_env("MAX_MODEL_LEN"), None)
    max_model_len = int(env_mml) if env_mml is not None else int(inf.get("max_model_len", 4096))

    env_gmu = _parse_float(_env("GPU_MEMORY_UTILIZATION"), None)
    gpu_memory_utilization = float(env_gmu) if env_gmu is not None else float(inf.get("gpu_memory_utilization", 0.90))

    env_mns = _parse_int(_env("MAX_NUM_SEQS"), None)
    max_num_seqs = int(env_mns) if env_mns is not None else inf.get("max_num_seqs", None)

    # Force-disable thinking (requested)
    cfg_thinking_default = bool((cfg.get("models", {}) or {}).get("enable_thinking", True))
    enable_thinking = False

    # Config per role
    student_model_id = cfg["models"]["student_model_id"]
    teacher_model_id = cfg["models"]["teacher_model_id"]

    cfg_K_student = int(cfg["caching"]["K_student"])
    cfg_K_teacher = int(cfg["caching"]["K_teacher"])
    cfg_max_new_student = int(cfg["caching"]["max_new_tokens_student"])
    cfg_max_new_teacher = int(cfg["caching"]["max_new_tokens_teacher_solve"])

    # Shared overrides (apply to both if set)
    env_K = _parse_int(_env("K"), None)
    env_max_new = _parse_int(_env("MAX_NEW"), None)

    K_student = int(env_K) if env_K is not None else cfg_K_student
    K_teacher = int(env_K) if env_K is not None else cfg_K_teacher
    max_new_student = int(env_max_new) if env_max_new is not None else cfg_max_new_student
    max_new_teacher = int(env_max_new) if env_max_new is not None else cfg_max_new_teacher

    temp = _parse_float(_env("TEMP"), None)
    if temp is None:
        temp = float(cfg["caching"]["temperature"])

    top_p = _parse_float(_env("TOP_P"), None)
    if top_p is None:
        top_p = float(cfg["caching"]["top_p"])

    stop = _resolve_stop_list(stop_mode)

    sp_student = SamplingParams(max_tokens=int(max_new_student), n=int(K_student), temperature=float(temp), top_p=float(top_p), stop=stop)
    sp_teacher = SamplingParams(max_tokens=int(max_new_teacher), n=int(K_teacher), temperature=float(temp), top_p=float(top_p), stop=stop)

    # JSONL shard path
    out_f = None
    shard_path = ""
    if jsonl_out_base:
        Path(jsonl_out_base).parent.mkdir(parents=True, exist_ok=True)
        shard_path = f"{jsonl_out_base}.rank{rank}.jsonl"
        out_f = open(shard_path, "w", encoding="utf-8", buffering=1)

    tmp_dir = _tmp_dir_for(jsonl_out_base=jsonl_out_base, cfg=cfg)

    # Role totals (local)
    role_totals: Dict[str, Dict[str, int]] = {
        "student": {"comp": 0, "trunc": 0, "ctok": 0, "ttok": 0, "atok": 0, "rows": 0},
        "teacher": {"comp": 0, "trunc": 0, "ctok": 0, "ttok": 0, "atok": 0, "rows": 0},
    }

    try:
        if not picked_local:
            print(f"[rank {rank}] no assigned examples; skipping vLLM init.", flush=True)
        else:
            # Prepare local examples once
            ex_ids: List[str] = []
            qs: List[str] = []
            golds: List[int] = []
            for ex in picked_local:
                q = str(ex.get("question", "") or "").strip()
                a = ex.get("answer", None)
                if not q or a is None:
                    continue
                try:
                    gold = int(a)
                except Exception:
                    continue
                ex_ids.append(str(ex.get("id", "")))
                qs.append(q)
                golds.append(gold)

            if not qs:
                raise SystemExit(f"[rank {rank}] No usable (question, int answer) pairs after filtering.")

            batch_sz = args.batch if args.batch is not None else _parse_int(_env("BATCH"), None)
            if batch_sz is None or int(batch_sz) <= 0:
                batch_sz = len(qs)
            batch_sz = max(1, int(batch_sz))

            roles_to_run = ["student", "teacher"] if who == "both" else [who]

            print(
                "=== inspect_rollouts ===\n"
                f"rank={rank} world_size={world_size} (NO torch.distributed)\n"
                f"config={config_path}\n"
                f"dataset={dataset_mode} pool_size={len(pool)}\n"
                f"selected_global={len(picked_global)} selected_local={len(qs)}\n"
                f"who={who} roles_to_run={roles_to_run}\n"
                f"student_model={student_model_id}\n"
                f"teacher_model={teacher_model_id}\n"
                f"student: K={K_student} max_new={max_new_student}\n"
                f"teacher: K={K_teacher} max_new={max_new_teacher}\n"
                f"batch={batch_sz} (RTT_INSPECT_BATCH)\n"
                f"thinking={enable_thinking} (cfg_default={cfg_thinking_default}, forced_off=True)\n"
                f"stop_mode={stop_mode} stop={stop}\n"
                f"temp={temp} top_p={top_p}\n"
                f"max_model_len={max_model_len}\n"
                f"gpu_memory_utilization={gpu_memory_utilization}\n"
                f"max_num_seqs={max_num_seqs}\n"
                f"max_num_batched_tokens={inf.get('max_num_batched_tokens', None)}\n"
                f"truncate_on_context={truncate_on_context}\n"
                f"print_only_truncated={only_trunc_print}\n"
                f"jsonl_out_base={jsonl_out_base or '<none>'} shard={shard_path or '<none>'} merge_jsonl={merge_jsonl}\n"
                f"jsonl_only_truncated={jsonl_only_trunc} fsync_every={fsync_every}\n"
                f"prompt_extra_instruction={STEP_BY_STEP_INSTRUCTION!r}\n",
                flush=True,
            )

            # For memory safety: run one model at a time over all local questions,
            # store metas, then print combined per question.
            metas_by_role: Dict[str, List[Dict[str, Any]]] = {}

            for role in roles_to_run:
                runner: Optional[VLLMChatRunner] = None
                think_end_ids: List[int] = []
                model_id = student_model_id if role == "student" else teacher_model_id
                sp = sp_student if role == "student" else sp_teacher

                try:
                    runner = VLLMChatRunner(
                        model_id=model_id,
                        tp_size=1,
                        dtype=str(inf.get("dtype", "bfloat16")),
                        max_model_len=max_model_len,
                        gpu_memory_utilization=gpu_memory_utilization,
                        enable_lora=False,
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                        max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=(inf.get("max_num_batched_tokens", None)),
                    )
                    try:
                        think_end_ids = _encode_no_special(runner.tokenizer, "</think>")
                    except Exception:
                        think_end_ids = []

                    role_metas: List[Dict[str, Any]] = []

                    for start in range(0, len(qs), batch_sz):
                        sub_qs = qs[start : start + batch_sz]

                        if role == "student":
                            sub_msgs = [_student_messages(q) for q in sub_qs]
                        else:
                            sub_msgs = []
                            for q in sub_qs:
                                tmsgs = build_teacher_solve_messages(q, cfg)
                                tmsgs = _append_stop_marker_instruction(tmsgs)
                                sub_msgs.append(tmsgs)

                        metas = runner.generate_from_messages(sub_msgs, sp, return_meta=True)
                        # attach tokenizer-derived ids for this role (used later)
                        for m in metas:
                            m["_think_end_ids"] = think_end_ids
                        role_metas.extend(metas)

                    metas_by_role[role] = role_metas
                finally:
                    _free_runner(runner)

            # Now print + JSONL per question, showing both roles for each question
            wrote_rows = 0
            for i, (ex_id, q, gold) in enumerate(zip(ex_ids, qs, golds)):
                q_show = q if show_q_chars <= 0 else q[: int(show_q_chars)]
                if show_q_chars > 0 and len(q) > int(show_q_chars):
                    q_show += f"\n...[+{len(q) - int(show_q_chars)} chars]"

                # Prefer prompt token info from whichever role exists first
                prompt_note = "prompt_tokens=unknown"
                for role in roles_to_run:
                    meta = metas_by_role[role][i]
                    pt = meta.get("prompt_tokens", None)
                    if isinstance(pt, int):
                        room = max_model_len - pt
                        prompt_note = f"prompt_tokens={pt} room_for_completion≈{room}"
                        break

                print(f"--- [rank {rank}] id={ex_id} gold={gold}  ({prompt_note})")
                print(textwrap.indent(q_show, "Q: "))
                print()

                for role in roles_to_run:
                    meta = metas_by_role[role][i]
                    completions = list(meta.get("completions", []))

                    role_totals[role]["comp"] += len(completions)

                    # Per-role max_new for trunc calc
                    role_max_new = max_new_student if role == "student" else max_new_teacher
                    role_K = K_student if role == "student" else K_teacher
                    model_id = student_model_id if role == "student" else teacher_model_id

                    prompt_tokens = meta.get("prompt_tokens", None)
                    think_end_ids = meta.get("_think_end_ids", []) or []

                    print(f"  === {role.upper()} (model={model_id}) ===")

                    for j, c in enumerate(completions):
                        text = str(c.get("text", ""))
                        fr = c.get("finish_reason", None)

                        thinking_trace, completion_text = _split_qwen3_thinking_text(text)

                        token_ids = c.get("token_ids", None)
                        total_toks, think_toks, ans_toks = _split_qwen3_thinking_tokens(
                            token_ids if isinstance(token_ids, list) else None,
                            think_end_ids,
                        )

                        ctoks = int(c.get("completion_tokens", 0) or 0)
                        if total_toks > 0:
                            ctoks = total_toks

                        role_totals[role]["ctok"] += int(ctoks)
                        role_totals[role]["ttok"] += int(think_toks)
                        role_totals[role]["atok"] += int(ans_toks)

                        is_trunc = _is_truncated(
                            fr,
                            ctoks,
                            int(role_max_new),
                            prompt_tokens=(prompt_tokens if isinstance(prompt_tokens, int) else None),
                            max_model_len=max_model_len,
                            truncate_on_context=truncate_on_context,
                        )
                        role_totals[role]["trunc"] += int(is_trunc)

                        pred = extract_final_int(completion_text)
                        ok = aime_score(completion_text, gold)

                        if out_f is not None and ((not jsonl_only_trunc) or is_trunc):
                            out_f.write(
                                json.dumps(
                                    {
                                        "rank": int(rank),
                                        "world_size": int(world_size),
                                        "id": ex_id,
                                        "who": role,
                                        "model_id": model_id,
                                        "gold": gold,
                                        "question": q,
                                        "k": int(role_K),
                                        "max_new": int(role_max_new),
                                        "temperature": float(temp),
                                        "top_p": float(top_p),
                                        "stop_mode": str(stop_mode),
                                        "stop": list(stop),
                                        "thinking": bool(enable_thinking),
                                        "dataset": str(dataset_mode),
                                        "batch_size": int(batch_sz),
                                        "prompt_tokens": prompt_tokens,
                                        "completion_index": j,
                                        "finish_reason": fr,
                                        "completion_tokens": int(ctoks),
                                        "thinking_tokens": int(think_toks),
                                        "answer_tokens": int(ans_toks),
                                        "truncated": bool(is_trunc),
                                        "pred_final_int": pred,
                                        "score": int(ok),
                                        "max_model_len": int(max_model_len),
                                        "gpu_memory_utilization": float(gpu_memory_utilization),
                                        "max_num_seqs": (int(max_num_seqs) if isinstance(max_num_seqs, int) else None),
                                        "max_num_batched_tokens": inf.get("max_num_batched_tokens", None),
                                        "thinking_trace": thinking_trace,
                                        "completion_text": completion_text,
                                        "text": text,
                                        "step_by_step_instruction": STEP_BY_STEP_INSTRUCTION,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            wrote_rows += 1
                            role_totals[role]["rows"] += 1
                            if wrote_rows % int(fsync_every) == 0:
                                out_f.flush()
                                try:
                                    os.fsync(out_f.fileno())
                                except Exception:
                                    pass

                        if only_trunc_print and not is_trunc:
                            continue

                        head = text
                        if (not full_output) and show_out_chars > 0 and len(text) > int(show_out_chars):
                            head = text[: int(show_out_chars)] + f"\n...[+{len(text) - int(show_out_chars)} chars]"

                        print(
                            f"    [{j}] finish_reason={fr!r} completion_tokens={ctoks} "
                            f"(thinking={think_toks} answer={ans_toks}) truncated={is_trunc} "
                            f"final_int={pred} score={ok}"
                        )
                        print(textwrap.indent(head, "      "))
                        print()

                print()

            # Local summary (by role)
            print("=== local summary (by role) ===", flush=True)
            for role in roles_to_run:
                comp = role_totals[role]["comp"]
                trunc = role_totals[role]["trunc"]
                rate = (trunc / comp) if comp > 0 else 0.0
                print(
                    f"{role}: completions={comp} truncated={trunc} trunc_rate={rate:.3f} "
                    f"ctok_total={role_totals[role]['ctok']} "
                    f"think_tok_total={role_totals[role]['ttok']} "
                    f"answer_tok_total={role_totals[role]['atok']} "
                    f"jsonl_rows={role_totals[role]['rows']}",
                    flush=True,
                )
            print("", flush=True)

    finally:
        if out_f is not None:
            try:
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except Exception:
                    pass
            finally:
                out_f.close()

    # Write per-rank stats + done marker (always)
    stats_path = tmp_dir / f"stats.rank{rank}.json"
    done_path = tmp_dir / f"done.rank{rank}.json"

    _write_json(
        stats_path,
        {
            "rank": int(rank),
            "world_size": int(world_size),
            "role_totals": role_totals,
            "shard_path": shard_path,
            "t": time.time(),
        },
    )
    _write_json(done_path, {"ok": True, "rank": int(rank), "world_size": int(world_size), "t": time.time()})

    if not _rank0(rank):
        return

    timeout_s = int((_env("TIMEOUT_S", None) or os.environ.get("RTT_INSPECT_TIMEOUT_S", "3600") or "3600"))
    _wait_all_done(tmp_dir, world_size=world_size, timeout_s=timeout_s)

    # Aggregate global stats (by role)
    g: Dict[str, Dict[str, int]] = {
        "student": {"comp": 0, "trunc": 0, "ctok": 0, "ttok": 0, "atok": 0, "rows": 0},
        "teacher": {"comp": 0, "trunc": 0, "ctok": 0, "ttok": 0, "atok": 0, "rows": 0},
    }

    for r in range(world_size):
        spath = tmp_dir / f"stats.rank{r}.json"
        if not spath.exists():
            raise SystemExit(f"[inspect_rollouts] Missing stats file: {spath}")
        d = json.loads(spath.read_text(encoding="utf-8"))
        rt = d.get("role_totals", {}) or {}
        for role in ("student", "teacher"):
            rr = rt.get(role, {}) or {}
            for k in ("comp", "trunc", "ctok", "ttok", "atok", "rows"):
                g[role][k] += int(rr.get(k, 0))

    print("=== global summary (all ranks; by role) ===", flush=True)
    for role in ("student", "teacher"):
        comp = g[role]["comp"]
        trunc = g[role]["trunc"]
        rate = (trunc / comp) if comp > 0 else 0.0
        print(
            f"{role}: completions={comp} truncated={trunc} trunc_rate={rate:.3f} "
            f"ctok_total={g[role]['ctok']} "
            f"think_tok_total={g[role]['ttok']} "
            f"answer_tok_total={g[role]['atok']} "
            f"jsonl_rows={g[role]['rows']}",
            flush=True,
        )
    print("", flush=True)

    # Merge JSONL shards
    if jsonl_out_base and merge_jsonl:
        merged_path = Path(jsonl_out_base)
        tmp_path = merged_path.with_suffix(merged_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as outm:
            for r in range(world_size):
                rp = Path(f"{jsonl_out_base}.rank{r}.jsonl")
                if not rp.exists():
                    continue
                with open(rp, "r", encoding="utf-8") as infp:
                    for line in infp:
                        outm.write(line)
        os.replace(tmp_path, merged_path)
        print(f"[rank 0] merged_jsonl -> {merged_path}", flush=True)

    # Best-effort cleanup of rendezvous dir
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