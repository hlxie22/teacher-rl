# src/prep_data.py
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import load_dataset

try:
    # available in datasets; may query hub
    from datasets import get_dataset_config_names  # type: ignore
except Exception:
    get_dataset_config_names = None  # type: ignore

from .utils import (
    ensure_dir,
    extract_final_answer_text,
    is_done,
    jsonl_line_count,
    load_config,
    parse_numeric_or_interval,
    sha256_file,
    write_done,
    write_jsonl,
)


def has_banned_substring(q: str, banned: List[str]) -> bool:
    ql = (q or "").lower()
    return any(s.lower() in ql for s in banned)


_LEVEL_INT_RE = re.compile(r"(\d+)")


def _level_to_int(level_str: str) -> int:
    s = (level_str or "").strip()
    m = _LEVEL_INT_RE.findall(s)
    if not m:
        return -1
    try:
        return int(m[-1])
    except Exception:
        return -1


def _normalize_math_configs(math_cfg: Dict[str, Any], hf_id: str) -> List[str]:
    """
    Accepts any of:
      - datasets.math_train.configs: [..]
      - datasets.math_train.config: "algebra"
      - datasets.math_train.subjects: [..]   (alias)
      - datasets.math_train.subject: "algebra" (alias)

    If none are provided, tries to auto-discover configs from HF.
    """
    raw = (
        math_cfg.get("configs", None)
        or math_cfg.get("config", None)
        or math_cfg.get("subjects", None)
        or math_cfg.get("subject", None)
    )

    cfgs: List[str] = []
    if isinstance(raw, str) and raw.strip():
        cfgs = [raw.strip()]
    elif isinstance(raw, (list, tuple)):
        cfgs = [str(x).strip() for x in raw if str(x).strip()]

    if cfgs:
        return cfgs

    if get_dataset_config_names is not None:
        try:
            discovered = get_dataset_config_names(hf_id)
            cfgs2 = [str(x).strip() for x in discovered if str(x).strip()]
            if cfgs2:
                return cfgs2
        except Exception:
            pass

    return [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]


def _normalize_levels(math_cfg: Dict[str, Any]) -> Set[int]:
    """
    Accepts any of:
      - datasets.math_train.level: 5
      - datasets.math_train.level: [1, 2, 3]
      - datasets.math_train.levels: [1, 2, 3]   (alias)

    Returns a non-empty set of ints.
    """
    raw = math_cfg.get("level", None)
    if raw is None:
        raw = math_cfg.get("levels", 5)

    if isinstance(raw, (list, tuple, set)):
        vals = {int(x) for x in raw}
    else:
        vals = {int(raw)}

    if not vals:
        raise SystemExit("datasets.math_train.level/levels must not be empty")

    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="Ignore done marker and rebuild outputs.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["data_dir"])

    done_path = Path(cfg["paths"]["data_dir"]) / "prep.done.json"
    outputs = [
        cfg["paths"]["train_pool_path"],
        cfg["paths"]["eval_aime24_path"],
        cfg["paths"]["eval_aime25_path"],
        cfg["paths"]["eval_hmmt25_path"],
        cfg["paths"]["eval_hmmt26_path"],
        cfg["paths"]["eval_amc_path"],
    ]

    sig: Dict[str, Any] = {"config_sha256": sha256_file(args.config)}
    if (not args.force) and is_done(done_path, sig, outputs):
        print(f"[prep_data] already done -> {done_path}")
        return

    banned = cfg.get("filtering", {}).get("banned_substrings", [])

    # ---- Train pool: Hendrycks MATH, using combined train+test ----
    math_cfg = cfg["datasets"]["math_train"]
    hf_id = str(math_cfg["hf_id"]).strip()
    level_targets = _normalize_levels(math_cfg)
    splits = list(math_cfg.get("splits", ["train", "test"]))

    math_configs = _normalize_math_configs(math_cfg, hf_id)
    if not math_configs:
        raise SystemExit(
            f"Could not determine configs for {hf_id}. "
            "Set datasets.math_train.configs in your YAML."
        )

    print(
        f"[prep_data] Loading {hf_id} "
        f"configs={math_configs} splits={splits} levels={sorted(level_targets)}"
    )

    items: List[Dict[str, Any]] = []
    kept = 0
    dropped_parse = 0
    dropped_no_ans = 0
    per_subject = Counter()
    per_level = Counter()

    for subject in math_configs:
        for sp in splits:
            ds = load_dataset(hf_id, subject, split=sp)

            for ex in ds:
                lvl = str(ex.get("level") or ex.get("Level") or "").strip()
                lvl_int = _level_to_int(lvl)
                if lvl_int not in level_targets:
                    continue

                q = (ex.get("problem") or ex.get("question") or ex.get("Question") or "").strip()
                sol = (ex.get("solution") or ex.get("Solution") or "").strip()
                if not q or not sol:
                    continue
                if has_banned_substring(q, banned):
                    continue
                if "[asy]" in q.lower():
                    continue

                ans = extract_final_answer_text(sol)
                if ans is None:
                    dropped_no_ans += 1
                    continue
                ans = str(ans).strip()

                # Keep only answers numeric_score can parse (prevents training-time scoring crashes)
                try:
                    parse_numeric_or_interval(ans)
                except Exception:
                    dropped_parse += 1
                    continue

                items.append(
                    {
                        "id": f"MATH{lvl_int}-{subject}-{sp}-{kept}",
                        "question": q,
                        "answer": ans,
                        "level": lvl,
                        "type": str(ex.get("type") or ex.get("Type") or ""),
                        "source": hf_id,
                        "subject": subject,
                        "split": sp,
                    }
                )
                kept += 1
                per_subject[subject] += 1
                per_level[lvl_int] += 1

    write_jsonl(cfg["paths"]["train_pool_path"], items)
    print(
        f"Wrote train pool: {cfg['paths']['train_pool_path']} (n={len(items)}), "
        f"dropped_no_ans={dropped_no_ans}, dropped_unparseable={dropped_parse}"
    )
    if per_subject:
        print(f"[prep_data] train pool per subject: {dict(per_subject)}")
    if per_level:
        print(f"[prep_data] train pool per level: {dict(sorted(per_level.items()))}")

    # ---- Eval AIME 2024 ----
    ds24 = load_dataset(cfg["datasets"]["aime24"]["hf_id"], split="train")
    eval24 = []
    for i, ex in enumerate(ds24):
        q = ex.get("Question") or ex.get("question") or ex.get("problem") or ex.get("Problem") or ""
        a = ex.get("Answer") or ex.get("answer")
        if a is None:
            continue
        if has_banned_substring(str(q), banned):
            continue
        eval24.append({"id": f"AIME24-{i}", "question": str(q).strip(), "answer": int(a)})
    write_jsonl(cfg["paths"]["eval_aime24_path"], eval24)
    print(f"Wrote AIME24: {cfg['paths']['eval_aime24_path']} (n={len(eval24)})")

    # ---- Eval AIME 2025 ----
    ds25 = load_dataset(cfg["datasets"]["aime25"]["hf_id"], split="train")
    eval25 = []
    for i, ex in enumerate(ds25):
        q = ex.get("Question") or ex.get("question") or ex.get("problem") or ex.get("Problem") or ""
        a = ex.get("Answer") or ex.get("answer")
        if a is None:
            continue
        if has_banned_substring(str(q), banned):
            continue
        eval25.append({"id": f"AIME25-{i}", "question": str(q).strip(), "answer": int(a)})
    write_jsonl(cfg["paths"]["eval_aime25_path"], eval25)
    print(f"Wrote AIME25: {cfg['paths']['eval_aime25_path']} (n={len(eval25)})")

    # ---- Eval hard OOD: HMMT Feb 2025 + Feb 2026 (MathArena) ----
    for key, out_key, prefix in [
        ("hmmt_feb_2025", "eval_hmmt25_path", "HMMT25"),
        ("hmmt_feb_2026", "eval_hmmt26_path", "HMMT26"),
    ]:
        dcfg = cfg["datasets"].get(key, {}) or {}
        hf2 = str(dcfg.get("hf_id", "")).strip()
        split = str(dcfg.get("split", "train"))
        if not hf2:
            raise SystemExit(f"Missing cfg.datasets.{key}.hf_id")

        ds_h = load_dataset(hf2, split=split)
        evalh = []
        for i, ex in enumerate(ds_h):
            q = ex.get("problem") or ex.get("question") or ex.get("Question") or ""
            a = ex.get("answer") or ex.get("Answer")
            if not q or a is None:
                continue
            q = str(q).strip()
            if has_banned_substring(q, banned):
                continue
            if "[asy]" in q.lower():
                continue
            evalh.append(
                {
                    "id": f"{prefix}-{int(ex.get('problem_idx', i))}",
                    "question": q,
                    "answer": str(a).strip(),
                    "source": hf2,
                }
            )

        write_jsonl(cfg["paths"][out_key], evalh)
        print(f"Wrote {prefix} OOD: {cfg['paths'][out_key]} (n={len(evalh)})")

    # ---- Eval OOD: AMC ----
    amc_cfg = (cfg.get("datasets", {}) or {}).get("amc_ood", {}) or {}
    amc_id = str(amc_cfg.get("hf_id", "")).strip()
    amc_split = str(amc_cfg.get("split", "train"))
    eval_amc = []
    if amc_id:
        ds_amc = load_dataset(amc_id, split=amc_split)
        for i, ex in enumerate(ds_amc):
            q = (ex.get("problem") or ex.get("question") or ex.get("Question") or "").strip()
            a = ex.get("answer") or ex.get("Answer")
            if not q or a is None:
                continue
            if has_banned_substring(q, banned) or "[asy]" in q.lower():
                continue
            try:
                aa = int(a)
            except Exception:
                continue
            eval_amc.append({"id": f"AMC-{i}", "question": q, "answer": aa, "source": amc_id})

    write_jsonl(cfg["paths"]["eval_amc_path"], eval_amc)
    print(f"Wrote AMC OOD: {cfg['paths']['eval_amc_path']} (n={len(eval_amc)})")

    write_done(
        done_path,
        sig,
        extra={"counts": {str(p): jsonl_line_count(p) for p in outputs}},
    )
    print(f"[prep_data] wrote done marker -> {done_path}")


if __name__ == "__main__":
    main()