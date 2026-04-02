# src/utils.py
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import yaml


RTT_STOP_SEQUENCE = "<RTT_END>"

_BOX_CMD_RE = re.compile(r"\\(?:boxed|fbox)\s*{")
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|\b)(?:final\s+answer|answer)\s*(?:is|=|:)?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_HINT_TAG_RE = re.compile(r"<FINAL_HINT>\s*(.*?)\s*</FINAL_HINT>", re.IGNORECASE | re.DOTALL)
_INTERVAL_RE = re.compile(r"^\s*([\[(])\s*(.+?)\s*,\s*(.+?)\s*([\])])\s*$", re.DOTALL)
_TRAILING_PUNCT_RE = re.compile(r"[\s\.;:,!]+$")
_GENERIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?![A-Za-z0-9_])"
)
_LATEX_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\s*{([^{}]+)}\s*{([^{}]+)}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*{([^{}]+)}")
_EXPRISH_HINT_RE = re.compile(r"[\\{}^/()]")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def jsonl_line_count(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def quick_sig(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": sha256_file(p),
    }


def is_done(done_path: str | Path, sig: Dict[str, Any], outputs: Iterable[str | Path]) -> bool:
    dp = Path(done_path)
    if not dp.exists():
        return False

    out_paths = [Path(x) for x in outputs]
    if not all(p.exists() for p in out_paths):
        return False

    try:
        payload = json.loads(dp.read_text(encoding="utf-8"))
    except Exception:
        return False

    return payload.get("sig") == sig


def write_done(done_path: str | Path, sig: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "sig": sig,
        "timestamp": float(time.time()),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(done_path, payload)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _find_last_boxed(text: str) -> Optional[str]:
    matches = list(_BOX_CMD_RE.finditer(text or ""))
    if not matches:
        return None

    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def _strip_wrapping_delims(text: str) -> str:
    s = str(text).strip()
    s = s.replace(RTT_STOP_SEQUENCE, "").strip()
    s = s.strip().strip("$").strip()
    s = _TRAILING_PUNCT_RE.sub("", s).strip()
    while len(s) >= 2 and ((s[0], s[-1]) == ("{", "}")):
        inner = s[1:-1].strip()
        if not inner:
            break
        s = inner
        s = _TRAILING_PUNCT_RE.sub("", s).strip()
    return s


def _normalize_final_span(text: str) -> str:
    s = _strip_wrapping_delims(text)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return s.strip()


def extract_final_answer_text(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None

    boxed = _find_last_boxed(raw)
    if boxed is not None:
        ans = _normalize_final_span(boxed)
        return ans or None

    answer_match = None
    for m in _FINAL_ANSWER_RE.finditer(raw):
        answer_match = m
    if answer_match is not None:
        ans = _normalize_final_span(answer_match.group(1))
        if ans:
            return ans

    cleaned = _normalize_final_span(raw)
    if _INTERVAL_RE.match(cleaned):
        return cleaned

    # Prefer parsing the whole cleaned span before guessing from a trailing number.
    try:
        parse_numeric_or_interval(cleaned)
        return cleaned
    except Exception:
        pass

    # Very conservative fallback: only plain-text, single-number answers.
    if not _EXPRISH_HINT_RE.search(cleaned):
        nums = _GENERIC_NUMBER_RE.findall(cleaned)
        if len(nums) == 1:
            return _normalize_final_span(nums[0])

    return cleaned or None


def extract_final_hint_text(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None

    m = None
    for m in _HINT_TAG_RE.finditer(raw):
        pass
    if m is None:
        return None

    hint = m.group(1).replace(RTT_STOP_SEQUENCE, "").strip()
    return hint or None


def _latex_to_expr(text: str) -> str:
    s = str(text).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^", "**")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")

    while True:
        ns = _LATEX_FRAC_RE.sub(r"((\1)/(\2))", s)
        if ns == s:
            break
        s = ns

    while True:
        ns = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s)
        if ns == s:
            break
        s = ns

    s = s.replace("\\pi", "pi")
    s = s.replace("{", "(").replace("}", ")")
    return s


_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
}
_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_expr(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("unsupported constant")
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp):
            lhs = _eval(n.left)
            rhs = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return lhs + rhs
            if isinstance(n.op, ast.Sub):
                return lhs - rhs
            if isinstance(n.op, ast.Mult):
                return lhs * rhs
            if isinstance(n.op, ast.Div):
                return lhs / rhs
            if isinstance(n.op, ast.Pow):
                return lhs ** rhs
            raise ValueError("unsupported binary operator")
        if isinstance(n, ast.UnaryOp):
            val = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +val
            if isinstance(n.op, ast.USub):
                return -val
            raise ValueError("unsupported unary operator")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function")
            if len(n.args) != 1:
                raise ValueError("unsupported function arity")
            return float(_ALLOWED_FUNCS[n.func.id](_eval(n.args[0])))
        if isinstance(n, ast.Name):
            if n.id not in _ALLOWED_NAMES:
                raise ValueError("unsupported name")
            return float(_ALLOWED_NAMES[n.id])
        raise ValueError("unsupported expression")

    return float(_eval(node))


def _parse_scalar(text: str) -> float:
    s = _normalize_final_span(text)
    s = _latex_to_expr(s)
    s = re.sub(r"\s+", "", s)
    if not s:
        raise ValueError("empty scalar")
    return _safe_eval_expr(s)


def parse_numeric_or_interval(text: Any) -> Dict[str, Any]:
    s = _normalize_final_span(str(text))
    if not s:
        raise ValueError("empty answer")

    m = _INTERVAL_RE.match(s)
    if m is not None:
        left_br, left, right, right_br = m.groups()
        return {
            "type": "interval",
            "left_closed": left_br == "[",
            "right_closed": right_br == "]",
            "left": _parse_scalar(left),
            "right": _parse_scalar(right),
        }

    return {
        "type": "scalar",
        "value": _parse_scalar(s),
    }


def _parsed_equal(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-6) -> bool:
    if a.get("type") != b.get("type"):
        return False
    if a.get("type") == "scalar":
        return math.isclose(float(a["value"]), float(b["value"]), rel_tol=tol, abs_tol=tol)
    return (
        bool(a["left_closed"]) == bool(b["left_closed"])
        and bool(a["right_closed"]) == bool(b["right_closed"])
        and math.isclose(float(a["left"]), float(b["left"]), rel_tol=tol, abs_tol=tol)
        and math.isclose(float(a["right"]), float(b["right"]), rel_tol=tol, abs_tol=tol)
    )


def numeric_score(pred_text: Any, gold_text: Any) -> int:
    gold = extract_final_answer_text(str(gold_text))
    pred = extract_final_answer_text(str(pred_text))
    if gold is None or pred is None:
        return 0

    try:
        gp = parse_numeric_or_interval(gold)
        pp = parse_numeric_or_interval(pred)
        return int(_parsed_equal(pp, gp))
    except Exception:
        return int(_normalize_final_span(pred) == _normalize_final_span(gold))


def aime_score(pred_text: Any, gold_answer: int) -> int:
    pred = extract_final_answer_text(str(pred_text))
    if pred is None:
        return 0
    try:
        parsed = parse_numeric_or_interval(pred)
        if parsed.get("type") != "scalar":
            return 0
        return int(int(round(float(parsed["value"]))) == int(gold_answer))
    except Exception:
        nums = _GENERIC_NUMBER_RE.findall(pred)
        if not nums:
            return 0
        try:
            return int(int(float(nums[-1])) == int(gold_answer))
        except Exception:
            return 0


def contains_answer_leak_any(hint_text: str, gold_text: Any) -> bool:
    hint = str(hint_text or "")
    gold = extract_final_answer_text(str(gold_text))
    if not hint or gold is None:
        return False

    hint_lower = hint.lower()
    gold_norm = _normalize_final_span(gold)
    if gold_norm and gold_norm.lower() in hint_lower:
        return True

    try:
        gold_parsed = parse_numeric_or_interval(gold_norm)
    except Exception:
        return False

    for tok in _GENERIC_NUMBER_RE.findall(hint):
        try:
            cand = parse_numeric_or_interval(tok)
        except Exception:
            continue
        if _parsed_equal(cand, gold_parsed):
            return True
    return False


def _cfg_get_str(cfg: Dict[str, Any], path: Sequence[str], default: str) -> str:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    if cur is None:
        return default
    s = str(cur).strip()
    return s if s else default


def get_role_runtime_cfg(cfg: Dict[str, Any], role: str) -> Dict[str, Any]:
    inf = (cfg.get("inference", {}) or {})

    if role == "student":
        role_cfg = (cfg.get("student_server", {}) or {})
    elif role == "teacher":
        role_cfg = (cfg.get("teacher_rollout", {}) or {})
    else:
        raise ValueError(f"unsupported runtime role: {role}")

    return {
        "dtype": str(role_cfg.get("dtype", inf.get("dtype", "bfloat16"))),
        "max_model_len": int(role_cfg.get("max_model_len", inf.get("max_model_len", 4096))),
        "gpu_memory_utilization": float(
            role_cfg.get("gpu_memory_utilization", inf.get("gpu_memory_utilization", 0.90))
        ),
        "attention_backend": inf.get("attention_backend", None),
        "language_model_only": bool(inf.get("language_model_only", False)),
        "cpu_offload_gb": float(inf.get("cpu_offload_gb", 0.0)),
        "enforce_eager": bool(inf.get("enforce_eager", False)),
        "disable_log_stats": bool(inf.get("disable_log_stats", True)),
        "max_num_seqs": inf.get("max_num_seqs", None),
        "max_num_batched_tokens": inf.get("max_num_batched_tokens", None),
        "max_loras": int(inf.get("max_loras", 8)),
        "max_lora_rank": int(inf.get("max_lora_rank", 64)),
    }


def build_student_attempt_messages(
    question: str,
    cfg: Dict[str, Any],
    hint: Optional[str] = None,
    include_stop_marker: bool = True,
) -> List[Dict[str, str]]:
    system = _cfg_get_str(
        cfg,
        ["prompting", "student_system_prompt"],
        "You are a careful competition math student. Solve the problem step by step and put the final answer in \\boxed{...}.",
    )
    parts = [f"Problem:\n{str(question).strip()}"]
    if hint:
        parts.append(f"Teacher hint:\n{str(hint).strip()}")
    if include_stop_marker:
        parts.append(
            "After you give the final answer in \\boxed{...}, output the stop marker "
            f"{RTT_STOP_SEQUENCE} on its own line."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_teacher_hint_messages(question: str, student_failure: str, cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    system = _cfg_get_str(
        cfg,
        ["prompting", "teacher_hint_system_prompt"],
        (
            "You are a math teacher writing short recovery hints. "
            "Help the student fix the mistake without revealing the final answer. "
            "Return exactly one hint inside <FINAL_HINT>...</FINAL_HINT>."
        ),
    )
    user = (
        f"Problem:\n{str(question).strip()}\n\n"
        f"Student failed attempt:\n{str(student_failure).strip()}\n\n"
        "Write one concise hint inside <FINAL_HINT>...</FINAL_HINT>. "
        "Do not reveal the exact final answer or a numerically equivalent expression."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_teacher_solve_messages(question: str, cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    system = _cfg_get_str(
        cfg,
        ["prompting", "teacher_solve_system_prompt"],
        "You are an expert competition math solver. Solve the problem carefully and put the final answer in \\boxed{...}.",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Problem:\n{str(question).strip()}"},
    ]


def build_teacher_solve_messages_numeric(question: str, cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    system = _cfg_get_str(
        cfg,
        ["prompting", "teacher_solve_numeric_system_prompt"],
        (
            "You are an expert competition math solver. Solve the problem carefully. "
            "The final answer should be numeric or an interval and should appear in \\boxed{...}."
        ),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Problem:\n{str(question).strip()}"},
    ]
