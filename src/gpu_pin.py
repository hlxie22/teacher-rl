# src/gpu_pin.py
from __future__ import annotations

import os
from typing import Optional


def pin_one_gpu_per_task_early(
    *,
    local_rank_envs: tuple[str, ...] = ("SLURM_LOCALID", "LOCAL_RANK"),
) -> Optional[str]:
    """
    Pin exactly one GPU per task/process by rewriting CUDA_VISIBLE_DEVICES.

    Intended for SLURM task-parallel (1 GPU per task) and torchrun-style launches.

    Rules:
      - If CUDA_VISIBLE_DEVICES is already a single device (no commas), do nothing.
      - Else pick devs[local_rank] where local_rank comes from SLURM_LOCALID/LOCAL_RANK.
      - If CUDA_VISIBLE_DEVICES is empty, set it to the local_rank index.
      - If local_rank is out of range for the visible device list, fail loudly.

    Returns:
      The chosen CUDA_VISIBLE_DEVICES string, or None if no change was made.
    """
    lr = None
    lr_key = None
    for k in local_rank_envs:
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            lr = v
            lr_key = k
            break
    if lr is None:
        return None

    try:
        idx = int(lr)
    except Exception as e:
        raise RuntimeError(
            f"Invalid local-rank value from {lr_key}: {lr!r}. Expected a non-negative integer."
        ) from e

    if idx < 0:
        raise RuntimeError(f"Invalid local-rank value from {lr_key}: {idx}. Expected a non-negative integer.")

    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()

    # Already pinned to a single device.
    if cvd and ("," not in cvd):
        return None

    # If not set, assume 0..N-1 and choose idx.
    if not cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        return os.environ["CUDA_VISIBLE_DEVICES"]

    devs = [d.strip() for d in cvd.split(",") if d.strip()]
    if not devs:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        return os.environ["CUDA_VISIBLE_DEVICES"]

    if idx >= len(devs):
        raise RuntimeError(
            "Local rank requested a GPU index outside CUDA_VISIBLE_DEVICES: "
            f"{lr_key}={idx}, CUDA_VISIBLE_DEVICES={cvd!r}, parsed_devices={devs}."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = devs[idx]
    return os.environ["CUDA_VISIBLE_DEVICES"]
