from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import torch


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    if is_dist():
        return torch.distributed.get_rank()
    return int(os.environ.get('RANK', '0') or '0')


def world_size() -> int:
    if is_dist():
        return torch.distributed.get_world_size()
    return int(os.environ.get('WORLD_SIZE', '1') or '1')


def local_rank_raw() -> int:
    v = os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0'))
    try:
        lr = int(v)
    except Exception as e:
        raise RuntimeError(f'Invalid LOCAL_RANK/SLURM_LOCALID value: {v!r}') from e
    if lr < 0:
        raise RuntimeError(f'Invalid LOCAL_RANK/SLURM_LOCALID value: {lr}. Expected >= 0.')
    return lr


def local_rank() -> int:
    lr = local_rank_raw()
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n <= 0:
            raise RuntimeError('torch.cuda.is_available() is True but no CUDA devices are visible.')
        if n == 1:
            return 0
        if lr >= n:
            raise RuntimeError(
                f'LOCAL_RANK/SLURM_LOCALID={lr} but only {n} CUDA device(s) are visible. '
                'Refusing to silently fall back to GPU 0. Check the launcher and CUDA_VISIBLE_DEVICES.'
            )
    return lr


def dist_barrier() -> None:
    if is_dist():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[local_rank()])
        else:
            torch.distributed.barrier()


def gather_objects_to_rank0(obj: Any) -> Optional[List[Any]]:
    if not is_dist():
        return [obj]

    ws = world_size()
    r = rank()

    if hasattr(torch.distributed, 'gather_object'):
        if r == 0:
            gathered: List[Any] = [None] * ws
            torch.distributed.gather_object(obj=obj, object_gather_list=gathered, dst=0)
            return gathered
        torch.distributed.gather_object(obj=obj, object_gather_list=None, dst=0)
        return None

    gathered = [None] * ws
    torch.distributed.all_gather_object(gathered, obj)
    return gathered if r == 0 else None


def scatter_object_from_rank0(rank_payloads: Sequence[Any]) -> Any:
    if not is_dist():
        return rank_payloads[0]

    r = rank()
    if hasattr(torch.distributed, 'scatter_object_list'):
        recv: List[Any] = [None]
        if r == 0:
            scatter_list = list(rank_payloads)
            if len(scatter_list) != world_size():
                raise RuntimeError(
                    f'scatter payload count mismatch: got {len(scatter_list)} payloads for world_size={world_size()}'
                )
            torch.distributed.scatter_object_list(
                scatter_object_output_list=recv,
                scatter_object_input_list=scatter_list,
                src=0,
            )
        else:
            torch.distributed.scatter_object_list(
                scatter_object_output_list=recv,
                scatter_object_input_list=None,
                src=0,
            )
        return recv[0]

    gathered = [None] * world_size()
    src_obj = list(rank_payloads) if r == 0 else None
    torch.distributed.all_gather_object(gathered, src_obj)
    src_payloads = gathered[0]
    if src_payloads is None:
        raise RuntimeError('rank 0 did not broadcast scatter payloads')
    return src_payloads[r]
