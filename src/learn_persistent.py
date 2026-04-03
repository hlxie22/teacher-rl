# src/learn_persistent.py
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from .coord import Coordinator
from .dist_utils import dist_barrier as _dist_barrier
from .dist_utils import is_dist as _is_dist
from .dist_utils import local_rank as _local_rank
from .dist_utils import rank as _rank
from .dist_utils import world_size as _world_size
from .distributed_rollout import DistributedRolloutCoordinator
from .models import LM, generate_text_batch, load_lm, maybe_apply_chat_template
from .remote_vllm import RemoteVLLMCompletionsClient
from .utils import (
    RTT_STOP_SEQUENCE,
    atomic_write_json,
    build_student_attempt_messages,
    build_teacher_hint_messages,
    contains_answer_leak_any,
    ensure_dir,
    extract_final_hint_text,
    load_config,
    numeric_score,
    read_jsonl,
    seed_everything,
)

_STOP_REQUESTED = False
_RESUME_CONTEXT_FILE = 'resume_context.json'
_TRL_COLOCATE_DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.30


def _handle_usr1(sig, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


try:
    signal.signal(signal.SIGUSR1, _handle_usr1)
    signal.signal(signal.SIGTERM, _handle_usr1)
    signal.signal(signal.SIGINT, _handle_usr1)
except Exception:
    pass


def _device() -> str:
    if torch.cuda.is_available():
        return f'cuda:{_local_rank()}'
    return 'cpu'


def _init_distributed_early() -> None:
    ws = int(os.environ.get('WORLD_SIZE', '1') or '1')
    if ws <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(_local_rank())
        return

    if not torch.distributed.is_available():
        raise RuntimeError('WORLD_SIZE>1 but torch.distributed is not available')

    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
        backend = 'nccl'
    else:
        backend = 'gloo'

    if not torch.distributed.is_initialized():
        kwargs: Dict[str, Any] = {'backend': backend, 'init_method': 'env://'}
        if backend == 'nccl':
            try:
                kwargs['device_id'] = torch.device('cuda', _local_rank())
            except Exception:
                pass
        torch.distributed.init_process_group(**kwargs)


def _extract_text_completion(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], dict) and 'content' in x[0]:
        return str(x[0].get('content', ''))
    if isinstance(x, dict) and 'content' in x:
        return str(x.get('content', ''))
    return str(x)


def _token_count(tokenizer: Any, text: Any) -> int:
    return int(len(tokenizer.encode(str(text), add_special_tokens=False)))


def _truncate_to_token_limit(tokenizer, text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ''
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text.strip()
        return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()
    except Exception:
        words = text.strip().split()
        if len(words) <= max_tokens:
            return text.strip()
        return ' '.join(words[:max_tokens]).strip()


def _get_grpo_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    top = (cfg.get('grpo', {}) or {})
    legacy = ((cfg.get('training', {}) or {}).get('grpo', {}) or {})
    if not top and legacy:
        return dict(legacy)
    return {**dict(legacy), **dict(top)}


def _use_trl_vllm_colocate(cfg: Dict[str, Any]) -> bool:
    mode = str(
        os.environ.get(
            'RTT_PL_MODE',
            (cfg.get('resources', {}) or {}).get('placement_mode', ''),
        )
    ).strip().lower()
    return mode == 'trl_colocate'


def _resolve_generation_batch_size_arg(grpo_cfg: Dict[str, Any], num_generations: int) -> Optional[int]:
    prompts_per_cycle = grpo_cfg.get('rollout_prompts_per_cycle', None)
    if prompts_per_cycle is None:
        prompts_per_cycle = grpo_cfg.get('prompts_per_cycle', None)
    if prompts_per_cycle is not None:
        prompts_per_cycle = int(prompts_per_cycle)
        if prompts_per_cycle < 1:
            raise ValueError('grpo.rollout_prompts_per_cycle must be >= 1 when set')
        return prompts_per_cycle * int(num_generations)

    generation_batch_size = grpo_cfg.get('generation_batch_size', None)
    if generation_batch_size is None:
        return None
    generation_batch_size = int(generation_batch_size)
    if generation_batch_size < 1:
        raise ValueError('grpo.generation_batch_size must be >= 1 when set')
    return generation_batch_size


def _read_resume_context(out_dir: Path) -> Dict[str, Any]:
    p = Path(out_dir) / _RESUME_CONTEXT_FILE
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_resume_context(out_dir: Path, *, step_offset: int, start_round: int, steps_per_round: int, max_steps_this_run: int) -> None:
    atomic_write_json(
        Path(out_dir) / _RESUME_CONTEXT_FILE,
        {
            'step_offset': int(step_offset),
            'start_round': int(start_round),
            'steps_per_round': int(steps_per_round),
            'max_steps_this_run': int(max_steps_this_run),
            'written_ts': float(time.time()),
        },
    )


def _load_student_lm_safe(model_id: str, device: str, load_in_4bit: bool, bf16: bool) -> LM:
    if device != 'cpu':
        return load_lm(model_id, device, load_in_4bit=load_in_4bit, bf16=bf16)

    if load_in_4bit:
        if _rank() == 0:
            print('[learner/grpo] WARNING: load_in_4bit requested on CPU; disabling (bnb requires CUDA).')
        load_in_4bit = False

    tok_kwargs: Dict[str, Any] = {'trust_remote_code': True}
    if Path(model_id).exists():
        tok_kwargs['local_files_only'] = True

    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs: Dict[str, Any] = {
        'trust_remote_code': True,
        'torch_dtype': torch.float32,
        'device_map': None,
    }
    if Path(model_id).exists():
        model_kwargs['local_files_only'] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to('cpu')
    model.eval()
    return LM(model=model, tokenizer=tok)


def _load_teacher_causallm(model_id: str, bf16: bool):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if bf16 else torch.float16
    else:
        dtype = torch.float32

    model_kwargs: Dict[str, Any] = {
        'trust_remote_code': True,
        'torch_dtype': dtype,
        'low_cpu_mem_usage': True,
    }
    if Path(model_id).exists():
        model_kwargs['local_files_only'] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model


def _read_trainer_checkpoint_global_step(ckpt_dir: Optional[str]) -> Optional[int]:
    if not ckpt_dir:
        return None
    p = Path(ckpt_dir) / 'trainer_state.json'
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return int(data.get('global_step', 0) or 0)
    except Exception:
        return None


def _load_teacher_policy_model(
    base_model_id: str,
    *,
    adapter_path: Optional[str],
    bf16: bool,
):
    model = _load_teacher_causallm(base_model_id, bf16=bf16)
    if not adapter_path:
        return model, False

    ap = Path(adapter_path)
    if not ap.exists():
        raise FileNotFoundError(f'Teacher adapter path does not exist: {adapter_path}')

    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        str(ap),
        is_trainable=True,
    )
    try:
        model.set_adapter('default')
    except Exception:
        pass
    return model, True


def _load_tokenizer_only(model_id: str):
    tok_kwargs: Dict[str, Any] = {'trust_remote_code': True}
    if Path(model_id).exists():
        tok_kwargs['local_files_only'] = True
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


class _RemoteStudentVLLMClient:
    def __init__(
        self,
        base_urls: Sequence[str],
        model_name: str,
        timeout_s: float = 120.0,
        aggregate_ranks_per_request: int = 1,
    ):
        urls = [str(u).strip().rstrip('/') for u in base_urls if str(u).strip()]
        if not urls:
            raise ValueError('Remote student vLLM client requires at least one base URL.')
        self.base_urls = urls
        self.model_name = str(model_name)
        self.timeout_s = float(timeout_s)
        self.aggregate_ranks_per_request = max(1, int(aggregate_ranks_per_request))
        self._rr = 0

        self._group_ranks: List[int] = [_rank()]
        self._group_leader_rank: int = _rank()
        self._assigned_base_url: str = self.base_urls[0]
        self._dist_group = None

        if _is_dist() and self.aggregate_ranks_per_request > 1:
            world = _world_size()
            my_rank = _rank()
            my_group = None
            for start in range(0, world, self.aggregate_ranks_per_request):
                ranks = list(range(start, min(world, start + self.aggregate_ranks_per_request)))
                grp = torch.distributed.new_group(ranks=ranks)
                if my_rank in ranks:
                    my_group = grp
                    self._group_ranks = ranks
                    self._group_leader_rank = ranks[0]
                    group_id = start // self.aggregate_ranks_per_request
                    self._assigned_base_url = self.base_urls[group_id % len(self.base_urls)]
            self._dist_group = my_group
        elif _is_dist():
            self._assigned_base_url = self.base_urls[_rank() % len(self.base_urls)]

    def _next_url(self) -> str:
        url = self.base_urls[self._rr % len(self.base_urls)]
        self._rr += 1
        return url

    @staticmethod
    def _chunks(xs: Sequence[str], bs: int) -> List[List[str]]:
        bs = max(1, int(bs))
        return [list(xs[i : i + bs]) for i in range(0, len(xs), bs)]

    def _one_batched_completion(
        self,
        *,
        base_url: str,
        prompts: Sequence[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        prompt_list = [str(p) for p in prompts]
        if not prompt_list:
            return []

        payload = {
            'model': self.model_name,
            'prompt': prompt_list,
            'max_tokens': int(max_new_tokens),
            'temperature': float(temperature),
            'top_p': float(top_p),
            'n': 1,
            'stream': False,
        }
        if stop:
            payload['stop'] = [str(s) for s in stop if str(s)]

        req = urlrequest.Request(
            url=f'{base_url}/v1/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except urlerror.HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            raise RuntimeError(f'student vLLM HTTPError {e.code} from {base_url}: {body}') from e
        except urlerror.URLError as e:
            raise RuntimeError(f'student vLLM URLError from {base_url}: {e}') from e

        choices = list(data.get('choices', []) or [])
        if len(choices) != len(prompt_list):
            raise RuntimeError(
                f'student vLLM batched response size mismatch from {base_url}: '
                f'expected {len(prompt_list)} choices, got {len(choices)}'
            )

        outs: List[Optional[str]] = [None] * len(prompt_list)
        for i, choice in enumerate(choices):
            idx = choice.get('index', i)
            if not isinstance(idx, int) or idx < 0 or idx >= len(prompt_list):
                raise RuntimeError(
                    f'student vLLM returned invalid choice index {idx} for batch of size {len(prompt_list)}'
                )
            outs[idx] = str(choice.get('text', ''))

        if any(x is None for x in outs):
            raise RuntimeError('student vLLM batched response had missing outputs after index mapping')

        return [str(x) for x in outs]

    def _generate_to_one_server(
        self,
        *,
        base_url: str,
        prompts: Sequence[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        prompts = [str(p) for p in prompts]
        if not prompts:
            return []
        request_batch_size = max(1, int(batch_size))
        outs: List[str] = []
        for chunk in self._chunks(prompts, request_batch_size):
            outs.extend(
                self._one_batched_completion(
                    base_url=base_url,
                    prompts=chunk,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
            )
        return outs

    def _generate_local_only(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        prompts = [str(p) for p in prompts]
        if not prompts:
            return []

        request_batch_size = max(1, int(batch_size))
        request_chunks = self._chunks(prompts, request_batch_size)
        results: List[Optional[List[str]]] = [None] * len(request_chunks)
        max_parallel = max(1, len(self.base_urls))

        for round_start in range(0, len(request_chunks), max_parallel):
            round_chunks = request_chunks[round_start : round_start + max_parallel]
            with ThreadPoolExecutor(max_workers=len(round_chunks)) as ex:
                futs = []
                for offset, chunk in enumerate(round_chunks):
                    chunk_idx = round_start + offset
                    base_url = self._next_url()
                    fut = ex.submit(
                        self._one_batched_completion,
                        base_url=base_url,
                        prompts=chunk,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                    )
                    futs.append((chunk_idx, fut))

                for chunk_idx, fut in futs:
                    results[chunk_idx] = fut.result()

        flat: List[str] = []
        for chunk_out in results:
            assert chunk_out is not None
            flat.extend(chunk_out)
        return flat

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        prompt_list = [str(p) for p in prompts]
        if not prompt_list:
            return []

        if (not _is_dist()) or self._dist_group is None or len(self._group_ranks) <= 1:
            if _is_dist():
                return self._generate_to_one_server(
                    base_url=self._assigned_base_url,
                    prompts=prompt_list,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    batch_size=batch_size,
                    stop=stop,
                )
            return self._generate_local_only(
                prompt_list,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                stop=stop,
            )

        my_rank = _rank()
        gathered: List[Optional[Dict[str, Any]]] = [None] * len(self._group_ranks)
        payload = {'rank': my_rank, 'prompts': prompt_list}
        torch.distributed.all_gather_object(gathered, payload, group=self._dist_group)

        result_map = None
        if my_rank == self._group_leader_rank:
            ordered = sorted((x for x in gathered if x is not None), key=lambda x: int(x['rank']))
            merged_prompts: List[str] = []
            spans: Dict[int, tuple[int, int]] = {}
            for item in ordered:
                rr = int(item['rank'])
                rr_prompts = [str(p) for p in item.get('prompts', [])]
                start = len(merged_prompts)
                merged_prompts.extend(rr_prompts)
                spans[rr] = (start, len(merged_prompts))

            merged_outs = self._generate_to_one_server(
                base_url=self._assigned_base_url,
                prompts=merged_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                stop=stop,
            )
            result_map = {rr: merged_outs[a:b] for rr, (a, b) in spans.items()}

        obj_list = [result_map]
        torch.distributed.broadcast_object_list(obj_list, src=self._group_leader_rank, group=self._dist_group)
        recv_map = obj_list[0]
        if not isinstance(recv_map, dict) or my_rank not in recv_map:
            raise RuntimeError(
                f'remote student aggregation broadcast missing outputs for rank={my_rank} '
                f'group_ranks={self._group_ranks}'
            )
        my_outs = list(recv_map[my_rank])
        if len(my_outs) != len(prompt_list):
            raise RuntimeError(
                f'remote student aggregation size mismatch for rank={my_rank}: '
                f'expected {len(prompt_list)} outputs, got {len(my_outs)}'
            )
        return [str(x) for x in my_outs]


def _write_deepspeed_config(cfg: Dict[str, Any], out_dir: Path) -> Path:
    tcfg = cfg.get('training', {}) or {}
    dscfg = (tcfg.get('deepspeed', {}) or {})

    zero_stage = int(dscfg.get('zero_stage', 3))
    off_opt = str(dscfg.get('offload_optimizer_device', 'none')).lower()
    off_param = str(dscfg.get('offload_param_device', 'none')).lower()
    bf16 = bool(tcfg.get('bf16', True))

    zero_opt: Dict[str, Any] = {'stage': zero_stage}

    if off_opt in ('cpu', 'nvme'):
        zero_opt['offload_optimizer'] = {'device': off_opt, 'pin_memory': True if off_opt == 'cpu' else False}
    if off_param in ('cpu', 'nvme'):
        zero_opt['offload_param'] = {'device': off_param, 'pin_memory': True if off_param == 'cpu' else False}

    if zero_stage == 3:
        zero_opt.update(
            {
                'stage3_gather_16bit_weights_on_model_save': True,
                'stage3_max_live_parameters': 1e9,
                'stage3_max_reuse_distance': 1e9,
                'stage3_prefetch_bucket_size': 5e7,
                'stage3_param_persistence_threshold': 1e6,
            }
        )

    ds: Dict[str, Any] = {
        'train_micro_batch_size_per_gpu': 'auto',
        'gradient_accumulation_steps': 'auto',
        'train_batch_size': 'auto',
        'steps_per_print': 2000,
        'wall_clock_breakdown': False,
        'zero_optimization': zero_opt,
        'zero_allow_untested_optimizer': True,
    }

    if bf16:
        ds['bf16'] = {'enabled': True}
        ds['fp16'] = {'enabled': False}
    else:
        ds['bf16'] = {'enabled': False}
        ds['fp16'] = {'enabled': True}

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'deepspeed_config.json'
    tmp = out_dir / f'.deepspeed_config.tmp.{os.getpid()}'
    tmp.write_text(json.dumps(ds, indent=2), encoding='utf-8')
    os.replace(tmp, path)
    return path


class GracefulStopCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if _STOP_REQUESTED:
            if _rank() == 0:
                print('[learner/grpo] SIGUSR1 received: requesting save+stop')
            control.should_save = True
            control.should_training_stop = True
        return control


def _rewrite_adapter_config_base_model_id(save_dir: Path, canonical_base_model_id: Optional[str]) -> None:
    if not canonical_base_model_id:
        return
    cfg_path = Path(save_dir) / 'adapter_config.json'
    if not cfg_path.exists():
        return
    try:
        data = json.loads(cfg_path.read_text(encoding='utf-8'))
        data['base_model_name_or_path'] = str(canonical_base_model_id)
        cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f'failed to rewrite {cfg_path} with canonical base model id {canonical_base_model_id}: {e}')


def save_peft_adapter_zero3_safe(model: Any, save_dir: Path, canonical_base_model_id: Optional[str] = None) -> None:
    if not _is_dist():
        model.save_pretrained(str(save_dir))
        _rewrite_adapter_config_base_model_id(save_dir, canonical_base_model_id)
        return

    import deepspeed  # noqa: F401
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    from peft.utils.save_and_load import get_peft_model_state_dict

    rank = _rank()
    adapter_name = getattr(model, 'active_adapter', 'default')

    named = [(n, p) for (n, p) in model.named_parameters() if getattr(p, 'requires_grad', False)]
    gathered_full: Dict[str, torch.Tensor] = {}

    for name, param in named:
        status = getattr(param, 'ds_status', None)
        needs_gather = hasattr(param, 'ds_id') and status == ZeroParamStatus.NOT_AVAILABLE
        if needs_gather:
            with zero.GatheredParameters([param], modifier_rank=0):
                if rank == 0:
                    gathered_full[name] = param.detach().cpu().clone()
        else:
            if rank == 0:
                gathered_full[name] = param.detach().cpu().clone()

    if rank != 0:
        _dist_barrier()
        return

    adapter_sd = get_peft_model_state_dict(model, state_dict=gathered_full, adapter_name=adapter_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.peft_config[adapter_name].save_pretrained(str(save_dir))
    _rewrite_adapter_config_base_model_id(save_dir, canonical_base_model_id)
    torch.save(adapter_sd, save_dir / 'adapter_model.bin')
    _dist_barrier()


class LiveRolloutAdapterCallback(TrainerCallback):
    def __init__(
        self,
        coord: Coordinator,
        live_root: Path,
        steps_per_generation: int,
        keep_last: int,
        canonical_base_model_id: Optional[str],
        step_offset: int = 0,
    ):
        self.coord = coord
        self.live_root = Path(live_root)
        self.steps_per_generation = max(1, int(steps_per_generation))
        self.keep_last = max(1, int(keep_last))
        self.canonical_base_model_id = str(canonical_base_model_id) if canonical_base_model_id else None
        self.step_offset = int(step_offset)
        self.last_published_step: Optional[int] = None

    def _adapter_dir_for_step(self, step: int) -> Path:
        return self.live_root / f'step_{int(step):08d}'

    def _prune_old(self) -> None:
        if self.keep_last <= 0:
            return
        dirs = sorted(
            [p for p in self.live_root.glob('step_*') if p.is_dir()],
            key=lambda p: p.name,
        )
        for old in dirs[:-self.keep_last]:
            try:
                import shutil

                shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass

    def _publish(self, model_to_save: Any, step: int) -> None:
        step = int(step)
        out_dir = self._adapter_dir_for_step(step)
        tmp_dir = self.live_root / f'.{out_dir.name}.tmp_{os.getpid()}'

        if _rank() == 0:
            ensure_dir(self.live_root)
            if tmp_dir.exists():
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)
            ensure_dir(tmp_dir)

        save_peft_adapter_zero3_safe(
            model_to_save,
            tmp_dir,
            canonical_base_model_id=self.canonical_base_model_id,
        )

        if _rank() == 0:
            import shutil

            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            tmp_dir.rename(out_dir)
            self.coord.update_state(
                teacher_live_adapter_path=str(out_dir),
                teacher_live_adapter_name=out_dir.name,
                teacher_live_step=step,
            )
            self._prune_old()
            print(f'[learner/grpo] published live rollout adapter step={step} -> {out_dir}')

        _dist_barrier()
        self.last_published_step = step

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get('model', None)
        model_to_save = getattr(model, 'module', model)
        if model_to_save is None:
            return control
        step = self.step_offset + int(state.global_step or 0)
        if self.last_published_step != step:
            self._publish(model_to_save, step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        rel_step = int(state.global_step or 0)
        if rel_step <= 0 or (rel_step % self.steps_per_generation) != 0:
            return control
        step = self.step_offset + rel_step
        if self.last_published_step == step:
            return control
        model = kwargs.get('model', None)
        model_to_save = getattr(model, 'module', model)
        if model_to_save is None:
            return control
        self._publish(model_to_save, step)
        return control


class RoundSaverCallback(TrainerCallback):
    def __init__(
        self,
        coord: Coordinator,
        adapter_root: Path,
        steps_per_round: int,
        max_rounds: int,
        start_round: int = 0,
        round_offset: int = 0,
        canonical_base_model_id: Optional[str] = None,
    ):
        self.coord = coord
        self.adapter_root = Path(adapter_root)
        self.steps_per_round = max(1, int(steps_per_round))
        self.max_rounds = int(max_rounds)
        self.last_saved_round = int(start_round)
        self.round_offset = int(round_offset)
        self.canonical_base_model_id = str(canonical_base_model_id) if canonical_base_model_id else None

    def _adapter_dir_for_round(self, r: int) -> Path:
        return self.adapter_root / f'round_{r}' / 'final'

    def on_step_end(self, args, state, control, **kwargs):
        completed_rounds = self.round_offset + (int(state.global_step or 0) // self.steps_per_round)
        if completed_rounds <= self.last_saved_round:
            return control
        completed_rounds = min(completed_rounds, self.max_rounds)

        model = kwargs.get('model', None)
        model_to_save = getattr(model, 'module', model)

        if model_to_save is None:
            if _rank() == 0:
                print('[RoundSaverCallback] WARNING: model is None; cannot save adapter')
            self.last_saved_round = completed_rounds
            return control

        out_dir = self._adapter_dir_for_round(completed_rounds)
        tmp_dir = out_dir.parent / f'{out_dir.name}.tmp_{os.getpid()}'

        if _rank() == 0:
            ensure_dir(tmp_dir.parent)
            if tmp_dir.exists():
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)
            ensure_dir(tmp_dir)

        save_peft_adapter_zero3_safe(
            model_to_save,
            tmp_dir,
            canonical_base_model_id=self.canonical_base_model_id,
        )

        if _rank() == 0:
            import shutil

            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            tmp_dir.rename(out_dir)

            self.coord.update_state(
                round=completed_rounds,
                teacher_adapter=str(out_dir),
                phase=('done' if completed_rounds >= self.max_rounds else 'train'),
            )
            print(f'[learner/grpo] saved adapter for round {completed_rounds} -> {out_dir}')

        _dist_barrier()
        self.last_saved_round = completed_rounds
        return control


@dataclass
class _MetricAccum:
    delta_sum: float = 0.0
    leak_sum: float = 0.0
    hint_len_sum: float = 0.0
    reward_sum: float = 0.0
    reward_student_time_sum: float = 0.0
    reward_total_time_sum: float = 0.0

    teacher_new_sum: float = 0.0
    teacher_total_sum: float = 0.0
    student_new_sum: float = 0.0
    student_total_sum: float = 0.0

    teacher_new_max: float = 0.0
    teacher_total_max: float = 0.0
    student_new_max: float = 0.0
    student_total_max: float = 0.0

    teacher_trunc_count: int = 0
    student_trunc_count: int = 0
    n: int = 0

    def add(
        self,
        *,
        delta: float,
        leak: float,
        hint_len: float,
        reward: float,
        reward_student_time: float,
        reward_total_time: float,
        teacher_new: float,
        teacher_total: float,
        student_new: float,
        student_total: float,
        teacher_trunc: int,
        student_trunc: int,
    ) -> None:
        self.delta_sum += float(delta)
        self.leak_sum += float(leak)
        self.hint_len_sum += float(hint_len)
        self.reward_sum += float(reward)
        self.reward_student_time_sum += float(reward_student_time)
        self.reward_total_time_sum += float(reward_total_time)

        self.teacher_new_sum += float(teacher_new)
        self.teacher_total_sum += float(teacher_total)
        self.student_new_sum += float(student_new)
        self.student_total_sum += float(student_total)

        self.teacher_new_max = max(self.teacher_new_max, float(teacher_new))
        self.teacher_total_max = max(self.teacher_total_max, float(teacher_total))
        self.student_new_max = max(self.student_new_max, float(student_new))
        self.student_total_max = max(self.student_total_max, float(student_total))

        self.teacher_trunc_count += int(teacher_trunc)
        self.student_trunc_count += int(student_trunc)
        self.n += 1

    def snapshot_and_reset(self):
        out = (
            self.delta_sum,
            self.leak_sum,
            self.hint_len_sum,
            self.reward_sum,
            self.reward_student_time_sum,
            self.reward_total_time_sum,
            self.teacher_new_sum,
            self.teacher_total_sum,
            self.student_new_sum,
            self.student_total_sum,
            self.teacher_new_max,
            self.teacher_total_max,
            self.student_new_max,
            self.student_total_max,
            self.teacher_trunc_count,
            self.student_trunc_count,
            self.n,
        )
        self.delta_sum = self.leak_sum = self.hint_len_sum = self.reward_sum = 0.0
        self.reward_student_time_sum = self.reward_total_time_sum = 0.0
        self.teacher_new_sum = self.teacher_total_sum = 0.0
        self.student_new_sum = self.student_total_sum = 0.0
        self.teacher_new_max = self.teacher_total_max = 0.0
        self.student_new_max = self.student_total_max = 0.0
        self.teacher_trunc_count = self.student_trunc_count = 0
        self.n = 0
        return out


@dataclass
class _StepPerfAccum:
    step_time_sum: float = 0.0
    n_steps: int = 0

    def add(self, dt: float) -> None:
        self.step_time_sum += float(dt)
        self.n_steps += 1

    def snapshot_and_reset(self) -> tuple[float, int]:
        out = (self.step_time_sum, self.n_steps)
        self.step_time_sum = 0.0
        self.n_steps = 0
        return out


METRICS_ACCUM = _MetricAccum()
STEP_ACCUM = _StepPerfAccum()


class TrainMetricsEMACallback(TrainerCallback):
    def __init__(self, coord: Coordinator, ema_alpha: float, step_offset: int = 0):
        self.coord = coord
        self.alpha = float(ema_alpha)
        self.step_offset = int(step_offset)
        self.ema_delta: Optional[float] = None
        self.ema_leak: Optional[float] = None
        self.ema_hint_len: Optional[float] = None
        self.ema_reward: Optional[float] = None
        self.ema_reward_student_s: Optional[float] = None
        self.ema_reward_total_s: Optional[float] = None
        self.ema_step_time_s: Optional[float] = None
        self.ema_teacher_new: Optional[float] = None
        self.ema_teacher_total: Optional[float] = None
        self.ema_student_new: Optional[float] = None
        self.ema_student_total: Optional[float] = None
        self.ema_teacher_trunc_rate: Optional[float] = None
        self.ema_student_trunc_rate: Optional[float] = None
        self._t_step0: Optional[float] = None

    def _ema_update(self, old: Optional[float], x: float) -> float:
        if old is None:
            return float(x)
        return (1.0 - self.alpha) * float(old) + self.alpha * float(x)

    def _flush(self, global_step: int) -> None:
        (
            dsum,
            lsum,
            hsum,
            rsum,
            rs_s,
            rt_s,
            teacher_new_sum,
            teacher_total_sum,
            student_new_sum,
            student_total_sum,
            teacher_new_max,
            teacher_total_max,
            student_new_max,
            student_total_max,
            teacher_trunc_count,
            student_trunc_count,
            n,
        ) = METRICS_ACCUM.snapshot_and_reset()
        step_sum, step_n = STEP_ACCUM.snapshot_and_reset()

        dev = torch.device('cuda', _local_rank()) if torch.cuda.is_available() else torch.device('cpu')
        sum_t = torch.tensor(
            [
                dsum, lsum, hsum, rsum, rs_s, rt_s,
                teacher_new_sum, teacher_total_sum,
                student_new_sum, student_total_sum,
                float(teacher_trunc_count), float(student_trunc_count),
                float(n), step_sum, float(step_n),
            ],
            device=dev,
            dtype=torch.float64,
        )
        max_t = torch.tensor(
            [teacher_new_max, teacher_total_max, student_new_max, student_total_max],
            device=dev,
            dtype=torch.float64,
        )
        if _is_dist():
            torch.distributed.all_reduce(sum_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(max_t, op=torch.distributed.ReduceOp.MAX)

        n_tot = int(sum_t[12].item())
        step_n_tot = int(sum_t[14].item())
        if n_tot <= 0 and step_n_tot <= 0:
            return

        if n_tot > 0:
            delta_mean = float((sum_t[0] / sum_t[12]).item())
            leak_mean = float((sum_t[1] / sum_t[12]).item())
            hint_len_mean = float((sum_t[2] / sum_t[12]).item())
            reward_mean = float((sum_t[3] / sum_t[12]).item())
            rs_mean = float((sum_t[4] / sum_t[12]).item())
            rt_mean = float((sum_t[5] / sum_t[12]).item())
            teacher_new_mean = float((sum_t[6] / sum_t[12]).item())
            teacher_total_mean = float((sum_t[7] / sum_t[12]).item())
            student_new_mean = float((sum_t[8] / sum_t[12]).item())
            student_total_mean = float((sum_t[9] / sum_t[12]).item())
            teacher_trunc_rate = float((sum_t[10] / sum_t[12]).item())
            student_trunc_rate = float((sum_t[11] / sum_t[12]).item())
        else:
            delta_mean = leak_mean = hint_len_mean = reward_mean = rs_mean = rt_mean = float('nan')
            teacher_new_mean = teacher_total_mean = student_new_mean = student_total_mean = float('nan')
            teacher_trunc_rate = student_trunc_rate = float('nan')

        step_time_mean = float((sum_t[13] / sum_t[14]).item()) if step_n_tot > 0 else float('nan')
        steps_per_s = (1.0 / step_time_mean) if step_time_mean and step_time_mean > 0 else float('nan')
        abs_step = self.step_offset + int(global_step)

        if _rank() == 0:
            self.ema_delta = self._ema_update(self.ema_delta, delta_mean)
            self.ema_leak = self._ema_update(self.ema_leak, leak_mean)
            self.ema_hint_len = self._ema_update(self.ema_hint_len, hint_len_mean)
            self.ema_reward = self._ema_update(self.ema_reward, reward_mean)
            self.ema_reward_student_s = self._ema_update(self.ema_reward_student_s, rs_mean)
            self.ema_reward_total_s = self._ema_update(self.ema_reward_total_s, rt_mean)
            self.ema_step_time_s = self._ema_update(self.ema_step_time_s, step_time_mean)
            self.ema_teacher_new = self._ema_update(self.ema_teacher_new, teacher_new_mean)
            self.ema_teacher_total = self._ema_update(self.ema_teacher_total, teacher_total_mean)
            self.ema_student_new = self._ema_update(self.ema_student_new, student_new_mean)
            self.ema_student_total = self._ema_update(self.ema_student_total, student_total_mean)
            self.ema_teacher_trunc_rate = self._ema_update(self.ema_teacher_trunc_rate, teacher_trunc_rate)
            self.ema_student_trunc_rate = self._ema_update(self.ema_student_trunc_rate, student_trunc_rate)

            self.coord.update_state(
                train_metrics_step=int(abs_step),
                train_ema_teaching_delta=float(self.ema_delta),
                train_ema_leak_rate=float(self.ema_leak),
                train_ema_hint_len=float(self.ema_hint_len),
                train_ema_reward=float(self.ema_reward),
                train_ema_reward_student_s=float(self.ema_reward_student_s),
                train_ema_reward_total_s=float(self.ema_reward_total_s),
                train_ema_step_time_s=float(self.ema_step_time_s),
                train_steps_per_s=float(steps_per_s),
                train_ema_teacher_new_tokens=float(self.ema_teacher_new),
                train_ema_teacher_total_tokens=float(self.ema_teacher_total),
                train_teacher_new_tokens_max=float(max_t[0].item()),
                train_teacher_total_tokens_max=float(max_t[1].item()),
                train_ema_teacher_truncation_rate=float(self.ema_teacher_trunc_rate),
                train_ema_student_new_tokens=float(self.ema_student_new),
                train_ema_student_total_tokens=float(self.ema_student_total),
                train_student_new_tokens_max=float(max_t[2].item()),
                train_student_total_tokens_max=float(max_t[3].item()),
                train_ema_student_truncation_rate=float(self.ema_student_trunc_rate),
            )

    def on_step_begin(self, args, state, control, **kwargs):
        self._t_step0 = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._t_step0 is not None:
            STEP_ACCUM.add(float(time.perf_counter() - float(self._t_step0)))

        ls = int(getattr(args, 'logging_steps', 10) or 10)
        step = int(state.global_step or 0)
        if step > 0 and (step % ls == 0):
            self._flush(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        step = int(state.global_step or 0)
        self._flush(step)
        return control


def build_grpo_dataset(cfg: Dict[str, Any], teacher_tok) -> Dataset:
    buf = list(read_jsonl(cfg['paths']['replay_buffer_path']))
    if not buf:
        raise SystemExit(f"Replay buffer empty: {cfg['paths']['replay_buffer_path']}")

    grpo_cfg = _get_grpo_cfg(cfg)
    failures_per_ex = int(grpo_cfg.get('failures_per_example', 1))
    enable_thinking = bool((cfg.get('models', {}) or {}).get('enable_thinking', True))

    rows: List[Dict[str, Any]] = []
    for ex in buf:
        q = ex['question']
        a = str(ex['answer'])
        p_s_hat = float(ex.get('p_s_hat', 0.0))
        failures = list(ex.get('failures', []))[: max(1, failures_per_ex)]
        if not failures:
            continue
        for y_fail in failures:
            msgs = build_teacher_hint_messages(q, y_fail, cfg)
            prompt = maybe_apply_chat_template(teacher_tok, msgs, enable_thinking=enable_thinking)
            rows.append(
                {
                    'prompt': prompt,
                    'question': q,
                    'student_failure': y_fail,
                    'answer': a,
                    'p_s_hat': p_s_hat,
                }
            )

    if not rows:
        raise SystemExit('GRPO dataset is empty after expansion; check replay buffer and failures_per_example.')
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--local_rank', type=int, default=None)
    args, _unknown = ap.parse_known_args()

    if args.local_rank is not None:
        os.environ.setdefault('LOCAL_RANK', str(args.local_rank))

    _init_distributed_early()

    cfg = load_config(args.config)
    use_trl_vllm_colocate = _use_trl_vllm_colocate(cfg)
    seed_everything(int(cfg.get('training', {}).get('seed', 0)) + 2026)

    coord = Coordinator(cfg['paths']['coord_dir'])
    st = coord.read_state()
    max_rounds = int(st.get('max_rounds', cfg.get('persistent', {}).get('rounds', 1)))
    start_round = int(st.get('round', 0) or 0)
    phase = str(st.get('phase', 'train')).lower()

    if phase == 'done' or start_round >= max_rounds:
        if _rank() == 0:
            print(f'[learner/grpo] coord says done (round={start_round}/{max_rounds}); exiting.')
        return

    adapter_root = Path(cfg.get('persistent', {}).get('adapter_root', 'outputs/teacher_lora_persistent'))
    live_adapter_root = Path(cfg.get('persistent', {}).get('live_adapter_root', 'outputs/teacher_lora_live'))
    ensure_dir(adapter_root)
    ensure_dir(live_adapter_root)

    teacher_model_ref = os.environ.get(
        'RTT_TEACHER_MODEL_ID',
        os.environ.get('RTT_TEACHER_MODEL', cfg['models']['teacher_model_id']),
    )
    student_model_ref = os.environ.get(
        'RTT_STUDENT_MODEL_ID',
        os.environ.get('RTT_STUDENT_MODEL', cfg['models']['student_model_id']),
    )
    canonical_teacher_model_id = str(cfg['models']['teacher_model_id'])

    tok_kwargs: Dict[str, Any] = {'trust_remote_code': True}
    if Path(teacher_model_ref).exists():
        tok_kwargs['local_files_only'] = True

    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_ref, **tok_kwargs)
    if teacher_tok.pad_token_id is None:
        teacher_tok.pad_token = teacher_tok.eos_token
    teacher_tok.padding_side = 'left'

    enable_thinking = bool((cfg.get('models', {}) or {}).get('enable_thinking', True))
    train_ds = build_grpo_dataset(cfg, teacher_tok)

    grpo_cfg = _get_grpo_cfg(cfg)
    reward_student_device = str(grpo_cfg.get('reward_student_device', 'cuda')).lower()
    steps_per_generation_cfg = grpo_cfg.get('steps_per_generation', None)

    stud_device = _device() if reward_student_device == 'cuda' else 'cpu'

    student_server_urls = [
        x.strip()
        for x in (os.environ.get('RTT_STUDENT_VLLM_BASE_URLS', '') or '').split(',')
        if x.strip()
    ]
    student_server_model_name = (
        os.environ.get('RTT_STUDENT_VLLM_MODEL_NAME', '') or cfg['models']['student_model_id']
    ).strip()
    student_server_timeout = float(os.environ.get('RTT_STUDENT_VLLM_TIMEOUT_S', '120') or '120')
    use_remote_student = len(student_server_urls) > 0

    teacher_rollout_urls = [
        x.strip()
        for x in (os.environ.get('RTT_TEACHER_VLLM_BASE_URLS', '') or '').split(',')
        if x.strip()
    ]
    teacher_rollout_timeout = float(os.environ.get('RTT_TEACHER_VLLM_TIMEOUT_S', '180') or '180')
    teacher_rollout_max_retries = int(os.environ.get('RTT_TEACHER_VLLM_MAX_RETRIES', '8') or '8')
    teacher_rollout_retry_backoff = float(os.environ.get('RTT_TEACHER_VLLM_RETRY_BACKOFF_S', '5') or '5')
    teacher_rollout_max_logprobs = int(os.environ.get('RTT_TEACHER_VLLM_MAX_LOGPROBS', '1') or '1')
    teacher_rollout_extra_body = {}
    try:
        teacher_rollout_extra_body = json.loads(os.environ.get('RTT_TEACHER_VLLM_EXTRA_BODY', '{}') or '{}')
    except Exception:
        teacher_rollout_extra_body = {}
    use_remote_teacher_rollout = (len(teacher_rollout_urls) > 0) and (not use_trl_vllm_colocate)
    if not use_trl_vllm_colocate and not use_remote_teacher_rollout:
        raise RuntimeError(
            'Remote teacher rollout is required unless resources.placement_mode=trl_colocate. '
            'Launch through slurm/rl_learner_persistent.slurm or export teacher rollout URLs.'
        )

    student_tok = _load_tokenizer_only(student_model_ref)
    student_lm: Optional[LM] = None
    student_remote: Optional[_RemoteStudentVLLMClient] = None
    teacher_remote: Optional[RemoteVLLMCompletionsClient] = None

    if use_remote_student:
        student_remote = _RemoteStudentVLLMClient(
            base_urls=student_server_urls,
            model_name=student_server_model_name,
            timeout_s=student_server_timeout,
            aggregate_ranks_per_request=int(grpo_cfg.get('reward_student_aggregate_ranks_per_request', 1)),
        )
        if _rank() == 0:
            print(
                f'[learner/grpo] remote student vLLM enabled: '
                f'urls={student_server_urls} model={student_server_model_name} '
                f"reward_student_batch_size={int(grpo_cfg.get('reward_student_batch_size', 8))} "
                f"reward_student_aggregate_ranks_per_request={int(grpo_cfg.get('reward_student_aggregate_ranks_per_request', 1))}"
            )
    else:
        student_load_4bit = bool(cfg['models'].get('student_load_in_4bit', False))
        if stud_device == 'cpu' and student_load_4bit:
            if _rank() == 0:
                print('[learner/grpo] reward_student_device=cpu => disabling student 4-bit (bnb needs CUDA).')
            student_load_4bit = False

        student_lm = _load_student_lm_safe(
            student_model_ref,
            stud_device,
            load_in_4bit=student_load_4bit,
            bf16=bool(cfg.get('training', {}).get('bf16', True)),
        )
        student_tok = student_lm.tokenizer

    if use_remote_teacher_rollout:
        teacher_rollout_cfg = cfg.get('teacher_rollout', {}) or {}
        teacher_remote = RemoteVLLMCompletionsClient(
            base_urls=teacher_rollout_urls,
            timeout_s=teacher_rollout_timeout,
            max_retries=teacher_rollout_max_retries,
            retry_backoff_s=teacher_rollout_retry_backoff,
            request_logprobs=teacher_rollout_max_logprobs,
            extra_body=teacher_rollout_extra_body,
            max_requests_per_server=int(teacher_rollout_cfg.get('max_requests_per_server', 4)),
        )
        if _rank() == 0:
            print(
                f'[learner/grpo] remote teacher rollout enabled: '
                f'urls={teacher_rollout_urls} timeout_s={teacher_rollout_timeout} '
                f"max_requests_per_server={int(teacher_rollout_cfg.get('max_requests_per_server', 4))}"
            )

    beta_leak = float(cfg.get('reward', {}).get('beta_leak', 1.0))
    gamma_len = float(cfg.get('reward', {}).get('gamma_len', 0.0))
    beta_bad_hint_format = float(cfg.get('reward', {}).get('beta_bad_hint_format', 2.0))
    beta_overlong_hint = float(cfg.get('reward', {}).get('beta_overlong_hint', 2.0))

    stud_max = int(cfg['caching']['max_new_tokens_student'])
    stud_temp = float(cfg['caching']['temperature'])
    stud_top_p = float(cfg['caching']['top_p'])
    reward_student_batch_size = int(grpo_cfg.get('reward_student_batch_size', 8))

    tcfg = cfg.get('training', {}) or {}
    per_device_bs = int(tcfg.get('batch_size', 1))
    grad_accum = int(tcfg.get('grad_accum', 1))
    epochs_per_round = int(tcfg.get('epochs_per_round', 1))

    num_generations = int(grpo_cfg.get('num_generations', 8))
    global_batch = _world_size() * per_device_bs * grad_accum
    if global_batch % num_generations != 0:
        divisors = [d for d in range(1, num_generations + 1) if global_batch % d == 0]
        new_ng = max(divisors) if divisors else 1
        if _rank() == 0:
            print(
                f'[learner/grpo] WARNING: global_batch={global_batch} not divisible by num_generations={num_generations}. '
                f'Using num_generations={new_ng}.'
            )
        num_generations = new_ng

    dataset_len = len(train_ds)
    steps_per_epoch = max(1, int(math.ceil(dataset_len / max(1, global_batch))))
    steps_per_round = steps_per_epoch * max(1, epochs_per_round)

    total_round_aligned_steps = int(steps_per_round * max_rounds)
    cfg_max_steps = int(tcfg.get('max_steps', 0) or 0)
    if cfg_max_steps > 0:
        if cfg_max_steps != total_round_aligned_steps:
            raise ValueError(
                'training.max_steps must equal steps_per_round * persistent.rounds '
                'so persistent round checkpoints stay aligned: '
                f'got max_steps={cfg_max_steps}, '
                f'steps_per_round={steps_per_round}, '
                f'persistent.rounds={max_rounds}, '
                f'expected={total_round_aligned_steps}. '
                'Either set training.max_steps=0 or set it to the expected value.'
            )
        max_steps = cfg_max_steps
    else:
        max_steps = total_round_aligned_steps

    out_dir = Path(cfg['paths']['outputs_dir']) / 'grpo_teacher_run'
    ensure_dir(out_dir)

    last_ckpt = get_last_checkpoint(str(out_dir))
    ckpt_global_step = _read_trainer_checkpoint_global_step(last_ckpt)
    resume_ctx = _read_resume_context(out_dir)

    coord_adapter = st.get('teacher_adapter') or cfg.get('models', {}).get('teacher_init_adapter', None)

    if last_ckpt is not None:
        step_offset = int(resume_ctx.get('step_offset', 0) or 0)
        round_offset = int(resume_ctx.get('start_round', step_offset // max(1, steps_per_round)) or 0)
        ckpt_global_step_abs = (
            int(ckpt_global_step) + int(step_offset)
            if ckpt_global_step is not None
            else int(step_offset)
        )
        expected_min_step = int(start_round * steps_per_round)
        if ckpt_global_step is not None and ckpt_global_step_abs < expected_min_step:
            raise RuntimeError(
                f'Coordinator says round={start_round}, which implies at least step {expected_min_step}, '
                f'but latest trainer checkpoint only has effective global_step={ckpt_global_step_abs} '
                f'(checkpoint step={ckpt_global_step}, step_offset={step_offset}). '
                'Refusing ambiguous resume. Either resume from a newer trainer checkpoint or remove the stale checkpoint '
                'so adapter-based resume can be used.'
            )

        resume_round_from_ckpt = (
            min(max_rounds, int(ckpt_global_step_abs // steps_per_round))
            if ckpt_global_step is not None
            else start_round
        )

        load_adapter_path = None
        max_steps_this_run = int(
            resume_ctx.get(
                'max_steps_this_run',
                (steps_per_round * max(0, max_rounds - round_offset)) if step_offset > 0 else max_steps,
            )
            or 0
        )
        if max_steps_this_run <= 0:
            max_steps_this_run = int((steps_per_round * max(0, max_rounds - round_offset)) if step_offset > 0 else max_steps)
        callback_start_round = int(resume_round_from_ckpt)
    else:
        if start_round > 0 and not coord_adapter:
            raise RuntimeError(
                f'Coordinator says round={start_round}, but no trainer checkpoint and no teacher_adapter are available.'
            )

        load_adapter_path = str(coord_adapter) if coord_adapter else None
        round_offset = int(start_round)
        step_offset = int(round_offset * steps_per_round)
        max_steps_this_run = int(steps_per_round * max(0, max_rounds - start_round)) if start_round > 0 else int(max_steps)
        callback_start_round = int(start_round)

        if _rank() == 0:
            _write_resume_context(
                out_dir,
                step_offset=step_offset,
                start_round=round_offset,
                steps_per_round=steps_per_round,
                max_steps_this_run=max_steps_this_run,
            )
        _dist_barrier()

    teacher_model, teacher_model_is_peft = _load_teacher_policy_model(
        teacher_model_ref,
        adapter_path=load_adapter_path,
        bf16=bool(tcfg.get('bf16', True)),
    )

    beta_kl = float(grpo_cfg.get('beta_kl', 0.0))

    hint_token_cap = int(cfg['prompting']['hint_max_new_tokens'])
    hint_total_cap = int(cfg['prompting'].get('hint_max_total_tokens', hint_token_cap))
    hint_temp = float(cfg['prompting']['hint_temperature'])
    hint_top_p = float(cfg['prompting']['hint_top_p'])

    @torch.inference_mode()
    def teaching_reward_func(
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Optional[Sequence[Sequence[int]]] = None,
        question: Optional[Sequence[str]] = None,
        answer: Optional[Sequence[Any]] = None,
        p_s_hat: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> List[float]:
        qs = list(question or [])
        ans = list(answer or [])
        ps = list(p_s_hat or [])

        n = len(completions)
        if n == 0:
            return []

        golds: List[str] = []
        bases: List[float] = []
        leaks: List[float] = []
        hint_lens: List[float] = []
        bad_format_flags: List[float] = []
        overlong_flags: List[float] = []
        s_prompts: List[str] = []
        teacher_new_lens: List[int] = []
        teacher_total_lens: List[int] = []
        student_prompt_lens: List[int] = []

        for i, comp in enumerate(completions):
            raw_completion = _extract_text_completion(comp)
            extracted_hint = extract_final_hint_text(raw_completion)

            prompt_text = str(prompts[i]) if i < len(prompts) else ''
            teacher_prompt_len = _token_count(teacher_tok, prompt_text)
            if completion_ids is not None and i < len(completion_ids) and completion_ids[i] is not None:
                teacher_new_len = int(len(completion_ids[i]))
            else:
                teacher_new_len = _token_count(teacher_tok, raw_completion)
            teacher_total_len = teacher_prompt_len + teacher_new_len
            teacher_new_lens.append(teacher_new_len)
            teacher_total_lens.append(teacher_total_len)

            gold = str(ans[i]) if i < len(ans) else ''
            base = float(ps[i]) if i < len(ps) else 0.0
            q = qs[i] if i < len(qs) else ''

            bad_format = 1.0 if extracted_hint is None else 0.0
            hint_text = extracted_hint or ''

            try:
                full_hint_len = float(len(teacher_tok.encode(hint_text, add_special_tokens=False)))
            except Exception:
                full_hint_len = float(len(hint_text.split()))

            overlong = 1.0 if full_hint_len > float(hint_token_cap) else 0.0
            visible_hint = _truncate_to_token_limit(teacher_tok, hint_text, hint_token_cap)

            try:
                visible_hint_len = float(len(teacher_tok.encode(visible_hint, add_special_tokens=False)))
            except Exception:
                visible_hint_len = float(len(visible_hint.split()))

            golds.append(gold)
            bases.append(base)
            leaks.append(1.0 if contains_answer_leak_any(visible_hint, gold) else 0.0)
            hint_lens.append(visible_hint_len)
            bad_format_flags.append(bad_format)
            overlong_flags.append(overlong)

            s_msgs = build_student_attempt_messages(
                q,
                cfg,
                hint=visible_hint,
                include_stop_marker=True,
            )
            s_prompts.append(
                maybe_apply_chat_template(student_tok, s_msgs, enable_thinking=enable_thinking)
            )
            student_prompt_lens.append(_token_count(student_tok, s_prompts[-1]))

        t_s0 = time.perf_counter()
        if student_remote is not None:
            s_outs = student_remote.generate(
                s_prompts,
                max_new_tokens=stud_max,
                temperature=stud_temp,
                top_p=stud_top_p,
                batch_size=reward_student_batch_size,
                stop=[RTT_STOP_SEQUENCE],
            )
        else:
            assert student_lm is not None
            s_outs = generate_text_batch(
                student_lm,
                s_prompts,
                max_new_tokens=stud_max,
                temperature=stud_temp,
                top_p=stud_top_p,
                do_sample=True,
                batch_size=reward_student_batch_size,
                stop_strings=[RTT_STOP_SEQUENCE],
            )

        t_s_total = time.perf_counter() - t_s0
        per_item_student_s = t_s_total / max(1, len(s_outs))

        rewards: List[float] = []
        for i in range(n):
            t_reward0 = time.perf_counter()

            score = float(numeric_score(s_outs[i], golds[i]))
            delta = score - bases[i]
            reward = (
                delta
                - (beta_leak * leaks[i])
                - (gamma_len * hint_lens[i])
                - (beta_bad_hint_format * bad_format_flags[i])
                - (beta_overlong_hint * overlong_flags[i])
            )
            rewards.append(float(reward))

            student_new_len = _token_count(student_tok, s_outs[i])
            student_total_len = student_prompt_lens[i] + student_new_len
            teacher_trunc = int(teacher_new_lens[i] >= hint_total_cap)
            student_trunc = int(student_new_len >= stud_max)

            METRICS_ACCUM.add(
                delta=float(delta),
                leak=float(leaks[i]),
                hint_len=float(hint_lens[i]),
                reward=float(reward),
                reward_student_time=float(per_item_student_s),
                reward_total_time=float((time.perf_counter() - t_reward0) + per_item_student_s),
                teacher_new=float(teacher_new_lens[i]),
                teacher_total=float(teacher_total_lens[i]),
                student_new=float(student_new_len),
                student_total=float(student_total_len),
                teacher_trunc=teacher_trunc,
                student_trunc=student_trunc,
            )

        return rewards

    if _rank() == 0:
        _write_deepspeed_config(cfg, out_dir)
    _dist_barrier()
    ds_path = out_dir / 'deepspeed_config.json'

    ckpt_steps = int(tcfg.get('save_steps', 200))
    ckpt_keep = int(tcfg.get('save_total_limit', 2))
    want_gradient_checkpointing = bool(tcfg.get('gradient_checkpointing', True))

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_steps=int(max_steps_this_run),
        num_train_epochs=1,
        learning_rate=float(tcfg.get('lr', 1e-6)),
        weight_decay=float(tcfg.get('weight_decay', 0.0)),
        bf16=bool(tcfg.get('bf16', True)),
        logging_steps=10,
        report_to='none',
        remove_unused_columns=False,
        save_strategy='steps',
        save_steps=ckpt_steps,
        save_total_limit=ckpt_keep,
        save_on_each_node=False,
        num_generations=num_generations,
        max_completion_length=hint_total_cap,
        temperature=hint_temp,
        top_p=hint_top_p,
        beta=beta_kl,
        deepspeed=str(ds_path),
        gradient_checkpointing=bool(want_gradient_checkpointing),
    )

    grpo_sig_params = inspect.signature(GRPOConfig.__init__).parameters

    if use_trl_vllm_colocate:
        required_vllm_args = ['use_vllm', 'vllm_mode', 'vllm_tensor_parallel_size']
        missing_vllm_args = [name for name in required_vllm_args if name not in grpo_sig_params]
        if missing_vllm_args:
            raise RuntimeError(
                'trl_colocate mode requires a TRL version whose GRPOConfig supports '
                f'{missing_vllm_args}; found parameters={sorted(grpo_sig_params.keys())}'
            )

        colocate_mem_util = float(
            grpo_cfg.get(
                'vllm_gpu_memory_utilization',
                grpo_cfg.get(
                    'vllm_colocate_gpu_memory_utilization',
                    _TRL_COLOCATE_DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
                ),
            )
        )
        colocate_max_model_len = int(
            grpo_cfg.get(
                'vllm_max_model_length',
                grpo_cfg.get(
                    'vllm_colocate_max_model_length',
                    int((cfg.get('teacher_rollout', {}) or {}).get('max_model_len', 3072)),
                ),
            )
        )

        grpo_kwargs.update(
            use_vllm=True,
            vllm_mode='colocate',
            vllm_tensor_parallel_size=1,
        )
        if 'vllm_gpu_memory_utilization' in grpo_sig_params:
            grpo_kwargs['vllm_gpu_memory_utilization'] = colocate_mem_util
        if 'vllm_max_model_length' in grpo_sig_params:
            grpo_kwargs['vllm_max_model_length'] = colocate_max_model_len
        if 'ds3_gather_for_generation' in grpo_sig_params:
            grpo_kwargs['ds3_gather_for_generation'] = True

    if steps_per_generation_cfg is not None:
        try:
            if 'steps_per_generation' in grpo_sig_params:
                spg = int(steps_per_generation_cfg)
                if spg < 1:
                    raise ValueError('grpo.steps_per_generation must be >= 1')
                grpo_kwargs['steps_per_generation'] = spg
            else:
                if _rank() == 0:
                    print('[learner/grpo] WARNING: GRPOConfig has no steps_per_generation; ignoring YAML value.')
        except Exception as e:
            if _rank() == 0:
                print(f'[learner/grpo] WARNING: could not set steps_per_generation ({e}); ignoring YAML value.')

    generation_batch_size_arg = _resolve_generation_batch_size_arg(grpo_cfg, num_generations=num_generations)
    if generation_batch_size_arg is not None:
        if 'generation_batch_size' in grpo_sig_params:
            grpo_kwargs['generation_batch_size'] = int(generation_batch_size_arg)
        elif _rank() == 0:
            print(
                '[learner/grpo] WARNING: config requested an explicit generation batch size, '
                'but this GRPOConfig does not expose generation_batch_size.'
            )

    training_args = GRPOConfig(**grpo_kwargs)

    from peft import LoraConfig

    lora = cfg.get('models', {}).get('lora', {}) or {}
    peft_config = LoraConfig(
        r=int(lora.get('r', 16)),
        lora_alpha=int(lora.get('alpha', 32)),
        lora_dropout=float(lora.get('dropout', 0.05)),
        target_modules=list(lora.get('target_modules', [])),
        bias='none',
        task_type='CAUSAL_LM',
    )

    if _rank() == 0:
        coord.update_state(phase='train')

    ema_alpha = float((cfg.get('eval', {}) or {}).get('train_metrics', {}).get('ema_alpha', 0.05))
    steps_per_generation = int(getattr(training_args, 'steps_per_generation', steps_per_generation_cfg or 1) or 1)
    callbacks = [
        RoundSaverCallback(
            coord,
            Path(cfg.get('persistent', {}).get('adapter_root', adapter_root)),
            steps_per_round=steps_per_round,
            max_rounds=max_rounds,
            start_round=callback_start_round,
            round_offset=round_offset,
            canonical_base_model_id=canonical_teacher_model_id,
        ),
        TrainMetricsEMACallback(coord, ema_alpha=ema_alpha, step_offset=step_offset),
        GracefulStopCallback(),
    ]

    if not use_trl_vllm_colocate:
        callbacks.insert(
            0,
            LiveRolloutAdapterCallback(
                coord,
                live_root=live_adapter_root,
                steps_per_generation=steps_per_generation,
                keep_last=int(cfg.get('persistent', {}).get('live_keep_last', 3) or 3),
                canonical_base_model_id=canonical_teacher_model_id,
                step_offset=step_offset,
            ),
        )

    if want_gradient_checkpointing:
        try:
            if hasattr(teacher_model, 'gradient_checkpointing_enable'):
                teacher_model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            if hasattr(teacher_model, 'enable_input_require_grads'):
                teacher_model.enable_input_require_grads()
        except Exception:
            pass
        try:
            if hasattr(teacher_model, 'config'):
                setattr(teacher_model.config, 'gradient_checkpointing', True)
                setattr(teacher_model.config, 'use_cache', False)
        except Exception:
            pass

    trainer_kwargs: Dict[str, Any] = dict(
        model=teacher_model,
        args=training_args,
        train_dataset=train_ds,
        reward_funcs=teaching_reward_func,
        processing_class=teacher_tok,
        peft_config=(None if teacher_model_is_peft else peft_config),
        callbacks=callbacks,
    )

    if not use_trl_vllm_colocate:
        assert teacher_remote is not None
        rollout_coordinator = DistributedRolloutCoordinator(
            teacher_remote=teacher_remote,
            teacher_tokenizer=teacher_tok,
            per_device_train_batch_size=per_device_bs,
            steps_per_generation=steps_per_generation,
            base_seed=int(cfg.get('training', {}).get('seed', 0)),
            step_offset=step_offset,
        )

        def remote_teacher_rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
            prompt_strs = [str(p) for p in prompts]
            return rollout_coordinator.generate(prompt_strs, trainer)

        trainer_kwargs['rollout_func'] = remote_teacher_rollout_func

    if _rank() == 0:
        print(f'[learner/grpo] teacher model source: {teacher_model_ref}')
        print(f'[learner/grpo] student model source: {student_model_ref}')
        print(f'[learner/grpo] teacher model class: {teacher_model.__class__.__name__}')
        print(
            f'[learner/grpo] teacher rollout mode: '
            f'{"trl_vllm_colocate" if use_trl_vllm_colocate else "remote_server"} '
            f'student_remote={use_remote_student}'
        )
        if use_trl_vllm_colocate:
            print(
                f'[learner/grpo] trl colocate settings: '
                f'use_vllm={getattr(training_args, "use_vllm", None)} '
                f'vllm_mode={getattr(training_args, "vllm_mode", None)} '
                f'vllm_tensor_parallel_size={getattr(training_args, "vllm_tensor_parallel_size", None)} '
                f'vllm_gpu_memory_utilization={getattr(training_args, "vllm_gpu_memory_utilization", None)} '
                f'vllm_max_model_length={getattr(training_args, "vllm_max_model_length", None)}'
            )
        print(
            f'[learner/grpo] rollout config: steps_per_generation={steps_per_generation} '
            f"generation_batch_size={getattr(training_args, 'generation_batch_size', None)}"
        )

    trainer = GRPOTrainer(**trainer_kwargs)

    try:
        m = getattr(trainer, 'model', None)
        if want_gradient_checkpointing:
            if m is not None and hasattr(m, 'gradient_checkpointing_enable'):
                m.gradient_checkpointing_enable()
            if m is not None and hasattr(m, 'config'):
                setattr(m.config, 'gradient_checkpointing', True)
                setattr(m.config, 'use_cache', False)
        else:
            if m is not None and hasattr(m, 'gradient_checkpointing_disable'):
                m.gradient_checkpointing_disable()
            if m is not None and hasattr(m, 'config'):
                setattr(m.config, 'gradient_checkpointing', False)
    except Exception:
        pass

    if _rank() == 0:
        print(f'[learner/grpo] last checkpoint: {last_ckpt}')
        print(f'[learner/grpo] startup adapter: {load_adapter_path}')
        print(
            f'[learner/grpo] start_round={start_round} callback_start_round={callback_start_round} '
            f'max_rounds={max_rounds} steps_per_round={steps_per_round}'
        )
        print(
            f'[learner/grpo] step_offset={step_offset} round_offset={round_offset} '
            f'max_steps_this_run={max_steps_this_run} full_run_max_steps={max_steps}'
        )
        print(
            f'[learner/grpo] gradient checkpointing: '
            f"{'ENABLED' if want_gradient_checkpointing else 'DISABLED'}"
        )

    trainer.train(resume_from_checkpoint=last_ckpt)

    if _rank() == 0:
        st2 = coord.read_state()
        r = int(st2.get('round', 0))
        if r >= max_rounds:
            coord.update_state(phase='done')
        print(f"[learner/grpo] finished: round={r}/{max_rounds} phase={coord.read_state().get('phase')}")

    time.sleep(2)


if __name__ == '__main__':
    main()
