from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any, Dict

import yaml

from .placement_core import build_components, summarize_components


DEFAULT_CPUS_PER_GPU = 3
DEFAULT_MEM_PER_NODE_GB = 96
DEFAULT_PACKED_ISOLATE_TEACHER = True
TRL_COLOCATE_CPUS_PER_GPU = 6
TRL_COLOCATE_MEM_PER_NODE_GB = 96


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _require(res: Dict[str, Any], key: str) -> Any:
    if key not in res:
        raise KeyError(f'Missing required config key: {key}')
    return res[key]


def _normalize_mode(raw: Any) -> str:
    s = str(raw or 'legacy').strip().lower()
    if s not in {'legacy', 'packed', 'trl_colocate'}:
        raise ValueError('resources.placement_mode must be one of: legacy, packed, trl_colocate')
    return s


def _coerce_replicas(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == 'auto':
        return 'auto'
    return int(raw)


def _coerce_bool(raw: Any, *, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if raw is None:
        raise ValueError(f'{name} must be a boolean value, got {raw!r}')
    s = str(raw).strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'{name} must be a boolean value, got {raw!r}')


def _packed_isolate_teacher(res: Dict[str, Any]) -> bool:
    raw = res.get('isolate_teacher_rollout', DEFAULT_PACKED_ISOLATE_TEACHER)
    return _coerce_bool(raw, name='resources.isolate_teacher_rollout')


def get_resources(cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = (cfg.get('resources', {}) or {})
    mode = _normalize_mode(res.get('placement_mode', 'legacy'))

    scfg = (cfg.get('student_server', {}) or {})
    tcfg = (cfg.get('teacher_rollout', {}) or {})
    inf = (cfg.get('inference', {}) or {})

    if mode == 'trl_colocate':
        return {
            'placement_mode': mode,
            'packing_policy': 'trainer_only_trl_colocate',
            'gpus_per_node': 2,
            'gpus_per_job': 2,
            'cpus_per_gpu': TRL_COLOCATE_CPUS_PER_GPU,
            'mem_per_node_gb': TRL_COLOCATE_MEM_PER_NODE_GB,
            'trainer_nodes': 1,
            'student_server_nodes': 0,
            'teacher_rollout_nodes': 0,
            'trainer_gpus': 2,
            'student_server_gpus': 0,
            'teacher_rollout_gpus': 0,
            'student_replicas': 0,
            'teacher_replicas': 0,
            'teacher_tp_size': 1,
            'total_nodes': 1,
            'total_gpus': 2,
        }

    teacher_tp_size = int(tcfg.get('tensor_parallel_size', 4))
    student_cpus_per_replica = int(scfg.get('cpus_per_replica', 4))
    teacher_cpus_per_replica = int(tcfg.get('cpus_per_replica', 12))
    min_required_cpus_per_gpu = max(
        DEFAULT_CPUS_PER_GPU,
        student_cpus_per_replica,
        int(math.ceil(teacher_cpus_per_replica / max(1, teacher_tp_size))),
    )
    raw_cpg = res.get('cpus_per_gpu', None)
    cpus_per_gpu = int(raw_cpg if raw_cpg is not None else min_required_cpus_per_gpu)
    mem_per_node_gb = int(res.get('mem_per_node_gb', DEFAULT_MEM_PER_NODE_GB))
    if cpus_per_gpu < min_required_cpus_per_gpu:
        raise ValueError(
            f'resources.cpus_per_gpu={cpus_per_gpu} is too small for the configured roles; need at least {min_required_cpus_per_gpu}'
        )
    if mem_per_node_gb < 1:
        raise ValueError('resources.mem_per_node_gb must be >= 1')

    if teacher_tp_size < 1:
        raise ValueError('teacher_rollout.tensor_parallel_size must be >= 1')

    if mode == 'legacy':
        trainer_nodes = int(_require(res, 'trainer_nodes'))
        student_server_nodes = int(res.get('student_server_nodes', 1))
        teacher_rollout_nodes = int(res.get('teacher_rollout_nodes', 1))
        gpus_per_job = int(res.get('gpus_per_job', 4))
        gpus_per_node = int(res.get('gpus_per_node', gpus_per_job))

        if trainer_nodes < 1:
            raise ValueError('resources.trainer_nodes must be >= 1')
        if student_server_nodes < 0:
            raise ValueError('resources.student_server_nodes must be >= 0')
        if teacher_rollout_nodes < 0:
            raise ValueError('resources.teacher_rollout_nodes must be >= 0')
        if gpus_per_job < 1:
            raise ValueError('resources.gpus_per_job must be >= 1')
        if gpus_per_node < 1:
            raise ValueError('resources.gpus_per_node must be >= 1')

        trainer_gpus = trainer_nodes * gpus_per_job
        student_server_reserved_gpus = student_server_nodes * gpus_per_job
        teacher_rollout_reserved_gpus = teacher_rollout_nodes * gpus_per_job
        total_nodes = trainer_nodes + student_server_nodes + teacher_rollout_nodes
        total_reserved_gpus = total_nodes * gpus_per_job

        student_replicas = _coerce_replicas(scfg.get('replicas', 1))
        if student_replicas in (None, 'auto'):
            if student_server_reserved_gpus <= 0:
                student_replicas = 0
            else:
                student_replicas = min(gpus_per_job, student_server_reserved_gpus)
        student_replicas = int(student_replicas)
        if student_replicas < 0:
            raise ValueError('student_server.replicas must be >= 0')
        if student_replicas > student_server_reserved_gpus:
            raise ValueError(
                f'student_server.replicas={student_replicas} exceeds legacy student capacity={student_server_reserved_gpus}'
            )

        teacher_replicas = _coerce_replicas(tcfg.get('replicas', None))
        if teacher_replicas in (None, 'auto'):
            teacher_replicas = teacher_rollout_nodes
        teacher_replicas = int(teacher_replicas)
        if teacher_replicas != teacher_rollout_nodes:
            raise ValueError(
                'legacy mode preserves one teacher rollout replica per teacher_rollout node; '
                f'got teacher_rollout_nodes={teacher_rollout_nodes}, replicas={teacher_replicas}'
            )
        if teacher_tp_size > gpus_per_job:
            raise ValueError(
                f'teacher_rollout.tensor_parallel_size={teacher_tp_size} exceeds resources.gpus_per_job={gpus_per_job}'
            )

        return {
            'placement_mode': mode,
            'packing_policy': 'whole_node_legacy',
            'gpus_per_node': gpus_per_node,
            'gpus_per_job': gpus_per_job,
            'cpus_per_gpu': cpus_per_gpu,
            'mem_per_node_gb': mem_per_node_gb,
            'trainer_nodes': trainer_nodes,
            'student_server_nodes': student_server_nodes,
            'teacher_rollout_nodes': teacher_rollout_nodes,
            'trainer_gpus': trainer_gpus,
            'student_server_gpus': student_server_reserved_gpus,
            'teacher_rollout_gpus': teacher_rollout_reserved_gpus,
            'student_replicas': student_replicas,
            'teacher_replicas': teacher_replicas,
            'teacher_tp_size': teacher_tp_size,
            'total_nodes': total_nodes,
            'total_gpus': total_reserved_gpus,
        }

    gpus_per_node = int(res.get('gpus_per_node', res.get('gpus_per_job', 4)))
    trainer_gpus = int(_require(res, 'trainer_gpus'))
    student_server_gpus = int(res.get('student_server_gpus', 0))
    teacher_rollout_gpus = int(res.get('teacher_rollout_gpus', 0))
    isolate_teacher_rollout = _packed_isolate_teacher(res)

    if gpus_per_node < 1:
        raise ValueError('resources.gpus_per_node must be >= 1')
    if trainer_gpus < 1:
        raise ValueError('resources.trainer_gpus must be >= 1')
    if student_server_gpus < 0:
        raise ValueError('resources.student_server_gpus must be >= 0')
    if teacher_rollout_gpus < 0:
        raise ValueError('resources.teacher_rollout_gpus must be >= 0')

    raw_student_replicas = _coerce_replicas(scfg.get('replicas', 'auto'))
    if raw_student_replicas in (None, 'auto'):
        student_replicas = student_server_gpus
    else:
        student_replicas = int(raw_student_replicas)
        if student_replicas != student_server_gpus:
            raise ValueError(
                'packed mode expects one student server replica per reserved student GPU; '
                f'set student_server.replicas=auto or {student_server_gpus}'
            )

    if teacher_rollout_gpus % teacher_tp_size != 0:
        raise ValueError(
            'packed mode requires teacher_rollout_gpus to be divisible by teacher_rollout.tensor_parallel_size; '
            f'got {teacher_rollout_gpus} and {teacher_tp_size}'
        )
    teacher_replicas = teacher_rollout_gpus // teacher_tp_size

    raw_teacher_replicas = _coerce_replicas(tcfg.get('replicas', 'auto'))
    if raw_teacher_replicas not in (None, 'auto') and int(raw_teacher_replicas) != teacher_replicas:
        raise ValueError(
            'packed mode derives teacher rollout replicas from teacher_rollout_gpus / tensor_parallel_size; '
            f'expected {teacher_replicas}, got replicas={raw_teacher_replicas}'
        )

    total_gpus = trainer_gpus + student_server_gpus + teacher_rollout_gpus
    packing_policy = 'teacher_isolated' if isolate_teacher_rollout else 'minimal'
    packed_components = build_components(
        gpus_per_node=gpus_per_node,
        trainer_gpus=trainer_gpus,
        student_replicas=student_replicas,
        teacher_replicas=teacher_replicas,
        teacher_tp_size=teacher_tp_size,
        isolate_teacher_rollout=isolate_teacher_rollout,
    )
    packed_node_counts = summarize_components(packed_components)

    return {
        'placement_mode': mode,
        'packing_policy': packing_policy,
        'gpus_per_node': gpus_per_node,
        'gpus_per_job': int(res.get('gpus_per_job', gpus_per_node)),
        'cpus_per_gpu': cpus_per_gpu,
        'mem_per_node_gb': mem_per_node_gb,
        'trainer_nodes': int(packed_node_counts['trainer_nodes']),
        'student_server_nodes': int(packed_node_counts['student_server_nodes']),
        'teacher_rollout_nodes': int(packed_node_counts['teacher_rollout_nodes']),
        'trainer_gpus': trainer_gpus,
        'student_server_gpus': student_server_gpus,
        'teacher_rollout_gpus': teacher_rollout_gpus,
        'student_replicas': student_replicas,
        'teacher_replicas': teacher_replicas,
        'teacher_tp_size': teacher_tp_size,
        'total_nodes': int(packed_node_counts['total_nodes']),
        'total_gpus': total_gpus,
    }


def get_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models = (cfg.get('models', {}) or {})
    teacher_model_id = str(_require(models, 'teacher_model_id'))
    student_model_id = str(_require(models, 'student_model_id'))
    return {
        'teacher_model_id': teacher_model_id,
        'student_model_id': student_model_id,
    }


def get_student_server(cfg: Dict[str, Any]) -> Dict[str, Any]:
    scfg = (cfg.get('student_server', {}) or {})
    return {
        'replicas': scfg.get('replicas', 1),
        'base_port': int(scfg.get('base_port', 8100)),
        'dtype': str(scfg.get('dtype', 'bfloat16')),
        'max_model_len': int(scfg.get('max_model_len', 4096)),
        'gpu_memory_utilization': float(scfg.get('gpu_memory_utilization', 0.85)),
        'cpus_per_replica': int(scfg.get('cpus_per_replica', 4)),
        'startup_timeout_s': int(scfg.get('startup_timeout_s', 1800)),
    }


def get_teacher_rollout(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tcfg = (cfg.get('teacher_rollout', {}) or {})
    return {
        'replicas': tcfg.get('replicas', 'auto'),
        'base_port': int(tcfg.get('base_port', 8200)),
        'dtype': str(tcfg.get('dtype', 'bfloat16')),
        'tensor_parallel_size': int(tcfg.get('tensor_parallel_size', 4)),
        'max_model_len': int(tcfg.get('max_model_len', 6144)),
        'gpu_memory_utilization': float(tcfg.get('gpu_memory_utilization', 0.90)),
        'cpus_per_replica': int(tcfg.get('cpus_per_replica', 12)),
        'startup_timeout_s': int(tcfg.get('startup_timeout_s', 900)),
        'request_timeout_s': float(tcfg.get('request_timeout_s', 180.0)),
        'request_max_retries': int(tcfg.get('request_max_retries', 8)),
        'request_retry_backoff_s': float(tcfg.get('request_retry_backoff_s', 5.0)),
        'max_logprobs': int(tcfg.get('max_logprobs', 1)),
        'max_requests_per_server': int(tcfg.get('max_requests_per_server', 4)),
        'max_num_seqs': int(tcfg.get('max_num_seqs', 32)),
        'extra_body': tcfg.get('extra_body', {}) or {},
    }


def get_teacher_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ecfg = ((cfg.get('eval', {}) or {}).get('teacher_eval', {}) or {})
    res = get_resources(cfg)

    schedule = str(ecfg.get('schedule', 'online') or 'online').strip().lower()
    if schedule not in {'online', 'post_train'}:
        raise ValueError("eval.teacher_eval.schedule must be one of: online, post_train")

    if res['placement_mode'] == 'trl_colocate' and schedule == 'online':
        raise ValueError(
            'eval.teacher_eval.schedule=online is not supported in trl_colocate mode; use post_train or disable teacher_eval'
        )

    sweep = str(ecfg.get('sweep', 'final_only') or 'final_only').strip().lower()
    if sweep not in {'final_only', 'all_rounds'}:
        raise ValueError("eval.teacher_eval.sweep must be one of: final_only, all_rounds")

    raw_gpus = ecfg.get('gpus', 'auto')
    if isinstance(raw_gpus, str) and raw_gpus.strip().lower() == 'auto':
        if res['placement_mode'] == 'packed':
            gpus = int(res['gpus_per_node'])
        else:
            gpus = int(res['gpus_per_job'])
    else:
        gpus = int(raw_gpus)

    if gpus < 1:
        raise ValueError('eval.teacher_eval.gpus must be >= 1')

    if schedule == 'online' and gpus > int(res['gpus_per_node']):
        raise ValueError(
            f'eval.teacher_eval.gpus={gpus} exceeds resources.gpus_per_node={res["gpus_per_node"]}. '
            'The online teacher_eval_once path currently runs on one node only.'
        )

    return {
        'gpus': gpus,
        'schedule': schedule,
        'sweep': sweep,
    }


def build_shell_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = get_resources(cfg)
    models = get_models(cfg)
    scfg = get_student_server(cfg)
    tcfg = get_teacher_rollout(cfg)
    inf = (cfg.get('inference', {}) or {})
    jobs = (cfg.get('jobs', {}) or {})
    paths = (cfg.get('paths', {}) or {})
    persistent = (cfg.get('persistent', {}) or {})
    teacher_eval = ((cfg.get('eval', {}) or {}).get('teacher_eval', {}) or {})
    teacher_eval_cfg = get_teacher_eval(cfg)

    return {
        'RTT_CFG_PLACEMENT_MODE': res['placement_mode'],
        'RTT_CFG_PACKING_POLICY': res['packing_policy'],
        'RTT_CFG_GPUS_PER_NODE': res['gpus_per_node'],
        'RTT_CFG_GPUS_PER_JOB': res['gpus_per_job'],
        'RTT_CFG_CPUS_PER_GPU': res['cpus_per_gpu'],
        'RTT_CFG_MEM_PER_NODE_GB': res['mem_per_node_gb'],
        'RTT_CFG_TRAINER_NODES': res['trainer_nodes'],
        'RTT_CFG_STUDENT_SERVER_NODES': res['student_server_nodes'],
        'RTT_CFG_TEACHER_ROLLOUT_NODES': res['teacher_rollout_nodes'],
        'RTT_CFG_TRAINER_GPUS': res['trainer_gpus'],
        'RTT_CFG_STUDENT_SERVER_GPUS': res['student_server_gpus'],
        'RTT_CFG_TEACHER_ROLLOUT_GPUS': res['teacher_rollout_gpus'],
        'RTT_CFG_STUDENT_REPLICAS': res['student_replicas'],
        'RTT_CFG_TEACHER_REPLICAS': res['teacher_replicas'],
        'RTT_CFG_TOTAL_NODES': res['total_nodes'],
        'RTT_CFG_TOTAL_GPUS': res['total_gpus'],
        'RTT_CFG_TEACHER_MODEL_ID': models['teacher_model_id'],
        'RTT_CFG_STUDENT_MODEL_ID': models['student_model_id'],
        'RTT_CFG_STUDENT_BASE_PORT': scfg['base_port'],
        'RTT_CFG_STUDENT_DTYPE': scfg['dtype'],
        'RTT_CFG_STUDENT_MAX_MODEL_LEN': scfg['max_model_len'],
        'RTT_CFG_STUDENT_GPU_MEM_UTIL': scfg['gpu_memory_utilization'],
        'RTT_CFG_STUDENT_CPUS_PER_REPLICA': scfg['cpus_per_replica'],
        'RTT_CFG_STUDENT_STARTUP_TIMEOUT_S': scfg['startup_timeout_s'],
        'RTT_CFG_TEACHER_ROLLOUT_BASE_PORT': tcfg['base_port'],
        'RTT_CFG_TEACHER_ROLLOUT_DTYPE': tcfg['dtype'],
        'RTT_CFG_TEACHER_ROLLOUT_REPLICAS': res['teacher_replicas'],
        'RTT_CFG_TEACHER_ROLLOUT_TP_SIZE': res['teacher_tp_size'],
        'RTT_CFG_TEACHER_ROLLOUT_MAX_MODEL_LEN': tcfg['max_model_len'],
        'RTT_CFG_TEACHER_ROLLOUT_GPU_MEM_UTIL': tcfg['gpu_memory_utilization'],
        'RTT_CFG_TEACHER_ROLLOUT_CPUS_PER_REPLICA': tcfg['cpus_per_replica'],
        'RTT_CFG_TEACHER_ROLLOUT_STARTUP_TIMEOUT_S': tcfg['startup_timeout_s'],
        'RTT_CFG_TEACHER_ROLLOUT_REQUEST_TIMEOUT_S': tcfg['request_timeout_s'],
        'RTT_CFG_TEACHER_ROLLOUT_REQUEST_MAX_RETRIES': tcfg['request_max_retries'],
        'RTT_CFG_TEACHER_ROLLOUT_REQUEST_RETRY_BACKOFF_S': tcfg['request_retry_backoff_s'],
        'RTT_CFG_TEACHER_ROLLOUT_MAX_LOGPROBS': tcfg['max_logprobs'],
        'RTT_CFG_TEACHER_ROLLOUT_MAX_REQUESTS_PER_SERVER': tcfg['max_requests_per_server'],
        'RTT_CFG_TEACHER_ROLLOUT_MAX_NUM_SEQS': tcfg['max_num_seqs'],
        'RTT_CFG_TEACHER_ROLLOUT_EXTRA_BODY_JSON': json.dumps(tcfg['extra_body'], ensure_ascii=False, separators=(',', ':')),
        'RTT_CFG_INFER_LANGUAGE_MODEL_ONLY': int(bool(inf.get('language_model_only', False))),
        'RTT_CFG_INFER_CPU_OFFLOAD_GB': float(inf.get('cpu_offload_gb', 0.0)),
        'RTT_CFG_INFER_ENFORCE_EAGER': int(bool(inf.get('enforce_eager', False))),
        'RTT_CFG_INFER_DISABLE_LOG_STATS': int(bool(inf.get('disable_log_stats', True))),
        'RTT_CFG_INFER_ATTN_BACKEND': str(inf.get('attention_backend', '') or ''),
        'RTT_CFG_CACHE_PARALLELISM': str((inf.get('cache_parallelism', 'slurm') or 'slurm')).lower(),
        'RTT_CFG_INFER_MAX_LORAS': int(inf.get('max_loras', 8)),
        'RTT_CFG_INFER_MAX_LORA_RANK': int(inf.get('max_lora_rank', 64)),
        'RTT_CFG_CACHE_STUDENT_JOBS': int(jobs.get('cache_student_jobs', 1)),
        'RTT_CFG_CACHE_TEACHER_JOBS': int(jobs.get('cache_teacher_jobs', 1)),
        'RTT_CFG_OUTPUTS_DIR': str(paths.get('outputs_dir', 'outputs')),
        'RTT_CFG_COORD_DIR': str(paths.get('coord_dir', 'data/coord_rl')),
        'RTT_CFG_EVAL_OUTDIR': str(Path(paths.get('outputs_dir', 'outputs')) / 'eval'),
        'RTT_CFG_PERSISTENT_ROUNDS': int(persistent.get('rounds', 1)),
        'RTT_CFG_PERSISTENT_CLEAN_COORD': int(bool(persistent.get('clean_coord', False))),
        'RTT_CFG_PERSISTENT_ADAPTER_ROOT': str(persistent.get('adapter_root', 'outputs/teacher_lora_persistent')),
        'RTT_CFG_PERSISTENT_LIVE_ADAPTER_ROOT': str(persistent.get('live_adapter_root', 'outputs/teacher_lora_live')),
        'RTT_CFG_TEACHER_EVAL_ENABLED': int(bool(teacher_eval.get('enabled', True))),
        'RTT_CFG_TEACHER_EVAL_EVERY_ROUNDS': int(teacher_eval.get('every_rounds', 1)),
        'RTT_CFG_TEACHER_EVAL_GPUS': int(teacher_eval_cfg['gpus']),
        'RTT_CFG_TEACHER_EVAL_SCHEDULE': str(teacher_eval_cfg['schedule']),
        'RTT_CFG_TEACHER_EVAL_SWEEP': str(teacher_eval_cfg['sweep']),
    }


def emit_shell_assignments(vars_dict: Dict[str, Any]) -> str:
    lines = []
    for k, v in vars_dict.items():
        lines.append(f'{k}={shlex.quote(str(v))}')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    shell = sub.add_parser('shell', help='Emit shell-safe assignments for config values.')
    shell.add_argument('--config', required=True)

    args = ap.parse_args()

    if args.cmd == 'shell':
        cfg = load_config(args.config)
        print(emit_shell_assignments(build_shell_vars(cfg)))


if __name__ == '__main__':
    main()
