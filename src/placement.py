from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_helpers import get_resources, get_student_server, get_teacher_eval, load_config
from .placement_core import build_components
from .utils import atomic_write_json, ensure_dir


def _manifest_path(cfg: Dict[str, Any]) -> Path:
    coord_dir = Path((cfg.get('paths', {}) or {}).get('coord_dir', 'data/coord_rl'))
    ensure_dir(coord_dir)
    return coord_dir / 'placement.json'


def _teacher_replicas_from_resources(res: Dict[str, Any]) -> int:
    return int(res['teacher_replicas'])


def _default_eval_plan() -> Dict[str, Any]:
    return {
        'enabled': False,
        'colocate_with_student': False,
        'component_id': None,
        'reserved_gpus': 0,
        'reason': 'disabled',
    }


def _pick_colocated_eval_plan(cfg: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    ecfg = ((cfg.get('eval', {}) or {}).get('teacher_eval', {}) or {})
    if not bool(ecfg.get('enabled', True)):
        return _default_eval_plan()

    schedule = str(ecfg.get('schedule', 'online') or 'online').strip().lower()
    if schedule != 'online':
        plan = _default_eval_plan()
        plan['enabled'] = True
        plan['reason'] = f'schedule_{schedule}'
        return plan

    if str(manifest.get('mode', '')) != 'packed':
        plan = _default_eval_plan()
        plan['reason'] = 'not_packed'
        return plan

    student_cfg = get_student_server(cfg)
    gpus_per_node = int(manifest['gpus_per_node'])
    cpus_per_gpu = int(manifest['cpus_per_gpu'])
    node_cpu_capacity = gpus_per_node * cpus_per_gpu
    student_cpus_per_replica = int(student_cfg['cpus_per_replica'])

    raw_eval_gpus = ecfg.get('gpus', 'auto')
    explicit_eval_gpus: Optional[int]
    auto_eval_gpus = isinstance(raw_eval_gpus, str) and raw_eval_gpus.strip().lower() == 'auto'
    if auto_eval_gpus:
        explicit_eval_gpus = None
    else:
        explicit_eval_gpus = int(get_teacher_eval(cfg)['gpus'])

    best: Optional[Dict[str, Any]] = None
    for comp in manifest.get('components', []):
        student_count = len(comp.get('student_replica_ids', []))
        if student_count <= 0:
            continue
        if comp.get('trainer_rank_ids') or comp.get('teacher_replica_ids'):
            continue

        spare_gpus = max(0, gpus_per_node - int(comp.get('gpu_count', 0)))
        if spare_gpus <= 0:
            continue

        eval_gpus = int(spare_gpus if auto_eval_gpus else explicit_eval_gpus or 0)
        if eval_gpus <= 0 or eval_gpus > spare_gpus:
            continue

        student_cpu_need = student_count * student_cpus_per_replica
        eval_cpu_need = eval_gpus * cpus_per_gpu
        if (student_cpu_need + eval_cpu_need) > node_cpu_capacity:
            continue

        candidate = {
            'enabled': True,
            'colocate_with_student': True,
            'component_id': int(comp['component_id']),
            'reserved_gpus': int(eval_gpus),
            'reason': 'student_component_reserved',
            '_spare_gpus': int(spare_gpus),
        }
        if best is None:
            best = candidate
            continue
        if candidate['reserved_gpus'] > best['reserved_gpus']:
            best = candidate
            continue
        if candidate['reserved_gpus'] == best['reserved_gpus'] and candidate['component_id'] < best['component_id']:
            best = candidate

    if best is None:
        plan = _default_eval_plan()
        plan['enabled'] = True
        plan['reason'] = 'no_student_component_fit'
        return plan

    chosen_cid = int(best['component_id'])
    for comp in manifest['components']:
        if int(comp['component_id']) == chosen_cid:
            comp['gpu_count'] = int(comp['gpu_count']) + int(best['reserved_gpus'])
            break

    best.pop('_spare_gpus', None)
    return best


def build_legacy_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = get_resources(cfg)
    if res['placement_mode'] != 'legacy':
        raise ValueError('build_legacy_manifest called for non-legacy config')

    gpus_per_job = int(res['gpus_per_job'])
    teacher_tp_size = int(res['teacher_tp_size'])
    student_replicas = int(res['student_replicas'])
    teacher_replicas = _teacher_replicas_from_resources(res)

    components: List[Dict[str, Any]] = []
    comp_id = 0
    next_rank = 0
    next_student = 0
    next_teacher = 0

    for _ in range(int(res['trainer_nodes'])):
        comp = {
            'component_id': int(comp_id),
            'gpu_count': gpus_per_job,
            'trainer_rank_ids': list(range(next_rank, next_rank + gpus_per_job)),
            'student_replica_ids': [],
            'teacher_replica_ids': [],
        }
        comp_id += 1
        next_rank += gpus_per_job
        components.append(comp)

    student_capacity = int(res['student_server_nodes']) * gpus_per_job
    if student_replicas > student_capacity:
        raise ValueError(
            f'student_server.replicas={student_replicas} exceeds legacy student capacity={student_capacity}'
        )
    for _ in range(int(res['student_server_nodes'])):
        comp = {
            'component_id': int(comp_id),
            'gpu_count': gpus_per_job,
            'trainer_rank_ids': [],
            'student_replica_ids': [],
            'teacher_replica_ids': [],
        }
        comp_id += 1
        take = min(gpus_per_job, max(0, student_replicas - next_student))
        comp['student_replica_ids'] = list(range(next_student, next_student + take))
        next_student += take
        components.append(comp)

    if teacher_replicas != int(res['teacher_rollout_nodes']):
        raise ValueError(
            'legacy mode preserves one teacher rollout replica per teacher_rollout node; '
            f"expected {int(res['teacher_rollout_nodes'])}, got {teacher_replicas}"
        )
    if teacher_tp_size > gpus_per_job:
        raise ValueError(
            f'teacher_rollout.tensor_parallel_size={teacher_tp_size} exceeds resources.gpus_per_job={gpus_per_job}'
        )
    for _ in range(int(res['teacher_rollout_nodes'])):
        comp = {
            'component_id': int(comp_id),
            'gpu_count': gpus_per_job,
            'trainer_rank_ids': [],
            'student_replica_ids': [],
            'teacher_replica_ids': [next_teacher],
        }
        comp_id += 1
        next_teacher += 1
        components.append(comp)

    return {
        'mode': 'legacy',
        'packing_policy': 'whole_node_legacy',
        'gpus_per_node': int(res['gpus_per_node']),
        'cpus_per_gpu': int(res['cpus_per_gpu']),
        'mem_per_node_gb': int(res['mem_per_node_gb']),
        'teacher_tp_size': teacher_tp_size,
        'trainer_world_size': next_rank,
        'student_replica_count': student_replicas,
        'teacher_replica_count': teacher_replicas,
        'components': components,
    }


def build_packed_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = get_resources(cfg)
    if res['placement_mode'] != 'packed':
        raise ValueError('build_packed_manifest called for non-packed config')

    gpus_per_node = int(res['gpus_per_node'])
    trainer_gpus = int(res['trainer_gpus'])
    student_replicas = int(res['student_replicas'])
    teacher_replicas = _teacher_replicas_from_resources(res)
    teacher_tp_size = int(res['teacher_tp_size'])
    packing_policy = str(res['packing_policy'])

    if teacher_tp_size > gpus_per_node:
        raise ValueError(
            f'teacher_rollout.tensor_parallel_size={teacher_tp_size} exceeds resources.gpus_per_node={gpus_per_node}'
        )

    if packing_policy not in ('teacher_isolated', 'minimal'):
        raise ValueError(f'unsupported packed policy: {packing_policy}')

    components = build_components(
        gpus_per_node=gpus_per_node,
        trainer_gpus=trainer_gpus,
        student_replicas=student_replicas,
        teacher_replicas=teacher_replicas,
        teacher_tp_size=teacher_tp_size,
        isolate_teacher_rollout=(packing_policy == 'teacher_isolated'),
    )

    return {
        'mode': 'packed',
        'packing_policy': packing_policy,
        'gpus_per_node': gpus_per_node,
        'cpus_per_gpu': int(res['cpus_per_gpu']),
        'mem_per_node_gb': int(res['mem_per_node_gb']),
        'teacher_tp_size': teacher_tp_size,
        'trainer_world_size': trainer_gpus,
        'student_replica_count': student_replicas,
        'teacher_replica_count': teacher_replicas,
        'components': components,
    }


def build_trl_colocate_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = get_resources(cfg)
    if res['placement_mode'] != 'trl_colocate':
        raise ValueError('build_trl_colocate_manifest called for non-trl_colocate config')

    return {
        'mode': 'trl_colocate',
        'packing_policy': 'trainer_only_trl_colocate',
        'gpus_per_node': int(res['gpus_per_node']),
        'cpus_per_gpu': int(res['cpus_per_gpu']),
        'mem_per_node_gb': int(res['mem_per_node_gb']),
        'teacher_tp_size': 1,
        'trainer_world_size': 2,
        'student_replica_count': 0,
        'teacher_replica_count': 0,
        'components': [
            {
                'component_id': 0,
                'gpu_count': 2,
                'trainer_rank_ids': [0, 1],
                'student_replica_ids': [],
                'teacher_replica_ids': [],
            }
        ],
    }


def build_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mode = str((cfg.get('resources', {}) or {}).get('placement_mode', 'legacy')).strip().lower()
    if mode == 'legacy':
        manifest = build_legacy_manifest(cfg)
    elif mode == 'packed':
        manifest = build_packed_manifest(cfg)
    elif mode == 'trl_colocate':
        manifest = build_trl_colocate_manifest(cfg)
    else:
        raise ValueError(f'unsupported placement mode: {mode}')

    for comp in manifest['components']:
        comp['trainer_rank_ids'] = sorted(int(x) for x in comp.get('trainer_rank_ids', []))
        comp['student_replica_ids'] = sorted(int(x) for x in comp.get('student_replica_ids', []))
        comp['teacher_replica_ids'] = sorted(int(x) for x in comp.get('teacher_replica_ids', []))

    if manifest['mode'] == 'trl_colocate':
        manifest['eval_plan'] = {
            'enabled': False,
            'colocate_with_student': False,
            'component_id': None,
            'reserved_gpus': 0,
            'reason': 'no_spare_gpus_trl_colocate',
        }
    else:
        manifest['eval_plan'] = _pick_colocated_eval_plan(cfg, manifest)
    manifest['component_gpu_counts'] = [int(c['gpu_count']) for c in manifest['components']]
    manifest['component_count'] = len(manifest['components'])
    return manifest


def write_manifest(cfg: Dict[str, Any]) -> Path:
    path = _manifest_path(cfg)
    atomic_write_json(path, build_manifest(cfg))
    return path


def emit_shell_for_config(cfg: Dict[str, Any]) -> str:
    manifest_path = write_manifest(cfg)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    eval_plan = dict(manifest.get('eval_plan', {}) or {})
    lines = {
        'RTT_PLACEMENT_MANIFEST': str(manifest_path),
        'RTT_PLACEMENT_MODE': str(manifest['mode']),
        'RTT_PLACEMENT_PACKING_POLICY': str(manifest.get('packing_policy', '')),
        'RTT_PLACEMENT_COMPONENT_GPU_COUNTS': ','.join(str(x) for x in manifest['component_gpu_counts']),
        'RTT_PLACEMENT_COMPONENT_COUNT': str(len(manifest['components'])),
        'RTT_PLACEMENT_TOTAL_TRAINER_RANKS': str(manifest['trainer_world_size']),
        'RTT_PLACEMENT_TOTAL_STUDENT_REPLICAS': str(manifest['student_replica_count']),
        'RTT_PLACEMENT_TOTAL_TEACHER_REPLICAS': str(manifest['teacher_replica_count']),
        'RTT_PLACEMENT_CPUS_PER_GPU': str(manifest['cpus_per_gpu']),
        'RTT_PLACEMENT_MEM_PER_NODE_GB': str(manifest['mem_per_node_gb']),
        'RTT_PLACEMENT_EVAL_COLOCATE': str(int(bool(eval_plan.get('colocate_with_student', False)))),
        'RTT_PLACEMENT_EVAL_COMPONENT_ID': str(eval_plan.get('component_id', '') or ''),
        'RTT_PLACEMENT_EVAL_GPUS': str(int(eval_plan.get('reserved_gpus', 0) or 0)),
        'RTT_PLACEMENT_EVAL_REASON': str(eval_plan.get('reason', '')),
    }
    return '\n'.join(f'{k}={shlex.quote(v)}' for k, v in lines.items())


def emit_runtime_shell(manifest_path: str) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    eval_plan = dict(manifest.get('eval_plan', {}) or {})
    lines: List[str] = []
    lines.append(f"RTT_PL_MODE={shlex.quote(str(manifest['mode']))}")
    lines.append(f"RTT_PL_PACKING_POLICY={shlex.quote(str(manifest.get('packing_policy', '')))}")
    lines.append(f"RTT_PL_COMPONENT_IDS={shlex.quote(' '.join(str(c['component_id']) for c in manifest['components']))}")
    lines.append(f"RTT_PL_TRAINER_WORLD_SIZE={shlex.quote(str(manifest['trainer_world_size']))}")
    lines.append(f"RTT_PL_TEACHER_TP_SIZE={shlex.quote(str(manifest['teacher_tp_size']))}")
    lines.append(f"RTT_PL_COMPONENT_GPU_COUNTS={shlex.quote(','.join(str(c['gpu_count']) for c in manifest['components']))}")
    lines.append(f"RTT_PL_EVAL_COLOCATE={shlex.quote(str(int(bool(eval_plan.get('colocate_with_student', False)))))}")
    lines.append(f"RTT_PL_EVAL_COMPONENT_ID={shlex.quote(str(eval_plan.get('component_id', '') or ''))}")
    lines.append(f"RTT_PL_EVAL_GPUS={shlex.quote(str(int(eval_plan.get('reserved_gpus', 0) or 0)))}")
    lines.append(f"RTT_PL_EVAL_REASON={shlex.quote(str(eval_plan.get('reason', '')))}")
    for comp in manifest['components']:
        cid = int(comp['component_id'])
        trainer_ids = ','.join(str(x) for x in comp.get('trainer_rank_ids', []))
        student_ids = ','.join(str(x) for x in comp.get('student_replica_ids', []))
        teacher_ids = ','.join(str(x) for x in comp.get('teacher_replica_ids', []))
        lines.append(f"RTT_PL_COMP_{cid}_GPU_COUNT={shlex.quote(str(comp['gpu_count']))}")
        lines.append(f"RTT_PL_COMP_{cid}_TRAINER_IDS={shlex.quote(trainer_ids)}")
        lines.append(f"RTT_PL_COMP_{cid}_TRAINER_COUNT={shlex.quote(str(len(comp.get('trainer_rank_ids', []))))}")
        lines.append(f"RTT_PL_COMP_{cid}_STUDENT_IDS={shlex.quote(student_ids)}")
        lines.append(f"RTT_PL_COMP_{cid}_STUDENT_COUNT={shlex.quote(str(len(comp.get('student_replica_ids', []))))}")
        lines.append(f"RTT_PL_COMP_{cid}_TEACHER_IDS={shlex.quote(teacher_ids)}")
        lines.append(f"RTT_PL_COMP_{cid}_TEACHER_COUNT={shlex.quote(str(len(comp.get('teacher_replica_ids', []))))}")
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_shell = sub.add_parser('shell', help='Write placement manifest and emit shell-safe assignments.')
    p_shell.add_argument('--config', required=True)

    p_plan = sub.add_parser('plan', help='Write placement manifest and print it.')
    p_plan.add_argument('--config', required=True)

    p_runtime = sub.add_parser('runtime-shell', help='Emit shell-safe assignments from an existing manifest.')
    p_runtime.add_argument('--manifest', required=True)

    args = ap.parse_args()

    if args.cmd == 'shell':
        cfg = load_config(args.config)
        print(emit_shell_for_config(cfg))
    elif args.cmd == 'plan':
        cfg = load_config(args.config)
        path = write_manifest(cfg)
        print(path.read_text(encoding='utf-8'))
    elif args.cmd == 'runtime-shell':
        print(emit_runtime_shell(args.manifest))


if __name__ == '__main__':
    main()
