from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PlacementComponent:
    component_id: int
    gpu_count: int = 0
    trainer_rank_ids: List[int] = field(default_factory=list)
    student_replica_ids: List[int] = field(default_factory=list)
    teacher_replica_ids: List[int] = field(default_factory=list)

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            'component_id': int(self.component_id),
            'gpu_count': int(self.gpu_count),
            'trainer_rank_ids': sorted(int(x) for x in self.trainer_rank_ids),
            'student_replica_ids': sorted(int(x) for x in self.student_replica_ids),
            'teacher_replica_ids': sorted(int(x) for x in self.teacher_replica_ids),
        }


class Packer:
    def __init__(self, gpus_per_node: int):
        self.gpus_per_node = int(gpus_per_node)
        if self.gpus_per_node < 1:
            raise ValueError('gpus_per_node must be >= 1')
        self.components: List[PlacementComponent] = []

    def _fits(self, comp: PlacementComponent, width: int) -> bool:
        return int(comp.gpu_count) + int(width) <= self.gpus_per_node

    def _candidate(self, width: int) -> PlacementComponent | None:
        candidates = [c for c in self.components if self._fits(c, width)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (self.gpus_per_node - (int(c.gpu_count) + int(width)), c.component_id),
        )

    def _create(self) -> PlacementComponent:
        comp = PlacementComponent(component_id=len(self.components))
        self.components.append(comp)
        return comp

    def place(self, role: str, width: int, item_id: int) -> None:
        if int(width) < 1:
            raise ValueError(f'placement width must be >= 1, got {width}')
        if int(width) > self.gpus_per_node:
            raise ValueError(
                f'placement width {width} exceeds gpus_per_node={self.gpus_per_node}'
            )
        comp = self._candidate(int(width))
        if comp is None:
            comp = self._create()
        comp.gpu_count = int(comp.gpu_count) + int(width)
        if role == 'trainer':
            comp.trainer_rank_ids.append(int(item_id))
        elif role == 'student':
            comp.student_replica_ids.append(int(item_id))
        elif role == 'teacher':
            comp.teacher_replica_ids.append(int(item_id))
        else:
            raise ValueError(f'unsupported placement role: {role}')


def build_components(
    *,
    gpus_per_node: int,
    trainer_gpus: int,
    student_replicas: int,
    teacher_replicas: int,
    teacher_tp_size: int,
    isolate_teacher_rollout: bool,
) -> List[Dict[str, Any]]:
    if int(gpus_per_node) < 1:
        raise ValueError('gpus_per_node must be >= 1')
    if int(teacher_tp_size) < 1:
        raise ValueError('teacher_tp_size must be >= 1')
    if int(teacher_tp_size) > int(gpus_per_node):
        raise ValueError(
            f'teacher_rollout.tensor_parallel_size={teacher_tp_size} exceeds resources.gpus_per_node={gpus_per_node}'
        )

    if isolate_teacher_rollout:
        non_teacher = Packer(gpus_per_node=gpus_per_node)
        for rank_id in range(int(trainer_gpus)):
            non_teacher.place('trainer', 1, rank_id)
        for rep_id in range(int(student_replicas)):
            non_teacher.place('student', 1, rep_id)

        teacher_only = Packer(gpus_per_node=gpus_per_node)
        for rep_id in range(int(teacher_replicas)):
            teacher_only.place('teacher', int(teacher_tp_size), rep_id)

        comps = non_teacher.components + teacher_only.components
    else:
        packer = Packer(gpus_per_node=gpus_per_node)
        for rep_id in range(int(teacher_replicas)):
            packer.place('teacher', int(teacher_tp_size), rep_id)
        for rank_id in range(int(trainer_gpus)):
            packer.place('trainer', 1, rank_id)
        for rep_id in range(int(student_replicas)):
            packer.place('student', 1, rep_id)
        comps = packer.components

    out: List[Dict[str, Any]] = []
    for cid, comp in enumerate(comps):
        comp.component_id = int(cid)
        out.append(comp.to_manifest_dict())
    return out


def summarize_components(components: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        'trainer_nodes': sum(1 for c in components if len(c.get('trainer_rank_ids', [])) > 0),
        'student_server_nodes': sum(1 for c in components if len(c.get('student_replica_ids', [])) > 0),
        'teacher_rollout_nodes': sum(1 for c in components if len(c.get('teacher_replica_ids', [])) > 0),
        'total_nodes': len(components),
    }
