from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .dist_utils import (
    dist_barrier,
    gather_objects_to_rank0,
    is_dist,
    rank,
    scatter_object_from_rank0,
    world_size,
)
from .remote_vllm import RemoteVLLMCompletionsClient, build_rollout_model_name
from .utils import RTT_STOP_SEQUENCE


@dataclass
class _PromptEnvelope:
    owner_rank: int
    owner_local_index: int
    generation_step_index: int
    prompt_text: str


class DistributedRolloutCoordinator:
    """
    Rank-0 rollout coordinator.

    Each rank contributes its raw local prompt slice. Rank 0 gathers all prompts,
    dispatches remote rollout generation centrally, then scatters each rank's
    completion payload back in the original local order.
    """

    def __init__(
        self,
        *,
        teacher_remote: RemoteVLLMCompletionsClient,
        teacher_tokenizer: Any,
        per_device_train_batch_size: int,
        steps_per_generation: int,
        base_seed: int,
        step_offset: int,
    ):
        self.teacher_remote = teacher_remote
        self.teacher_tokenizer = teacher_tokenizer
        self.per_device_train_batch_size = max(1, int(per_device_train_batch_size))
        self.steps_per_generation = max(1, int(steps_per_generation))
        self.base_seed = int(base_seed)
        self.step_offset = int(step_offset)

    def _build_local_payload(self, prompts: Sequence[str]) -> Dict[str, Any]:
        local_prompts = [str(p) for p in prompts]
        my_rank = rank()
        items = []
        for local_idx, prompt in enumerate(local_prompts):
            items.append(
                {
                    'owner_rank': my_rank,
                    'owner_local_index': int(local_idx),
                    'generation_step_index': int(local_idx // self.per_device_train_batch_size),
                    'prompt_text': prompt,
                }
            )
        return {
            'rank': my_rank,
            'prompts': local_prompts,
            'items': items,
        }

    def _build_rank_payload_template(self) -> Dict[str, List[Any]]:
        return {
            'prompt_ids': [],
            'completion_ids': [],
            'logprobs': [],
            'teacher_rollout_model': [],
            'rollout_owner_rank': [],
            'rollout_owner_local_prompt_index': [],
            'rollout_generation_step_index': [],
        }

    def generate(self, prompts: Sequence[str], trainer: Any) -> Dict[str, List[Any]]:
        local_payload = self._build_local_payload(prompts)
        gathered = gather_objects_to_rank0(local_payload)

        rank_payloads: Optional[List[Dict[str, List[Any]]]] = None
        rollout_step = self.step_offset + int(getattr(trainer.state, 'global_step', 0) or 0)
        rollout_model = build_rollout_model_name(rollout_step)

        if rank() == 0:
            assert gathered is not None
            gathered_sorted = sorted(gathered, key=lambda x: int(x.get('rank', 0)))

            flat_items: List[_PromptEnvelope] = []
            rank_payloads = [self._build_rank_payload_template() for _ in range(world_size())]
            expected_local_prompt_counts = [0 for _ in range(world_size())]

            for payload in gathered_sorted:
                owner_rank = int(payload['rank'])
                items = list(payload.get('items', []))
                expected_local_prompt_counts[owner_rank] = len(items)
                for item in items:
                    flat_items.append(
                        _PromptEnvelope(
                            owner_rank=int(item['owner_rank']),
                            owner_local_index=int(item['owner_local_index']),
                            generation_step_index=int(item['generation_step_index']),
                            prompt_text=str(item['prompt_text']),
                        )
                    )

            flat_prompts = [item.prompt_text for item in flat_items]
            seed_base = self.base_seed + (rollout_step * 100003)
            remote_results = self.teacher_remote.generate(
                flat_prompts,
                model=rollout_model,
                n=int(trainer.num_generations),
                max_tokens=int(trainer.max_completion_length),
                temperature=float(trainer.temperature),
                top_p=float(trainer.top_p),
                stop=[RTT_STOP_SEQUENCE],
                seed_base=seed_base,
            )

            if len(remote_results) != len(flat_items):
                raise RuntimeError(
                    f'rank-0 rollout coordinator expected {len(flat_items)} prompt results, got {len(remote_results)}'
                )

            for item, res in zip(flat_items, remote_results):
                prompt_ids = list(self.teacher_tokenizer.encode(item.prompt_text, add_special_tokens=False))
                owner_payload = rank_payloads[item.owner_rank]
                for choice in res.choices:
                    completion_ids = list(self.teacher_tokenizer.encode(choice.text, add_special_tokens=False))
                    if len(completion_ids) != len(choice.token_logprobs):
                        raise RuntimeError(
                            'remote teacher completion tokenization length mismatch: '
                            f'model={rollout_model} owner_rank={item.owner_rank} '
                            f'owner_local_index={item.owner_local_index} encoded={len(completion_ids)} '
                            f'logprobs={len(choice.token_logprobs)}'
                        )
                    owner_payload['prompt_ids'].append(prompt_ids)
                    owner_payload['completion_ids'].append(completion_ids)
                    owner_payload['logprobs'].append(list(choice.token_logprobs))
                    owner_payload['teacher_rollout_model'].append(rollout_model)
                    owner_payload['rollout_owner_rank'].append(item.owner_rank)
                    owner_payload['rollout_owner_local_prompt_index'].append(item.owner_local_index)
                    owner_payload['rollout_generation_step_index'].append(item.generation_step_index)

            expected_completions_per_rank = [count * int(trainer.num_generations) for count in expected_local_prompt_counts]
            for rank_idx, expected_count in enumerate(expected_completions_per_rank):
                found_count = len(rank_payloads[rank_idx]['completion_ids'])
                if found_count != expected_count:
                    raise RuntimeError(
                        'rank-0 rollout coordinator produced an incomplete scatter payload: '
                        f'rank={rank_idx} expected={expected_count} found={found_count}'
                    )

        local_result = scatter_object_from_rank0(rank_payloads or [self._build_rank_payload_template()])
        if not isinstance(local_result, dict):
            raise RuntimeError(f'scatter returned unexpected payload type: {type(local_result).__name__}')
        dist_barrier()
        return local_result
