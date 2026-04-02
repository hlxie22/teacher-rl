# src/vllm_infer.py
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .chat_templates import render_messages

try:
    from vllm.tokenizers import get_tokenizer  # type: ignore
except Exception:
    from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore


class _CompletionMeta(TypedDict, total=False):
    text: str
    finish_reason: Optional[str]
    completion_tokens: int
    token_ids: List[int]


class _RequestMeta(TypedDict, total=False):
    prompt: str
    prompt_tokens: int
    prompt_token_ids: List[int]
    completions: List[_CompletionMeta]


def _stable_lora_id(path: str) -> int:
    return int(zlib.crc32(path.encode('utf-8')) & 0xFFFFFFFF)


@dataclass
class VLLMChatRunner:
    model_id: str
    tp_size: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float

    enable_lora: bool = False
    max_loras: int = 8
    max_lora_rank: int = 64
    trust_remote_code: bool = True
    attention_backend: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    language_model_only: bool = False
    cpu_offload_gb: float = 0.0
    enforce_eager: bool = False
    disable_log_stats: bool = True

    def __post_init__(self) -> None:
        self.tokenizer = get_tokenizer(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        kwargs: Dict[str, Any] = dict(
            model=self.model_id,
            tensor_parallel_size=int(self.tp_size),
            dtype=self.dtype,
            max_model_len=int(self.max_model_len),
            gpu_memory_utilization=float(self.gpu_memory_utilization),
            trust_remote_code=bool(self.trust_remote_code),
            language_model_only=bool(self.language_model_only),
            cpu_offload_gb=float(self.cpu_offload_gb),
            enforce_eager=bool(self.enforce_eager),
            disable_log_stats=bool(self.disable_log_stats),
        )

        if self.attention_backend:
            kwargs['attention_backend'] = str(self.attention_backend)

        if self.max_num_seqs is not None:
            kwargs['max_num_seqs'] = int(self.max_num_seqs)
        if self.max_num_batched_tokens is not None:
            kwargs['max_num_batched_tokens'] = int(self.max_num_batched_tokens)

        if self.enable_lora:
            kwargs.update(
                enable_lora=True,
                max_loras=int(self.max_loras),
                max_lora_rank=int(self.max_lora_rank),
            )

        self.llm = LLM(**kwargs)

    def generate_from_messages(
        self,
        batch_messages: List[List[Dict[str, str]]],
        sp: SamplingParams,
        lora_path: Optional[str] = None,
        lora_name: str = 'teacher_adapter',
        return_meta: bool = False,
    ):
        prompts = [
            render_messages(
                self.tokenizer,
                msgs,
                add_generation_prompt=True,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            for msgs in batch_messages
        ]

        lora_req = None
        if lora_path:
            lp = Path(lora_path)
            if not lp.exists():
                raise FileNotFoundError(f'LoRA adapter path does not exist: {lora_path}')
            lora_req = LoRARequest(lora_name, _stable_lora_id(str(lp)), str(lp))

        outputs = self.llm.generate(prompts, sp, lora_request=lora_req)

        if not return_meta:
            return [out.outputs[0].text for out in outputs]

        metas: List[_RequestMeta] = []
        for out in outputs:
            pm: _RequestMeta = {'prompt': getattr(out, 'prompt', '')}
            ptoks = getattr(out, 'prompt_token_ids', None)
            if ptoks is not None:
                pm['prompt_token_ids'] = list(ptoks)
                pm['prompt_tokens'] = int(len(ptoks))

            comps: List[_CompletionMeta] = []
            for c in getattr(out, 'outputs', []) or []:
                cm: _CompletionMeta = {'text': getattr(c, 'text', '')}
                cm['finish_reason'] = getattr(c, 'finish_reason', None)
                tids = getattr(c, 'token_ids', None)
                if tids is not None:
                    cm['token_ids'] = list(tids)
                    cm['completion_tokens'] = int(len(tids))
                comps.append(cm)

            pm['completions'] = comps
            metas.append(pm)

        return metas
