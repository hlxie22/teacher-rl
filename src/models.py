# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat_templates import render_messages

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


@dataclass
class LM:
    model: Any
    tokenizer: Any


def load_lm(
    model_id: str,
    device: str,
    load_in_4bit: bool,
    bf16: bool = True,
    trust_remote_code: bool = True,
) -> LM:
    torch_dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16

    quant_config = None
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError('BitsAndBytesConfig unavailable; install bitsandbytes/transformers.')
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                'load_in_4bit=True requires bitsandbytes to be installed in the train environment.'
            ) from e

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=(None if quant_config is not None else torch_dtype),
        quantization_config=quant_config,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return LM(model=model, tokenizer=tok)


def maybe_apply_chat_template(tokenizer, messages: List[dict], enable_thinking: bool = True) -> str:
    return render_messages(
        tokenizer,
        messages,
        add_generation_prompt=True,
        chat_template_kwargs={'enable_thinking': bool(enable_thinking)},
    )


def _truncate_at_stop_strings(text: str, stop_strings: Optional[Sequence[str]]) -> str:
    if not text or not stop_strings:
        return text

    cut = None
    for stop in stop_strings:
        s = str(stop)
        if not s:
            continue
        idx = text.find(s)
        if idx >= 0 and (cut is None or idx < cut):
            cut = idx

    if cut is None:
        return text
    return text[:cut]


@torch.inference_mode()
def generate_text(
    lm: LM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
    stop_strings: Optional[Sequence[str]] = None,
) -> str:
    tok = lm.tokenizer
    model = lm.model
    inputs = tok(prompt, return_tensors='pt', truncation=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0][inputs['input_ids'].shape[1] :]
    text = tok.decode(gen, skip_special_tokens=True)
    return _truncate_at_stop_strings(text, stop_strings)


def _chunks(xs: Sequence[str], bs: int) -> List[List[str]]:
    bs = max(1, int(bs))
    return [list(xs[i : i + bs]) for i in range(0, len(xs), bs)]


@torch.inference_mode()
def generate_text_batch(
    lm: LM,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
    batch_size: int = 8,
    stop_strings: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Batched version of generate_text() for decoder-only LMs.

    IMPORTANT: uses LEFT padding so HF generation appends correctly after prompts.
    Returns a list of decoded generations, one per prompt.
    """
    if not prompts:
        return []

    tok = lm.tokenizer
    model = lm.model

    old_side = getattr(tok, 'padding_side', 'right')
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    outs: List[str] = []
    try:
        for chunk in _chunks(prompts, batch_size):
            inputs = tok(
                list(chunk),
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            prompt_w = inputs['input_ids'].shape[1]
            gen_ids = out[:, prompt_w:]
            decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)
            outs.extend(_truncate_at_stop_strings(text, stop_strings) for text in decoded)
    finally:
        tok.padding_side = old_side

    return outs
