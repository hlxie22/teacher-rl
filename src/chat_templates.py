from __future__ import annotations

from typing import Any, Dict, List, Optional


def render_messages(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    if hasattr(tokenizer, 'apply_chat_template'):
        kwargs = dict(chat_template_kwargs or {})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    if add_generation_prompt:
        parts.append('ASSISTANT:')
    return '\n\n'.join(parts)
