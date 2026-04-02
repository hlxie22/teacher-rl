from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest


class RemoteVLLMError(RuntimeError):
    pass


@dataclass
class RemoteCompletionChoice:
    text: str
    token_logprobs: List[float]


@dataclass
class RemoteCompletionResult:
    prompt: str
    choices: List[RemoteCompletionChoice]


class RemoteVLLMCompletionsClient:
    def __init__(
        self,
        *,
        base_urls: Sequence[str],
        timeout_s: float = 180.0,
        max_retries: int = 8,
        retry_backoff_s: float = 5.0,
        max_workers: Optional[int] = None,
        request_logprobs: int = 1,
        extra_body: Optional[Dict[str, Any]] = None,
        max_requests_per_server: int = 4,
    ):
        urls = [str(u).strip().rstrip("/") for u in base_urls if str(u).strip()]
        if not urls:
            raise ValueError("Remote teacher rollout requires at least one base URL.")
        self.base_urls = urls
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.request_logprobs = int(request_logprobs)
        self.max_requests_per_server = max(1, int(max_requests_per_server))
        self.max_workers = int(max_workers or (len(urls) * self.max_requests_per_server))
        self.extra_body = dict(extra_body or {})
        self._rr = 0

    def _next_url(self) -> str:
        url = self.base_urls[self._rr % len(self.base_urls)]
        self._rr += 1
        return url

    @staticmethod
    def _extract_token_logprobs(choice: Dict[str, Any]) -> List[float]:
        lp = (choice.get("logprobs") or {}).get("token_logprobs", None)
        if lp is None:
            raise RemoteVLLMError(f"completion response missing logprobs.token_logprobs: {choice}")
        out: List[float] = []
        for x in lp:
            if x is None:
                raise RemoteVLLMError(f"completion response had None token logprob: {choice}")
            out.append(float(x))
        return out

    def _request_one(
        self,
        *,
        base_url: str,
        prompt: str,
        model: str,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> RemoteCompletionResult:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n": int(n),
            "stream": False,
            "logprobs": int(self.request_logprobs),
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)
        if self.extra_body:
            payload.update(self.extra_body)

        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url=f"{base_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = list(data.get("choices") or [])
                if len(choices) != int(n):
                    raise RemoteVLLMError(
                        f"expected {n} choices from {base_url}, got {len(choices)}: {data}"
                    )
                parsed = [
                    RemoteCompletionChoice(
                        text=str(choice.get("text", "")),
                        token_logprobs=self._extract_token_logprobs(choice),
                    )
                    for choice in choices
                ]
                return RemoteCompletionResult(prompt=prompt, choices=parsed)
            except urlerror.HTTPError as e:
                body_txt = ""
                try:
                    body_txt = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_err = RemoteVLLMError(
                    f"teacher vLLM HTTPError {e.code} from {base_url}: {body_txt}"
                )
            except urlerror.URLError as e:
                last_err = RemoteVLLMError(f"teacher vLLM URLError from {base_url}: {e}")
            except Exception as e:
                last_err = e

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_s)

        assert last_err is not None
        raise RemoteVLLMError(str(last_err))

    def generate(
        self,
        prompts: Sequence[str],
        *,
        model: str,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[Sequence[str]] = None,
        seed_base: Optional[int] = None,
    ) -> List[RemoteCompletionResult]:
        prompts = [str(p) for p in prompts]
        if not prompts:
            return []

        results: List[Optional[RemoteCompletionResult]] = [None] * len(prompts)
        max_workers = max(1, min(self.max_workers, len(prompts)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for i, prompt in enumerate(prompts):
                base_url = self._next_url()
                seed = None if seed_base is None else int(seed_base) + i
                fut = ex.submit(
                    self._request_one,
                    base_url=base_url,
                    prompt=prompt,
                    model=model,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                )
                futs.append((i, fut))
            for i, fut in futs:
                results[i] = fut.result()
        return [x for x in results if x is not None]


def build_rollout_model_name(global_step: int) -> str:
    return f"step_{int(global_step):08d}"
