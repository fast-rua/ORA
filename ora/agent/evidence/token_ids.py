"""Token ids of every policy call, from the archived stream or the serving engine's ``/tokenize``.

A provider profile with ``request_options: {return_token_ids: true}`` (vLLM)
makes the engine stream the prompt's token ids in its first chunk and each
sampled token's id in its delta, so the archived response already holds both.
For any other run, ``Tokenizer`` sends each archived wire request's chat-template
inputs to an engine's ``/tokenize`` (vLLM and sglang both take ``messages`` and
``tools``); that gives the prompt ids, and the sampled tokens stay a count.

Nothing here knows a model. Every call is checked against the usage the engine
reported for it: the prompt ids must have its ``prompt_tokens``, either
directly or after each image's single placeholder token is counted as the
engine's ``prompt_tokens_details.image_tokens`` (sglang tokenizes before it
expands images), and the sampled ids its ``completion_tokens``. A difference is
recorded on the row, never dropped.

Rows store a prompt as the longest prefix it shares with the previous call's
prompt plus its own suffix, with the sha256 of the whole sequence as
little-endian uint32; ``prompt_ids`` rebuilds it.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Callable, Iterable, Iterator, Mapping

import httpx

from .canonical import TrajectoryError

#: The wire request fields an engine's chat template reads (the rest are sampling and transport).
TEMPLATE_FIELDS = ("model", "messages", "tools", "chat_template_kwargs", "reasoning_effort")
#: ``reasoning_effort`` values a chat endpoint treats as unset (vLLM ``renderers.merge_kwargs``).
UNSET_EFFORTS = (None, "auto")


def tokenize_payload(body: Mapping[str, Any]) -> dict[str, Any]:
    """The tokenize request for one chat request, rendered the way the chat endpoint renders it.

    A tokenize request has no ``reasoning_effort``: vLLM's chat endpoint merges
    it into the template kwargs (and ``enable_thinking`` unless the request set
    it), so the payload carries it there. Sent as a top-level field it is
    ignored and the template renders its default effort, a prompt of the same
    length with one token changed (found on 2026-10-29 against streamed ids).
    """

    payload = {field: body[field] for field in ("model", "messages", "tools") if field in body}
    kwargs = dict(body.get("chat_template_kwargs") or {})
    effort = body.get("reasoning_effort")
    if effort not in UNSET_EFFORTS:
        kwargs["reasoning_effort"] = effort
        kwargs.setdefault("enable_thinking", effort != "none")
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    return payload


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def _ids(value: Any) -> list[int] | None:
    if isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                                       for item in value):
        return value
    return None


def stream_token_ids(response: Any) -> tuple[list[int] | None, list[int] | None]:
    """The prompt and sampled token ids an archived Chat Completions stream or body carries, else None."""

    if isinstance(response, Mapping):
        choices = response.get("choices") if isinstance(response.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
        return _ids(response.get("prompt_token_ids")), _ids(choice.get("token_ids"))
    if not isinstance(response, list):
        return None, None
    prompt: list[int] | None = None
    output: list[int] = []
    sampled = False
    for chunk in response:
        if not isinstance(chunk, Mapping):
            continue
        if prompt is None:
            prompt = _ids(chunk.get("prompt_token_ids"))
        for choice in chunk.get("choices") or []:
            # vLLM sends "token_ids": null in every delta when the request did not ask for them.
            if isinstance(choice, Mapping) and isinstance(choice.get("token_ids"), list):
                sampled = True
                output += _ids(choice["token_ids"]) or []
    return prompt, (output if sampled else None)


def without_token_ids(response: Any, *, prompt: bool, sampled: bool) -> tuple[Any, bool]:
    """An archived stream or body with its prompt and/or sampled token ids null, and whether it had any."""

    dropped = False

    def strip(chunk: Any) -> Any:
        nonlocal dropped
        if not isinstance(chunk, Mapping):
            return chunk
        out = dict(chunk)
        if prompt and out.get("prompt_token_ids") is not None:
            out["prompt_token_ids"], dropped = None, True
        if sampled and isinstance(out.get("choices"), list):
            choices = []
            for choice in out["choices"]:
                if isinstance(choice, Mapping) and choice.get("token_ids") is not None:
                    choice, dropped = {**choice, "token_ids": None}, True
                choices.append(choice)
            out["choices"] = choices
        return out

    out = [strip(chunk) for chunk in response] if isinstance(response, list) else strip(response)
    return (out, True) if dropped else (response, False)


class Tokenizer:
    """An engine's ``/tokenize`` for the requests of one run (a chat-template request, not text)."""

    def __init__(self, url: str, api_key: str | None = None, *, timeout: float = 300.0,
                 client: httpx.Client | None = None) -> None:
        self.url = url
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def describe(self) -> dict[str, Any]:
        return {"url": self.url, "fields": list(TEMPLATE_FIELDS), "reasoning_effort": "chat_template_kwargs"}

    def __call__(self, call_id: str, body: Mapping[str, Any]) -> list[int]:
        payload = tokenize_payload(body)
        try:
            response = self._client.post(self.url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise TrajectoryError(f"tokenize failed for {call_id}: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise TrajectoryError(f"tokenize failed for {call_id}: HTTP {response.status_code}: {response.text[:200]}")
        ids = _ids(response.json().get("tokens"))
        if ids is None:
            raise TrajectoryError(f"tokenize answered {call_id} without a token id list")
        return ids

    def close(self) -> None:
        self._client.close()


def _usage(reply: Mapping[str, Any] | None) -> dict[str, int | None]:
    usage = (reply or {}).get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}

    def count(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    return {"input_tokens": count(usage.get("prompt_tokens")), "output_tokens": count(usage.get("completion_tokens")),
            "image_tokens": count(details.get("image_tokens"))}


def _images(body: Mapping[str, Any]) -> int:
    return sum(1 for message in body.get("messages") or [] if isinstance(message, Mapping)
               and isinstance(message.get("content"), list)
               for part in message["content"] if isinstance(part, Mapping) and part.get("type") == "image_url")


def _image_form(length: int, images: int, usage: Mapping[str, int | None]) -> str | None:
    expected = usage["input_tokens"]
    if expected is None:
        return None
    if length == expected:
        return "expanded"
    if images and usage["image_tokens"] is not None and length - images + usage["image_tokens"] == expected:
        return "placeholder"
    return None


def _common_prefix(a: list[int], b: list[int]) -> int:
    size = min(len(a), len(b))
    for position in range(size):
        if a[position] != b[position]:
            return position
    return size


class TokenChain:
    """The token rows of one run's policy calls, fed one call at a time in trace order.

    ``tokenize(model_call_id, body)`` gives prompt ids for a call whose stream
    has none; a call whose stream has them and that is tokenized too records
    whether both agree, and the tokenized ids when they do not. A call that is
    not ``tokenizable`` (another wire protocol) keeps only what its stream has.
    """

    def __init__(self, tokenize: Callable[[str, Mapping[str, Any]], list[int] | None] | None = None) -> None:
        self.tokenize = tokenize
        self.previous: tuple[str, list[int], list[int] | None] | None = None

    def row(self, call_id: str, body: Mapping[str, Any], response: Any,
            reply: Mapping[str, Any] | None, *, tokenizable: bool = True) -> dict[str, Any]:
        usage = _usage(reply)
        streamed, output = stream_token_ids(response)
        tokenized = self.tokenize(call_id, body) if self.tokenize is not None and tokenizable else None
        prompt = streamed if streamed is not None else tokenized
        previous = self.previous
        row: dict[str, Any] = {"model_call_id": call_id, "usage": usage, "prompt": None, "output": None}
        checks: dict[str, Any] = {"prompt_length": None, "output_length": None, "tokenize_equals_stream": None}
        if prompt is not None:
            images = _images(body)
            form = _image_form(len(prompt), images, usage)
            prefix = 0 if previous is None else _common_prefix(previous[1], prompt)
            row["prompt"] = {
                "source": "stream" if streamed is not None else "tokenize",
                "length": len(prompt), "sha256": ids_sha256(prompt), "images": images, "image_form": form,
                "prefix": None if previous is None else {"model_call_id": previous[0], "length": prefix},
                "suffix": prompt[prefix:],
            }
            checks["prompt_length"] = None if usage["input_tokens"] is None else form is not None
            if streamed is not None and tokenized is not None:
                checks["tokenize_equals_stream"] = streamed == tokenized
                if streamed != tokenized:
                    row["prompt"]["tokenized"] = {"length": len(tokenized), "sha256": ids_sha256(tokenized),
                                                  "ids": tokenized}
        if output is not None:
            row["output"] = {"source": "stream", "length": len(output), "sha256": ids_sha256(output), "ids": output}
            checks["output_length"] = None if usage["output_tokens"] is None else len(output) == usage["output_tokens"]
        # The token-level append-only relation: this prompt starts with the previous prompt and what it sampled.
        extends = None
        if prompt is not None and previous is not None and previous[2] is not None:
            head = previous[1] + previous[2]
            extends = prompt[:len(head)] == head
        row["checks"] = checks
        row["extends_previous"] = extends
        if prompt is not None:
            self.previous = (call_id, prompt, output)
        return row


def prompt_ids(rows: Iterable[Mapping[str, Any]]) -> Iterator[tuple[str, list[int] | None]]:
    """Every row's whole prompt, rebuilt from its prefix and suffix; raise when a row does not chain."""

    previous: tuple[str, list[int]] | None = None
    for row in rows:
        prompt = row.get("prompt")
        if prompt is None:
            yield str(row.get("model_call_id")), None
            continue
        prefix = prompt.get("prefix")
        if (prefix is None) != (previous is None) or (
                prefix is not None and (prefix.get("model_call_id") != previous[0]
                                        or not 0 <= prefix.get("length", -1) <= len(previous[1]))):
            raise TrajectoryError(f"{row.get('model_call_id')}: prompt prefix does not chain")
        ids = (previous[1][:prefix["length"]] if prefix else []) + list(prompt.get("suffix") or [])
        previous = (str(row.get("model_call_id")), ids)
        yield previous[0], ids


def summarize(rows: list[Mapping[str, Any]], tokenizer: Mapping[str, Any] | None) -> dict[str, Any]:
    def tally(values: Iterable[Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for value in values:
            key = "none" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    return {
        "calls": len(rows),
        "prompt_ids": tally((row["prompt"] or {}).get("source") for row in rows),
        "output_ids": sum(1 for row in rows if row["output"] is not None),
        "image_form": tally((row["prompt"] or {}).get("image_form") for row in rows if row["prompt"]),
        "prompt_length": tally(row["checks"]["prompt_length"] for row in rows),
        "output_length": tally(row["checks"]["output_length"] for row in rows),
        "tokenize_equals_stream": tally(row["checks"]["tokenize_equals_stream"] for row in rows),
        "extends_previous": tally(row["extends_previous"] for row in rows),
        "tokenizer": None if tokenizer is None else dict(tokenizer),
    }


__all__ = ["TEMPLATE_FIELDS", "TokenChain", "Tokenizer", "ids_sha256", "prompt_ids", "stream_token_ids", "summarize",
           "tokenize_payload", "without_token_ids"]
