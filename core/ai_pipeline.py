from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, List, Optional, Sequence

import requests


try:
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


HARD_JSON_GUARDRAIL = (
    "NON-NEGOTIABLE OUTPUT CONTRACT:\n"
    "- Reply with one valid JSON array only.\n"
    "- No markdown fences, no prose, no notes.\n"
    "- Each item must be exactly [start, end, score].\n"
    "- start/end/score must be numeric.\n"
    "- If no valid clips exist, return []."
)

RELEVANCE_GUARDRAIL = (
    "RELEVANCE CONTRACT:\n"
    "- Pick clips with meaningful spoken sentence-level content that matches the user query.\n"
    "- Reject clips that are mostly music, intro/outro filler, grunts, or non-verbal effort sounds.\n"
    "- Respect explicit duration constraints written in the user query.\n"
    "- Prefer coherent complete thoughts with natural start/end boundaries.\n"
    "- For 'at least X seconds', treat X as a minimum floor, not an exact target."
)

NUMERIC_TRIPLE_PATTERN = re.compile(
    r"\[\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*,\s*"
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*,\s*"
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*\]"
)


def estimate_tokens(text: str) -> int:
    text = str(text)
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    return max(1, math.floor(len(text) / 3.5))


def chunk_transcript(
    transcript: Sequence[Sequence[Any]],
    max_tokens: int,
    safety_margin: int = 300,
    overlap_segments: int = 0,
) -> List[List[List[Any]]]:
    """
    Split transcript into chunk lists while respecting a max token budget.
    Transcript format: [[start, end, text], ...]
    """
    budget = max(256, int(max_tokens) - max(0, int(safety_margin)))
    chunks: List[List[List[Any]]] = []
    current_chunk: List[List[Any]] = []
    current_tokens = 0

    for segment in transcript:
        if len(segment) < 3:
            continue
        start, end, text = segment[0], segment[1], str(segment[2])
        serialized = f"{start} {end} {text}"
        seg_tokens = estimate_tokens(serialized)

        if seg_tokens > budget:
            words = text.split()
            if not words:
                continue
            words_per_window = max(8, int(budget * 0.8))
            for index in range(0, len(words), words_per_window):
                mini_words = words[index : index + words_per_window]
                mini_text = " ".join(mini_words)
                mini_tokens = estimate_tokens(mini_text)
                if current_tokens + mini_tokens > budget and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append([start, end, mini_text])
                current_tokens += mini_tokens
            continue

        if current_tokens + seg_tokens > budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append([start, end, text])
        current_tokens += seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    overlap = max(0, int(overlap_segments))
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped_chunks: List[List[List[Any]]] = [chunks[0]]
    for index in range(1, len(chunks)):
        previous = chunks[index - 1]
        current = chunks[index]
        prefix = [list(item) for item in previous[-overlap:]]
        merged = prefix + [list(item) for item in current]
        while prefix and estimate_tokens(json.dumps(merged, ensure_ascii=True)) > budget:
            prefix = prefix[1:]
            merged = prefix + [list(item) for item in current]
        overlapped_chunks.append(merged)

    return overlapped_chunks


def _clean_model_output(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Model returned empty response.")

    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", maxsplit=1)[-1].strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _extract_json_text(raw_text: str) -> str:
    cleaned = _clean_model_output(raw_text)

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")
    return cleaned[start : end + 1]


def _validate_clip_schema(payload: Any) -> List[List[float]]:
    if not isinstance(payload, list):
        raise ValueError("Model output must be a JSON list.")

    validated: List[List[float]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"Item #{index} is not a list.")
        if len(item) != 3:
            raise ValueError(f"Item #{index} must match [start, end, score].")

        try:
            start = float(item[0])
            end = float(item[1])
            score = float(item[2])
        except Exception as exc:
            raise ValueError(f"Item #{index} has non-numeric values: {exc}") from exc

        if end <= start:
            raise ValueError(f"Item #{index} has invalid timing: end <= start.")

        normalized_score = max(1.0, min(10.0, score))
        validated.append([start, end, normalized_score])

    deduped: List[List[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for start, end, score in sorted(validated, key=lambda row: (row[0], row[1], -row[2])):
        key = (round(start, 3), round(end, 3), round(score, 2))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([start, end, score])

    return deduped


def _parse_from_lines(cleaned: str) -> List[List[float]]:
    parsed_items: List[List[float]] = []
    for line in cleaned.splitlines():
        candidate = line.strip().rstrip(",")
        if not candidate:
            continue
        try:
            line_payload = json.loads(candidate)
        except Exception:
            continue

        if isinstance(line_payload, list) and line_payload and isinstance(line_payload[0], (int, float)):
            parsed_items.append(line_payload)
            continue
        if isinstance(line_payload, list):
            for item in line_payload:
                if isinstance(item, list):
                    parsed_items.append(item)

    if not parsed_items:
        return []
    return _validate_clip_schema(parsed_items)


def _parse_from_numeric_triples(cleaned: str) -> List[List[float]]:
    matches = NUMERIC_TRIPLE_PATTERN.findall(cleaned)
    if not matches:
        return []
    triples: List[List[float]] = []
    for match in matches:
        try:
            payload = json.loads(match)
        except Exception:
            continue
        if isinstance(payload, list):
            triples.append(payload)
    if not triples:
        return []
    return _validate_clip_schema(triples)


def parse_clip_response(raw_text: str) -> List[List[float]]:
    cleaned = _clean_model_output(raw_text)
    parse_errors: List[str] = []

    candidates: List[str] = [cleaned]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return _validate_clip_schema(parsed)
        except Exception as exc:
            parse_errors.append(str(exc))

    parsed_from_lines = _parse_from_lines(cleaned)
    parsed_from_triples = _parse_from_numeric_triples(cleaned)
    if parsed_from_lines and parsed_from_triples:
        merged = parsed_from_lines + parsed_from_triples
        return _validate_clip_schema(merged)
    if parsed_from_lines:
        return parsed_from_lines
    if parsed_from_triples:
        return parsed_from_triples

    details = "; ".join(parse_errors[-2:]) if parse_errors else "No valid JSON structure detected."
    raise ValueError(details)


class AIPipeline:
    """Ollama chat + strict JSON parsing with retry/repair logic."""

    def __init__(
        self,
        model: str,
        ollama_url: str,
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
        max_context_tokens: int,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens
        self.logger = logger

    def _resolve_think_setting(self) -> Any:
        """
        Request lowest-thinking mode for latency across models.
        If a model/runtime does not support `think`, `_chat` retries without it.
        """
        return "low"

    def _chat(
        self,
        prompt: str,
        system_message: str,
        force_json: bool = False,
        num_predict_override: Optional[int] = None,
    ) -> str:
        temperature = self.temperature if not force_json else min(self.temperature, 0.15)
        predict_tokens = int(num_predict_override or self.max_output_tokens)
        predict_tokens = max(32, predict_tokens)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": self._resolve_think_setting(),
            "options": {
                "temperature": temperature,
                "num_predict": predict_tokens,
                "num_ctx": self.max_context_tokens,
            },
        }
        if force_json:
            payload["format"] = "json"

        endpoint = f"{self.ollama_url}/api/chat"
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=180,
        )
        if response.status_code >= 400 and "think" in payload:
            self.logger.debug("Model rejected explicit think setting; retrying chat request without `think`.")
            fallback_payload = dict(payload)
            fallback_payload.pop("think", None)
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(fallback_payload),
                timeout=180,
            )
        response.raise_for_status()
        body = response.json()
        message = body.get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            content = str(body.get("response", "")).strip()
        return content

    def _scan_chunk_once(
        self,
        chunk: Sequence[Sequence[Any]],
        user_query: str,
        strict_mode: bool = False,
    ) -> str:
        chunking_context = (
            "Chunking context:\n"
            "- You are seeing one chunk from a larger transcript.\n"
            "- Avoid selecting clips from trailing chunk edges if context appears incomplete.\n"
            "- Output only JSON."
        )
        prompt = (
            f"Transcript JSON chunk:\n{json.dumps(chunk, ensure_ascii=True)}\n\n"
            f"User query:\n{user_query}\n\n"
            "Return only JSON list of clips: [[start, end, score], ...]."
        )
        system_message = f"{self.system_prompt}\n\n{chunking_context}\n\n{RELEVANCE_GUARDRAIL}"
        if strict_mode:
            system_message = f"{system_message}\n\n{HARD_JSON_GUARDRAIL}"
        return self._chat(
            prompt=prompt,
            system_message=system_message,
            force_json=strict_mode,
            num_predict_override=min(self.max_output_tokens, 220),
        )

    def scan_chunk_with_retries(
        self,
        chunk: Sequence[Sequence[Any]],
        user_query: str,
        max_retries: int = 3,
    ) -> List[List[float]]:
        prompt_hint = ""
        for attempt in range(1, max_retries + 1):
            strict_mode = attempt > 1
            try:
                raw = self._scan_chunk_once(
                    chunk,
                    f"{user_query}\n{prompt_hint}".strip(),
                    strict_mode=strict_mode,
                )
            except Exception as exc:
                self.logger.warning("Ollama request failed on attempt %s: %s", attempt, exc)
                prompt_hint = (
                    "Previous request failed. Follow the non-negotiable JSON output contract exactly."
                )
                continue

            self.logger.debug(
                "Raw model output (attempt %s, %s chars): %s",
                attempt,
                len(raw),
                raw[:2500],
            )

            try:
                return parse_clip_response(raw)
            except Exception as exc:
                self.logger.warning("Failed to parse AI response on attempt %s: %s", attempt, exc)
                prompt_hint = (
                    "Your previous response violated the format. "
                    "Return exactly one valid JSON array with schema: [[start, end, score], ...]."
                )
                continue

        self.logger.error("Failed to parse valid AI clip JSON after %s attempts.", max_retries)
        return []

    def scan_all_chunks(
        self,
        chunks: Sequence[Sequence[Sequence[Any]]],
        user_query: str,
        ai_loops: int,
        progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    ) -> List[List[List[float]]]:
        all_outputs: List[List[List[float]]] = []
        loop_count = max(1, int(ai_loops))
        total_chunks = len(chunks)
        for chunk_index, chunk in enumerate(chunks, start=1):
            combined: List[List[float]] = []
            for loop_index in range(1, loop_count + 1):
                output = self.scan_chunk_with_retries(chunk=chunk, user_query=user_query)
                if output:
                    combined.extend(output)
                if progress_callback:
                    progress_callback(
                        chunk_index,
                        total_chunks,
                        loop_index,
                        loop_count,
                        len(output),
                    )
            all_outputs.append(combined)
        return all_outputs
