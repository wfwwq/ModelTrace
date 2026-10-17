from __future__ import annotations

import json
import math
import os
import random
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from fingerprint import analyze_global_outputs, generate_challenges, parse_numbers
from bank_builder import build_bank, read_rows
from challenge_suite import fingerprint_suite


PROJECT = Path(__file__).resolve().parent
DATA_FILE = PROJECT / "data" / "gpt_reference.jsonl"
BANK_FILE = PROJECT / "data" / "gpt_bank.json"

# urllib 默认的 Python-urllib User-Agent 会被 Cloudflare/WAF 网关直接拦成 403，
# 因此伪装成真实客户端（与 gpt56 检测器使用的 UA 一致）。
DEFAULT_UPSTREAM_USER_AGENT = (
    "Codex Desktop/0.147.0-alpha.1.2 (Windows 10.0.26200; x86_64) unknown "
    "(codex_exec; 0.147.0-alpha.1.2)"
)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0


def upstream_user_agent() -> str:
    override = (
        os.environ.get("MODELTRACE_USER_AGENT")
        or os.environ.get("GPT56_USER_AGENT")
        or ""
    ).strip()
    return override or DEFAULT_UPSTREAM_USER_AGENT


def bank_summary(bank: dict) -> dict:
    return {
        "model_count": len(bank["models"]),
        "response_count": sum(model["response_count"] for model in bank["models"]),
        "number_count": sum(model["valid_number_count"] for model in bank["models"]),
        "models": [
            {
                "id": model["id"],
                "display_name": model["display_name"],
                "responses": model["response_count"],
                "valid_numbers": model["valid_number_count"],
                "channels": model.get("channels", {}),
            }
            for model in bank["models"]
        ],
    }


def make_row(
    model_label: str,
    text: str,
    condition: str,
    challenge_id: str,
    expected_count: int = 0,
    temperature: str | float = "unknown",
    bank_id: str = "reference-bank",
    wrapper_transport: str | None = None,
    provider: str = "api",
    prompt: str | None = None,
    base_prompt: str | None = None,
    system_prompt: str | None = None,
    user_prefix: str | None = None,
    channel_id: str | None = None,
    channel_name: str | None = None,
) -> dict:
    numbers = parse_numbers(text)
    threshold = max(80, math.ceil(expected_count * 0.55)) if expected_count else 80
    row_id = f"{condition}-{secrets.token_hex(8)}"
    return {
        "row_id": row_id,
        "parent_row_id": row_id,
        "bank_id": bank_id,
        "source": model_label,
        "model_id": model_label,
        "exact_version": model_label,
        "condition_id": condition,
        "nuisance_condition_id": condition,
        "wrapper_id": condition,
        "wrapper_transport": wrapper_transport or ("manual_import" if condition == "manual" else "clean"),
        "provider": "manual" if condition == "manual" else provider,
        "challenge_id": challenge_id,
        "task_index": 0,
        "requested_count": expected_count,
        "parsed_count": len(numbers),
        "strict_threshold": threshold,
        "strict_valid": len(numbers) >= threshold,
        "temperature": temperature,
        "config_id": "enrollment",
        "prompt": prompt,
        "base_prompt": base_prompt,
        "system_prompt": system_prompt,
        "user_prefix": user_prefix,
        **({"channel_id": channel_id, "channel_name": channel_name or channel_id} if channel_id else {}),
        "text": text,
        "error": None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def append_rows(rows: list[dict], data_file: Path = DATA_FILE) -> None:
    with data_file.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rebuild_bank(data_file: Path = DATA_FILE, bank_file: Path = BANK_FILE) -> dict:
    previous = json.loads(bank_file.read_text(encoding="utf-8")) if bank_file.exists() else None
    bank = build_bank(read_rows(data_file), previous["calibration"] if previous else None)
    bank_file.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bank


def enroll_manual(
    model_label: str,
    pasted: str,
    data_file: Path = DATA_FILE,
    bank_file: Path = BANK_FILE,
    bank_id: str = "reference-bank",
) -> dict:
    blocks = [
        block.strip()
        for block in re.split(r"(?m)^\s*===OUTPUT===\s*$", pasted)
        if block.strip()
    ]
    rows = [
        make_row(
            model_label=model_label,
            text=block,
            condition="manual",
            challenge_id=f"manual-{secrets.token_hex(6)}",
            bank_id=bank_id,
        )
        for block in blocks
    ]
    accepted = [row for row in rows if row["strict_valid"]]
    append_rows(accepted, data_file)
    bank = rebuild_bank(data_file, bank_file)
    return {
        "submitted": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "parsed_numbers": [row["parsed_count"] for row in rows],
        "bank": bank_summary(bank),
    }


def completion_url(base_url: str, api_format: str = "openai") -> str:
    normalized = base_url.rstrip("/")
    if api_format == "anthropic":
        if normalized.endswith("/messages"):
            return normalized
        if normalized.endswith("/v1"):
            return normalized + "/messages"
        return normalized + "/v1/messages"
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def _looks_like_waf_block(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("cloudflare", "just a moment", "cf-ray", "access denied", "attention required")
    )


def _compact_upstream_error(details: str, fallback: str) -> str:
    text = (details or fallback or "").strip()
    if not text:
        return "上游接口返回错误"
    if _looks_like_waf_block(text):
        return (
            "请求被上游网关拦截（Cloudflare/WAF 拦截页）。"
            "请确认 base_url 指向 API 端点而非网页地址、API Key 有效，"
            "或该服务是否限制当前网络/IP"
        )
    if text.startswith("<") or "{" not in text and "html" in text.lower():
        return text[:200]
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
            return json.dumps(payload, ensure_ascii=False)[:500]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return text[:500]


def _request_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str,
    system_prompt: str = "",
) -> str:
    if api_format == "anthropic":
        body_data = {
            "model": api_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body_data["system"] = system_prompt
        headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    else:
        body_data = {
            "model": api_model,
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": upstream_user_agent(),
        }
    if temperature is not None:
        body_data["temperature"] = temperature
    body = json.dumps(body_data).encode("utf-8")
    url = completion_url(base_url, api_format)
    payload = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            message = _compact_upstream_error(details, error.reason)
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            if attempt < MAX_ATTEMPTS and error.code in RETRYABLE_STATUS:
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            raise RuntimeError(f"HTTP {error.code}: {message}{retried}") from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", str(error))
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5))
                continue
            retried = f"（已自动重试 {attempt - 1} 次）" if attempt > 1 else ""
            raise RuntimeError(f"无法连接接口：{reason}{retried}") from error
    if api_format == "anthropic":
        content = "".join(
            block.get("text", "")
            for block in payload["content"]
            if block.get("type") == "text"
        )
        stop_reason = payload.get("stop_reason")
        if stop_reason == "refusal":
            raise RuntimeError("模型拒绝生成，本次回答不计入")
        if stop_reason == "max_tokens":
            raise RuntimeError("回答因 max_tokens 截断，本次回答不计入")
    else:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise RuntimeError(f"回答未正常完成（{choice['finish_reason']}），本次回答不计入")
    return str(content)


def request_completion(
    base_url: str,
    api_key: str,
    api_model: str,
    prompt: str,
    temperature: float | None,
    api_format: str = "auto",
    system_prompt: str = "",
) -> str:
    if api_format != "auto":
        return _request_completion(
            base_url, api_key, api_model, prompt, temperature, api_format, system_prompt
        )
    formats = ("openai", "anthropic")
    errors = []
    for candidate in formats:
        try:
            return _request_completion(
                base_url, api_key, api_model, prompt, temperature, candidate, system_prompt
            )
        except RuntimeError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("接口格式自动探测失败；" + "；".join(errors))


def test_automatic(
    base_url: str,
    api_key: str,
    api_model: str,
    temperature: float | None,
    bank: dict,
    api_format: str = "openai",
) -> dict:
    target_count = 3
    max_attempts = 6
    challenges = generate_challenges(max_attempts)
    outputs = []
    errors = []
    for challenge in challenges:
        try:
            text = request_completion(
                base_url,
                api_key,
                api_model,
                challenge["prompt"],
                temperature,
                api_format,
            )
            minimum = max(80, math.ceil(challenge["expected_count"] * 0.55))
            parsed_count = len(parse_numbers(text))
            if parsed_count >= minimum:
                outputs.append(
                    {
                        "text": text,
                        "expected_count": challenge["expected_count"],
                    }
                )
            else:
                errors.append(f"有效数字不足：{parsed_count}/{minimum}")
        except Exception as error:
            errors.append(str(error))
        if len(outputs) == target_count:
            break
    result = analyze_global_outputs(outputs, bank)
    result["api_test"] = {
        "requested": target_count,
        "attempted": len(outputs) + len(errors),
        "max_attempts": max_attempts,
        "received": len(outputs),
        "errors": errors,
    }
    return result


def select_suite(sample_count: int) -> list[dict]:
    suite = fingerprint_suite()
    if sample_count < 3 or sample_count > len(suite):
        raise ValueError(f"采集回答数必须在 3 到 {len(suite)} 之间")
    return [suite[int(index * len(suite) / sample_count)] for index in range(sample_count)]


def collect_suite(
    base_url: str,
    api_key: str,
    api_model: str,
    model_label: str,
    tasks: list[dict],
    temperature: float | None,
    api_format: str = "auto",
    bank_id: str = "reference-bank",
    provider: str = "api",
    channel_id: str | None = None,
    channel_name: str | None = None,
    workers: int = 2,
) -> tuple[list[dict], list[str]]:
    """按标准挑战集向一个渠道采集回答，返回 (有效行, 错误)。"""
    rows: list[dict] = []
    errors: list[str] = []
    label = f"[{channel_name}] " if channel_name else ""

    def collect(task: dict) -> tuple[dict | None, list[str]]:
        task_errors = []
        base_prompt = task["prompt"]
        prompt = base_prompt
        if task["user_prefix"]:
            prompt = task["user_prefix"] + "\n\nFinal task:\n" + prompt
        system_prompt = task["system"]
        for _ in range(2):
            try:
                text = request_completion(
                    base_url, api_key, api_model, prompt, temperature, api_format, system_prompt,
                )
                row = make_row(
                    model_label=model_label,
                    text=text,
                    condition=task["condition"],
                    challenge_id=task["challenge_id"],
                    expected_count=task["expected_count"],
                    temperature=temperature if temperature is not None else "provider_default",
                    bank_id=bank_id,
                    wrapper_transport=task["transport"],
                    provider=provider,
                    prompt=prompt,
                    base_prompt=base_prompt,
                    system_prompt=system_prompt,
                    user_prefix=task["user_prefix"],
                    channel_id=channel_id,
                    channel_name=channel_name,
                )
                if row["strict_valid"]:
                    row["response_model"] = api_model
                    return row, task_errors
                task_errors.append(
                    f"{label}{task['challenge_id']} 有效数字不足：{row['parsed_count']}/{row['strict_threshold']}"
                )
            except Exception as error:
                task_errors.append(f"{label}{task['challenge_id']}: {error}")
        return None, task_errors

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect, task) for task in tasks]
        for future in as_completed(futures):
            row, task_errors = future.result()
            errors.extend(task_errors)
            if row is not None:
                rows.append(row)
    return rows, errors


def enroll_automatic(
    base_url: str,
    api_key: str,
    api_model: str,
    model_label: str,
    sample_count: int,
    temperature: float | None,
    api_format: str = "openai",
    data_file: Path = DATA_FILE,
    bank_file: Path = BANK_FILE,
    bank_id: str = "reference-bank",
    provider: str = "api",
) -> dict:
    selected = select_suite(sample_count)
    rows, errors = collect_suite(
        base_url, api_key, api_model, model_label, selected, temperature, api_format,
        bank_id=bank_id, provider=provider,
    )
    accepted = [row for row in rows if row["strict_valid"]]
    if not accepted:
        raise ValueError("没有获得可用回答，指纹库未修改")
    append_rows(accepted, data_file)
    bank = rebuild_bank(data_file, bank_file)
    return {
        "requested": sample_count,
        "received": len(accepted),
        "accepted": len(accepted),
        "rejected": sample_count - len(accepted),
        "errors": errors,
        "bank": bank_summary(bank),
    }


def group_channels_by_host(channels: list[dict]) -> dict[str, list[dict]]:
    """按 base_url 主机分组：同一网关上的渠道要排队，免得自己把自己的限流打满。"""
    groups: dict[str, list[dict]] = {}
    for channel in channels:
        host = urllib.parse.urlparse(str(channel.get("base_url") or "").strip()).netloc.lower()
        groups.setdefault(host, []).append(channel)
    return groups


def enroll_multi_channel(
    channels: list[dict],
    model_label: str,
    sample_count: int,
    temperature: float | None,
    bank_id: str,
    data_file: Path,
    bank_file: Path,
    keep_suspect: bool = False,
    dry_run: bool = False,
    api_format: str = "auto",
    channel_workers: int = 2,
) -> dict:
    """从多个渠道采集同一模型的回答，校验渠道一致性后整合并重新校准指纹库。

    channels：[{"name": "官方", "base_url": ..., "api_key": ..., "api_model": ...}, ...]

    多个渠道常常指向同一台中转站：按主机分组，组内串行、组间并行，
    避免同一网关被自己的并发打到限流（403/429）。
    """
    from channel_calibration import integrate_channels, slugify_channel

    if not channels:
        raise ValueError("至少需要一个渠道")
    seen = set()
    prepared = []
    for index, channel in enumerate(channels, start=1):
        name = (channel.get("name") or "").strip() or f"渠道{index}"
        channel_id = slugify_channel(name)
        if channel_id in seen:
            raise ValueError(f"渠道名称重复：{name}")
        seen.add(channel_id)
        for key in ("base_url", "api_key", "api_model"):
            if not str(channel.get(key) or "").strip():
                raise ValueError(f"渠道 {name} 缺少 {key}")
        prepared.append({**channel, "name": name, "channel_id": channel_id})
    tasks = select_suite(sample_count)

    def run(channel: dict) -> tuple[dict, list[dict], list[str]]:
        rows, errors = collect_suite(
            channel["base_url"].strip(),
            channel["api_key"],
            channel["api_model"].strip(),
            model_label,
            tasks,
            temperature,
            api_format,
            bank_id=bank_id,
            provider=channel["channel_id"],
            channel_id=channel["channel_id"],
            channel_name=channel["name"],
        )
        return channel, rows, errors

    groups = group_channels_by_host(prepared)

    def run_group(members: list[dict]) -> list[tuple[dict, list[dict], list[str]]]:
        return [run(channel) for channel in members]

    outcomes: list[tuple[dict, list[dict], list[str]]] = []
    with ThreadPoolExecutor(max_workers=min(channel_workers, len(groups))) as pool:
        for group_result in pool.map(run_group, groups.values()):
            outcomes.extend(group_result)

    all_rows: list[dict] = []
    collection = []
    for channel, rows, errors in outcomes:
        all_rows.extend(rows)
        collection.append(
            {
                "channel": channel["channel_id"],
                "name": channel["name"],
                "requested": sample_count,
                "received": len(rows),
                "errors": errors,
            }
        )
    if not all_rows:
        raise ValueError("所有渠道都没有获得可用回答，指纹库未修改")
    result = integrate_channels(
        bank_id, model_label, all_rows, keep_suspect=keep_suspect, dry_run=dry_run,
        data_file=data_file, bank_file=bank_file,
    )
    result["collection"] = collection
    return result
