from __future__ import annotations

import asyncio
import inspect
import os
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from ablation_study.common.cot_helpers import construct_preference_prompt
from ablation_study.common.io import dump_json, load_json

SAVE_INTERVAL = 100
REQUEST_TIMEOUT_SECONDS = 120
INSUFFICIENT_BALANCE_STOP_THRESHOLD = 3


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_async_client() -> tuple[AsyncOpenAI, str, str]:
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    key_source = "OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else ("SILICONFLOW_API_KEY" if os.getenv("SILICONFLOW_API_KEY") else "NONE")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or SILICONFLOW_API_KEY before generating preferences.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url), base_url, key_source


def _has_preference(item: dict) -> bool:
    value = item.get("preference")
    return isinstance(value, str) and bool(value.strip())


def _is_insufficient_balance_error(error: str | None) -> bool:
    if not error:
        return False
    text = str(error).lower()
    return ("insufficient" in text and "balance" in text) or ("code': 30001" in text) or ('"code": 30001' in text)


async def _safe_close_client(client: AsyncOpenAI) -> None:
    close_method = getattr(client, "close", None)
    if callable(close_method):
        result = close_method()
        if inspect.isawaitable(result):
            await result
        return

    aclose_method = getattr(client, "aclose", None)
    if callable(aclose_method):
        result = aclose_method()
        if inspect.isawaitable(result):
            await result


async def _generate_one(
    client: AsyncOpenAI,
    model_name: str,
    item: dict,
    index: int,
    request_timeout_seconds: int,
) -> tuple[int, str | None, str | None]:
    target_timestamp = item["result"]["timestamp"]
    dt_object = datetime.fromisoformat(target_timestamp)
    specific_time_str = dt_object.strftime("%A %H:%M")
    system_prompt, user_prompt = construct_preference_prompt(
        history_data=item["history"],
        target_day_type=item.get("result", {}).get("day_type", ""),
        target_time_of_day=item.get("result", {}).get("time_of_day", ""),
        specific_time=specific_time_str,
    )
    retries = 3
    for attempt in range(retries):
        try:
            rsp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                ),
                timeout=request_timeout_seconds,
            )
            return index, rsp.choices[0].message.content.strip(), None
        except asyncio.TimeoutError:
            exc = RuntimeError(f"Timeout after {request_timeout_seconds}s")
            if attempt == retries - 1:
                return index, None, str(exc)
            await asyncio.sleep(5 * (attempt + 1))
        except Exception as exc:
            if attempt == retries - 1:
                return index, None, str(exc)
            await asyncio.sleep(5 * (attempt + 1))
    return index, None, "Unknown error"


async def add_preferences_to_file(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_concurrent_requests: int = 10,
    save_interval: int = SAVE_INTERVAL,
    request_timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> dict:
    data = load_json(input_path)
    existing = load_json(output_path) if output_path.exists() else data
    if len(existing) != len(data):
        print(
            f"[preference:{input_path.stem}] warning: output len({len(existing)}) != input len({len(data)}), "
            "will align by index and continue."
        )
        aligned = data
        limit = min(len(existing), len(aligned))
        for idx in range(limit):
            if _has_preference(existing[idx]):
                aligned[idx]["preference"] = existing[idx]["preference"]
        existing = aligned

    already_done = sum(1 for item in existing if _has_preference(item))
    items_to_process = [(idx, item) for idx, item in enumerate(existing) if not _has_preference(item)]
    print(
        f"[preference:{input_path.stem}] total={len(existing)}, existing={already_done}, "
        f"to_generate={len(items_to_process)}, concurrent={max_concurrent_requests}"
    )
    if not items_to_process:
        if not output_path.exists():
            dump_json(existing, output_path)
        return {"total": len(existing), "generated": 0, "failed": 0, "skipped_existing": already_done}

    client, base_url, key_source = build_async_client()
    existing_errors = sum(1 for item in existing if item.get("preference_error"))
    print(
        f"[preference:{input_path.stem}] start generation, output -> {output_path}, "
        f"base_url={base_url}, key_source={key_source}, existing_error_rows={existing_errors}"
    )
    try:
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _bounded(index: int, item: dict):
            async with semaphore:
                return await _generate_one(client, model_name, item, index, request_timeout_seconds)

        tasks = [asyncio.create_task(_bounded(idx, item)) for idx, item in items_to_process]
        failed = 0
        completed = 0
        last_save_count = 0
        stopped_due_to_balance = False
        insufficient_balance_errors = 0
        sample_error_printed = False
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"preference:{input_path.stem}"):
            idx, preference, error = await future
            completed += 1
            if error is None:
                existing[idx]["preference"] = preference
                existing[idx].pop("preference_error", None)
            else:
                existing[idx]["preference_error"] = error
                failed += 1
                if not sample_error_printed:
                    print(f"[preference:{input_path.stem}] sample_error(idx={idx}): {error}")
                    sample_error_printed = True

                # Fail fast on account-balance errors: save progress and stop generating new calls.
                if _is_insufficient_balance_error(error):
                    insufficient_balance_errors += 1
                    print(
                        f"[preference:{input_path.stem}] insufficient-balance error idx={idx}, "
                        f"count={insufficient_balance_errors}/{INSUFFICIENT_BALANCE_STOP_THRESHOLD}"
                    )
                    if insufficient_balance_errors >= INSUFFICIENT_BALANCE_STOP_THRESHOLD:
                        stopped_due_to_balance = True
                        print(
                            f"[preference:{input_path.stem}] stop early due to repeated insufficient-balance errors. "
                            "Progress will be saved for resume."
                        )
                        break

            if completed - last_save_count >= save_interval:
                dump_json(existing, output_path)
                print(
                    f"[preference:{input_path.stem}] checkpoint saved: processed={completed}/{len(tasks)}, "
                    f"failed={failed}"
                )
                last_save_count = completed

        if stopped_due_to_balance:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        dump_json(existing, output_path)
        if stopped_due_to_balance:
            remaining = sum(1 for item in existing if not _has_preference(item))
            print(
                f"[preference:{input_path.stem}] paused: generated={completed - failed}, failed={failed}, "
                f"remaining={remaining}"
            )
        else:
            print(f"[preference:{input_path.stem}] finished: generated={len(tasks) - failed}, failed={failed}")
        return {
            "total": len(existing),
            "generated": (completed - failed) if stopped_due_to_balance else (len(tasks) - failed),
            "failed": failed,
            "skipped_existing": already_done,
            "stopped_due_to_balance": stopped_due_to_balance,
        }
    finally:
        await _safe_close_client(client)
