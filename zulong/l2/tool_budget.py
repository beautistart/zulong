"""User-controlled tool-call budget helpers.

This module is an execution guard for TSD §23.2/§23.3: the L2 model still
decides whether to call tools, but explicit user limits are enforced before
any tool call is executed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional


_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _parse_small_int(raw: str) -> Optional[int]:
    raw = str(raw or "").strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in _CN_DIGITS:
        return _CN_DIGITS[raw]
    if raw == "十":
        return 10
    if raw.startswith("十"):
        tail = raw[1:]
        return 10 + _CN_DIGITS.get(tail, 0)
    if "十" in raw:
        head, _, tail = raw.partition("十")
        tens = _CN_DIGITS.get(head, 0)
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        if tens:
            return tens * 10 + ones
    return None


_NO_TOOL_PATTERNS = (
    r"(?:不要|别|禁止|无需|不用|不需要)(?:再)?(?:调用|使用)?(?:任何)?工具",
    r"(?:no|without)\s+tools?",
    r"do\s+not\s+use\s+tools?",
)

_BUDGET_PATTERNS = (
    r"(?:最多|至多|不超过|不要超过|别超过|限制(?:为|在)?|只用|仅用|最多使用|最多调用)"
    r"\s*([0-9一二两三四五六七八九十〇零]+)\s*(?:个|次|项)?\s*(?:工具调用|调用工具|工具)",
    r"(?:工具调用|调用工具|工具)\s*(?:最多|至多|不超过|不要超过|别超过|限制(?:为|在)?)"
    r"\s*([0-9一二两三四五六七八九十〇零]+)\s*(?:个|次|项)?",
    r"(?:at\s+most|no\s+more\s+than|limit(?:ed)?\s+to|use\s+at\s+most)"
    r"\s*([0-9]+)\s*(?:tool\s+calls?|tools?)",
    r"(?:tool\s+calls?|tools?)\s*(?:at\s+most|no\s+more\s+than|limit(?:ed)?\s+to)"
    r"\s*([0-9]+)",
)


def detect_tool_call_budget(user_text: str) -> Optional[int]:
    """Return an explicit max number of tool calls, or None when absent."""
    text = str(user_text or "")
    if not text.strip():
        return None
    lowered = text.lower()
    for pattern in _NO_TOOL_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return 0

    budgets = []
    for pattern in _BUDGET_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            value = _parse_small_int(match.group(1))
            if value is not None and value >= 0:
                budgets.append(value)
    if not budgets:
        return None
    return min(budgets)


def sync_engine_tool_budget(engine: object, user_text: str) -> Optional[int]:
    """Initialize or reuse a per-user-turn budget on the engine instance."""
    budget = detect_tool_call_budget(user_text)
    key_src = f"{budget}:{user_text or ''}"
    key = hashlib.sha1(key_src.encode("utf-8", errors="ignore")).hexdigest()
    if getattr(engine, "_tool_budget_key", "") != key:
        setattr(engine, "_tool_budget_key", key)
        setattr(engine, "_tool_call_budget", budget)
        setattr(engine, "_tool_calls_used_for_budget", 0)
    return getattr(engine, "_tool_call_budget", budget)


def get_engine_tool_budget(engine: object) -> Optional[int]:
    return getattr(engine, "_tool_call_budget", None)


def get_engine_tool_calls_used(engine: object) -> int:
    try:
        return int(getattr(engine, "_tool_calls_used_for_budget", 0) or 0)
    except Exception:
        return 0


def record_engine_tool_calls_used(engine: object, count: int) -> None:
    if count <= 0:
        return
    used = get_engine_tool_calls_used(engine) + int(count)
    setattr(engine, "_tool_calls_used_for_budget", used)


def engine_tool_budget_exhausted(engine: object) -> bool:
    budget = get_engine_tool_budget(engine)
    if budget is None:
        return False
    return get_engine_tool_calls_used(engine) >= budget
