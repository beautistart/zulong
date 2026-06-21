"""Message visibility helpers for separating control context from UX events.

The LLM needs internal correction prompts, but Web/InteractionStore/MemoryGraph
should only see structured public events.  These helpers make that boundary
explicit at the message source instead of relying on frontend text filtering.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict


VISIBILITY_KEY = "_zulong_visibility"
CHANNEL_KEY = "_zulong_channel"
UX_VISIBILITY_KEY = "_zulong_ux_visibility"

PUBLIC = "public"
INTERNAL = "internal"

CHANNEL_CONTROL = "control"
CHANNEL_LEDGER = "ledger"
CHANNEL_STATUS = "status"
CHANNEL_FINAL = "final"

UX_MAIN = "main"
UX_DETAILS = "details"
UX_HIDDEN = "hidden"

_OPENAI_MESSAGE_KEYS = {
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "function_call",
}


def internal_control_message(content: str, role: str = "user") -> Dict[str, Any]:
    """Return a message that is valid LLM context but never user-visible."""
    return {
        "role": role,
        "content": content,
        VISIBILITY_KEY: INTERNAL,
        CHANNEL_KEY: CHANNEL_CONTROL,
    }


def mark_public_payload(
    payload: Dict[str, Any],
    channel: str,
    ux_visibility: str = UX_MAIN,
) -> Dict[str, Any]:
    """Mark an EventBus/Web payload as public and assign its UX channel."""
    payload[VISIBILITY_KEY] = PUBLIC
    payload[CHANNEL_KEY] = channel
    payload[UX_VISIBILITY_KEY] = ux_visibility
    return payload


def mark_hidden_payload(payload: Dict[str, Any], channel: str = CHANNEL_STATUS) -> Dict[str, Any]:
    """Mark a payload as server-side only for normal UX rendering."""
    return mark_public_payload(payload, channel, ux_visibility=UX_HIDDEN)


def _visibility_candidates(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield payload
    nested = payload.get("payload")
    if isinstance(nested, dict):
        yield nested
    data = payload.get("data")
    if isinstance(data, dict):
        yield data
    interaction = payload.get("interaction")
    if isinstance(interaction, dict):
        yield interaction
    nested_interaction = nested.get("interaction") if isinstance(nested, dict) else None
    if isinstance(nested_interaction, dict):
        yield nested_interaction
    data_interaction = data.get("interaction") if isinstance(data, dict) else None
    if isinstance(data_interaction, dict):
        yield data_interaction


def is_public_payload(payload: Any) -> bool:
    """Return False for explicitly internal/control/hidden payloads."""
    if not isinstance(payload, dict):
        return True
    for candidate in _visibility_candidates(payload):
        visibility = str(candidate.get(VISIBILITY_KEY) or PUBLIC).lower()
        channel = str(candidate.get(CHANNEL_KEY) or candidate.get("channel") or "").lower()
        ux_visibility = str(
            candidate.get(UX_VISIBILITY_KEY)
            or candidate.get("ux_visibility")
            or UX_MAIN
        ).lower()
        if visibility == INTERNAL:
            return False
        if channel == CHANNEL_CONTROL:
            return False
        if ux_visibility == UX_HIDDEN:
            return False
    return True


def is_main_ux_payload(payload: Any) -> bool:
    """Return True when a public payload should create/update main chat UI."""
    if not is_public_payload(payload):
        return False
    if not isinstance(payload, dict):
        return True
    for candidate in _visibility_candidates(payload):
        ux_visibility = str(
            candidate.get(UX_VISIBILITY_KEY)
            or candidate.get("ux_visibility")
            or UX_MAIN
        ).lower()
        if ux_visibility == UX_DETAILS:
            return False
    return True


def strip_llm_message_metadata(message_or_messages: Any) -> Any:
    """Remove Zulong/Web-only metadata before sending messages to an LLM API."""
    if isinstance(message_or_messages, dict):
        return {
            key: value
            for key, value in message_or_messages.items()
            if key in _OPENAI_MESSAGE_KEYS
        }
    if isinstance(message_or_messages, Iterable) and not isinstance(
        message_or_messages, (str, bytes)
    ):
        return [
            strip_llm_message_metadata(msg)
            for msg in message_or_messages
            if isinstance(msg, dict)
        ]
    return message_or_messages
