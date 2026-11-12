"""Pure conversion between the harness's chat-style messages and Responses API items."""

from __future__ import annotations

from typing import Any


def tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-style function tool specs as flat Responses function tools."""
    flat = []
    for tool in tools:
        spec = tool["function"]
        flat.append({"type": "function", "name": spec["name"],
                     "description": spec.get("description", ""),
                     "parameters": spec["parameters"]})  # fmt: skip
    return flat


def messages_to_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conversation messages as Responses input items. System messages are skipped."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            items.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": call["id"],
                              "name": call["function"]["name"],
                              "arguments": call["function"]["arguments"]})  # fmt: skip
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})  # fmt: skip
    return items


def output_to_message(output: list[dict[str, Any]]) -> dict[str, Any]:
    """Responses output items as one chat-style assistant message."""
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        elif item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"],
                                       "arguments": item.get("arguments", "")}})  # fmt: skip
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
    if calls:
        message["tool_calls"] = calls
    return message


def usage_from_responses(usage: dict[str, Any]) -> dict[str, int]:
    """Map Responses usage fields to the prompt/completion/total/reasoning keys we record."""
    prompt = int(usage.get("input_tokens", 0))
    completion = int(usage.get("output_tokens", 0))
    reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "reasoning_tokens": reasoning}  # fmt: skip
