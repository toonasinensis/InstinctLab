"""Tool registry: @tool decorator that attaches Anthropic tool schema to a function.

Usage:
    @tool("connect_task", "创建或续接 task 工作区")
    async def connect_task(*, name: str, _root: Path) -> str:
        ...

Convention:
- Parameters starting with `_` are injected by the agent loop — NOT in schema.
  Declare the ones you need; the framework injects them before calling your function.
- Keyword-only parameters (after `*`) that don't start with `_` become the
  tool's input_schema — the only thing the model sees and fills.

Injectable parameters (closed set — add needs_* here + injection in _run_tools):
    _root          : Path    app launch dir (TARSAI_LAUNCH_CWD); subprocess cwd
    _ctx                     SessionContext (task_workspace / conversation_workspace / active_agents / …)
    _emit_command            async callback to surface subprocess command to UI
    _delegation              ask_<key> only: {parent_use_id, sink} for event forwarding
    _session                 in-session schedule tools: the live ClaudeSession

Adapted from reference/robotCenter-master-backend/backend/ai/core/tool_registry.py
"""
from __future__ import annotations

import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _python_type_to_json_schema(tp: Any) -> dict:
    origin = get_origin(tp)
    # Optional[X] / Union[X, None] / X | None ——先解包非 None 分支再递归,
    # 否则 Optional[dict]/Optional[list[...]] 会落到兜底 string(types.UnionType 兼容 PEP604 的 X|None)。
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
    if origin is list:
        (inner,) = get_args(tp) or (str,)
        return {"type": "array", "items": _python_type_to_json_schema(inner)}
    if origin is dict:
        return {"type": "object"}
    if tp in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[tp]}
    return {"type": "string"}


def _build_schema(fn: Callable, description: str, params: dict | None = None) -> dict:
    """Build Anthropic input_schema by reflecting the function signature.

    params: optional {param_name: description} map — each entry is written into
    the field's JSON-schema `description` so the model sees the contract
    (shape, allowed values, examples) on every call, without reading SKILL.md.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    param_docs = params or {}
    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # Skip injected params: anything starting with `_` (_root, _ctx…).
        if name.startswith("_"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        tp = hints.get(name, str)
        schema = _python_type_to_json_schema(tp)

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = param.default

        # Field description from the params map → model sees the contract.
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(name: str, description: str, params: dict | None = None):
    """Decorator — attaches .tool_name / .tool_description / .tool_schema to fn.

    params: optional {param_name: description} map written into each field's
    JSON-schema description (the contract the model sees every call).
    """
    def decorate(fn: Callable) -> Callable:
        fn.tool_name = name
        fn.tool_description = description
        fn.tool_schema = _build_schema(fn, description, params)
        return fn
    return decorate


def to_anthropic_schema(fn: Callable) -> dict:
    """Return the Anthropic API-ready tool definition dict."""
    return {
        "name": fn.tool_name,
        "description": fn.tool_description,
        "input_schema": fn.tool_schema,
    }


def needs_command_sink(fn: Callable) -> bool:
    """True if the function has an `_emit_command` parameter.

    Such tools run a subprocess and want to surface the real executed command
    to the UI (like Claude Code shows the actual bash command), separate from
    the model-facing tool result.
    """
    return "_emit_command" in inspect.signature(fn).parameters


def needs_context(fn: Callable) -> bool:
    """True if the function has a `_ctx` parameter (needs the SessionContext injected).

    会话级状态的单一容器：工具要 session 级状态（task_workspace /
    conversation_workspace / profile / active_round …）声明 `_ctx`，
    agent loop 注入 live SessionContext。模型看不到该参数。
    """
    return "_ctx" in inspect.signature(fn).parameters


def needs_delegation(fn: Callable) -> bool:
    """True if the function has a `_delegation` parameter.

    agent-as-tool（ask_<key>）专用：声明 `_delegation` 的工具会拿到父调用的
    {parent_use_id, sink}，用来把子循环事件打上 parent_use_id 标签后转发回父 ws，
    实现子 agent 内部步骤的实时可视化 + 回放。模型看不到该参数。
    """
    return "_delegation" in inspect.signature(fn).parameters


def needs_session(fn: Callable) -> bool:
    """True if the function declares a `_session` parameter.

    会话内嵌定时（session_schedule_*）专用：声明 `_session` 的工具拿到当前
    ClaudeSession 本体，用来挂/停内存自唤醒任务（绑会话生命周期，不入 DB）。
    与 _ctx 同列，按声明注入、模型看不到。
    """
    return "_session" in inspect.signature(fn).parameters


def needs_root(fn: Callable) -> bool:
    """True if the function declares a `_root` parameter.

    `_root` 是 app 启动目录（TARSAI_LAUNCH_CWD，进程级常量）的注入参数——与
    _ctx / _emit_command / _delegation / _session 同列，按声明注入、模型看不到。
    需要"项目根 / 子进程 cwd"的工具（data_ops_agent 业务工具、bash）声明它；不需要的（如纯
    SQL 查询 query_database）不声明，框架不对首参做任何假设。
    """
    return "_root" in inspect.signature(fn).parameters
