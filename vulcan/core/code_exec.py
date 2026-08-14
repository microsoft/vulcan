"""Python code execution helpers for loading and testing generated API classes.

!!! SECURITY WARNING — THIS MODULE EXECUTES MODEL-GENERATED CODE !!!

    ``load_api_class`` and ``execute_class_code`` call ``exec()`` on Python
    source written by a language model, **in this process, with no sandbox**.
    ``safe_execute_tool`` then calls into the objects that code defines.  The
    generated code runs with the full privileges of the interpreter: it can
    read and write any file the process can reach, open network connections,
    spawn subprocesses, and import arbitrary modules.  Nothing here inspects,
    restricts, or contains it.

    ``safe_execute_tool``'s ``timeout`` bounds how long the *caller* waits for
    a tool call to return.  It does not, and cannot, stop what the call is
    doing — the worker thread keeps running after the timeout fires.

    Run this pipeline only against models you trust, and only inside a
    disposable, isolated environment (a throwaway container or VM with no
    credentials, secrets, or private data mounted).  Read the SECURITY section
    of the README before running it anywhere else.

Public API:
    Tool                          — abstract base class for generated tools
    load_api_class(code, ...)     — exec code, return first suitable class
    execute_class_code(code)      — exec code, return first class defined
    safe_execute_tool(env, ...)   — call env tool safely, return result dict
    find_unused_invoke_params     — AST analysis helper
    strip_get_info                — strip get_info from generated code
"""

from __future__ import annotations

import abc
import ast
import inspect
import re
import threading
import types
from typing import Any, Dict, Type

from .exceptions import EnvironmentError


class Tool(abc.ABC):
    """Base class for generated API tool implementations."""

    @staticmethod
    def invoke(args, kwargs):  # noqa: ANN001
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError


_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+[.\w\d_, ]+|from\s+[.\w\d_]+\s+import\s+[.\w\d_, ]+)",
    re.MULTILINE,
)


def _extract_imports(code: str) -> list[str]:
    return [m.lstrip() for m in _IMPORT_RE.findall(code)]


def load_api_class(
    code: str,
    *,
    base_tool: Type[Tool] = Tool,
    strict: bool = True,
) -> Type | None:
    """Execute *code* and return the first class that looks like an API tool.

    Search order:
    1. Any subclass of *base_tool* or a class named 'Tool' in the snippet.
    2. Duck-type: any class with callable ``invoke`` and ``get_info`` attrs.
    3. Most lenient: any class with a callable ``invoke`` attr.

    Raises:
        EnvironmentError: when *strict=True* and no suitable class is found.
        TypeError:        when *code* is not a string.
    """
    if not isinstance(code, str):
        raise TypeError("load_api_class expects a string of Python source")

    ns: Dict[str, Any] = {}

    for stmt in _extract_imports(code):
        try:
            exec(stmt, ns)  # noqa: S102
        except ImportError:
            pass

    if "Tool" not in ns:
        ns["Tool"] = base_tool

    exec(code, ns)  # noqa: S102

    candidate_bases = {base_tool}
    user_tool = ns.get("Tool")
    if inspect.isclass(user_tool):
        candidate_bases.add(user_tool)

    # Pass 1: subclass of base
    for obj in ns.values():
        if inspect.isclass(obj) and obj not in candidate_bases:
            if any(issubclass(obj, b) for b in candidate_bases):
                return obj

    # Pass 2: duck-type
    for obj in ns.values():
        if inspect.isclass(obj) and obj not in candidate_bases:
            if callable(getattr(obj, "invoke", None)) and callable(
                getattr(obj, "get_info", None)
            ):
                return obj

    # Pass 3: any class with invoke
    for obj in ns.values():
        if inspect.isclass(obj) and obj not in candidate_bases:
            if callable(getattr(obj, "invoke", None)):
                return obj

    if strict:
        raise EnvironmentError("No suitable API class found in generated code")
    return None


def execute_class_code(code: str) -> Type | None:
    """Execute *code* and return the first class defined in it.

    The abstract ``Tool`` base class (if the model emits it) is skipped so the
    environment class is returned rather than the non-instantiable base.
    """
    class_name = None
    fallback_name = None
    for node in ast.parse(code).body:
        if isinstance(node, ast.ClassDef):
            if fallback_name is None:
                fallback_name = node.name
            if node.name != "Tool":
                class_name = node.name
                break
    if class_name is None:
        class_name = fallback_name
    if class_name is None:
        return None

    module = types.ModuleType("dynamic_module")
    exec(code, module.__dict__)  # noqa: S102
    return getattr(module, class_name, None)


def safe_execute_tool(env: Any, tool_name: str, args: dict, timeout: int = 20) -> dict:
    """Execute a tool on an environment instance, catching any errors.

    The call runs on a daemon worker thread and is awaited for at most
    *timeout* seconds of wall-clock time, so a generated tool that blocks
    forever (an infinite loop, a hanging socket) cannot stall the pipeline.
    Python cannot kill a running thread: on timeout the call is abandoned, not
    stopped, and it keeps consuming CPU until the process exits.  The worker is
    a daemon precisely so that an abandoned call cannot hold up interpreter
    shutdown.  See the security warning at the top of this module.

    Args:
        env:       instantiated environment object
        tool_name: name from function_name_mapping()
        args:      input arguments dict
        timeout:   wall-clock seconds to wait for the call; values <= 0 (or
                   None) wait indefinitely, as before this bound existed.

    Returns:
        {"output": …, "success": True}  or  {"error": "…", "success": False}
        — a timeout returns the error form with a message naming the limit.
    """
    import json as _json

    try:
        mapping = env.function_name_mapping()
        if tool_name not in mapping:
            return {
                "error": f"tool '{tool_name}' not in function_name_mapping",
                "success": False,
            }

        func = mapping[tool_name]

        def _call() -> Any:
            return func(**args) if args else func()

        if timeout and timeout > 0:
            # Captures whichever of the two outcomes the worker reaches; the
            # exception is re-raised here so callers see identical behaviour to
            # the untimed path.
            outcome: dict[str, Any] = {}

            def _worker() -> None:
                try:
                    outcome["result"] = _call()
                except BaseException as exc:  # noqa: BLE001 — re-raised below
                    outcome["exc"] = exc

            thread = threading.Thread(
                target=_worker,
                name=f"vulcan-tool-{tool_name}",
                daemon=True,
            )
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                return {
                    "error": (
                        f"tool '{tool_name}' timed out after {timeout}s "
                        "(call abandoned; it may still be running)"
                    ),
                    "success": False,
                }
            if "exc" in outcome:
                raise outcome["exc"]
            result = outcome["result"]
        else:
            result = _call()

        if isinstance(result, str):
            try:
                result = _json.loads(result)
            except _json.JSONDecodeError:
                pass

        return {"output": result, "success": True}

    except Exception as exc:
        return {"error": str(exc), "success": False}


def find_unused_invoke_params(code: str) -> list[str]:
    """Return names of invoke() parameters not referenced in its body.

    Uses the LAST invoke() definition (the subclass implementation, not the
    abstract base stub).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    invoke_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "invoke":
            invoke_nodes.append(node)

    if not invoke_nodes:
        return []

    node = invoke_nodes[-1]

    params: set[str] = set()
    for arg in node.args.args:
        if arg.arg not in ("self", "data"):
            params.add(arg.arg)
    for arg in node.args.kwonlyargs:
        if arg.arg not in ("self", "data"):
            params.add(arg.arg)
    if not params:
        return []

    used: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)

    body_start = node.body[0].lineno
    body_end = getattr(node, "end_lineno", None)
    if body_end:
        lines = code.splitlines()
        body_text = "\n".join(lines[body_start - 1: body_end])
        for p in list(params):
            if re.search(r"\b" + re.escape(p) + r"\b", body_text):
                used.add(p)

    return sorted(params - used)


def strip_get_info(code: str) -> str:
    """Extract only the invoke method from a generated API tool class.

    Removes the base Tool class, get_info methods, class wrappers, and
    duplicate imports.  Returns just the invoke method body.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name != "Tool":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "invoke":
                    start = item.lineno - 1
                    end = getattr(item, "end_lineno", item.lineno)
                    while start > 0 and lines[start - 1].strip().startswith("@"):
                        start -= 1
                    return "".join(lines[start:end])

    # Fallback: remove get_info methods only
    regions_to_remove = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "get_info":
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno)
                while start > 0 and lines[start - 1].strip().startswith("@"):
                    start -= 1
                regions_to_remove.append((start, end))

    if not regions_to_remove:
        return code

    for start, end in sorted(regions_to_remove, reverse=True):
        del lines[start:end]

    return "".join(lines)
