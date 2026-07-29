"""`grapharc serve` — run the HTTP API.

The API is `grapharc.server`, imported at call time: it pulls in FastAPI, which
the rest of GraphARC does not, so a checkout without the `server` extra keeps a
working CLI. The server package's own `serve()` is preferred over calling
uvicorn from here — how the app is hosted is its decision, not the CLI's.

Graphs are supplied with `--registry module:attr`. Without one the app starts
with an empty registry, which is a legitimate way to check the process comes up
and a useless way to run anything, so the CLI says which of the two you got.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from grapharc.cli import optional, style
from grapharc.cli.output import EXIT_OK, emit, fail

SERVER_HINT = "Install the server extra with: uv sync --extra server"
UVICORN_HINT = "Install uvicorn with: uv sync --extra server"

#: `serve` lines its one label up at nine characters, not the ten the report
#: commands use. All three of its lines are printed verbatim in
#: `docs/cookbook/06-serving-and-ops.md`, so the number is not up for revision.
LABEL_WIDTH = 9


def resolve_registry(target: str) -> Any:
    """Import `module:attr` and return the graph registry it names.

    A callable attribute is called: `--registry mypkg:build_registry` and
    `--registry mypkg:REGISTRY` are both natural things to write, and a
    `GraphRegistry` instance is not callable, so the two cannot be confused.
    """
    module_name, separator, attribute = target.partition(":")
    if not separator or not attribute:
        raise optional.Unavailable(
            f"--registry expects module:attr, got {target!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise optional.Unavailable(f"--registry {target!r}: {exc}") from exc
    registry = getattr(module, attribute, None)
    if registry is None:
        raise optional.Unavailable(f"--registry {target!r}: {module_name} has no {attribute!r}")
    return registry() if callable(registry) else registry


def _uvicorn_runner(uvicorn: Any) -> Any:
    """A runner shaped like `grapharc.server.serve`, so both paths call the same way."""

    def run(app: Any, **kwargs: Any) -> None:
        uvicorn.run(app, **kwargs)

    return run


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
    registry_target: str | None = None,
    as_json: bool = False,
) -> int:
    try:
        module = optional.load("grapharc.server", needed_for="grapharc serve", hint=SERVER_HINT)
        _, create_app = optional.pick(module, ("create_app",), needed_for="grapharc serve")
        registry = resolve_registry(registry_target) if registry_target else None
    except optional.Unavailable as exc:
        return fail(str(exc), as_json=as_json, command="serve")

    try:
        app = create_app(registry=registry) if registry is not None else create_app()
    except Exception as exc:  # noqa: BLE001 — a config error, reported as one
        return fail(f"could not build the app: {exc}", as_json=as_json, command="serve")

    names = registry.names() if hasattr(registry, "names") else []
    runner = getattr(module, "serve", None)
    if runner is None:
        try:
            uvicorn = optional.load("uvicorn", needed_for="grapharc serve", hint=UVICORN_HINT)
        except optional.Unavailable as exc:
            return fail(str(exc), as_json=as_json, command="serve")
        runner = _uvicorn_runner(uvicorn)

    payload = {
        "ok": True,
        "command": "serve",
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "log_level": log_level,
        "registry": registry_target,
        "graphs": names,
    }
    graphs = (
        ", ".join(names)
        if names
        else "none registered — pass --registry module:attr, or every "
        "create-session request will 404"
    )
    lines = [
        f"serving grapharc.server on {style.accent(f'http://{host}:{port}')}",
        # An empty registry is a legitimate way to check the process comes up and
        # a useless way to run anything, so on a terminal it is amber.
        style.kv("graphs", graphs, width=LABEL_WIDTH, tint=None if names else style.warn),
        style.dim("ctrl-c to stop"),
    ]
    # Printed *and flushed* before the server blocks: a caller watching stdout for
    # the URL would otherwise wait for the process to exit to learn it. Nothing
    # here is buffered, deferred, or drawn on a timer for the same reason.
    emit(payload, lines, as_json=as_json)
    sys.stdout.flush()

    runner(app, host=host, port=port, log_level=log_level)
    return EXIT_OK


__all__ = ["resolve_registry", "serve"]
