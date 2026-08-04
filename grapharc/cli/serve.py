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
import json
import os
import sys
from pathlib import Path
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
    live_root: str | None = None,
    live_token: str | None = None,
    as_json: bool = False,
) -> int:
    try:
        module = optional.load("grapharc.server", needed_for="grapharc serve", hint=SERVER_HINT)
        _, create_app = optional.pick(module, ("create_app",), needed_for="grapharc serve")
        registry = resolve_registry(registry_target) if registry_target else None
    except optional.Unavailable as exc:
        return fail(str(exc), as_json=as_json, command="serve")

    if live_root is not None and not Path(live_root).is_dir():
        return fail(
            f"--live-root is not a directory: {live_root}", as_json=as_json, command="serve"
        )

    kwargs: dict[str, Any] = {}
    if registry is not None:
        kwargs["registry"] = registry
    if live_root is not None:
        kwargs["live_root"] = live_root
        if live_token:
            kwargs["live_token"] = live_token
    try:
        app = create_app(**kwargs)
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
        "live_root": live_root,
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
    # The three lines above are printed verbatim in the cookbook; anything the
    # live view adds appears only when the operator asked for it.
    if live_root is not None:
        lines.insert(
            1,
            style.kv(
                "live view",
                f"http://{host}:{port}/live  (root: {live_root})",
                width=LABEL_WIDTH,
            ),
        )
        if host not in ("127.0.0.1", "localhost"):
            lines.insert(
                2,
                style.kv(
                    "warning",
                    "binding beyond loopback exposes run telemetry; put a tunnel "
                    "(Tailscale/cloudflared) or --live-token in front",
                    width=LABEL_WIDTH,
                    tint=style.warn,
                ),
            )
    # A discovery marker, so `grapharc plan`/`go` can print the exact live-view
    # URL for their trace. Best-effort in both directions: an unwritable cwd
    # must not stop the server, and the marker never holds the token — it is a
    # convenience pointer, not a credential store. Written in cwd only, the
    # same rule grapharc.toml follows.
    marker: Path | None = None
    if live_root is not None:
        try:
            marker = Path(".grapharc") / "live-server.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "url": f"http://{host}:{port}",
                        "host": host,
                        "port": port,
                        "live_root": str(Path(live_root).resolve()),
                        "pid": os.getpid(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            payload["live_marker"] = str(marker)
        except OSError:
            marker = None

    # Printed *and flushed* before the server blocks: a caller watching stdout for
    # the URL would otherwise wait for the process to exit to learn it. Nothing
    # here is buffered, deferred, or drawn on a timer for the same reason.
    emit(payload, lines, as_json=as_json)
    sys.stdout.flush()

    try:
        runner(app, host=host, port=port, log_level=log_level)
    finally:
        # Only our own marker: a second serve that replaced it owns it now.
        if marker is not None:
            try:
                written = json.loads(marker.read_text(encoding="utf-8"))
                if written.get("pid") == os.getpid():
                    marker.unlink()
            except (OSError, ValueError):
                pass
    return EXIT_OK


__all__ = ["resolve_registry", "serve"]
