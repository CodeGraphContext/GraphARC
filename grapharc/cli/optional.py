"""Lazy access to components the CLI drives but does not own.

The core toolset, the HTTP API and the replay engine live in packages that may
not exist in a given checkout. Importing them at module scope would make every
command — including the ones that do not need them — fail to import at all, so
each is loaded at call time and a missing one is reported as a plain message
naming what is missing and why it was wanted.

`Unavailable` is deliberately not an `ImportError`: it means "this install
cannot run this command", which is a CLI outcome with an exit code, not a
programming error to propagate.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class Unavailable(Exception):
    """A component this command delegates to is not importable or not usable."""


def load(name: str, *, needed_for: str, hint: str = "") -> ModuleType:
    """Import `name`, or raise `Unavailable` with the reason attached.

    The underlying `ImportError` text is kept: "no module named grapharc.tools"
    and "grapharc.tools imports something that is not installed" are different
    problems with different fixes, and collapsing them hides which one you have.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        message = f"{needed_for} needs {name}, which is not importable ({exc})"
        raise Unavailable(f"{message}. {hint}" if hint else message) from exc


def pick(module: ModuleType, names: tuple[str, ...], *, needed_for: str) -> tuple[str, Any]:
    """First attribute of `module` present in `names`, with the name it was found under.

    Returning the name matters: the caller reports which entry point it used, so
    a run's record says what it actually called rather than what it hoped for.
    """
    for name in names:
        attr = getattr(module, name, None)
        if attr is not None:
            return name, attr
    raise Unavailable(
        f"{needed_for} imported {module.__name__} but found none of its expected "
        f"entry points ({', '.join(names)})"
    )


__all__ = ["Unavailable", "load", "pick"]
