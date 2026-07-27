import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from grapharc.observe.trace import TraceRecorder


@pytest.fixture
def trace(tmp_path):
    return TraceRecorder(tmp_path / "trace.jsonl")


@pytest.fixture
def checkpointer(tmp_path):
    conn = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    yield saver
    conn.close()
