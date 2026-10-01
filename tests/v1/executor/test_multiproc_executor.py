# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import threading
import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


def test_recoverable_rpc_error_keeps_failure_pipe_open():
    """A rejected utility RPC must not invalidate the worker group."""
    reader, writer = multiprocessing.Pipe(duplex=False)
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 1
    worker_proc.error_writer = writer
    outputs: list[Any] = []

    def reject_adapter():
        raise ValueError("invalid adapter")

    worker_proc.worker = SimpleNamespace(add_lora=reject_adapter)
    worker_proc.handle_output = outputs.append
    try:
        worker_proc._execute_worker_rpc(("add_lora", (), {}, None))
        assert isinstance(outputs.pop(), ValueError)
        assert not reader.poll(), "Recoverable errors must not notify the monitor"
    finally:
        writer.close()
        reader.close()


@pytest.fixture
def worker_failure_monitor(monkeypatch):
    """Run the real monitor/shutdown with CPU-only child processes and pipes."""
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "0")
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    writers = []
    workers: list[SimpleNamespace] = []
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.workers = workers
    failed = threading.Event()
    monitor_errors = []
    executor.failure_callback = failed.set

    def monitor():
        try:
            executor.start_worker_monitor(inline=True)
        except Exception as error:
            monitor_errors.append(error)

    thread = threading.Thread(target=monitor, daemon=True)
    try:
        for rank in range(2):
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(target=release.wait, name=f"test-worker-{rank}")
            process.start()
            writers.append(writer)
            workers.append(
                SimpleNamespace(
                    proc=process,
                    error_reader=reader,
                    death_writer=None,
                    worker_response_mq=None,
                )
            )
        thread.start()
        yield executor, writers, failed, monitor_errors
    finally:
        for writer in writers:
            writer.close()
        release.set()
        if thread.ident is not None:
            thread.join(timeout=10)
        executor.shutdown()
        for worker in workers:
            worker.proc.join(timeout=5)
            if worker.proc.is_alive():
                worker.proc.kill()
                worker.proc.join(timeout=5)
            worker.proc.close()
        assert not thread.is_alive(), "Worker monitor failed to terminate"


def test_non_output_rank_failure_stops_local_worker_group(worker_failure_monitor):
    """A rank-1 forward error must reach the monitor without a rank-0 reply."""
    executor, writers, failed, monitor_errors = worker_failure_monitor
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 1
    worker_proc.error_writer = writers[1]

    def fail_forward():
        raise RuntimeError("injected forward failure")

    worker_proc.worker = SimpleNamespace(execute_model=fail_forward)
    worker_proc._execute_worker_rpc(("execute_model", (), {}, 0))

    assert failed.wait(timeout=10), "Inference failure did not reach the executor"
    assert not monitor_errors
    assert executor.is_failed
    assert all(not worker.proc.is_alive() for worker in executor.workers)


def test_partial_error_message_cannot_block_shutdown(worker_failure_monitor):
    """Pipe readiness must trigger cleanup without waiting for a framed payload."""
    executor, writers, failed, monitor_errors = worker_failure_monitor
    # A Connection frame advertising 1024 bytes, with only one payload byte.
    # Keep the writer open: recv() would wait indefinitely for the remainder.
    os.write(writers[1].fileno(), b"\x00\x00\x04\x00x")

    assert failed.wait(timeout=10), "Incomplete diagnostics blocked failure handling"
    assert not monitor_errors
    assert executor.is_failed
    assert all(not worker.proc.is_alive() for worker in executor.workers)
