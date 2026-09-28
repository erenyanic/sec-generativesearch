"""One-ahead background fetch for the ingest loops.

Ingest runs fetch → parse → chunk → embed → store one filing at a time,
and the EDGAR fetch (network, seconds per 10-K) and the embed (GPU) never
overlapped.  Both ingest loops — :class:`~sec_generative_search.api.tasks.TaskManager`
and the ``sec-rag ingest`` commands — start the *next* filing's fetch on
a :class:`BackgroundFetch` while the current one is processed, and
collect it when they reach that filing.

The contract every caller relies on:

- **One ahead, never more.**  A caller holds at most one pending
  :class:`BackgroundFetch`, so at most two filings' HTML are resident
  (the one being processed and the one prefetched).
- **Same failure surface.**  :meth:`BackgroundFetch.result` re-raises the
  call's exception unchanged on the caller's thread, so the per-filing
  ``except`` ladders produce the same ``filing_failed`` envelopes as an
  inline fetch.
- **Nothing outlives the caller.**  A caller that stops early (cancel,
  filing ceiling, an unexpected error) MUST :meth:`BackgroundFetch.wait`
  in a ``finally`` — the API worker drops the task's EDGAR identity
  resolver right after its loop returns, and a fetch still running (or
  not yet started) past that point would fetch under whatever identity
  is process-global by then.
- The thread is a daemon (like the ingest worker) and runs in a copy of
  the starting thread's context, so its log records keep the task's
  correlation ID.
"""

from __future__ import annotations

import contextvars
import itertools
import threading
from collections.abc import Callable, Sequence

__all__ = ["BackgroundFetch", "OneAheadFetches"]


class BackgroundFetch[T]:
    """Run one blocking call on a daemon thread and collect it later."""

    def __init__(self, call: Callable[[], T], *, name: str) -> None:
        self._result: T | None = None
        self._error: BaseException | None = None
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run,
            args=(self._run, call),
            name=name,
            daemon=True,
        )
        self._thread.start()

    def _run(self, call: Callable[[], T]) -> None:
        try:
            self._result = call()
        except BaseException as exc:  # re-raised on the collecting thread
            self._error = exc

    def result(self) -> T:
        """Wait for the call, then return its value or re-raise its error."""
        self._thread.join()
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        result, self._result = self._result, None
        return result  # type: ignore[return-value]

    def wait(self) -> None:
        """Wait for the call and discard its outcome (an abandoned prefetch)."""
        self._thread.join()
        self._result = None
        self._error = None


def _never() -> bool:
    return False


class OneAheadFetches[K, T]:
    """One-ahead fetch state for a loop over *pending* items.

    *pending* is exactly the items the loop will reach its fetch step for,
    in order — the work list minus known duplicates — so a duplicate is
    never fetched.  The loop calls :meth:`fetch` for the item in hand (the
    prefetched result when it was started one ahead, else an inline call),
    then :meth:`start_after` before processing it, and :meth:`close` in a
    ``finally``.  At most one :class:`BackgroundFetch` is in flight.

    ``should_stop`` (the task's cancel flag) stops new prefetches; ``name``
    names the prefetch thread.  As a context manager it closes on exit —
    waiting, except when the exit is a ``KeyboardInterrupt`` (CLI Ctrl-C).
    """

    def __init__(
        self,
        fetch: Callable[[K], T],
        pending: Sequence[K],
        *,
        name: str,
        should_stop: Callable[[], bool] = _never,
    ) -> None:
        self._fetch = fetch
        self._name = name
        self._should_stop = should_stop
        self._next_after = {id(item): nxt for item, nxt in itertools.pairwise(pending)}
        self._in_flight: tuple[K, BackgroundFetch[T]] | None = None

    def fetch(self, item: K) -> T:
        """The result for *item*: the prefetched one if it is in flight."""
        in_flight, self._in_flight = self._in_flight, None
        if in_flight is not None:
            prefetched_for, background = in_flight
            if prefetched_for is item:
                return background.result()
            background.wait()  # not the item the loop reached — never expected
        return self._fetch(item)

    def start_after(self, item: K) -> None:
        """Start fetching the pending item that follows *item*, if any."""
        nxt = self._next_after.get(id(item))
        if nxt is None or self._in_flight is not None or self._should_stop():
            return
        fetch = self._fetch
        self._in_flight = (nxt, BackgroundFetch(lambda: fetch(nxt), name=self._name))

    def __enter__(self) -> OneAheadFetches[K, T]:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        # An interactive Ctrl-C must not wait out a network fetch; every
        # other exit (normal, error) waits.
        self.close(wait=exc_type is not KeyboardInterrupt)

    def close(self, *, wait: bool = True) -> None:
        """Drop an in-flight prefetch, waiting for it unless ``wait=False``.

        The API worker MUST wait (see the module docstring); only an
        interactive caller abandoning on Ctrl-C may skip it — the thread is
        a daemon and dies with the process.
        """
        in_flight, self._in_flight = self._in_flight, None
        if in_flight is not None and wait:
            in_flight[1].wait()
