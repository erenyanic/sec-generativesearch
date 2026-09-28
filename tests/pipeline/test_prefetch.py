"""Tests for :mod:`sec_generative_search.pipeline.prefetch` (F21)."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from sec_generative_search.core.correlation import bind_correlation_id, get_correlation_id
from sec_generative_search.pipeline.prefetch import BackgroundFetch, OneAheadFetches


class TestBackgroundFetch:
    def test_result_returns_the_call_value(self) -> None:
        assert BackgroundFetch(lambda: ("id", "<html/>"), name="t").result() == ("id", "<html/>")

    def test_result_reraises_the_original_exception_on_the_caller(self) -> None:
        """Same object, same type: the ingest loops' ``except`` ladders then
        build exactly the envelope an inline fetch would have produced."""
        error = LookupError("Filing not found")

        def _boom() -> None:
            raise error

        background = BackgroundFetch(_boom, name="t")
        with pytest.raises(LookupError) as excinfo:
            background.result()
        assert excinfo.value is error

    def test_base_exceptions_are_carried_too(self) -> None:
        def _exit() -> None:
            raise SystemExit(3)

        with pytest.raises(SystemExit):
            BackgroundFetch(_exit, name="t").result()

    def test_wait_discards_the_outcome(self) -> None:
        def _boom() -> None:
            raise RuntimeError("abandoned prefetch")

        background = BackgroundFetch(_boom, name="t")
        background.wait()  # never raises — the result is thrown away
        assert not background._thread.is_alive()

    def test_runs_on_a_named_daemon_thread(self) -> None:
        seen: list[threading.Thread] = []
        BackgroundFetch(lambda: seen.append(threading.current_thread()), name="p-1").result()
        assert seen[0] is not threading.current_thread()
        assert seen[0].name == "p-1"
        assert seen[0].daemon

    def test_keeps_the_callers_correlation_id(self) -> None:
        """Log records from the prefetch thread must carry the task's ID —
        a fresh thread would otherwise start with no bound ID."""
        with bind_correlation_id("prefetch-cid-1234"):
            background = BackgroundFetch(get_correlation_id, name="t")
        assert background.result() == "prefetch-cid-1234"


class TestOneAheadFetches:
    def test_first_item_inline_then_one_ahead(self) -> None:
        threads: dict[str, str] = {}

        def _fetch(item: str) -> str:
            threads[item] = threading.current_thread().name
            return item.upper()

        fetches = OneAheadFetches(_fetch, ["a", "b", "c"], name="pf")
        assert fetches.fetch("a") == "A"
        fetches.start_after("a")
        assert fetches.fetch("b") == "B"
        fetches.start_after("b")
        assert fetches.fetch("c") == "C"
        fetches.start_after("c")  # last item: nothing to prefetch
        fetches.close()

        assert threads["a"] == threading.current_thread().name
        assert threads["b"] == threads["c"] == "pf"

    def test_only_pending_items_are_ever_fetched(self) -> None:
        fetched: list[str] = []
        fetches = OneAheadFetches(lambda i: fetched.append(i) or i, ["a", "c"], name="pf")
        fetches.fetch("a")
        fetches.start_after("a")  # "b" is not pending (a known duplicate)
        fetches.fetch("c")
        fetches.close()
        assert fetched == ["a", "c"]

    def test_should_stop_prevents_a_new_prefetch(self) -> None:
        fetched: list[str] = []
        stop = threading.Event()
        fetches = OneAheadFetches(
            lambda i: fetched.append(i) or i, ["a", "b"], name="pf", should_stop=stop.is_set
        )
        fetches.fetch("a")
        stop.set()
        fetches.start_after("a")
        fetches.close()
        assert fetched == ["a"]

    @staticmethod
    def _second_item_blocks(gate: threading.Event, done: list[str]) -> Callable[[str], str]:
        def _fetch(item: str) -> str:
            if item == "b":  # the one fetched ahead
                gate.wait(timeout=5.0)
                done.append(item)
            return item

        return _fetch

    def test_close_waits_for_the_in_flight_fetch(self) -> None:
        gate, done = threading.Event(), []
        fetches = OneAheadFetches(self._second_item_blocks(gate, done), ["a", "b"], name="pf")
        fetches.fetch("a")
        fetches.start_after("a")
        threading.Timer(0.1, gate.set).start()
        fetches.close()
        assert done == ["b"]

    def test_ctrl_c_exit_does_not_wait(self) -> None:
        gate, done = threading.Event(), []
        fetch = self._second_item_blocks(gate, done)
        try:
            with (
                pytest.raises(KeyboardInterrupt),
                OneAheadFetches(fetch, ["a", "b"], name="pf-int") as fetches,
            ):
                fetches.fetch("a")
                fetches.start_after("a")
                raise KeyboardInterrupt
            assert done == []  # left the block while the prefetch was still blocked
        finally:
            gate.set()

    def test_any_other_exit_waits(self) -> None:
        gate, done = threading.Event(), []
        fetch = self._second_item_blocks(gate, done)
        with (
            pytest.raises(RuntimeError),
            OneAheadFetches(fetch, ["a", "b"], name="pf-err") as fetches,
        ):
            fetches.fetch("a")
            fetches.start_after("a")
            threading.Timer(0.1, gate.set).start()
            raise RuntimeError("processing failed")
        assert done == ["b"]
