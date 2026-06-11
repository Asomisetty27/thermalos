"""
Tests for daemon.py stage-isolation (_stage context manager).

Before _stage existed, a deterministic exception in any advisory pipeline
stage aborted the remainder of _process_sample for that tick — a broken
enrichment stage silently killed the state machine and alert routing.
These tests pin the isolation contract without constructing a full daemon
(PrometheusExporter binds a port), using __new__ + the minimal attrs the
helper touches.
"""

import asyncio

import pytest

from theta.agent.daemon import ThetaAgent


def make_agent_shell():
    """Bare ThetaAgent with only the state _stage() needs."""
    agent = ThetaAgent.__new__(ThetaAgent)
    agent._stage_errors = {}
    return agent


class TestStageIsolation:
    def test_stage_passes_through_success(self):
        agent = make_agent_shell()
        ran = []
        with agent._stage("demo", gpu=0):
            ran.append(True)
        assert ran == [True]
        assert agent._stage_errors == {}

    def test_stage_swallows_exception(self):
        agent = make_agent_shell()
        with agent._stage("fault_classifier", gpu=0):
            raise RuntimeError("synthetic stage failure")
        # No exception propagated; failure counted
        assert agent._stage_errors == {"fault_classifier": 1}

    def test_downstream_stages_still_run_after_failure(self):
        """The contract: stage N failing must not stop stage N+1."""
        agent = make_agent_shell()
        executed = []

        with agent._stage("a", gpu=0):
            executed.append("a-start")
            raise ValueError("boom")
        with agent._stage("b", gpu=0):
            executed.append("b")
        with agent._stage("c", gpu=0):
            executed.append("c")

        assert executed == ["a-start", "b", "c"]
        assert agent._stage_errors == {"a": 1}

    def test_error_counts_accumulate_per_stage(self):
        agent = make_agent_shell()
        for _ in range(3):
            with agent._stage("telemetry", gpu=1):
                raise OSError("flush failed")
        with agent._stage("critic", gpu=1):
            raise KeyError("x")
        assert agent._stage_errors == {"telemetry": 3, "critic": 1}

    def test_stage_works_around_await(self):
        """Stages wrap awaits (router.route etc.) — verify async bodies isolate too."""
        agent = make_agent_shell()
        executed = []

        async def failing_coro():
            raise ConnectionError("webhook down")

        async def main():
            with agent._stage("alert_route", gpu=0):
                await failing_coro()
            with agent._stage("after", gpu=0):
                executed.append("after-ran")

        asyncio.run(main())
        assert executed == ["after-ran"]
        assert agent._stage_errors == {"alert_route": 1}

    def test_keyboard_interrupt_not_swallowed(self):
        """BaseException (shutdown signals) must escape — only Exception isolates."""
        agent = make_agent_shell()
        with pytest.raises(KeyboardInterrupt):
            with agent._stage("any", gpu=0):
                raise KeyboardInterrupt()
        assert agent._stage_errors == {}
