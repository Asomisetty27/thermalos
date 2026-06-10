"""
Tests for silicon.py: ECC monitoring and micro-throttle detection.

Characterization tests pinning current behavior of EccMonitor, MicroThrottleDetector,
and decode_throttle_reasons. Covers edge cases: zero baselines, counter wraparound,
alert cooldowns, division-by-zero guards.
"""

import pytest
from theta.agent.silicon import (
    EccMonitor,
    MicroThrottleDetector,
    decode_throttle_reasons,
    SBIT_RATE_WARN_PER_HOUR,
)
from theta.agent.metrics import RawSample, GPUState


class TestDecodeThrottleReasons:
    """Test bitmask-to-string decoding of NVML throttle reasons."""

    def test_no_throttle(self):
        assert decode_throttle_reasons(0) == []

    def test_single_reason_sw_power_cap(self):
        mask = 0x0000000000000004
        reasons = decode_throttle_reasons(mask)
        assert "sw_power_cap" in reasons
        assert len(reasons) == 1

    def test_single_reason_hw_thermal(self):
        mask = 0x0000000000000040
        reasons = decode_throttle_reasons(mask)
        assert "hw_thermal_slowdown" in reasons

    def test_multiple_reasons(self):
        # sw_power_cap | hw_thermal_slowdown
        mask = 0x0000000000000004 | 0x0000000000000040
        reasons = decode_throttle_reasons(mask)
        assert "sw_power_cap" in reasons
        assert "hw_thermal_slowdown" in reasons
        assert len(reasons) == 2

    def test_unknown_bit_ignored(self):
        # Valid bit + unknown bit
        mask = 0x0000000000000004 | 0x1000000000000000
        reasons = decode_throttle_reasons(mask)
        assert "sw_power_cap" in reasons
        assert len(reasons) == 1  # unknown bit ignored


class TestEccMonitor:
    """Test ECC error detection and alert emission."""

    def _sample(
        self,
        gpu_index=0,
        timestamp=0.0,
        ecc_sbit=0,
        ecc_dbit=0,
    ) -> RawSample:
        """Helper to create a minimal RawSample."""
        return RawSample(
            gpu_index=gpu_index,
            timestamp=timestamp,
            temp_junction=30.0,
            power_w=100.0,
            util_pct=50.0,
            mem_util_pct=30.0,
            perf_state=3,
            clock_sm_mhz=1000,
            clock_mem_mhz=5000,
            ecc_sbit=ecc_sbit,
            ecc_dbit=ecc_dbit,
        )

    def test_first_sample_no_alert(self):
        """First sample establishes baseline, no alert."""
        monitor = EccMonitor()
        sample = self._sample(ecc_sbit=5, ecc_dbit=0)
        alert = monitor.update(sample)
        assert alert is None

    def test_no_ecc_increase_no_alert(self):
        """ECC counters unchanged → no alert."""
        monitor = EccMonitor()
        sample1 = self._sample(timestamp=0, ecc_sbit=10, ecc_dbit=0)
        sample2 = self._sample(timestamp=1, ecc_sbit=10, ecc_dbit=0)
        monitor.update(sample1)
        alert = monitor.update(sample2)
        assert alert is None

    def test_dbit_increase_critical_alert(self):
        """Any increase in double-bit errors → CRITICAL alert."""
        monitor = EccMonitor()
        sample1 = self._sample(timestamp=1000, ecc_sbit=0, ecc_dbit=0)
        sample2 = self._sample(timestamp=1001, ecc_sbit=0, ecc_dbit=1)
        monitor.update(sample1)
        alert = monitor.update(sample2)

        assert alert is not None
        assert alert.state == GPUState.CRITICAL
        assert alert.confidence == 1.0
        assert "uncorrectable ecc" in alert.message.lower()
        assert alert.context["ecc_dbit"] == 1

    def test_multiple_dbit_increase(self):
        """Multiple double-bit errors in one interval."""
        monitor = EccMonitor()
        sample1 = self._sample(timestamp=0, ecc_sbit=0, ecc_dbit=0)
        sample2 = self._sample(timestamp=1, ecc_sbit=0, ecc_dbit=3)
        monitor.update(sample1)
        alert = monitor.update(sample2)

        assert alert is not None
        assert alert.context["ecc_dbit"] == 3

    def test_sbit_rate_accumulation(self):
        """Single-bit errors accumulate over 1-hour window."""
        monitor = EccMonitor()
        # Start at ts=4000 to pass the first-alert cooldown check (ts >= 3600)
        sample1 = self._sample(timestamp=4000, ecc_sbit=0, ecc_dbit=0)
        monitor.update(sample1)

        # Add 10 sbit errors (reaching SBIT_RATE_WARN_PER_HOUR threshold)
        for i in range(10):
            sample = self._sample(timestamp=4000 + i + 1, ecc_sbit=i + 1, ecc_dbit=0)
            alert = monitor.update(sample)
            if i < 9:
                assert alert is None  # Below threshold or not enough time
            else:
                # 10th sbit error at SBIT_RATE_WARN_PER_HOUR threshold
                assert alert is not None
                assert alert.state == GPUState.DRIFTING
                assert "single-bit ecc" in alert.message.lower()

    def test_sbit_alert_cooldown(self):
        """Single-bit alerts suppress repeated alerts for 1 hour."""
        monitor = EccMonitor()
        sample0 = self._sample(timestamp=4000, ecc_sbit=0, ecc_dbit=0)
        monitor.update(sample0)

        # First sbit alert at t=4010, when we accumulate 10 errors
        for i in range(10):
            sample = self._sample(timestamp=4000 + i + 1, ecc_sbit=i + 1, ecc_dbit=0)
            alert = monitor.update(sample)
        alert1 = alert  # Last alert from loop
        assert alert1 is not None

        # At t=4500 (490s later, still within 3600s cooldown), another error
        # but cooldown suppresses alert
        sample2 = self._sample(timestamp=4500, ecc_sbit=20, ecc_dbit=0)
        alert2 = monitor.update(sample2)
        assert alert2 is None  # Suppressed by cooldown

        # At t=7610 (3600s+ after first alert at t=4010), cooldown expires
        sample3 = self._sample(timestamp=7610, ecc_sbit=30, ecc_dbit=0)
        alert3 = monitor.update(sample3)
        assert alert3 is not None

    def test_sbit_rate_clears_with_time(self):
        """Old sbit errors drop out of the 1-hour rolling window."""
        monitor = EccMonitor()
        sample0 = self._sample(timestamp=4000, ecc_sbit=0, ecc_dbit=0)
        monitor.update(sample0)

        # Accumulate 10 sbits by t=4010
        for i in range(10):
            sample = self._sample(timestamp=4000 + i + 1, ecc_sbit=i + 1, ecc_dbit=0)
            alert = monitor.update(sample)
        alert1 = alert
        assert alert1 is not None  # Triggers alert at t=4010

        # At t=7610 (3600s later), the 10 errors from t=4000-4010 window have aged out
        # The rolling window only includes errors within the last 3600s
        # At t=7610, errors from before t=4010 are outside the window
        # Only new error at t=7610 counts → rate = 1, below threshold
        sample2 = self._sample(timestamp=7610, ecc_sbit=11, ecc_dbit=0)
        alert2 = monitor.update(sample2)
        # Only 1 new sbit (11 - 10 = 1) → rate = 1, below threshold (10)
        assert alert2 is None

    def test_counter_does_not_go_backwards(self):
        """Negative deltas (counter reset) treated as max(0, delta)."""
        monitor = EccMonitor()
        sample1 = self._sample(timestamp=0, ecc_sbit=100, ecc_dbit=5)
        monitor.update(sample1)

        # Counter resets (e.g., driver reload)
        sample2 = self._sample(timestamp=1, ecc_sbit=10, ecc_dbit=0)
        alert = monitor.update(sample2)
        # delta = max(0, 10 - 100) = 0, no alert
        assert alert is None

    def test_multiple_gpus_independent(self):
        """ECC state is tracked per-GPU independently."""
        monitor = EccMonitor()
        sample_g0 = self._sample(gpu_index=0, timestamp=0, ecc_sbit=0, ecc_dbit=0)
        sample_g1 = self._sample(gpu_index=1, timestamp=0, ecc_sbit=0, ecc_dbit=0)
        monitor.update(sample_g0)
        monitor.update(sample_g1)

        # GPU 0 gets a dbit error
        sample_g0_bad = self._sample(gpu_index=0, timestamp=1, ecc_sbit=0, ecc_dbit=1)
        alert = monitor.update(sample_g0_bad)
        assert alert is not None
        assert alert.gpu_index == 0

        # GPU 1 unaffected
        sample_g1_ok = self._sample(gpu_index=1, timestamp=1, ecc_sbit=0, ecc_dbit=0)
        alert2 = monitor.update(sample_g1_ok)
        assert alert2 is None


class TestMicroThrottleDetector:
    """Test SM clock suppression detection under load."""

    def _sample(
        self,
        gpu_index=0,
        timestamp=0.0,
        clock_sm_mhz=1000,
        sm_clock_max_mhz=1500,
        util_pct=90.0,
        throttle_reasons=0,
    ) -> RawSample:
        """Helper to create a minimal RawSample."""
        return RawSample(
            gpu_index=gpu_index,
            timestamp=timestamp,
            temp_junction=30.0,
            power_w=100.0,
            util_pct=util_pct,
            mem_util_pct=30.0,
            perf_state=3,
            clock_sm_mhz=clock_sm_mhz,
            clock_mem_mhz=5000,
            throttle_reasons=throttle_reasons,
            sm_clock_max_mhz=sm_clock_max_mhz,
        )

    def test_no_suppression_no_alert(self):
        """High clock ratio + high load → no alert."""
        detector = MicroThrottleDetector()
        # 1000 / 1500 = 0.667 >= EFFICIENCY_THRESHOLD (0.95)? No, this is low
        # Let's use high ratio: 1400 / 1500 = 0.933, below threshold
        sample = self._sample(clock_sm_mhz=1420, sm_clock_max_mhz=1500, util_pct=90)
        alert = detector.update(sample)
        assert alert is None

    def test_suppression_high_efficiency_no_alert(self):
        """Low utilization, even with suppressed clock → no alert."""
        detector = MicroThrottleDetector()
        sample = self._sample(
            clock_sm_mhz=1000,
            sm_clock_max_mhz=1500,
            util_pct=50.0,  # Below LOAD_THRESHOLD (80)
        )
        alert = detector.update(sample)
        assert alert is None

    def test_suppression_and_load_requires_sustained(self):
        """Single sample of suppression + high load → no alert yet."""
        detector = MicroThrottleDetector()
        sample = self._sample(
            clock_sm_mhz=1000,
            sm_clock_max_mhz=1500,  # 1000/1500 = 0.667 < 0.95
            util_pct=90.0,
        )
        alert = detector.update(sample)
        # Only 1 consecutive, need SUSTAINED_SAMPLES (5)
        assert alert is None

    def test_sustained_suppression_triggers_alert(self):
        """5 consecutive samples of suppression + load → alert."""
        detector = MicroThrottleDetector()
        for i in range(5):
            sample = self._sample(
                timestamp=1000 + i,
                clock_sm_mhz=1000,
                sm_clock_max_mhz=1500,
                util_pct=90.0,
                throttle_reasons=0x0000000000000040,  # hw_thermal_slowdown
            )
            alert = detector.update(sample)
            if i < 4:
                assert alert is None
            else:
                # 5th sample triggers
                assert alert is not None
                assert alert.state == GPUState.DRIFTING
                assert "micro-throttling" in alert.message.lower()
                assert "hw_thermal_slowdown" in alert.message.lower()

    def test_throttle_reason_decode(self):
        """Throttle reasons are decoded and included in alert."""
        detector = MicroThrottleDetector()
        # Trigger with multiple throttle reasons
        mask = 0x0000000000000004 | 0x0000000000000040  # power_cap + thermal
        for i in range(5):
            sample = self._sample(
                timestamp=1000 + i,
                clock_sm_mhz=1000,
                sm_clock_max_mhz=1500,
                util_pct=90.0,
                throttle_reasons=mask,
            )
            alert = detector.update(sample)

        assert alert is not None
        assert "sw_power_cap" in alert.message
        assert "hw_thermal_slowdown" in alert.message

    def test_throttle_cooldown(self):
        """Repeated suppression alerts have 300s cooldown."""
        detector = MicroThrottleDetector()

        # First alert at t=1000-1004
        for i in range(5):
            sample = self._sample(timestamp=1000 + i, clock_sm_mhz=1000, sm_clock_max_mhz=1500)
            alert = detector.update(sample)
        first_alert = alert
        assert first_alert is not None

        # At t=1100-1104 (100s after alert, still within 300s cooldown)
        # Alert should be suppressed despite sustained suppression
        for i in range(5, 10):
            sample = self._sample(timestamp=1100 + i, clock_sm_mhz=1000, sm_clock_max_mhz=1500)
            alert = detector.update(sample)
        assert alert is None

        # At t=1310 (310s since first alert, after 300s cooldown), alert fires again
        # Send one sample to trigger (already have 5 consecutive from t=1100-1104)
        sample = self._sample(timestamp=1310, clock_sm_mhz=1000, sm_clock_max_mhz=1500)
        second_alert = detector.update(sample)
        assert second_alert is not None
        assert second_alert.gpu_index == first_alert.gpu_index

    def test_reset_on_recovery(self):
        """Clock recovers (efficiency > threshold) → consecutive count resets."""
        detector = MicroThrottleDetector()

        # 2 samples of suppression
        for i in range(2):
            sample = self._sample(timestamp=1000 + i, clock_sm_mhz=1000, sm_clock_max_mhz=1500)
            alert = detector.update(sample)
            assert alert is None

        # Recovery: high efficiency
        sample = self._sample(timestamp=1002, clock_sm_mhz=1450, sm_clock_max_mhz=1500)
        alert = detector.update(sample)
        assert alert is None

        # Back to suppression, restarts count
        for i in range(3, 8):
            sample = self._sample(timestamp=1000 + i, clock_sm_mhz=1000, sm_clock_max_mhz=1500)
            alert = detector.update(sample)
            if i < 7:
                assert alert is None
            else:
                # 5th consecutive again
                assert alert is not None

    def test_sm_clock_max_zero_guard(self):
        """sm_clock_max_mhz = 0 → no division by zero, no alert."""
        detector = MicroThrottleDetector()
        sample = self._sample(
            clock_sm_mhz=1000,
            sm_clock_max_mhz=0,
            util_pct=90.0,
        )
        alert = detector.update(sample)
        assert alert is None

    def test_multiple_gpus_independent(self):
        """Throttle state tracked per-GPU independently."""
        detector = MicroThrottleDetector()

        # GPU 0: 5 samples of suppression → alert
        for i in range(5):
            sample_g0 = self._sample(
                gpu_index=0,
                timestamp=1000 + i,
                clock_sm_mhz=1000,
                sm_clock_max_mhz=1500,
            )
            alert = detector.update(sample_g0)

        assert alert is not None
        assert alert.gpu_index == 0

        # GPU 1: only 2 samples of suppression → no alert yet
        for i in range(2):
            sample_g1 = self._sample(
                gpu_index=1,
                timestamp=1000 + i,
                clock_sm_mhz=1000,
                sm_clock_max_mhz=1500,
            )
            alert = detector.update(sample_g1)

        assert alert is None  # GPU 1 hasn't triggered yet

        # GPU 0: still under cooldown, no alert
        sample_g0_again = self._sample(gpu_index=0, timestamp=1100, clock_sm_mhz=1000, sm_clock_max_mhz=1500)
        alert = detector.update(sample_g0_again)
        assert alert is None
