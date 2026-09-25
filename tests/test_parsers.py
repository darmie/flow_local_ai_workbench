"""Unit tests for telemetry parsers that cannot run on the dev machine's hardware.

Run: python -m unittest discover tests
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))

from summarize import _is_throttled  # noqa: E402
from hostload import parse_typeperf_line  # noqa: E402
from telemetry import parse_powermetrics  # noqa: E402


class PowerMetricsTest(unittest.TestCase):
    def test_apple_silicon_sample(self):
        raw = open(os.path.join(ROOT, "tests", "fixtures", "powermetrics_sample.plist"), "rb").read()
        docs = [d for d in raw.split(b"\0") if d.strip()]
        self.assertEqual(len(docs), 2)
        s = parse_powermetrics(docs[0])
        self.assertEqual(s["cpu_pkg_power_w"], 4.25)
        self.assertEqual(s["gpu_power_w"], 11.8)
        self.assertEqual(s["soc_power_w"], 16.05)
        self.assertEqual(s["gpu_util_pct"], 88.0)
        self.assertEqual(s["gpu_sm_clock_mhz"], 1398)

    def test_garbage_is_ignored(self):
        self.assertEqual(parse_powermetrics(b"not a plist"), {})


class TypeperfTest(unittest.TestCase):
    def test_rows(self):
        header = '"(PDH-CSV 4.0)","\\\\PC\\Processor(_Total)\\% Processor Time","\\\\PC\\Memory\\Available MBytes"'
        self.assertIsNone(parse_typeperf_line(header))
        self.assertIsNone(parse_typeperf_line(""))
        self.assertEqual(parse_typeperf_line('"09/25/2026 19:30:01.123","12.5","8123.000000"'), (12.5, 8123.0))
        self.assertIsNone(parse_typeperf_line('"09/25/2026 19:30:02.123"," ","8123"'))


class ThrottleTest(unittest.TestCase):
    def test_idle_is_not_throttle(self):
        self.assertFalse(_is_throttled("0x0000000000000001"))

    def test_thermal_is_throttle(self):
        self.assertTrue(_is_throttled("0x0000000000000000|0x0000000000000040"))


if __name__ == "__main__":
    unittest.main()
