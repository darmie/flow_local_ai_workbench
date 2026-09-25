"""Unit tests for telemetry parsers that cannot run on the dev machine's hardware.

Run: python -m unittest discover tests
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))

from summarize import _is_throttled  # noqa: E402
from fingerprint import accelerator, machine_class, suggest_tier  # noqa: E402
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


def _fp(cpu="x", arch="x86_64", ram=16.0, gpus=(), os_name="Linux-6.8-x86_64"):
    return {"cpu": {"model": cpu, "arch": arch}, "memory": {"total_gb": ram}, "gpus": list(gpus),
            "software": {"os": os_name}}


def _nv(name, mib):
    return {"vendor": "nvidia", "name": name, "memory": f"{mib} MiB"}


class TierTest(unittest.TestCase):
    def tier(self, fp):
        return suggest_tier(*accelerator(fp))

    def test_discrete(self):
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 4060", 8188)])), "t2-entry")
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 5090", 32607)])), "t3-pro")
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 6000 Ada", 49140)])), "t4-workstation")
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 4090", 24564)] * 2)), "t5-multi-gpu")

    def test_unified(self):
        mac = dict(arch="arm64", os_name="macOS-15.5-arm64-arm-64bit")
        self.assertEqual(self.tier(_fp(ram=8, **mac)), "t1-minimal")
        self.assertEqual(self.tier(_fp(ram=24, **mac)), "t2-entry")
        self.assertEqual(self.tier(_fp(ram=48, **mac)), "t3-pro")
        self.assertEqual(self.tier(_fp(ram=128, **mac)), "t4-workstation")
        self.assertEqual(self.tier(_fp(cpu="AMD RYZEN AI MAX+ 395", ram=124)), "t4-workstation")
        self.assertEqual(self.tier(_fp(ram=119, gpus=[{"vendor": "nvidia", "name": "NVIDIA GB10", "memory": "[N/A]"}])),
                         "t4-workstation")

    def test_cpu_only(self):
        self.assertEqual(self.tier(_fp()), "t1-minimal")


class MachineClassTest(unittest.TestCase):
    def cls(self, fp, form):
        fp["form_factor"] = form
        fp["memory"]["ecc"] = fp["memory"].get("ecc", False)
        kind, _, n = accelerator(fp)
        return machine_class(fp, kind, n)

    def test_classes(self):
        self.assertEqual(self.cls(_fp(gpus=[_nv("NVIDIA GeForce RTX 4070 Laptop GPU", 8188)]), "laptop"), "gaming-laptop")
        self.assertEqual(self.cls(_fp(gpus=[_nv("NVIDIA GeForce RTX 4090", 24564)]), "desktop"), "gaming-desktop")
        self.assertEqual(self.cls(_fp(gpus=[_nv("NVIDIA RTX 5000 Ada Generation", 32760)]), "desktop"), "pro-workstation")
        self.assertEqual(self.cls(_fp(gpus=[_nv("NVIDIA RTX A2000 Laptop GPU", 8192)]), "laptop"), "pro-laptop")
        self.assertEqual(self.cls(_fp(cpu="Intel Core i5-1235U"), "laptop"), "office-laptop")
        self.assertEqual(self.cls(_fp(cpu="AMD RYZEN AI MAX+ 395", ram=124), "desktop"), "uma-workstation")
        self.assertEqual(self.cls(_fp(arch="arm64", ram=36, os_name="macOS-15.5-arm64-arm-64bit"), "laptop"), "apple")
        self.assertEqual(self.cls(_fp(gpus=[_nv("NVIDIA L40S", 46068)] * 2), "server"), "server")


class ThrottleTest(unittest.TestCase):
    def test_idle_is_not_throttle(self):
        self.assertFalse(_is_throttled("0x0000000000000001"))

    def test_thermal_is_throttle(self):
        self.assertTrue(_is_throttled("0x0000000000000000|0x0000000000000040"))


if __name__ == "__main__":
    unittest.main()
