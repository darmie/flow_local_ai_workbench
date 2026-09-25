"""Unit tests for telemetry parsers that cannot run on the dev machine's hardware.

Run: python -m unittest discover tests
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))

from summarize import _is_throttled  # noqa: E402
from fingerprint import accelerator, machine_class, suggest_tier, udev_memory  # noqa: E402
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
        return suggest_tier(*accelerator(fp), fp["cpu"]["model"])

    def test_discrete(self):
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 4060", 8188)])), "t3-entry")
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 5090", 32607)])), "t4-pro")
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 6000 Ada", 49140)])), "t5-workstation")
        self.assertEqual(self.tier(_fp(gpus=[_nv("RTX 4090", 24564)] * 2)), "t6-multi-gpu")

    def test_unified(self):
        mac = dict(arch="arm64", os_name="macOS-15.5-arm64-arm-64bit")
        self.assertEqual(self.tier(_fp(ram=8, **mac)), "t1-minimal")
        self.assertEqual(self.tier(_fp(ram=16, **mac)), "t2-integrated")
        self.assertEqual(self.tier(_fp(ram=24, **mac)), "t2-integrated")
        self.assertEqual(self.tier(_fp(ram=48, **mac)), "t4-pro")
        self.assertEqual(self.tier(_fp(ram=128, **mac)), "t5-workstation")
        self.assertEqual(self.tier(_fp(cpu="AMD RYZEN AI MAX+ 395", ram=124)), "t5-workstation")
        self.assertEqual(self.tier(_fp(ram=119, gpus=[{"vendor": "nvidia", "name": "NVIDIA GB10", "memory": "[N/A]"}])),
                         "t5-workstation")

    def test_integrated(self):
        self.assertEqual(self.tier(_fp()), "t1-minimal")
        self.assertEqual(self.tier(_fp(cpu="12th Gen Intel(R) Core(TM) i7-1255U", ram=16)), "t1-minimal")
        self.assertEqual(self.tier(_fp(cpu="Intel(R) Core(TM) Ultra 7 258V", ram=32)), "t2-integrated")
        self.assertEqual(self.tier(_fp(cpu="AMD Ryzen AI 9 HX 370 w/ Radeon 890M", ram=32)), "t2-integrated")
        self.assertEqual(self.tier(_fp(cpu="Snapdragon(R) X Elite - X1E78100", arch="aarch64", ram=16)), "t2-integrated")
        self.assertEqual(self.tier(_fp(cpu="Intel(R) Core(TM) Ultra 5 125U", ram=8)), "t1-minimal")


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


class UdevMemoryTest(unittest.TestCase):
    def test_two_ddr5_modules(self):
        out = "\n".join(["MEMORY_ARRAY_ERROR_CORRECTION=None", "MEMORY_DEVICE_0_SIZE=17179869184",
                         "MEMORY_DEVICE_0_MEMORY_TYPE=DDR5", "MEMORY_DEVICE_0_CONFIGURED_SPEED_MTS=5600",
                         "MEMORY_DEVICE_1_SIZE=17179869184", "MEMORY_DEVICE_1_MEMORY_TYPE=DDR5",
                         "MEMORY_DEVICE_1_CONFIGURED_SPEED_MTS=5600", "MEMORY_DEVICE_2_SIZE=0"])
        d = udev_memory(out)
        self.assertEqual((d["type"], d["speed_mts"], d["populated_slots"], d["ecc"]), (["DDR5"], ["5600"], 2, False))

    def test_absent(self):
        self.assertIsNone(udev_memory("ID_VENDOR=x"))


class ModelsConfigTest(unittest.TestCase):
    def test_models_yaml_is_valid(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT, "harness", "check_models.py")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_rejects_unpinned(self):
        from check_models import check
        errs = check("new-model", {"hf": "org/name", "revision": "main", "quant": "awq", "params_b": 7,
                                   "tiers": ["t3-entry"]}, {"t3-entry": {}}, {})
        self.assertTrue(any("40-character" in e for e in errs))


class ThrottleTest(unittest.TestCase):
    def test_idle_is_not_throttle(self):
        self.assertFalse(_is_throttled("0x0000000000000001"))

    def test_thermal_is_throttle(self):
        self.assertTrue(_is_throttled("0x0000000000000000|0x0000000000000040"))


if __name__ == "__main__":
    unittest.main()
