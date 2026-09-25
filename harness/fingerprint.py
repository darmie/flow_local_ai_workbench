#!/usr/bin/env python3
"""Capture a machine + software fingerprint as JSON (hardware, OS, power state,
accelerators, vLLM version). Stored with every run so results are comparable.

Usage: python harness/fingerprint.py > run_dir/fingerprint.json
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys

import psutil


def sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def cpu_info():
    info = {"arch": platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True)}
    if sys.platform == "darwin":
        info["model"] = sh("sysctl -n machdep.cpu.brand_string")
        info["perf_cores"] = sh("sysctl -n hw.perflevel0.physicalcpu")
        info["eff_cores"] = sh("sysctl -n hw.perflevel1.physicalcpu")
        return info
    cpuinfo = read("/proc/cpuinfo")
    m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    info["model"] = m.group(1) if m else ""
    flags = set((re.search(r"flags\s*:\s*(.+)", cpuinfo) or [None, ""])[1].split())
    info["isa"] = sorted(f for f in flags if f in {
        "avx2", "avx512f", "avx512_bf16", "avx512_vnni", "avx_vnni", "amx_bf16", "amx_int8", "amx_tile"})
    info["governor"] = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["energy_perf_pref"] = read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
    info["smt_active"] = read("/sys/devices/system/cpu/smt/active")
    info["numa_nodes"] = len([d for d in os.listdir("/sys/devices/system/node")
                              if d.startswith("node")]) if os.path.isdir("/sys/devices/system/node") else ""
    return info


def memory_info():
    vm = psutil.virtual_memory()
    info = {"total_gb": round(vm.total / 2**30, 2), "swap_gb": round(psutil.swap_memory().total / 2**30, 2)}
    if sys.platform == "darwin":
        info["dimm"] = "unified (on-package)"
        return info
    # Memory type/speed needs root (dmidecode); record it when available.
    dmi = sh("dmidecode -t memory 2>/dev/null")
    if dmi:
        speeds = re.findall(r"Configured Memory Speed:\s*(\d+)", dmi)
        types = re.findall(r"^\s*Type:\s*(DDR\S*|LPDDR\S*)", dmi, re.M)
        sizes = re.findall(r"^\s*Size:\s*(\d+ [GM]B)", dmi, re.M)
        info["dimm"] = {"type": sorted(set(types)), "speed_mts": sorted(set(speeds)),
                        "populated_slots": len(sizes), "sizes": sizes}
    else:
        info["dimm"] = "unknown (run fingerprint as root to read dmidecode)"
    return info


def gpu_info():
    gpus = []
    if shutil.which("nvidia-smi"):
        out = sh("nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,"
                 "pcie.link.width.max,power.limit --format=csv,noheader")
        for line in out.splitlines():
            name, mem, drv, gen, width, plimit = [c.strip() for c in line.split(",")]
            gpus.append({"vendor": "nvidia", "name": name, "memory": mem, "driver": drv,
                         "pcie": f"gen{gen} x{width}", "power_limit": plimit})
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", sh("nvidia-smi"))
        if cuda and gpus:
            gpus[0]["cuda"] = cuda.group(1)
        topo = sh("nvidia-smi topo -m")
        if topo and len(gpus) > 1:
            gpus.append({"topology": topo})
    if shutil.which("rocm-smi"):
        gpus.append({"vendor": "amd", "rocm_smi": sh("rocm-smi --showproductname --showmeminfo vram")})
    if sys.platform == "darwin":
        gpus.append({"vendor": "apple", "detail": sh("system_profiler SPDisplaysDataType")[:2000]})
    elif shutil.which("lspci"):
        vga = sh("lspci | grep -Ei 'vga|3d|display'")
        if vga:
            gpus.append({"lspci": vga.splitlines()})
    return gpus


def power_state():
    info = {}
    batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if batt:
        info["on_ac"] = batt.power_plugged
        info["battery_pct"] = batt.percent
    if shutil.which("powerprofilesctl"):
        info["power_profile"] = sh("powerprofilesctl get")
    if sys.platform == "darwin":
        info["pmset"] = sh("pmset -g")
    return info


def software():
    sw = {"os": platform.platform(), "python": platform.python_version()}
    if sys.platform.startswith("linux"):
        sw["distro"] = (re.search(r'PRETTY_NAME="(.+)"', read("/etc/os-release")) or [None, ""])[1]
        sw["kernel"] = platform.release()
    sw["vllm"] = sh(f"{sys.executable} -c 'import vllm; print(vllm.__version__)' 2>/dev/null")
    sw["docker"] = sh("docker --version")
    sw["vllm_image"] = os.environ.get("VLLM_IMAGE", "")
    if sw["vllm_image"] and sw["docker"]:
        sw["vllm_image_digest"] = sh(f"docker image inspect --format '{{{{index .RepoDigests 0}}}}' {sw['vllm_image']}")
    here = os.path.dirname(os.path.abspath(__file__))
    sw["harness_commit"] = sh(f"git -C {here} rev-parse --short HEAD")
    sw["harness_dirty"] = bool(sh(f"git -C {here} status --porcelain"))
    return sw


def main():
    fp = {
        "machine_id": os.environ.get("BENCH_MACHINE_ID", platform.node()),
        "tier": os.environ.get("BENCH_TIER", ""),
        "operator": os.environ.get("BENCH_OPERATOR", ""),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpus": gpu_info(),
        "power": power_state(),
        "software": software(),
        "boot_time": psutil.boot_time(),
    }
    json.dump(fp, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
