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
        sw["wsl"] = "microsoft" in platform.release().lower()
    sw["vllm"] = sh(f"{sys.executable} -c 'import vllm; print(vllm.__version__)' 2>/dev/null")
    sw["docker"] = sh("docker --version")
    sw["vllm_image"] = os.environ.get("VLLM_IMAGE", "")
    if sw["vllm_image"] and sw["docker"]:
        sw["vllm_image_digest"] = sh(f"docker image inspect --format '{{{{index .RepoDigests 0}}}}' {sw['vllm_image']}")
    here = os.path.dirname(os.path.abspath(__file__))
    sw["harness_commit"] = sh(f"git -C {here} rev-parse --short HEAD")
    sw["harness_dirty"] = bool(sh(f"git -C {here} status --porcelain"))
    return sw


def system_model():
    """Vendor + product name of the machine (DMI on Linux, hw model on macOS)."""
    if sys.platform == "darwin":
        hw = sh("system_profiler SPHardwareDataType")
        name = re.search(r"Model Name:\s*(.+)", hw)
        ident = re.search(r"Model Identifier:\s*(.+)", hw)
        return " ".join(m.group(1).strip() for m in (name, ident) if m)
    parts = [read(f"/sys/class/dmi/id/{f}") for f in ("sys_vendor", "product_name", "product_version")]
    parts = [p for p in parts if p and p.lower() not in ("to be filled by o.e.m.", "default string", "none")]
    return " ".join(dict.fromkeys(parts))


def spec(fp):
    """Flat, human-readable specification used as report columns.

    BENCH_MACHINE_NAME and BENCH_RAM_DESC override detection; set them when
    DMI/dmidecode are unavailable (VMs, WSL2, no root).
    """
    cpu, mem = fp["cpu"], fp["memory"]
    dimm = mem.get("dimm") if isinstance(mem.get("dimm"), dict) else {}
    ram_desc = os.environ.get("BENCH_RAM_DESC") or " ".join(filter(None, [
        "/".join(dimm.get("type", [])),
        f"{'/'.join(dimm.get('speed_mts', []))} MT/s" if dimm.get("speed_mts") else "",
        f"{dimm['populated_slots']} DIMM" if dimm.get("populated_slots") else "",
    ])) or ("unified" if sys.platform == "darwin" else "(type unknown)")
    gpus = []
    for g in fp["gpus"]:
        if g.get("vendor") == "nvidia":
            gpus.append(f"{g['name']} {g['memory']}")
        elif g.get("lspci"):
            gpus += [re.sub(r"^\S+\s+(VGA compatible controller|3D controller|Display controller):\s*", "", l)
                     for l in g["lspci"]]
        elif g.get("vendor") == "amd":
            gpus.append("AMD " + " ".join(g.get("rocm_smi", "").split()[-6:]))
        elif g.get("vendor") == "apple":
            cores = re.search(r"Total Number of Cores:\s*(\d+)", g.get("detail", ""))
            gpus.append("Apple GPU" + (f" {cores.group(1)}-core" if cores else ""))
    nv = next((g for g in fp["gpus"] if g.get("vendor") == "nvidia"), {})
    out = {
        "machine_name": os.environ.get("BENCH_MACHINE_NAME") or system_model() or fp["machine_id"],
        "cpu_model": cpu.get("model", ""),
        "cpu_cores": f"{cpu.get('physical_cores')}C/{cpu.get('logical_cores')}T",
        "cpu_isa": " ".join(cpu.get("isa", [])),
        "ram_gb": mem.get("total_gb"),
        "ram_desc": ram_desc,
        "gpu": "; ".join(gpus) or "none",
        "gpu_driver": " ".join(filter(None, [nv.get("driver", ""), f"CUDA {nv['cuda']}" if nv.get("cuda") else ""])),
        "gpu_power_limit": nv.get("power_limit", ""),
        "gpu_pcie": nv.get("pcie", ""),
        "os": (fp["software"].get("distro") or fp["software"].get("os", ""))
              + (" (WSL2 on Windows)" if fp["software"].get("wsl") else ""),
        "kernel": fp["software"].get("kernel", ""),
        "power": "AC" if fp["power"].get("on_ac", True) else "battery",
        "power_profile": fp["power"].get("power_profile") or cpu.get("governor", ""),
    }
    out["summary"] = " | ".join(str(x) for x in [
        out["machine_name"], f"{out['cpu_model']} {out['cpu_cores']}",
        f"{out['ram_gb']} GB {out['ram_desc']}", out["gpu"], out["os"]] if x)
    return out


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
    fp["spec"] = spec(fp)
    json.dump(fp, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
