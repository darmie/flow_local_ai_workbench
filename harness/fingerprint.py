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


# SMBIOS memory type codes (Win32_PhysicalMemory.SMBIOSMemoryType).
SMBIOS_MEM = {18: "DDR", 19: "DDR2", 24: "DDR3", 26: "DDR4", 27: "LPDDR", 28: "LPDDR2", 29: "LPDDR3",
              30: "LPDDR4", 34: "DDR5", 35: "LPDDR5"}
PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def is_wsl():
    return "microsoft" in platform.release().lower()


def windows_host():
    """Maker, model and memory modules of the Windows host under WSL2 (no admin needed)."""
    if not is_wsl() or not os.path.exists(PS):
        return {}
    cmd = ("$c = Get-CimInstance Win32_ComputerSystem; "
           "$m = @(Get-CimInstance Win32_PhysicalMemory | Select-Object SMBIOSMemoryType, ConfiguredClockSpeed, Capacity); "
           "@{maker = $c.Manufacturer; model = $c.Model; mem = $m} | ConvertTo-Json -Depth 3 -Compress")
    try:
        # Single quotes: the shell must not expand PowerShell's $variables.
        return json.loads(sh(f"{PS} -NoProfile -Command '{cmd}'", timeout=30) or "{}")
    except ValueError:
        return {}


def udev_memory(out=None):
    """DIMM type/speed from udev's DMI properties (systemd >= 248), readable without root."""
    if out is None:
        out = sh("udevadm info --query=property --path=/sys/devices/virtual/dmi/id")
    props = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
    types, speeds, n = set(), set(), 0
    for key, value in props.items():
        m = re.match(r"MEMORY_DEVICE_(\d+)_SIZE$", key)
        if m and value not in ("0", ""):
            i = m.group(1)
            n += 1
            if props.get(f"MEMORY_DEVICE_{i}_MEMORY_TYPE"):
                types.add(props[f"MEMORY_DEVICE_{i}_MEMORY_TYPE"])
            speed = props.get(f"MEMORY_DEVICE_{i}_CONFIGURED_SPEED_MTS") or props.get(f"MEMORY_DEVICE_{i}_SPEED_MTS")
            if speed:
                speeds.add(speed)
    if not n:
        return None
    ecc = props.get("MEMORY_ARRAY_ERROR_CORRECTION", "")
    return {"type": sorted(types), "speed_mts": sorted(speeds), "populated_slots": n,
            "source": "udev", "ecc": bool(ecc) and ecc.lower() not in ("none", "unknown")}


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
        ecc = re.search(r"Error Correction Type:\s*(.+)", sh("dmidecode -t 16 2>/dev/null"))
        info["ecc"] = bool(ecc and "none" not in ecc.group(1).lower())
    elif udev_memory():
        info["dimm"] = udev_memory()
        info["ecc"] = info["dimm"].pop("ecc")
    elif windows_host().get("mem"):
        mods = windows_host()["mem"]
        info["dimm"] = {"type": sorted({SMBIOS_MEM.get(m.get("SMBIOSMemoryType"), "") for m in mods} - {""}),
                        "speed_mts": sorted({str(m["ConfiguredClockSpeed"]) for m in mods if m.get("ConfiguredClockSpeed")}),
                        "populated_slots": len(mods), "source": "windows"}
    else:
        info["dimm"] = "unknown"
    return info


def gpu_info():
    gpus = []
    if shutil.which("nvidia-smi"):
        out = sh("nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,"
                 "pcie.link.width.max,power.limit,power.default_limit,power.max_limit --format=csv,noheader")
        for line in out.splitlines():
            name, mem, drv, gen, width, plimit, pdefault, pmax = [c.strip() for c in line.split(",")]
            gpus.append({"vendor": "nvidia", "name": name, "memory": mem, "driver": drv,
                         "pcie": f"gen{gen} x{width}", "power_limit": plimit,
                         "power_default_limit": pdefault, "power_max_limit": pmax})
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
    if is_wsl():
        host = windows_host()
        parts = [host.get("maker", ""), host.get("model", "")]
    junk = ("to be filled by o.e.m.", "default string", "none", "system product name", "system manufacturer")
    parts = [p.strip() for p in parts if p and p.strip().lower() not in junk]
    return " ".join(dict.fromkeys(parts))


# SMBIOS chassis type codes (DMI /sys/class/dmi/id/chassis_type, readable without root).
LAPTOP_CHASSIS = {8, 9, 10, 11, 14, 30, 31, 32}
SERVER_CHASSIS = {17, 23, 25, 28, 29}
PRO_GPU = re.compile(r"RTX (A\d{3,4}|\d{4} Ada|PRO)|Quadro|Radeon (AI )?PRO|Arc Pro|Tesla|\b[AHL]\d{2,3}\b|GB10", re.I)
WORKSTATION_CPU = re.compile(r"Xeon|Threadripper|EPYC", re.I)


def chassis():
    """form factor: laptop | desktop | server | virtual | unknown."""
    virt = sh("systemd-detect-virt 2>/dev/null")
    if virt and virt != "none":
        return "virtual"
    if sys.platform == "darwin":
        model = sh("sysctl -n hw.model")
        return "laptop" if "MacBook" in model else "desktop"
    try:
        code = int(read("/sys/class/dmi/id/chassis_type"))
    except ValueError:
        return "unknown"
    return "laptop" if code in LAPTOP_CHASSIS else "server" if code in SERVER_CHASSIS else "desktop"


def machine_class(fp, kind, n_gpu):
    """Constraint profile beside the tier: what limits this kind of machine
    (power and cooling on laptops, VRAM vs system RAM on desktops, etc.)."""
    form = fp.get("form_factor", "unknown")
    names = " ".join(g.get("name", "") for g in fp["gpus"] if g.get("vendor") in ("nvidia", "amd"))
    names += " " + " ".join(" ".join(g.get("lspci", [])) for g in fp["gpus"] if g.get("lspci"))
    cpu_model = fp["cpu"].get("model", "")
    if kind == "unified":
        os_name = fp.get("software", {}).get("os", "")
        return "apple" if ("macOS" in os_name or "Darwin" in os_name) else "uma-workstation"
    if form in ("server", "virtual"):
        return form
    pro = bool(PRO_GPU.search(names)) or bool(WORKSTATION_CPU.search(cpu_model)) or fp["memory"].get("ecc")
    if kind == "discrete":
        if form == "laptop" or "Laptop GPU" in names:
            return "pro-laptop" if pro else "gaming-laptop"
        return "pro-workstation" if pro else "gaming-desktop"
    return "office-laptop" if form == "laptop" else ("pro-workstation" if pro else "office-desktop")


def accelerator(fp):
    """(kind, model-memory GB, discrete GPU count) for tier selection.

    kind: discrete (dedicated VRAM), unified (Apple Silicon, AMD Strix Halo,
    NVIDIA GB10 / DGX Spark: the GPU uses system RAM) or shared (CPU + iGPU)."""
    cpu_model = fp["cpu"].get("model", "")
    ram = fp["memory"].get("total_gb") or 0
    nvidia = [g for g in fp["gpus"] if g.get("vendor") == "nvidia" and g.get("name")]
    if any("GB10" in g["name"] for g in nvidia):
        return "unified", ram, 0
    os_name = fp.get("software", {}).get("os", "")
    if ("macOS" in os_name or "Darwin" in os_name) and fp["cpu"].get("arch") == "arm64":
        return "unified", ram, 0
    if re.search(r"Ryzen AI Max", cpu_model, re.I):
        return "unified", ram, 0
    vram = []
    for g in nvidia:
        m = re.search(r"([\d.]+)\s*MiB", g.get("memory", ""))
        if m:
            vram.append(float(m.group(1)) / 1024)
    for g in fp["gpus"]:
        if g.get("vendor") == "amd":
            vram += [int(b) / 2**30 for b in re.findall(r"VRAM Total Memory \(B\):\s*(\d+)", g.get("rocm_smi", ""))]
    if vram:
        return "discrete", round(max(vram), 1), len(vram)
    return "shared", ram, 0


AI_PC_CPU = re.compile(r"Core\(TM\) Ultra|Core Ultra|Ryzen AI|Snapdragon|\bX1[EP]\b", re.I)


def suggest_tier(kind, mem_gb, n_gpu, cpu_model=""):
    """Tier rule from configs/tiers.yaml (`fits`)."""
    if n_gpu >= 2:
        return "t6-multi-gpu"
    if kind == "discrete" and mem_gb >= 7.5:
        return "t5-workstation" if mem_gb >= 44 else "t4-pro" if mem_gb >= 19 else "t3-entry"
    if kind == "unified":
        return ("t5-workstation" if mem_gb >= 90 else "t4-pro" if mem_gb >= 44
                else "t2-integrated" if mem_gb >= 12 else "t1-minimal")
    # No usable discrete GPU: modern integrated GPU + NPU machines vs basic office machines.
    return "t2-integrated" if AI_PC_CPU.search(cpu_model) and mem_gb >= 12 else "t1-minimal"


def spec(fp):
    """Flat, human-readable specification used as report columns.

    BENCH_MACHINE_NAME and BENCH_RAM_DESC override detection; set them when
    DMI/dmidecode are unavailable (VMs, WSL2, no root).
    """
    cpu, mem = fp["cpu"], fp["memory"]
    dimm = mem.get("dimm") if isinstance(mem.get("dimm"), dict) else {}
    kind_speed = "-".join(filter(None, ["/".join(dimm.get("type", [])), "/".join(dimm.get("speed_mts", []))]))
    modules = dimm.get("populated_slots")
    # BENCH_RAM_DESC overrides detection (safe; only changes the label).
    ram_desc = os.environ.get("BENCH_RAM_DESC") or ", ".join(filter(None, [
        kind_speed, f"{modules} module{'s' if modules != 1 else ''}" if modules else ""])) \
        or ("unified" if sys.platform == "darwin" else "(type unknown)")
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
        "gpu_power_max": nv.get("power_max_limit", ""),
        "gpu_pcie": nv.get("pcie", ""),
        "os": (fp["software"].get("distro") or fp["software"].get("os", ""))
              + (" (WSL2 on Windows)" if fp["software"].get("wsl") else ""),
        "kernel": fp["software"].get("kernel", ""),
        "power": "AC" if fp["power"].get("on_ac", True) else "battery",
        "power_profile": fp["power"].get("power_profile") or cpu.get("governor", ""),
    }
    kind, mem_gb, n_gpu = accelerator(fp)
    out["form_factor"] = fp.get("form_factor", "unknown")
    # BENCH_MACHINE_CLASS overrides detection (e.g. a desktop whose DMI data is blank).
    out["machine_class"] = os.environ.get("BENCH_MACHINE_CLASS") or machine_class(fp, kind, n_gpu)
    out["gpu_count"] = n_gpu
    out["ecc_memory"] = bool(fp["memory"].get("ecc"))
    out["memory_kind"] = kind
    out["model_memory_gb"] = mem_gb
    if kind == "discrete" and mem_gb < 7.5:
        kind, mem_gb = "shared", fp["memory"].get("total_gb") or 0
    out["suggested_tier"] = suggest_tier(kind, mem_gb, n_gpu, fp["cpu"].get("model", ""))
    if kind == "unified" and out["ram_desc"] in ("(type unknown)", "unified"):
        out["ram_desc"] = "unified"
    out["summary"] = " | ".join(str(x) for x in [
        out["machine_name"], out["machine_class"], f"{out['cpu_model']} {out['cpu_cores']}",
        f"{out['ram_gb']} GB {out['ram_desc']}", out["gpu"], out["os"]] if x)
    return out


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def operator():
    """BENCH_OPERATOR, else the harness checkout's git user, else the login name."""
    import getpass
    here = os.path.dirname(os.path.abspath(__file__))
    return (os.environ.get("BENCH_OPERATOR") or sh(f"git -C {here} config user.name")
            or getpass.getuser())


def main():
    fp = {
        "tier": os.environ.get("BENCH_TIER", ""),
        "operator": operator(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpus": gpu_info(),
        "power": power_state(),
        "form_factor": chassis(),
        "software": software(),
        "boot_time": psutil.boot_time(),
    }
    fp["machine_id"] = "pending"
    fp["spec"] = spec(fp)
    # Stable id: hostname + make/model; BENCH_MACHINE_ID overrides.
    name = fp["spec"]["machine_name"]
    fp["machine_id"] = os.environ.get("BENCH_MACHINE_ID") or slug(
        platform.node() + ("-" + name if name and name != "pending" else ""))[:60]
    if fp["spec"]["machine_name"] == "pending":
        fp["spec"]["machine_name"] = fp["machine_id"]
    fp["spec"]["summary"] = fp["spec"]["summary"].replace("pending", fp["spec"]["machine_name"], 1)
    json.dump(fp, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
