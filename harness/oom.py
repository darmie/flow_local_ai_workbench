"""Classify why a vLLM server failed, from its log, the container state and the
kernel log.

Categories, most specific first:
  kv_cache_insufficient  model loaded but no room for the KV cache / context
  gpu_oom                accelerator memory exhausted (CUDA, HIP, XPU, Metal)
  host_oom               the OS killed the process for lack of RAM
  crashed                died for another reason
"""
import re
import subprocess

PATTERNS = [
    ("kv_cache_insufficient", re.compile(
        r"No available memory for the cache blocks"
        r"|larger than the maximum number of tokens that can be stored in KV cache"
        r"|KV cache is needed, which is larger than the available KV cache memory"
        r"|Insufficient memory for KV cache", re.I)),
    ("gpu_oom", re.compile(
        r"CUDA out of memory|torch\.(cuda\.)?OutOfMemoryError|HIP out of memory|XPU out of memory"
        r"|OUT_OF_DEVICE_MEMORY|hipErrorOutOfMemory|cudaErrorMemoryAllocation"
        r"|Insufficient Memory \(00000008|kIOGPUCommandBufferCallbackErrorOutOfMemory", re.I)),
    ("host_oom", re.compile(r"Cannot allocate memory|MemoryError|std::bad_alloc", re.I)),
]
KERNEL_OOM = re.compile(r"Out of memory: Killed process \d+ \((?P<proc>[^)]+)\)|oom-kill.*task=(?P<task>\S+)", re.I)


def kernel_oom_lines(since_s=None):
    """Recent kernel OOM-killer lines (best effort: dmesg/journalctl may need privileges)."""
    cmds = [["journalctl", "-k", "--no-pager", "-q"] + (["--since", f"-{int(since_s)}s"] if since_s else []),
            ["dmesg", "-T"]]
    for cmd in cmds:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            return [l for l in out.splitlines() if KERNEL_OOM.search(l)][-5:]
    return []


def classify(log_text="", oom_killed=False, exit_code=None, kernel_lines=()):
    """-> {"category", "evidence"} for a failed server."""
    for category, pattern in PATTERNS:
        for line in reversed(log_text.splitlines()):
            if pattern.search(line):
                return {"category": category, "evidence": line.strip()[:300]}
    if oom_killed or exit_code == 137:
        return {"category": "host_oom", "evidence": f"container OOM-killed (exit {exit_code})"}
    for line in kernel_lines:
        m = KERNEL_OOM.search(line)
        if m and re.search(r"python|vllm|VLLM", m.group("proc") or m.group("task") or ""):
            return {"category": "host_oom", "evidence": line.strip()[:300]}
    tail = [l for l in log_text.splitlines() if l.strip()][-1:] or [f"exit code {exit_code}"]
    return {"category": "crashed", "evidence": tail[0].strip()[:300]}
