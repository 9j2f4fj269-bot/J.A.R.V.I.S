"""Node memory profiling: watch heap growth over time and find retained leaks.

Two signals, because leaks look different depending on how obvious they are:

1. A timeline. A preloaded monitor samples `process.memoryUsage()` while the script
   runs, so a steadily climbing heap shows up as a slope in MB/s rather than "it feels slow".
2. A snapshot diff. Two V8 heap snapshots (early and late) are aggregated by
   constructor name; whatever grows in retained bytes between them is the suspect.

The monitor is a `--require` preload, so the target script runs unmodified.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path

from .node_debug import _base_dir, _resolve_target, _truncate, module_type, node_binary, pick_extension

MONITOR = r"""
const fs = require('fs');
const v8 = require('v8');

const out = process.env.JARVIS_MEM_OUT;
const interval = Number(process.env.JARVIS_MEM_INTERVAL || 100);
const limit = Number(process.env.JARVIS_MEM_LIMIT || 0);
const snap1 = process.env.JARVIS_MEM_SNAP1 || '';
const snap2 = process.env.JARVIS_MEM_SNAP2 || '';
const snap2At = Number(process.env.JARVIS_MEM_SNAP2_AT || 0);

const started = Date.now();
const samples = [];
let finished = false;
let stoppedByLimit = false;
let snapshots = { first: false, last: false };

function sample() {
  const m = process.memoryUsage();
  samples.push({
    t: Date.now() - started,
    rss: m.rss,
    heapUsed: m.heapUsed,
    heapTotal: m.heapTotal,
    external: m.external,
    arrayBuffers: m.arrayBuffers || 0,
  });
}

function takeSnapshot(path) {
  if (!path) return false;
  try {
    if (global.gc) global.gc();
    v8.writeHeapSnapshot(path);
    return true;
  } catch (error) {
    return false;
  }
}

function writeLastSnapshot() {
  if (!snapshots.last) snapshots.last = takeSnapshot(snap2);
}

function finish() {
  if (finished) return;
  finished = true;
  clearInterval(timer);
  sample();
  writeLastSnapshot();
  const payload = {
    samples,
    duration: Date.now() - started,
    exitCode: process.exitCode === undefined ? 0 : process.exitCode,
    stoppedByLimit,
    snapshots,
  };
  try {
    fs.writeFileSync(out, JSON.stringify(payload));
  } catch (error) {
    // nothing useful to do if the report cannot be written
  }
}

sample();
const timer = setInterval(sample, Math.max(20, interval));
if (timer.unref) timer.unref();

// Baseline snapshot before the target runs: short scripts never reach a timer,
// and anything allocated later is exactly what the diff should count.
snapshots.first = takeSnapshot(snap1);

if (limit > 0) {
  const stopTimer = setTimeout(() => {
    stoppedByLimit = true;
    finish();            // snapshot while the target's objects are still reachable
    process.exitCode = 0;
    process.exit(0);
  }, limit);
  if (stopTimer.unref) stopTimer.unref();
}

// A mid-run snapshot catches a leak while the target still holds the objects;
// beforeExit and exit are the fallbacks for scripts that end early.
if (snap2At > 0) {
  const snapTimer = setTimeout(writeLastSnapshot, snap2At);
  if (snapTimer.unref) snapTimer.unref();
}

// beforeExit fires while the target's state is still alive; exit is the safety net.
process.on('beforeExit', finish);
process.on('exit', finish);
process.on('SIGTERM', () => {
  process.exitCode = 143;
  process.exit(0);
});
"""


def _mb(value: float | int | None) -> float:
    return round((value or 0) / (1024 * 1024), 2)


def _trend(times: list[float], values: list[float]) -> tuple[float, float, float]:
    """Least-squares slope (units/s), change, and ratio of flat-or-rising steps."""
    mean_t = sum(times) / len(times)
    mean_v = sum(values) / len(values)
    numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    denominator = sum((t - mean_t) ** 2 for t in times) or 1e-6
    rises = sum(1 for previous, current in zip(values, values[1:]) if current >= previous * 0.98)
    ratio = rises / max(1, len(values) - 1)
    return numerator / denominator, values[-1] - values[0], ratio


def analyse(samples: list[dict], leak_slope_mb_s: float = 0.35) -> dict:
    """Look for a climbing heap *and* climbing off-heap memory.

    Buffers live outside the V8 heap, so a heap-only view calls a Buffer leak "stable".
    """
    points = [s for s in samples if "t" in s and "heapUsed" in s]
    if len(points) < 4:
        heaps = [p["heapUsed"] / (1024 * 1024) for p in points]
        return {
            "verdict": "no trend",
            "reason": (
                f"script finished in about {round(points[-1]['t'] / 1000, 2)}s, too short for a timeline"
                if points
                else "the monitor never sampled: the event loop was probably blocked by synchronous work"
            ),
            "samples": len(points),
            "heap_first_mb": round(heaps[0], 2) if heaps else 0.0,
            "heap_last_mb": round(heaps[-1], 2) if heaps else 0.0,
            "growth_mb": round(heaps[-1] - heaps[0], 2) if len(heaps) > 1 else 0.0,
            "peak_rss_mb": _mb(max((p.get("rss") or 0) for p in points)) if points else 0.0,
        }

    warmup = max(1, len(points) // 5)
    body = points[warmup:]
    t0 = body[0]["t"]
    times = [(p["t"] - t0) / 1000.0 for p in body]

    heap = [p["heapUsed"] / (1024 * 1024) for p in body]
    offheap = [((p.get("external") or 0) + (p.get("arrayBuffers") or 0)) / (1024 * 1024) for p in body]
    total = [h + o for h, o in zip(heap, offheap)]

    heap_slope, heap_change, heap_ratio = _trend(times, heap)
    off_slope, off_change, off_ratio = _trend(times, offheap)
    total_slope, total_change, total_ratio = _trend(times, total)
    span = times[-1] or 1e-6

    heap_leaking = heap_slope >= leak_slope_mb_s and heap_ratio >= 0.7
    off_leaking = off_slope >= leak_slope_mb_s and off_ratio >= 0.7

    if heap_leaking and off_leaking:
        verdict = "leak suspected"
        area = "heap and buffers"
        slope, ratio = max(heap_slope, off_slope), max(heap_ratio, off_ratio)
        reason = (
            f"heap climbed {heap_slope:.2f} MB/s and off-heap memory {off_slope:.2f} MB/s"
        )
    elif heap_leaking:
        verdict, area, slope, ratio = "leak suspected", "heap", heap_slope, heap_ratio
        reason = f"heap climbed steadily at {heap_slope:.2f} MB/s ({heap_ratio:.0%} of samples flat or rising)"
    elif off_leaking:
        verdict, area, slope, ratio = "leak suspected", "buffers/external", off_slope, off_ratio
        reason = (
            f"off-heap memory climbed at {off_slope:.2f} MB/s ({off_ratio:.0%} of samples flat or rising); "
            "Buffers and ArrayBuffers are the usual cause"
        )
    elif abs(total_slope) < leak_slope_mb_s / 2 and abs(total_change) < leak_slope_mb_s * span:
        verdict, area, slope, ratio = "stable", "none", total_slope, total_ratio
        reason = f"memory stayed flat (total slope {total_slope:+.2f} MB/s) across {span:.1f}s"
    else:
        verdict, area, slope, ratio = "inconclusive", "unclear", total_slope, total_ratio
        reason = (
            f"memory moved at {total_slope:+.2f} MB/s with {total_ratio:.0%} rising samples; "
            "run longer or leak faster to separate noise from a trend"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "growth_area": area,
        "samples": len(points),
        "duration_s": round(span, 2),
        "heap_first_mb": round(heap[0], 2),
        "heap_last_mb": round(heap[-1], 2),
        "growth_mb": round(total_change, 2),
        "heap_growth_mb": round(heap_change, 2),
        "offheap_growth_mb": round(off_change, 2),
        "slope_mb_per_s": round(slope, 3),
        "heap_slope_mb_per_s": round(heap_slope, 3),
        "offheap_slope_mb_per_s": round(off_slope, 3),
        "rising_ratio": round(ratio, 2),
        "peak_heap_mb": round(max(heap), 2),
        "peak_offheap_mb": round(max(offheap), 2),
        "peak_rss_mb": _mb(max((p.get("rss") or 0) for p in points)),
        "timeline": [
            {
                "t_s": round((p["t"] - t0) / 1000.0, 1),
                "heap_mb": round(h / 1, 2),
                "offheap_mb": round(o, 2),
            }
            for p, h, o in zip(body[:: max(1, len(body) // 40)], heap, offheap)
        ],
    }


def load_snapshot(path: Path) -> dict:
    """Aggregate a .heapsnapshot by (type, constructor name): count and retained bytes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("snapshot", {}).get("meta", {})
    fields = meta.get("node_fields") or ["type", "name", "id", "self_size", "edge_count"]
    types = (meta.get("node_types") or [[]])[0] or []
    nodes = data.get("nodes") or []
    strings = data.get("strings") or []
    width = len(fields)
    index = {name: position for position, name in enumerate(fields)}

    def read(offset: int, field: str):
        position = index.get(field)
        return None if position is None else nodes[offset + position]

    aggregate: dict[tuple[str, str], dict] = {}
    total = 0
    for offset in range(0, len(nodes) - width + 1, width):
        type_index = read(offset, "type") or 0
        name_index = read(offset, "name") or 0
        size = read(offset, "self_size") or 0
        type_name = types[type_index] if 0 <= type_index < len(types) else "object"
        name = strings[name_index] if 0 <= name_index < len(strings) else ""
        name = " ".join(str(name).split())
        if len(name) > 80:
            name = name[:80] + "..."
        key = (type_name, name or "(anonymous)")
        entry = aggregate.setdefault(key, {"type": type_name, "name": name or "(anonymous)", "count": 0, "bytes": 0})
        entry["count"] += 1
        entry["bytes"] += size
        total += size
    return {"by_name": aggregate, "total_bytes": total, "nodes": len(nodes) // width}


def compare_snapshots(first: dict, last: dict, top: int = 8) -> dict:
    grows = []
    for key, entry in last["by_name"].items():
        before = first["by_name"].get(key, {"count": 0, "bytes": 0})
        count_delta = entry["count"] - before["count"]
        byte_delta = entry["bytes"] - before["bytes"]
        if count_delta > 0 or byte_delta > 0:
            grows.append(
                {
                    "type": entry["type"],
                    "name": entry["name"],
                    "count_delta": count_delta,
                    "retained_kb": round(byte_delta / 1024, 1),
                    "now_count": entry["count"],
                }
            )
    grows.sort(key=lambda row: row["retained_kb"], reverse=True)
    internals = {"code", "hidden", "synthetic"}
    interesting = [
        row for row in grows if row["type"] not in internals and not row["name"].startswith("system /")
    ]
    rows = interesting or grows
    return {
        "compared": True,
        "total_retained_kb_first": round(first["total_bytes"] / 1024, 1),
        "total_retained_kb_last": round(last["total_bytes"] / 1024, 1),
        "retained_growth_kb": round((last["total_bytes"] - first["total_bytes"]) / 1024, 1),
        "suspects": rows[: max(1, top)],
        "internals_hidden": max(0, len(grows) - len(rows)),
    }


def memory_profile(
    workspace: Path,
    code: str | None = None,
    path: str | None = None,
    cwd: str = ".",
    seconds: int = 6,
    interval_ms: int = 100,
    snapshots: bool = True,
    top: int = 8,
    timeout: int | None = None,
    node_flags: list[str] | None = None,
) -> dict:
    node = node_binary()
    if not node:
        return {"ok": False, "error": "node is not installed; install Node.js 18+ to profile JavaScript"}

    seconds = max(1, min(int(seconds), 120))
    base = _base_dir(workspace, cwd)
    created: Path | None = None
    try:
        if path:
            target = _resolve_target(base, path)
        elif code is not None:
            target = base / f".jarvis_mem_{secrets.token_hex(4)}{pick_extension(code)}"
            target.write_text(code, encoding="utf-8")
            created = target
        else:
            return {"ok": False, "error": "pass either code or path"}
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}

    token = secrets.token_hex(4)
    monitor = base / f".jarvis_mem_{token}.cjs"
    report_file = base / f".jarvis_mem_{token}.json"
    snap_a = base / f".jarvis_heap_{token}_first.heapsnapshot"
    snap_b = base / f".jarvis_heap_{token}_last.heapsnapshot"
    monitor.write_text(MONITOR, encoding="utf-8")

    limit_ms = seconds * 1000
    env = {
        **os.environ,
        "JARVIS_MEM_OUT": str(report_file),
        "JARVIS_MEM_INTERVAL": str(max(20, min(int(interval_ms), 5000))),
        "JARVIS_MEM_LIMIT": str(limit_ms),
        "JARVIS_MEM_SNAP1": str(snap_a) if snapshots else "",
        "JARVIS_MEM_SNAP2": str(snap_b) if snapshots else "",
        "JARVIS_MEM_SNAP2_AT": str(int(limit_ms * 0.65)) if snapshots else "0",
    }
    esm = module_type(target) == "esm"
    flags = ["--expose-gc", "--require", str(monitor), *(node_flags or [])]
    command = [node, *flags, str(target)]

    hard_timeout = timeout or seconds + 12
    stopped_early = False
    stdout = stderr = ""
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            command, cwd=str(base), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        try:
            stdout, stderr = proc.communicate(timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            stopped_early = True
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable loop
                proc.kill()
                stdout, stderr = proc.communicate()
        exit_code = proc.returncode
    except OSError as exc:  # pragma: no cover - node vanished mid-run
        for leftover in (monitor, report_file, snap_a, snap_b):
            leftover.unlink(missing_ok=True)
        if created is not None:
            created.unlink(missing_ok=True)
        return {"ok": False, "error": f"could not start node: {exc}"}

    report: dict = {}
    if report_file.is_file():
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            report = {}

    analysis = analyse(report.get("samples") or [])
    result: dict = {
        "ok": bool(report.get("samples")),
        "runtime": "node",
        "module_type": "esm" if esm else "cjs",
        "target": str(target),
        "exit_code": exit_code,
        "stopped_early": stopped_early,
        "stopped_by_limit": bool(report.get("stoppedByLimit")),
        "measured_s": round(time.monotonic() - started, 2),
        "stdout": _truncate(stdout),
        "stderr": _truncate(stderr),
        **analysis,
    }

    if snapshots:
        snapshot_info: dict = {"compared": False}
        first = last = None
        if snap_a.is_file():
            try:
                first = load_snapshot(snap_a)
            except (ValueError, OSError):
                first = None
        if snap_b.is_file():
            try:
                last = load_snapshot(snap_b)
            except (ValueError, OSError):
                last = None
        if first and last:
            snapshot_info = compare_snapshots(first, last, top)
        elif last:
            snapshot_info["note"] = "only the exit snapshot was captured (the baseline one failed; try a larger seconds=)"
        else:
            snapshot_info["note"] = "no heap snapshot captured; run with --expose-gc available and snapshots=true"
        result["snapshots"] = snapshot_info

    if analysis.get("growth_area") == "buffers/external":
        result["snapshot_caveat"] = (
            "heap snapshots under-count Buffer backing stores; trust count_delta here more than retained_kb"
        )

    if analysis["verdict"] == "leak suspected" and not result.get("snapshots", {}).get("suspects"):
        result["suggestion"] = (
            "the heap climbs but nothing obviously retained it: check timers, listeners and caches; "
            "run again with snapshots=true and a longer seconds= window"
        )

    for leftover in (monitor, report_file, snap_a, snap_b):
        leftover.unlink(missing_ok=True)
    if created is not None:
        created.unlink(missing_ok=True)
    return result


LEAK_HINTS = re.compile(r"(?:retain|leak|grow|cache|accumulat)", re.IGNORECASE)


def register(registry, config) -> None:
    ws = config.workspace

    @registry.tool(
        "node_memory",
        "Profile Node memory to catch leaks. Samples heap and off-heap usage while the script runs "
        "(slope in MB/s, peak, verdict) and, with snapshots=true, diffs V8 heap snapshots to name the "
        "objects whose retained memory keeps growing. Use it when code works but memory climbs, or "
        "after node_profile kind='heap' points at allocations without explaining what is retained.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Script to profile (relative to cwd)"},
                "code": {"type": "string", "description": "Snippet to run instead of a file"},
                "cwd": {"type": "string", "default": "."},
                "seconds": {"type": "integer", "default": 6, "description": "How long to watch (1-120)"},
                "interval_ms": {"type": "integer", "default": 100, "description": "Sampling interval"},
                "snapshots": {"type": "boolean", "default": True, "description": "Diff two heap snapshots"},
                "top": {"type": "integer", "default": 8, "description": "How many suspects to list"},
            },
        },
    )
    def node_memory_tool(
        path: str | None = None,
        code: str | None = None,
        cwd: str = ".",
        seconds: int = 6,
        interval_ms: int = 100,
        snapshots: bool = True,
        top: int = 8,
    ):
        return memory_profile(
            Path(ws),
            code=code,
            path=path,
            cwd=cwd,
            seconds=seconds,
            interval_ms=interval_ms,
            snapshots=snapshots,
            top=top,
        )
