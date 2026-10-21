"""
live_metrics.py
---------------
Real, live readings from the computer running the dashboard (via psutil).

Used by the dashboard's "This computer (live)" data source:
    CPU %, Memory %, Disk % (system drive) and Network throughput (Mbps).

Response time and error rate come from an application, not the operating
system, so they are only available in the simulated data source.

Run directly to print a few readings:
    python live_metrics.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import psutil

LIVE_FEATURES = ["cpu_usage", "memory_usage", "network_mbps"]


def system_drive() -> str:
    """C:\\ on Windows (or whatever the system drive is), / elsewhere."""
    if os.name == "nt":
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"


class SystemSampler:
    """Takes one reading of this computer's resource usage per call."""

    def __init__(self, disk_path: str | None = None):
        self.disk_path = disk_path or system_drive()
        psutil.cpu_percent(interval=None)          # first call is always 0
        self._last_net = psutil.net_io_counters()
        self._last_time = time.monotonic()
        self._first = True

    def read(self) -> dict:
        reading = self._read()
        self._first = False
        return reading

    def _read(self) -> dict:
        now = time.monotonic()
        net = psutil.net_io_counters()
        seconds = max(now - self._last_time, 1e-3)
        moved = ((net.bytes_sent - self._last_net.bytes_sent)
                 + (net.bytes_recv - self._last_net.bytes_recv))
        self._last_net, self._last_time = net, now

        return {
            "timestamp": pd.Timestamp.now().floor("s"),
            # The first reading measures over 0.5 s; later ones measure the
            # time since the previous reading (the refresh interval)
            "cpu_usage": float(psutil.cpu_percent(interval=0.5 if self._first else None)),
            "memory_usage": float(psutil.virtual_memory().percent),
            "network_mbps": round(max(moved, 0) * 8 / seconds / 1e6, 3),
            "disk_usage": float(psutil.disk_usage(self.disk_path).percent),
        }


def top_processes(by: str = "memory_percent", n: int = 3) -> list[tuple[str, float]]:
    """The n processes using the most memory (percent of RAM)."""
    procs = []
    for p in psutil.process_iter(["name", by]):
        try:
            value = p.info.get(by)
            if value is not None:
                procs.append((p.info.get("name") or "?", float(value)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(procs, key=lambda x: x[1], reverse=True)[:n]


if __name__ == "__main__":
    sampler = SystemSampler()
    print(f"Disk measured on: {sampler.disk_path}")
    for _ in range(5):
        time.sleep(1)
        print(sampler.read())
    print("Top memory users:", top_processes())
