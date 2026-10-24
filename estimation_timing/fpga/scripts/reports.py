#!/usr/bin/env python3
"""Copy synthesis reports and extract only the kernel latency in seconds."""
import csv
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TOPS = {"local": "local_spin_flip_operation", "uniform": "uniform_move_operation"}


def collect(root):
    config = root / "build/run_config.json"
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "summary.csv").unlink(missing_ok=True)
    (reports / "fits.json").unlink(missing_ok=True)  # Remove results from the previous package.
    for n in json.loads(config.read_text())["synthesis_sizes"]:
        for move, top in TOPS.items():
            project = root / "build" / f"synth_{move}_N{n}"
            matches = list(project.rglob(f"{top}_csynth.xml"))
            if len(matches) != 1:
                raise ValueError(f"Expected one synthesis report in {project}")
            destination = reports / f"{move}_N{n}"
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir()
            for path in matches[0].parent.iterdir():
                if path.suffix in (".xml", ".rpt"):
                    shutil.copy2(path, destination / path.name)
            print(f"Copied {move}, n={n}")
    shutil.copy2(config, reports / "run_config.json")


def parse(root):
    reports = root / "reports"
    rows = []
    for n in json.loads((reports / "run_config.json").read_text())["synthesis_sizes"]:
        for move, top in TOPS.items():
            xml = ET.parse(reports / f"{move}_N{n}" / f"{top}_csynth.xml")
            # Vitis already gives the absolute latency, e.g. "0.273 us".
            value, unit = xml.findtext(".//Worst-caseRealTimeLatency").split()
            seconds = float(value) * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[unit]
            rows.append((move, n, seconds))
    with (reports / "summary.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("move", "n", "latency_s"))
        writer.writerows(rows)
    print("move       n    latency_s")
    for move, n, seconds in rows:
        print(f"{move:8s} {n:3d} {seconds:12.6g}")
    return rows


def clean(root):
    # Vitis creates journals internally; retain reports, not logs.
    for path in (root / "build").rglob("*"):
        if path.is_file() and path.suffix.lower() in (".log", ".jou", ".str"):
            path.unlink()
    for path in sorted((root / "build").rglob("logs"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
    for pattern in ("vitis*.log", "vitis*.jou", "vitis*.str"):
        for path in root.glob(pattern):
            path.unlink()


if __name__ == "__main__":
    {"collect": collect, "parse": parse, "clean": clean}[sys.argv[1]](ROOT)
