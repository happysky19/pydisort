#!/usr/bin/env python3
"""Run the FP64 one-column-per-thread solver benchmark and write a bar chart.

``EXOFMS_SOURCE_ROOT`` is optional.  When it names an Exo-FMS checkout, the
script also compiles and measures its single-threaded Toon solvers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


DTYPE = torch.float64
PDISORT_STREAMS = (4, 8)


@dataclass
class Result:
    solver: str
    device: str
    profiles: int
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", type=int, nargs="+", default=(1000, 10000, 100000)
    )
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    return parser.parse_args()


def time_call(call, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / 1_000.0 / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - start) / repeats


def boundary_conditions(nprofile: int, device: torch.device) -> dict[str, torch.Tensor]:
    options = {"device": device, "dtype": DTYPE}
    return {
        "umu0": torch.full((nprofile,), 0.5, **options),
        "fbeam": torch.ones((1, nprofile), **options),
        "albedo": torch.full((1, nprofile), 0.1, **options),
    }


def benchmark_pydisort(
    nprofile: int,
    nlayer: int,
    nstr: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Result:
    try:
        from pydisort.pydisort import Disort, DisortOptions
    except ModuleNotFoundError:
        from pydisort import Disort, DisortOptions

    options = {"device": device, "dtype": DTYPE}
    disort_options = DisortOptions()
    disort_options.upward(True)
    disort_options.flags("onlyfl,lamber,quiet")
    disort_options.nwave(1)
    disort_options.ncol(nprofile)
    disort_options.ds().nlyr = nlayer
    disort_options.ds().nstr = nstr
    disort_options.ds().nmom = nstr
    disort_options.ds().nphase = nstr
    solver = Disort(disort_options)
    prop = torch.empty((1, nprofile, nlayer, 2 + nstr), **options)
    prop[..., 0] = 0.1
    prop[..., 1] = 0.5
    for moment in range(nstr):
        prop[..., 2 + moment] = 0.5 ** (moment + 1)
    bc = boundary_conditions(nprofile, device)
    seconds = time_call(lambda: solver(prop, **bc), device, warmup, repeats)
    return Result(
        f"pydisort DISORT {nstr}-stream", device.type.upper(), nprofile, seconds
    )


def benchmark_pyharp(
    nprofile: int, nlayer: int, device: torch.device, warmup: int, repeats: int
) -> Result:
    import pyharp

    options = {"device": device, "dtype": DTYPE}
    solver = pyharp.ToonMcKay89(pyharp.ToonMcKay89Options())
    prop = torch.empty((1, nprofile, nlayer, 3), **options)
    prop[..., 0] = 0.1
    prop[..., 1] = 0.5
    prop[..., 2] = 0.5
    bc = boundary_conditions(nprofile, device)
    seconds = time_call(lambda: solver(prop, **bc), device, warmup, repeats)
    return Result("pyharp Toon", device.type.upper(), nprofile, seconds)


def benchmark_exofms(
    root: Path, profiles: list[int], nlayer: int, warmup: int, repeats: int
) -> list[Result]:
    runner = Path(__file__).with_name("run_exofms_toon_benchmark.sh")
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "EXOFMS_LAYERS": str(nlayer),
        "EXOFMS_WARMUP": str(warmup),
        "EXOFMS_REPEATS": str(repeats),
    }
    completed = subprocess.run(
        [str(runner), str(root), *(str(value) for value in profiles)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = []
    current_profiles = None
    for line in completed.stdout.splitlines():
        fields = dict(item.split("=", 1) for item in line.split(",") if "=" in item)
        if "nprofile" in fields:
            current_profiles = int(fields["nprofile"])
        elif current_profiles and "exofms_sw_toon_seconds" in fields:
            result.append(
                Result(
                    "Exo-FMS SW Toon",
                    "CPU",
                    current_profiles,
                    float(fields["exofms_sw_toon_seconds"]),
                )
            )
        elif current_profiles and "exofms_lw_toon_5node_seconds" in fields:
            result.append(
                Result(
                    "Exo-FMS LW Toon",
                    "CPU",
                    current_profiles,
                    float(fields["exofms_lw_toon_5node_seconds"]),
                )
            )
    if len(result) != 2 * len(profiles):
        raise RuntimeError(
            f"could not parse Exo-FMS benchmark output:\n{completed.stdout}"
        )
    return result


def plot(results: list[Result], output: Path) -> None:
    import matplotlib.pyplot as plt

    series = list(dict.fromkeys((result.solver, result.device) for result in results))
    solver_colors = {
        name: index
        for index, name in enumerate(dict.fromkeys(key[0] for key in series))
    }
    profiles = sorted({result.profiles for result in results})
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(
        1, len(profiles), figsize=(4 * len(profiles), 4), sharey=True
    )
    axes = (axes,) if len(profiles) == 1 else axes
    for axis, nprofile in zip(axes, profiles):
        values = [
            next(
                (
                    1e6 * item.seconds / item.profiles
                    for item in results
                    if (item.solver, item.device) == key and item.profiles == nprofile
                ),
                None,
            )
            for key in series
        ]
        positions = [index for index, value in enumerate(values) if value is not None]
        bars = axis.bar(
            positions,
            [values[index] for index in positions],
            color=[
                colors[solver_colors[series[index][0]] % len(colors)]
                for index in positions
            ],
            hatch=["//" if series[index][1] == "CPU" else "" for index in positions],
        )
        axis.bar_label(bars, fmt="%.2g", padding=2, fontsize=8)
        axis.set_title(f"{nprofile:,} profiles")
        axis.set_xticks(
            positions,
            [f"{series[index][0]}\n{series[index][1]}" for index in positions],
            rotation=35,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("time per profile (µs, log scale)")
    axes[0].set_yscale("log")
    figure.tight_layout()
    figure.savefig(output / "fp64_solver_benchmark.png", dpi=200)


def main() -> None:
    args = parse_args()
    if min(*args.profiles, args.layers, args.repeats) < 1 or args.warmup < 0:
        raise ValueError("profiles, layers, and repeats must be positive")

    torch.set_num_threads(1)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    results = []
    for nprofile in args.profiles:
        for device in devices:
            for nstr in PDISORT_STREAMS:
                results.append(
                    benchmark_pydisort(
                        nprofile,
                        args.layers,
                        nstr,
                        device,
                        args.warmup,
                        args.repeats,
                    )
                )

    try:
        import pyharp  # noqa: F401
    except ImportError:
        print("pyharp not found; skipping pyharp Toon")
    else:
        for nprofile in args.profiles:
            for device in devices:
                results.append(
                    benchmark_pyharp(
                        nprofile, args.layers, device, args.warmup, args.repeats
                    )
                )

    exofms_root = os.environ.get("EXOFMS_SOURCE_ROOT")
    exofms_files = ("src/WENO4_mod.f90", "src/sw_Toon_mod.f90", "src/lw_Toon_mod.f90")
    if (
        exofms_root
        and shutil.which(os.environ.get("FC", "gfortran"))
        and all((Path(exofms_root) / name).is_file() for name in exofms_files)
    ):
        results.extend(
            benchmark_exofms(
                Path(exofms_root), args.profiles, args.layers, args.warmup, args.repeats
            )
        )
    else:
        print(
            "Exo-FMS Toon sources or Fortran compiler not found; skipping Exo-FMS Toon"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fp64_solver_benchmark.json").open("w") as stream:
        json.dump([asdict(result) for result in results], stream, indent=2)
    with (args.output / "fp64_solver_benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=Result.__annotations__)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    plot(results, args.output)
    for result in results:
        print(
            f"{result.solver:24s} {result.device:4s} {result.profiles:7d} {1e6 * result.seconds / result.profiles:10.3f} µs/profile"
        )


if __name__ == "__main__":
    main()
