#!/usr/bin/env python3
"""Compare one-thread CUDA FP64 flux solves for pydisort and Toon.

The timed region contains solver forward calls only: solver construction and
input preparation are deliberately excluded.  ``nwave=1`` and ``ncol=1``
ensure that each solver launch processes exactly one radiative-transfer
column.  Both solvers receive float64 CUDA tensors.

Example:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_cuda_fp64.py \
        --nlyr 30 --nstr 8 --warmup 10 --repeats 100
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class Timing:
    name: str
    wall_ms: float
    device_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nwave", type=int, default=1)
    parser.add_argument("--ncol", type=int, default=1)
    parser.add_argument("--nlyr", type=int, default=30)
    parser.add_argument("--nstr", type=int, default=8, choices=(4, 8, 16, 32))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--ssalb", type=float, default=0.5)
    parser.add_argument("--asymmetry", type=float, default=0.5)
    parser.add_argument(
        "--check-cpu",
        action="store_true",
        help="compare one untimed CPU DISORT solve with the CUDA result",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser.parse_args()


def time_solver(name: str, solve, warmup: int, repeats: int) -> Timing:
    for _ in range(warmup):
        solve()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    for _ in range(repeats):
        solve()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1_000.0 / repeats
    device_ms = start.elapsed_time(end) / repeats
    return Timing(name=name, wall_ms=wall_ms, device_ms=device_ms)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.nwave, args.ncol, args.nlyr, args.repeats) < 1 or args.warmup < 0:
        raise ValueError(
            "nwave, ncol, nlyr, and repeats must be >= 1; warmup must be >= 0"
        )

    # Load pydisort before pyharp.  Both projects expose a
    # ``libdisort_release.so``; importing pyharp first can make the dynamic
    # linker reuse pyharp's copy for this extension instead of this build's
    # sibling library.
    try:
        from pydisort.pydisort import Disort, DisortOptions
    except ModuleNotFoundError:
        # ``setup.py build_ext --inplace`` exposes the extension as the
        # top-level module, whereas an installed wheel exposes a package.
        from pydisort import Disort, DisortOptions
    import pyharp

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.float64
    nwave = args.nwave
    ncol = args.ncol
    options = {"device": device, "dtype": dtype}

    toon_options = pyharp.ToonMcKay89Options()
    toon = pyharp.ToonMcKay89(toon_options)
    toon_prop = torch.empty((nwave, ncol, args.nlyr, 3), **options)
    toon_prop[..., 0] = args.tau
    toon_prop[..., 1] = args.ssalb
    toon_prop[..., 2] = args.asymmetry

    disort_options = DisortOptions()
    disort_options.upward(True)
    disort_options.flags("onlyfl,lamber,quiet")
    disort_options.nwave(nwave)
    disort_options.ncol(ncol)
    disort_options.ds().nlyr = args.nlyr
    disort_options.ds().nstr = args.nstr
    disort_options.ds().nmom = args.nstr
    disort_options.ds().nphase = args.nstr
    disort = Disort(disort_options)
    disort_prop = torch.empty((nwave, ncol, args.nlyr, 2 + args.nstr), **options)
    disort_prop[..., 0] = args.tau
    disort_prop[..., 1] = args.ssalb
    for moment in range(args.nstr):
        disort_prop[..., 2 + moment] = args.asymmetry ** (moment + 1)

    bc = {
        "umu0": torch.full((ncol,), 0.5, **options),
        "fbeam": torch.ones((nwave, ncol), **options),
        "albedo": torch.full((nwave, ncol), 0.1, **options),
    }

    toon_out = toon(toon_prop, **bc)
    disort_out = disort(disort_prop, **bc)
    torch.cuda.synchronize()
    scale = torch.maximum(disort_out.abs().max(), torch.tensor(1.0, **options))
    relative_linf = float((toon_out - disort_out).abs().max() / scale)
    cpu_gpu_max_abs = None
    if args.check_cpu:
        cpu_bc = {name: value.cpu() for name, value in bc.items()}
        cpu_out = disort(disort_prop.cpu(), **cpu_bc)
        cpu_gpu_max_abs = float((cpu_out - disort_out.cpu()).abs().max())

    timings = [
        time_solver("toon", lambda: toon(toon_prop, **bc), args.warmup, args.repeats),
        time_solver("pydisort", lambda: disort(disort_prop, **bc), args.warmup, args.repeats),
    ]
    result = {
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype).removeprefix("torch."),
        "nwave": nwave,
        "ncol": ncol,
        "nlyr": args.nlyr,
        "nstr": args.nstr,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "work_items": nwave * ncol,
        "relative_linf_flux_difference": relative_linf,
        "cpu_gpu_max_abs_flux_difference": cpu_gpu_max_abs,
        "timings": [asdict(timing) for timing in timings],
        "pydisort_to_toon_device_ratio": timings[1].device_ms / timings[0].device_ms,
        "pydisort_to_toon_wall_ratio": timings[1].wall_ms / timings[0].wall_ms,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"GPU: {result['device']}")
    print("Each work item is one (wavelength, column) solver thread; dtype=float64")
    print(
        f"nwave={nwave}, ncol={ncol}, nlyr={args.nlyr}, nstr={args.nstr}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )
    print("solver     wall ms/solve    device ms/solve")
    for timing in timings:
        print(f"{timing.name:9s}{timing.wall_ms:14.3f}{timing.device_ms:19.3f}")
    print(f"pydisort / Toon device ratio: {result['pydisort_to_toon_device_ratio']:.2f}x")
    print(f"pydisort / Toon wall ratio:   {result['pydisort_to_toon_wall_ratio']:.2f}x")
    print(f"Flux relative L-infinity difference: {relative_linf:.3e}")


if __name__ == "__main__":
    main()
