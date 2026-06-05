#!/usr/bin/env python3
"""
CLI wrapper that submits a kernel to the Modal B200 evaluator and writes
results.json in markdown format the agent can parse.

Usage:
    python run_eval.py submission.py -o results.json
    python run_eval.py submission.py -o results.json --mode test   # correctness only
"""

import argparse
import json
import sys
import threading

import modal

# Speed-of-light targets (µs) at 1.5 GHz on B200
SOL = {
    (7168, 16384, 1): 8.622,
    (4096, 7168,  8): 17.275,
    (7168, 2048,  4): 4.317,
}


def format_results_markdown(res: dict, mode: str = "leaderboard") -> str:
    gpu = res.get("gpu_name", "NVIDIA B200")
    torch_ver = res.get("torch_version", "unknown")
    plat = res.get("platform", "unknown")

    if res["success"]:
        status_line = "**B200 on Modal ✅ success**"
    else:
        status_line = "**B200 on Modal ❌ failure**"

    lines = [status_line]

    if res["success"]:
        lines.append("> ✅ Testing successful")
        lines.append("> ✅ Benchmarking successful")
    elif res.get("tests_passed", 0) == res.get("tests_total", 1):
        lines.append("> ✅ Testing successful")
        lines.append("> ❌ Benchmarking failed")
    else:
        lines.append("> ❌ Testing failed")

    lines += [
        "",
        "Running on:",
        f"* GPU: `{gpu}`",
        f"* Runtime: `CUDA`",
        f"* Platform: `{plat}`",
        f"* Torch: `{torch_ver}`",
        "",
    ]

    passed = res.get("tests_passed", 0)
    total = res.get("tests_total", 0)
    lines.append(f"## {'✅' if passed == total else '❌'} Passed {passed}/{total} tests:")
    lines.append("```")
    for td in res.get("test_details", []):
        icon = "✅" if td["passed"] else "❌"
        lines.append(f"{icon} m={td['m']} k={td['k']} l={td['l']}")
        if td.get("error"):
            lines.append(f"   ERROR: {td['error']}")
    lines.append("```")

    if res.get("error") and not res["success"]:
        lines += ["", "## Error:", "```", res["error"], "```"]

    bm = res.get("benchmark")
    if bm and mode == "leaderboard":
        geomean = bm["geomean_us"]
        lines += ["", "## Benchmarks:", "```", f"Geometric mean: ⏱ {geomean} µs", ""]
        for bd in res.get("benchmark_details", []):
            m, k, l = bd["m"], bd["k"], bd["l"]
            sol_us = SOL.get((m, k, l), None)
            sol_str = f" (SOL: {sol_us} µs, ratio: {bd['mean_us']/sol_us:.3f}x)" if sol_us else ""
            lines.append(
                f"  m={m} k={k} l={l}: ⏱ {bd['mean_us']} ± {bd['stderr_us']} µs"
                f"  ⚡ {bd['min_us']} µs  🐌 {bd['max_us']} µs{sol_str}"
            )
        lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an nvfp4_gemv kernel on Modal B200")
    parser.add_argument("submission", help="Path to submission.py")
    parser.add_argument("-o", "--output", default="results.json")
    parser.add_argument(
        "--mode",
        choices=["test", "leaderboard"],
        default="leaderboard",
        help="'test' for correctness only, 'leaderboard' for correctness + benchmark",
    )
    args = parser.parse_args()

    try:
        with open(args.submission) as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.submission} not found")
        sys.exit(1)

    # Prepend a cache-clear, CUDA context reset, and load_inline patch so Modal
    # containers never serve stale artifacts, don't carry dirty CUDA contexts
    # from a previous failed kernel, and every kernel gets a fresh isolated
    # build directory regardless of whether it sets build_directory itself.
    _CACHE_CLEAR = (
        "import shutil as _shutil, os as _os\n"
        "_shutil.rmtree(_os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)\n"
        "del _shutil, _os\n"
        "try:\n"
        "    import torch as _t\n"
        "    if _t.cuda.is_available():\n"
        "        _t.cuda.synchronize()\n"
        "    del _t\n"
        "except Exception:\n"
        "    try:\n"
        "        import ctypes as _ct, torch.cuda as _tc\n"
        "        for _lib in ('libcudart.so', 'libcudart.so.12', 'libcudart.so.11.0'):\n"
        "            try:\n"
        "                _ct.CDLL(_lib).cudaDeviceReset(); break\n"
        "            except OSError:\n"
        "                continue\n"
        "        _tc._initialized = False\n"
        "        del _ct, _tc\n"
        "    except Exception:\n"
        "        pass\n"
        "import torch.utils.cpp_extension as _cpp_ext\n"
        "_orig_load_inline = _cpp_ext.load_inline\n"
        "def _load_inline_patched(*_a, build_directory=None, **_kw):\n"
        "    if build_directory is None:\n"
        "        import tempfile as _t; build_directory = _t.mkdtemp(prefix='gemv_build_')\n"
        "    return _orig_load_inline(*_a, build_directory=build_directory, **_kw)\n"
        "_cpp_ext.load_inline = _load_inline_patched\n"
        "del _cpp_ext\n"
    )
    kernel_code = _CACHE_CLEAR + kernel_code

    print(f"Submitting {args.submission} to Modal B200 ({args.mode} mode)...")

    evaluate_kernel = modal.Function.from_name("cuda-kernel-eval-nvfp4-gemv", "evaluate_kernel")

    MODAL_TIMEOUT = 300
    result_holder = [None]
    error_holder = [None]

    def _call():
        try:
            if args.mode == "test":
                result_holder[0] = evaluate_kernel.remote(kernel_code, warmup_iters=0, eval_iters=0)
            else:
                result_holder[0] = evaluate_kernel.remote(kernel_code)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=MODAL_TIMEOUT)

    if t.is_alive():
        print(f"Error: Modal call timed out after {MODAL_TIMEOUT}s", file=sys.stderr)
        sys.exit(2)
    if error_holder[0] is not None:
        print(f"Error: Modal call failed: {error_holder[0]}", file=sys.stderr)
        sys.exit(1)

    raw = result_holder[0]
    res = json.loads(raw)
    md = format_results_markdown(res, mode=args.mode)

    with open(args.output, "w") as f:
        json.dump(md, f)

    print(md)
    sys.exit(0 if res["success"] else 1)


if __name__ == "__main__":
    main()
