"""
Deployable Modal A100 evaluator for the grayscale kernel task.

Deploy once:
    uv run modal deploy eval_modal_grayscale_kernel.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

BENCHMARK_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
TEST_SIZES = [512, 1024, 2048]

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("grayscale-kernel-eval")


@app.function(gpu="A100", image=image, timeout=300)
def evaluate_kernel(kernel_code: str, warmup_iters: int = 5, eval_iters: int = 20) -> str:
    import json as _json
    import math
    import traceback
    import types

    import torch

    def ref_kernel(data):
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device, dtype=data.dtype)
        return torch.sum(data * weights, dim=-1)

    def generate_input(size: int, seed: int):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        return torch.rand(size, size, 3, device="cuda", dtype=torch.float32, generator=gen).contiguous()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__

    module = types.ModuleType("submission")
    module.__dict__["__builtins__"] = __builtins__
    try:
        exec(kernel_code, module.__dict__)
    except Exception:
        return _json.dumps({
            "success": False,
            "error": f"Failed to load submission:\n{traceback.format_exc()}",
            "tests_passed": 0,
            "tests_total": len(TEST_SIZES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-a100",
        })

    custom_kernel = getattr(module, "custom_kernel", None)
    if custom_kernel is None:
        return _json.dumps({
            "success": False,
            "error": "submission.py does not define custom_kernel(data)",
            "tests_passed": 0,
            "tests_total": len(TEST_SIZES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-a100",
        })

    # Correctness tests
    test_details = []
    tests_passed = 0
    for size in TEST_SIZES:
        data = generate_input(size, seed=42)
        try:
            expected = ref_kernel(data)
            actual = custom_kernel(data)
            torch.cuda.synchronize()
            if actual.shape != expected.shape:
                test_details.append({
                    "size": size,
                    "passed": False,
                    "error": f"shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}",
                })
            elif not torch.allclose(actual, expected, rtol=1e-4, atol=1e-4):
                max_diff = (actual - expected).abs().max().item()
                test_details.append({
                    "size": size,
                    "passed": False,
                    "error": f"values differ, max abs diff: {max_diff:.6f}",
                })
            else:
                test_details.append({"size": size, "passed": True})
                tests_passed += 1
        except Exception:
            test_details.append({"size": size, "passed": False, "error": traceback.format_exc()[:600]})

    if tests_passed < len(TEST_SIZES):
        return _json.dumps({
            "success": False,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_SIZES),
            "test_details": test_details,
            "error": "Correctness check failed — see test_details",
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-a100",
        })

    if warmup_iters == 0 and eval_iters == 0:
        return _json.dumps({
            "success": True,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_SIZES),
            "test_details": test_details,
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-a100",
        })

    # Benchmarks
    benchmark_details = []
    times_us = []
    for size in BENCHMARK_SIZES:
        data = generate_input(size, seed=0)
        try:
            for _ in range(warmup_iters):
                custom_kernel(data)
            torch.cuda.synchronize()

            iter_times = []
            for _ in range(eval_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                custom_kernel(data)
                end.record()
                torch.cuda.synchronize()
                iter_times.append(start.elapsed_time(end) * 1000.0)  # ms → µs

            mean_us = sum(iter_times) / len(iter_times)
            min_us = min(iter_times)
            max_us = max(iter_times)
            stderr_us = (sum((t - mean_us) ** 2 for t in iter_times) / len(iter_times)) ** 0.5

            benchmark_details.append({
                "size": size,
                "mean_us": round(mean_us, 3),
                "min_us": round(min_us, 3),
                "max_us": round(max_us, 3),
                "stderr_us": round(stderr_us, 3),
            })
            times_us.append(mean_us)
        except Exception:
            return _json.dumps({
                "success": False,
                "tests_passed": tests_passed,
                "tests_total": len(TEST_SIZES),
                "test_details": test_details,
                "error": f"Benchmark failed at size={size}:\n{traceback.format_exc()}",
                "gpu_name": gpu_name,
                "torch_version": torch_ver,
                "platform": "modal-a100",
            })

    geomean = math.exp(sum(math.log(t) for t in times_us) / len(times_us))

    return _json.dumps({
        "success": True,
        "tests_passed": tests_passed,
        "tests_total": len(TEST_SIZES),
        "test_details": test_details,
        "benchmark": {"geomean_us": round(geomean, 3)},
        "benchmark_details": benchmark_details,
        "gpu_name": gpu_name,
        "torch_version": torch_ver,
        "platform": "modal-a100",
    })
