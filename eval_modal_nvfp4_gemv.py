"""
Modal app for evaluating nvfp4 GEMV kernels on a B200 GPU.

Input tuple to custom_kernel:
  (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)

  a:             M x K//2 x L  float4_e2m1fn_x2
  b:             128 x K//2 x L  float4_e2m1fn_x2  (N padded to 128)
  sfa:           M x (K//16) x L  float8_e4m3fn
  sfb:           128 x (K//16) x L  float8_e4m3fn
  sfa_permuted:  (32, 4, ceil(M/128), 4, ceil(K//16,4), L)  float8_e4m3fn  CuBLAS layout
  sfb_permuted:  same layout for b
  c:             M x 1 x L  float16  (output buffer)

Returns a JSON string to avoid pickle/torch issues on the client.
"""

import modal

app = modal.App("cuda-kernel-eval-nvfp4-gemv")

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ninja-build")
    .pip_install(
        "torch", "numpy", "triton", "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "10.0a",
    })
)


@app.function(gpu="B200", image=gpu_image, timeout=600)
def evaluate_kernel(
    kernel_code: str,
    warmup_iters: int = 10,
    eval_iters: int = 10,
) -> str:
    import torch
    import platform
    import math
    import json
    import time
    import traceback
    import importlib.util
    import sys
    import types
    import tempfile
    import os

    # ── reference implementation ────────────────────────────────────────────

    sf_vec_size = 16

    def ceil_div(a, b):
        return (a + b - 1) // b

    # fp4 e2m1 lookup table: 4-bit index -> float32 value
    # bit3=sign, bits[2:1]=exponent, bit0=mantissa
    _FP4_LUT_VALUES = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,     # positive (sign=0)
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative (sign=1)
    ]

    def ref_kernel(data):
        # data: (a_packed, b_packed, sfa, sfb, sfa_permuted, sfb_permuted, c)
        # a_packed: [m, k//2, l]  fp4x2 (2 fp4 values per byte, lo nibble = lower k)
        # b_packed: [128, k//2, l] fp4x2 (only row 0 is the actual b vector)
        # sfa:      [m, k//16, l] fp8   (one scale per 16 k-elements)
        # sfb:      [128, k//16, l] fp8  (row 0 is the b scale vector)
        # c:        [m, 1, l] fp16       (output, written in place)
        a_packed, b_packed, sfa, sfb, _, _, c = data
        a_u8 = a_packed.view(torch.uint8)  # [m, k//2, l]
        b_u8 = b_packed.view(torch.uint8)  # [128, k//2, l]
        m, k_half, l = a_u8.shape
        k = k_half * 2

        lut = torch.tensor(_FP4_LUT_VALUES, dtype=torch.float32, device=a_u8.device)

        # Unpack fp4: lo nibble = even k-index, hi nibble = odd k-index
        a_f32 = torch.empty(m, k, l, dtype=torch.float32, device=a_u8.device)
        a_f32[:, 0::2, :] = lut[(a_u8 & 0xF).long()]
        a_f32[:, 1::2, :] = lut[((a_u8 >> 4) & 0xF).long()]

        b_f32 = torch.empty(1, k, l, dtype=torch.float32, device=b_u8.device)
        b_f32[:, 0::2, :] = lut[(b_u8[0:1] & 0xF).long()]
        b_f32[:, 1::2, :] = lut[((b_u8[0:1] >> 4) & 0xF).long()]

        # fp8 scale factors -> float32, broadcast over the 16 k-elements each covers
        sfa_f32 = sfa.float().repeat_interleave(16, dim=1)      # [m, k, l]
        sfb_f32 = sfb[0:1].float().repeat_interleave(16, dim=1)  # [1, k, l]

        # GEMV: c[m, 0, l] = sum_k( a[m,k,l]*sfa[m,k//16,l] * b[0,k,l]*sfb[0,k//16,l] )
        c_f32 = (a_f32 * sfa_f32 * b_f32 * sfb_f32).sum(dim=1, keepdim=True)  # [m, 1, l]
        c.copy_(c_f32.to(c.dtype))
        return c

    def generate_input(m, k, l, seed):
        torch.manual_seed(seed)
        n_padded_128 = 128

        a_ref = torch.randint(0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        b_ref = torch.randint(0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        if hasattr(torch, "float4_e2m1fn_x2"):
            a_ref = a_ref.view(torch.float4_e2m1fn_x2)
            b_ref = b_ref.view(torch.float4_e2m1fn_x2)

        c_ref = torch.randn((l, m, 1), dtype=torch.float16, device="cuda").permute(1, 2, 0)

        def create_scale_factor_tensors(l, mn, sf_k):
            ref_shape = (l, mn, sf_k)
            ref_f8_random_int = torch.randint(0, 3, ref_shape, dtype=torch.int8, device="cuda")
            ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
            ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(1, 2, 0)

            atom_m = (32, 4)
            atom_k = 4
            mma_shape = (
                l,
                ceil_div(mn, atom_m[0] * atom_m[1]),
                ceil_div(sf_k, atom_k),
                atom_m[0],
                atom_m[1],
                atom_k,
            )

            rand_int_tensor = torch.randint(0, 3, mma_shape, dtype=torch.int8, device="cuda")
            reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
            reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(3, 4, 1, 5, 2, 0)

            i_idx = torch.arange(mn, device="cuda")
            j_idx = torch.arange(sf_k, device="cuda")
            b_idx = torch.arange(l, device="cuda")
            i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

            mm = i_grid // (atom_m[0] * atom_m[1])
            mm32 = i_grid % atom_m[0]
            mm4 = (i_grid % 128) // atom_m[0]
            kk = j_grid // atom_k
            kk4 = j_grid % atom_k

            reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

            return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

        sf_k = ceil_div(k, sf_vec_size)
        sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
        sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

        sfa_ref = sfa_ref_cpu.to("cuda")
        sfb_ref = sfb_ref_cpu.to("cuda")

        return (a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref)

    # ── test / benchmark cases ──────────────────────────────────────────────
    # All m must be multiples of 128 (for to_blocked), k multiples of 64 (spec).

    CORRECTNESS_CASES = [
        {"m": 128, "k": 2048,  "l": 1, "seed": 42},   # 1024 special path
        {"m": 256, "k": 7168,  "l": 1, "seed": 100},  # 3584 special path
        {"m": 128, "k": 16384, "l": 1, "seed": 200},  # 8192 special path
    ]

    BENCHMARK_CASES = [
        {"m": 7168, "k": 16384, "l": 1, "seed": 1001},
        {"m": 4096, "k": 7168,  "l": 8, "seed": 1002},
        {"m": 7168, "k": 2048,  "l": 4, "seed": 1003},
    ]

    # ── kernel loader ───────────────────────────────────────────────────────

    def load_kernel(code):
        class DeterministicContext:
            def __enter__(self): return self
            def __exit__(self, *a): pass

        task_mod = types.ModuleType("task")
        task_mod.input_t = tuple
        task_mod.output_t = torch.Tensor
        sys.modules["task"] = task_mod

        utils_mod = types.ModuleType("utils")
        utils_mod.DeterministicContext = DeterministicContext
        def make_match_reference(ref_fn, rtol=1e-3, atol=1e-3):
            def checker(output, expected):
                return torch.allclose(output.float(), expected.float(), rtol=rtol, atol=atol)
            return checker
        utils_mod.make_match_reference = make_match_reference
        sys.modules["utils"] = utils_mod

        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "submission.py")
        with open(path, "w") as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location("submission", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = mod

        # Capture subprocess output (nvcc/ninja) at the fd level so build
        # errors land in the returned JSON rather than disappearing into
        # container stdout.
        build_log = ""
        load_exc = None
        log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        log_fd = log_file.fileno()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            load_exc = exc
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            log_file.seek(0)
            build_log = log_file.read().decode("utf-8", errors="replace")
            log_file.close()
            os.unlink(log_file.name)

        if load_exc is not None:
            import io as _io
            tb_buf = _io.StringIO()
            traceback.print_exception(type(load_exc), load_exc, load_exc.__traceback__, file=tb_buf)
            raise RuntimeError(
                f"Build log (last 4000 chars):\n{build_log[-4000:] or '(empty — ninja may have crashed before producing output)'}\n\n"
                f"Traceback:\n{tb_buf.getvalue()}"
            )

        if not hasattr(mod, "custom_kernel"):
            raise AttributeError("submission.py must define a `custom_kernel` function")
        return mod.custom_kernel

    # ── main eval logic ─────────────────────────────────────────────────────

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    result = {
        "success": False,
        "tests_passed": 0,
        "tests_total": len(CORRECTNESS_CASES),
        "test_details": [],
        "benchmark": None,
        "benchmark_details": [],
        "gpu_name": gpu_name,
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "error": None,
    }

    try:
        custom_kernel = load_kernel(kernel_code)
    except Exception as _load_exc:
        result["error"] = f"Failed to load kernel:\n{traceback.format_exc()}"
        return json.dumps(result)

    # Flush any deferred CUDA errors from a warm container's previous run.
    # A buggy kernel raises cudaErrorIllegalAddress asynchronously; on warm
    # Modal containers the error surfaces at the first CUDA sync of the *next*
    # evaluation (typically inside torch.manual_seed), making it look like the
    # current kernel crashed.  We synchronize eagerly here and, if the context
    # is dirty, reset the device so the test loop runs on a clean context.
    try:
        torch.cuda.synchronize()
    except Exception as _ctx_err:
        _reset_ok = False
        try:
            import ctypes as _ct
            for _lib in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
                try:
                    _ct.CDLL(_lib).cudaDeviceReset()
                    torch.cuda._initialized = False  # force PyTorch to reinit on next op
                    _reset_ok = True
                    break
                except OSError:
                    continue
        except Exception:
            pass
        if not _reset_ok:
            result["error"] = (
                "CUDA context contamination: a previous kernel on this warm Modal "
                "container left a deferred illegal-memory-access error that surfaced "
                "here.  This failure is NOT from the submitted kernel.  "
                f"Original error: {_ctx_err}"
            )
            return json.dumps(result)

    for case in CORRECTNESS_CASES:
        m, k, l = case["m"], case["k"], case["l"]
        detail = {"m": m, "k": k, "l": l, "passed": False, "error": None}
        try:
            # Run ref and custom on separate identical-seed inputs so neither
            # sees the other's in-place writes to c_ref.
            ref_out = ref_kernel(generate_input(m, k, l, case["seed"]))
            custom_out = custom_kernel(generate_input(m, k, l, case["seed"]))
            torch.cuda.synchronize()

            if torch.allclose(ref_out.float(), custom_out.float(), atol=1.0, rtol=1e-2):
                detail["passed"] = True
                result["tests_passed"] += 1
            else:
                diff = (ref_out.float() - custom_out.float()).abs().max().item()
                detail["error"] = f"Mismatch: max_diff={diff:.6f}"
        except Exception:
            detail["error"] = traceback.format_exc()

        result["test_details"].append(detail)

    if result["tests_passed"] < result["tests_total"]:
        result["error"] = "Testing failed"
        return json.dumps(result)

    # Anti-exploit: different inputs must produce different outputs.
    try:
        chk_m, chk_k, chk_l = 128, 2048, 1
        out_x = custom_kernel(generate_input(chk_m, chk_k, chk_l, seed=77001))
        out_y = custom_kernel(generate_input(chk_m, chk_k, chk_l, seed=77002))
        torch.cuda.synchronize()
        t_x = out_x[0] if isinstance(out_x, (list, tuple)) else out_x
        t_y = out_y[0] if isinstance(out_y, (list, tuple)) else out_y
        if torch.allclose(t_x.float(), t_y.float(), atol=0.5):
            result["error"] = "Exploit detected: kernel returns identical outputs for different inputs"
            return json.dumps(result)
    except Exception:
        pass

    if eval_iters <= 0:
        result["success"] = True
        return json.dumps(result)

    try:
        case_times = []

        # Timed GPU warmup: run the first benchmark shape for at least 2 seconds
        # so the B200 frequency ramps to boost before any measurements begin.
        _wm, _wk, _wl = BENCHMARK_CASES[0]["m"], BENCHMARK_CASES[0]["k"], BENCHMARK_CASES[0]["l"]
        _warmup_start = time.time()
        _wi = 0
        while time.time() - _warmup_start < 2.0:
            _ = custom_kernel(generate_input(_wm, _wk, _wl, 99999 + _wi))
            torch.cuda.synchronize()
            _wi += 1

        for bi, bcase in enumerate(BENCHMARK_CASES):
            m, k, l = bcase["m"], bcase["k"], bcase["l"]

            for i in range(warmup_iters):
                data = generate_input(m, k, l, bcase["seed"] + i)
                _ = custom_kernel(data)
                torch.cuda.synchronize()

            timings_us = []
            for i in range(eval_iters):
                data = generate_input(m, k, l, bcase["seed"] + warmup_iters + i)
                torch.cuda.synchronize()

                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                start_evt.record()
                _ = custom_kernel(data)
                end_evt.record()
                torch.cuda.synchronize()

                timings_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

            mean_us = sum(timings_us) / len(timings_us)
            variance = sum((t - mean_us) ** 2 for t in timings_us) / len(timings_us)
            std_us = math.sqrt(variance)
            stderr_us = std_us / math.sqrt(len(timings_us))

            result["benchmark_details"].append({
                "case_idx": bi,
                "m": m,
                "k": k,
                "l": l,
                "mean_us":   round(mean_us, 1),
                "std_us":    round(std_us, 2),
                "stderr_us": round(stderr_us, 1),
                "min_us":    round(min(timings_us), 1),
                "max_us":    round(max(timings_us), 1),
            })
            case_times.append(mean_us)

        geomean = math.exp(sum(math.log(t) for t in case_times) / len(case_times))
        result["benchmark"] = {
            "geomean_us": round(geomean, 1),
            "case_times_us": [round(t, 1) for t in case_times],
            "num_cases": len(case_times),
        }
        result["success"] = True

    except Exception:
        result["error"] = f"Benchmark failed:\n{traceback.format_exc()}"

    return json.dumps(result)
