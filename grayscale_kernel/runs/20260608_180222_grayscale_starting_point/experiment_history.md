# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-08 18:02:25 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark before any agent changes

**Result:** 90.13 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #2 — 2026-06-08 18:03:23 UTC ❌ DISCARD

**Hypothesis:** Increase pixels-per-thread from 4 to 8 (two float4 outputs per thread) with cache-streaming stores, 256 threads/block. Hypothesis: more work per thread reduces kernel launch overhead and improves ILP on H100.

**Result:** 90.15 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #3 — 2026-06-08 18:04:12 UTC ❌ DISCARD

**Hypothesis:** Use cache-streaming (ld.global.cs) for both reads AND writes, 4 pixels/thread, 256 threads/block. Hypothesis: large sequential reads bypass L1 allocation reducing cache pressure and improving HBM bandwidth on H100.

**Result:** 96.83 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #4 — 2026-06-08 18:05:07 UTC ✅ KEEP

**Hypothesis:** Switch to pure PyTorch mv (cuBLAS GEMV): reshape rgb to (H*W, 3), multiply by [0.2989, 0.5870, 0.1140] weight vector via mv(), copy result into pre-allocated output. Hypothesis: cuBLAS GEMV is heavily optimized for H100 bandwidth and eliminates custom kernel overhead.

**Result:** 89.24 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #5 — 2026-06-08 18:05:44 UTC ❌ DISCARD

**Hypothesis:** Use torch.mv with out= parameter pointing to flattened output buffer to eliminate intermediate allocation and copy_() call. Hypothesis: direct write avoids one memory pass.

**Result:** 90.00 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #6 — 2026-06-08 18:06:21 UTC ❌ DISCARD

**Hypothesis:** Use torch.einsum('hwc,c->hw', out=output) to fuse weighted channel sum in a single pass, avoiding reshape+copy overhead. Hypothesis: einsum dispatches a more efficient reduction kernel than GEMV+copy.

**Result:** 90.01 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #7 — 2026-06-08 18:06:59 UTC ❌ DISCARD

**Hypothesis:** Use torch.mm with weight as (3,1) matrix — triggers GEMM path (N×3)@(3×1) which might use Tensor Cores on H100. Hypothesis: GEMM path faster than GEMV for this workload.

**Result:** 90.72 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #8 — 2026-06-08 18:07:40 UTC ❌ DISCARD

**Hypothesis:** Triton kernel with BLOCK_SIZE=1024, scalar loads at stride-3 offsets (base+0, base+1, base+2), compile-time constant weights, direct output write. Hypothesis: large blocks + constant folding beats mv()+copy_().

**Result:** 102.76 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #9 — 2026-06-08 18:08:12 UTC ❌ DISCARD

**Hypothesis:** Wrap mv()+copy_() in torch.compile(mode='max-autotune') to fuse GEMV+copy into a single kernel via inductor, eliminating intermediate allocation and second memory pass.

**Result:** 90.67 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #10 — 2026-06-08 18:08:50 UTC ❌ DISCARD

**Hypothesis:** Use rgb.mul(weights).sum(-1) — fused elementwise multiply + channel reduction — to avoid cuBLAS GEMV launch overhead. Hypothesis: single pointwise kernel faster than GEMV+copy for this workload.

**Result:** 90.41 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #11 — 2026-06-08 18:09:43 UTC ✅ KEEP

**Hypothesis:** Custom CUDA kernel: 512 threads/block (up from 256), __ldg reads + st.global.wt (write-through) stores, 4 pixels/thread with float4 vectorization, writes directly to output buffer (no copy_). Hypothesis: larger blocks better saturate H100's 132 SMs and direct write eliminates copy overhead.

**Result:** 86.75 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #12 — 2026-06-08 18:10:15 UTC ❌ DISCARD

**Hypothesis:** Increase block size from 512 to 1024 threads/block to further saturate H100 SMs. All else same: __ldg reads, st.global.wt writes, 4 pixels/thread, float4 vectorization.

**Result:** 87.43 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #13 — 2026-06-08 18:11:05 UTC ❌ DISCARD

**Hypothesis:** 8 pixels/thread (6 float4 reads, 2 float4 writes) with 512 threads/block. More work per thread to hide memory latency and reduce grid overhead vs best 4-pixel/thread config.

**Result:** 86.90 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #14 — 2026-06-08 18:11:54 UTC ❌ DISCARD

**Hypothesis:** Switch stores from st.global.wt (write-through) to st.global.wb (write-back, allocates in L2 cache). 512 threads/block, 4 pixels/thread, __ldg reads. Hypothesis: L2 write-back allocation might improve store throughput on H100.

**Result:** 100.90 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #15 — 2026-06-08 18:12:39 UTC ❌ DISCARD

**Hypothesis:** Use explicit ld.global.nc.v4 (non-coherent read-only/texture cache) instead of __ldg for reads, with st.global.wt stores, 512 threads/block, 4 pixels/thread. Hypothesis: explicit nc path routes through texture hw with higher bandwidth than __ldg on H100.

**Result:** 100.93 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #16 — 2026-06-08 18:13:29 UTC ❌ DISCARD

**Hypothesis:** Grid-stride loop with fixed 1056 blocks (132 SMs × 8 blocks/SM), 512 threads/block, __ldg reads, st.global.wt stores, 4 pixels/thread. Hypothesis: exactly one wave of blocks keeps all SMs busy with minimal tail overhead.

**Result:** 86.78 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #17 — 2026-06-08 18:15:00 UTC ✅ KEEP

**Hypothesis:** Add CUDA graph capture/replay around the best kernel (512 threads/block, __ldg reads, st.global.wt stores, 4 pixels/thread). Graph cached per (rgb_ptr, out_ptr, n_pixels) key. Eliminates CPU-side kernel launch overhead on repeated calls.

**Result:** 86.16 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #18 — 2026-06-08 18:15:54 UTC ❌ DISCARD

**Hypothesis:** Add L2 prefetch hints (prefetch.global.L2) for next block's data to hide memory latency, combined with CUDA graphs. 512 threads/block, __ldg reads, st.global.wt stores, 4 pixels/thread.

**Result:** 86.77 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #19 — 2026-06-08 18:17:03 UTC ✅ KEEP

**Hypothesis:** Combine CUDA graphs with cuBLAS GEMV (mv + copy_). mv() was 89.24us without graphs; with graph capture eliminating launch overhead, cuBLAS GEMV gains dramatically on large sizes (4096², 8192²) which dominate the geomean.

**Result:** 83.33 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #20 — 2026-06-08 18:17:41 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + torch.mv with out=output.view(-1) to write directly into output buffer, eliminating intermediate allocation and copy_ kernel. Hypothesis: single memory pass faster than mv+copy_ even inside graph.

**Result:** 86.60 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #21 — 2026-06-08 18:18:14 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + torch.mm (GEMM) with (3,1) weight matrix instead of mv (GEMV). Hypothesis: GEMM uses Tensor Core path on H100, faster for large N×3 matrices at 4096² and 8192².

**Result:** 86.01 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #22 — 2026-06-08 18:18:56 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + mv + copy_ with allow_tf32=False to force full FP32 GEMV path. Hypothesis: disabling TF32 overhead speeds up the memory-bound GEMV. Result: completely loses the large-size speedup — TF32 was responsible for the 4096²/8192² wins in #19.

**Result:** 86.21 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #23 — 2026-06-08 18:19:57 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + torch.mm with (1,3) weight row @ (3,N) transposed rgb matrix. Hypothesis: column-major (3,N) layout is cuBLAS-native and triggers a faster GEMV/GEMM kernel than (N,3)@(3,) mv.

**Result:** 86.82 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #24 — 2026-06-08 18:20:30 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + half-precision (fp16) mv: convert rgb to fp16, run HGEMV, convert result back to fp32. Hypothesis: halving memory bandwidth via fp16 outweighs conversion cost on H100.

**Result:** 86.91 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #25 — 2026-06-08 18:21:07 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + torch.addmv(out_flat, mat, vec, beta=0, alpha=1, out=out_flat) to write GEMV directly into output buffer without intermediate allocation or copy_. Hypothesis: eliminates one memory pass vs mv+copy_.

**Result:** 88.94 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

---

## Experiment #26 — 2026-06-08 18:21:52 UTC ❌ DISCARD

**Hypothesis:** CUDA graphs + mv + copy_ captured on a high-priority CUDA stream (priority=-1). Hypothesis: high-priority stream reduces GPU scheduling latency, especially for small sizes.

**Result:** 86.96 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END

```

