# Grayscale Autoresearch

An autonomous agent that iteratively optimizes a CUDA kernel for RGB-to-grayscale conversion on NVIDIA A100. Each iteration the agent makes exactly one change to `submission.py`, evaluates it on an A100 via Modal, logs the result, and stops. The outer loop drives the next iteration.

## Task

Convert a square RGB image to grayscale using the standard luminance coefficients:

```
Y = 0.2989 R + 0.5870 G + 0.1140 B
```

`custom_kernel` receives an RGB tensor and returns a grayscale tensor:

| Argument | Shape | Dtype |
|---|---|---|
| input | `H × W × 3` | `float32` |
| output | `H × W` | `float32` |

**Benchmark shapes:**

| Size | Image |
|---|---|
| 512 | 512 × 512 × 3 |
| 1024 | 1024 × 1024 × 3 |
| 2048 | 2048 × 2048 × 3 |
| 4096 | 4096 × 4096 × 3 |
| 8192 | 8192 × 8192 × 3 |
| 16384 | 16384 × 16384 × 3 |

Ranked by geometric mean latency across all six shapes (lower is better).

## Setup

```bash
uv sync

# Configure Modal credentials
uv run modal token set --token-id <token-id> --token-secret <token-secret>

# Deploy the A100 evaluator (once, before any agent runs)
uv run modal deploy eval_modal_grayscale_kernel.py
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-opus-4-7   # optional, this is the default
```

## Running the agent

```bash
uv run grayscale_kernel/agent.py --iterations 20
```

Start from a specific baseline file instead of the current `submission.py`:

```bash
uv run grayscale_kernel/agent.py --baseline path/to/baseline.py --iterations 20
```

Quick correctness check without a full benchmark:

```bash
cd grayscale_kernel
uv run python run_eval.py submission.py -o results.json --mode test
```

## Structure

```
eval_modal_grayscale_kernel.py  — deployable Modal A100 evaluator
grayscale_kernel/
├── agent.py        — agentic loop (LangChain + DeepAgents)
├── program.md      — system prompt: task spec, constraints, optimization hints
├── submission.py   — the kernel file the agent edits each iteration
├── run_eval.py     — submits submission.py to the deployed Modal evaluator
├── tools.py        — log_experiment and get_experiment_history tools
└── runs/           — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count
- `iterations.png` — best latency per agent iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `conversation_history/` — full agent conversation saved on exit

## LLM Call Counter

The agent tracks how many times the LLM is invoked (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the yield summary line: `--- Agent yielded (N messages, K LLM calls, T total) ---`
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
