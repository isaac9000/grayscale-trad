"""
NVfp4 GEMV kernel optimization agent powered by LangChain Deep Agents.

Runs autoresearch for X iterations, checkpointing every Y iterations.

Usage:
    uv run agent.py
    uv run agent.py --iterations 100 --checkpoint-every 10
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

import tools as _tools
from tools import (
    log_experiment,
    get_experiment_history,
    _update_plot,
    _get_next_iteration,
    _log_experiment_direct,
    set_run_directory,
    set_agent_iteration,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_DIR)


def load_system_prompt() -> str:
    with open(os.path.join(PROJECT_DIR, "program.md")) as f:
        return f.read()


def read_results_summary() -> str:
    if not os.path.exists(_tools.TSV_FILE):
        return "No experiments run yet."
    with open(_tools.TSV_FILE) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return "No experiments run yet."

    total = len(lines) - 1
    keeps, discards, crashes = [], 0, 0
    best_time = float("inf")
    best_desc = ""
    last_5 = []

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        it, _, _, time_str, status = parts[0], parts[1], parts[2], parts[3], parts[4]
        desc = parts[5] if len(parts) > 5 else ""
        try:
            t = float(time_str)
        except ValueError:
            t = 0.0

        if status == "keep" and t > 0:
            keeps.append((int(it) if it.isdigit() else 0, t, desc))
            if t < best_time:
                best_time = t
                best_desc = desc
        elif status == "discard":
            discards += 1
        elif status == "crash":
            crashes += 1

        last_5.append(f"  #{it}: {t:.2f}μs ({status}) — {desc[:60]}")

    last_5 = last_5[-5:]
    summary = f"=== EXPERIMENT SUMMARY ({total} total) ===\n"
    summary += f"Best time: {best_time:.2f} μs" if best_time < float("inf") else "Best time: none yet"
    if best_desc:
        summary += f" — {best_desc[:80]}\n"
    else:
        summary += "\n"
    summary += f"Keeps: {len(keeps)} | Discards: {discards} | Crashes: {crashes}\n"
    if keeps:
        summary += "Keep history (experiment -> time):\n"
        for it, t, d in keeps[-10:]:
            summary += f"  #{it}: {t:.2f}μs — {d[:60]}\n"
    summary += "\nLast 5 experiments:\n" + "\n".join(last_5) + "\n"
    return summary


def build_agent():
    load_dotenv()
    model_name = os.environ.get("AUTORESEARCH_MODEL", "claude-opus-4-7")
    system_prompt = load_system_prompt()
    checkpointer = MemorySaver()

    if model_name.startswith("claude-"):
        model = ChatAnthropic(model=model_name, timeout=120, max_retries=2)
    else:
        model = ChatOpenAI(model=model_name, use_responses_api=False, timeout=120, max_retries=2)

    venv_path = os.path.join(REPO_ROOT, ".venv", "bin")
    current_path = os.environ.get("PATH", "")
    env = {
        "PATH": f"{venv_path}:{current_path}",
        "VIRTUAL_ENV": os.path.join(REPO_ROOT, ".venv"),
        "PYTHONPATH": PROJECT_DIR,
    }
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]:
        if key in os.environ:
            env[key] = os.environ[key]

    agent = create_deep_agent(
        model=model,
        tools=[log_experiment, get_experiment_history],
        system_prompt=system_prompt,
        backend=LocalShellBackend(root_dir=PROJECT_DIR, virtual_mode=False, env=env),
        checkpointer=checkpointer,
    )
    return agent, checkpointer


def print_checkpoint(iteration: int, total_iterations: int, start_time: float):
    elapsed_min = (time.time() - start_time) / 60
    rate = iteration / elapsed_min if elapsed_min > 0 else 0
    summary = read_results_summary()
    print(f"\n{'#'*60}")
    print(f"  CHECKPOINT — Iteration {iteration}/{total_iterations}")
    print(f"  Elapsed: {elapsed_min:.1f} min | Rate: {rate:.1f} iter/min")
    print(f"{'#'*60}")
    print(summary)
    try:
        _update_plot()
        print(f"  Plot updated: {_tools.PLOT_FILE}")
    except Exception as e:
        print(f"  Plot update failed: {e}")
    print(f"{'#'*60}\n")


def print_final_report(total_iterations: int, actual_iterations: int, start_time: float):
    elapsed_min = (time.time() - start_time) / 60
    summary = read_results_summary()
    print(f"\n{'='*60}")
    print(f"  FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Iterations completed: {actual_iterations}/{total_iterations}")
    print(f"  Total time: {elapsed_min:.1f} min")
    print(f"{'='*60}")
    print(summary)
    try:
        _update_plot()
    except Exception:
        pass
    print(f"{'='*60}")


def save_conversation_history(checkpointer, config: dict, run_dir: str) -> None:
    try:
        checkpoint_tuple = checkpointer.get_tuple(config)
        if not checkpoint_tuple:
            return
        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        if not messages:
            return
        save_dir = os.path.join(run_dir, "conversation_history")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "conversation.md")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(path, "w") as f:
            f.write(f"# Conversation History\n\nSaved: {timestamp} | Messages: {len(messages)}\n\n")
            for i, msg in enumerate(messages, 1):
                f.write(f"---\n\n## Message {i} — {type(msg).__name__}\n\n")
                content = msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                f.write(block["text"] + "\n\n")
                            elif block.get("type") == "tool_use":
                                args_str = json.dumps(block.get("input", {}), indent=2)
                                if len(args_str) > 2000:
                                    args_str = args_str[:2000] + "\n... (truncated)"
                                f.write(f"**Tool call:** `{block['name']}`\n```json\n{args_str}\n```\n\n")
                elif content:
                    f.write(content + "\n\n")
        print(f"Conversation history saved to: {path}", flush=True)
    except Exception as e:
        print(f"Warning: could not save conversation history: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="NVfp4 GEMV Autoresearch Agent")
    parser.add_argument("--iterations", "-n", type=int, default=20)
    parser.add_argument("--checkpoint-every", "-c", type=int, default=5)
    parser.add_argument(
        "--baseline", "-b",
        default=None,
        help="Path to a baseline file to copy into submission.py before the run starts. "
             "Each run gets a fresh history — no cross-run visibility.",
    )
    args = parser.parse_args()

    load_dotenv()

    model_name = os.environ.get("AUTORESEARCH_MODEL", "claude-opus-4-7")
    if model_name.startswith("claude-"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set")
            sys.exit(1)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set")
            sys.exit(1)

    # Resolve and validate baseline file.
    baseline_path = None
    baseline_name = "scratch"
    if args.baseline:
        baseline_path = os.path.abspath(args.baseline)
        if not os.path.isfile(baseline_path):
            print(f"Error: baseline file not found: {baseline_path}")
            sys.exit(1)
        baseline_name = os.path.splitext(os.path.basename(baseline_path))[0]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROJECT_DIR, "runs", f"{timestamp}_nvfp4_gemv_{baseline_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Copy baseline into submission.py so this run starts from a known state.
    # Each run gets its own run_dir and a fresh experiment history, so runs are
    # fully isolated from each other.
    submission_path = os.path.join(PROJECT_DIR, "submission.py")
    if baseline_path:
        import shutil
        shutil.copy2(baseline_path, submission_path)
        print(f"Copied baseline '{baseline_name}' -> submission.py", flush=True)
        # Patch known PTX issues in the copy (not in the original baseline):
        # ptxas for sm_100a rejects (1) multiple cache eviction modifiers on one
        # ld instruction and (2) v4.u64 (256-bit) vector loads (max is v2.u64 / 128-bit).
        # We split each v4.u64 load into two v2.u64 loads and strip conflicting modifiers.
        with open(submission_path) as _f:
            _code = _f.read()
        _patched = (
            _code
            # load_block_32x2fp4 line 1: 256-bit A-tile load with dual L2 cache modifiers.
            # Replace the 256-bit v4.u64 load with two 128-bit v2.u64 loads (ptxas limit)
            # and strip the conflicting L2 cache modifiers.
            # Note: \\n\\t inside the strings is the CUDA PTX escape (literal \n\t);
            # the actual newline (\n) between strings is valid C adjacent-string concatenation.
            .replace(
                '"ld.global.L1::no_allocate.L2::evict_first.L2::256B.v4.u64 {%0, %1, %2, %3}, [%8];\\n\\t"',
                '"ld.global.v2.u64 {%0, %1}, [%8];\\n\\t"\n        "ld.global.v2.u64 {%2, %3}, [%8+16];\\n\\t"',
            )
            # load_block_32x2fp4 line 2: 256-bit B-tile load with dual L1+L2 eviction hints.
            .replace(
                '"ld.global.L1::evict_last.L2::evict_last.v4.u64 {%4, %5, %6, %7}, [%9];\\n\\t"',
                '"ld.global.v2.u64 {%4, %5}, [%9];\\n\\t"\n        "ld.global.v2.u64 {%6, %7}, [%9+16];\\n\\t"',
            )
        )
        if _patched != _code:
            with open(submission_path, "w") as _f:
                _f.write(_patched)
            print("  Applied PTX v4.u64→v2.u64 patch to submission.py", flush=True)
        else:
            print("  No PTX patch needed (strings not found — baseline may differ)", flush=True)

        # Patch load_inline calls that lack build_directory to avoid stale
        # torch_extensions cache on warm Modal containers.  The cache persists
        # between invocations; if a previous build partially failed the .so is
        # missing but the metadata says it exists, causing an ImportError.
        # Injecting a unique tempfile dir bypasses this entirely.
        with open(submission_path) as _f:
            _code2 = _f.read()
        if "load_inline(" in _code2 and "build_directory" not in _code2:
            # Try specific baseline10 structure first, then fall back to generic.
            _patched2 = _code2.replace(
                "load_inline(\n    name='gemv_cuda',",
                "load_inline(\n    build_directory=__import__('tempfile').mkdtemp(prefix='gemv_build_'),\n    name='gemv_cuda',",
            )
            if _patched2 == _code2:
                # Generic fallback: inject after "load_inline(\n    name=".
                _patched2 = _code2.replace(
                    "load_inline(\n    name=",
                    "load_inline(\n    build_directory=__import__('tempfile').mkdtemp(prefix='gemv_build_'),\n    name=",
                )
            if _patched2 != _code2:
                with open(submission_path, "w") as _f:
                    _f.write(_patched2)
                print("  Applied build_directory patch to load_inline in submission.py", flush=True)
            else:
                print("  build_directory patch: load_inline structure unrecognized — skipping", flush=True)
        else:
            print("  No build_directory patch needed (already present or load_inline not found)", flush=True)
    else:
        print("No baseline specified — using current submission.py as-is.", flush=True)

    set_run_directory(run_dir)

    agent, checkpointer = build_agent()

    print(f"Starting nvfp4_gemv optimization agent...")
    print(f"  Model:      {model_name}")
    print(f"  Baseline:   {baseline_name}")
    print(f"  Run dir:    {run_dir}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Checkpoint every: {args.checkpoint_every} iterations")
    print()

    # Each run uses a unique thread_id so LangChain's MemorySaver never leaks
    # conversation state between runs.
    thread_id = f"nvfp4-gemv-{baseline_name}-{timestamp}"
    config = {"configurable": {"thread_id": thread_id}}

    def _sigterm_handler(signum, frame):
        print("\n--- SIGTERM received ---", flush=True)
        save_conversation_history(checkpointer, config, run_dir)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    start_time = time.time()

    submission_path = os.path.join(PROJECT_DIR, "submission.py")
    results_path = os.path.join(PROJECT_DIR, "results.json")

    if baseline_path:
        # Benchmark the baseline before the agent loop so iteration 1 is always
        # logged, regardless of what the agent does.
        print(f"Benchmarking baseline '{baseline_name}'...", flush=True)
        venv_python = os.path.join(REPO_ROOT, ".venv", "bin", "python")
        ret = os.system(
            f"cd {PROJECT_DIR} && {venv_python} run_eval.py {submission_path} -o {results_path} 2>&1"
        )
        try:
            with open(submission_path) as f:
                baseline_code = f.read()
            with open(results_path) as f:
                import re as _re
                md = json.load(f)
            m = _re.search(r"Geometric mean: ⏱ ([\d.]+)", md if isinstance(md, str) else "")
            time_us = float(m.group(1)) if m else 0.0
            if ret == 0 and time_us > 0:
                _log_experiment_direct(
                    kernel_code=baseline_code,
                    hypothesis=f"Baseline '{baseline_name}' — initial benchmark before any agent changes",
                    time_us=time_us,
                    status="keep",
                )
                print(f"Baseline logged: {time_us:.1f} µs", flush=True)
                kickoff_note = (
                    f"The '{baseline_name}' baseline has been benchmarked and logged as "
                    f"experiment #1 ({time_us:.1f} µs). Your job is to beat it. "
                )
            else:
                error_snippet = md[:600] if isinstance(md, str) else f"run_eval exited {ret}"
                _log_experiment_direct(
                    kernel_code=baseline_code,
                    hypothesis=f"Baseline '{baseline_name}' — initial benchmark (CRASHED)",
                    time_us=0.0,
                    status="crash",
                    error_message=error_snippet,
                )
                print(f"Baseline crashed — logged as crash.", flush=True)
                kickoff_note = (
                    f"The '{baseline_name}' baseline was benchmarked but CRASHED (logged as experiment #1). "
                    "Read the crash error in get_experiment_history and fix the kernel before anything else. "
                )
        except Exception as e:
            print(f"Warning: could not log baseline: {e}", flush=True)
            kickoff_note = (
                f"submission.py has been pre-loaded with '{baseline_name}'. "
                "Benchmark it first, log the result, then improve. "
            )
    else:
        kickoff_note = (
            "submission.py is a stub with NotImplementedError — "
            "your first task is to implement a working kernel. "
        )

    kickoff_message = (
        "Read program.md for full instructions. Then call get_experiment_history "
        "to review any prior attempts. "
        f"{kickoff_note}"
        "Make exactly ONE meaningful change to submission.py, evaluate it "
        "with `python run_eval.py submission.py -o results.json`, "
        "log the result with log_experiment, then stop.\n\n"
        + read_results_summary()
    )

    iteration = 0
    try:
        msg = kickoff_message
        while iteration < args.iterations:
            iteration += 1
            set_agent_iteration(iteration)
            print(f"\n{'='*60}")
            print(f"  Agent iteration {iteration}/{args.iterations}")
            print(f"{'='*60}\n", flush=True)

            log_count_before = _get_next_iteration() - 1

            result = None
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": msg}]},
                config=config,
                stream_mode="values",
            ):
                result = chunk
                last_msg = chunk["messages"][-1]
                msg_type = type(last_msg).__name__
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"[tool_call] {tc['name']}({str(tc.get('args',''))[:120]})", flush=True)
                elif hasattr(last_msg, "tool_call_id"):
                    print(f"[tool_result] {str(getattr(last_msg,'content',''))[:200]}", flush=True)
                elif msg_type == "AIMessage":
                    preview = str(getattr(last_msg, "content", ""))[:200]
                    if preview.strip():
                        print(f"[llm] {preview}", flush=True)

            n_msgs = len(result["messages"])
            final = result["messages"][-1]
            content = final.content if hasattr(final, "content") else str(final)
            print(f"\n--- Agent yielded ({n_msgs} messages) ---")
            print(content[:500] if content else "(empty)", flush=True)

            log_count_after = _get_next_iteration() - 1
            logged_this_iter = log_count_after > log_count_before
            if not logged_this_iter:
                print(f"[WARNING] Iteration {iteration} produced no log entry!", flush=True)

            if iteration % args.checkpoint_every == 0:
                print_checkpoint(iteration, args.iterations, start_time)

            recent_summary = read_results_summary()
            enforce_log = (
                "CRITICAL: You did NOT call log_experiment in the previous iteration. "
                "Before making any new change, call log_experiment NOW for whatever you did last — "
                "use status='crash' and time_us=0.0 if the eval failed or was skipped. "
                "Logging is mandatory for every single attempt.\n\n"
                if not logged_this_iter else ""
            )
            msg = (
                f"{enforce_log}"
                f"Iteration {iteration + 1}/{args.iterations}. "
                "Make exactly ONE meaningful algorithmic change to submission.py, "
                "evaluate it, log the result with log_experiment, then stop.\n\n"
                f"{recent_summary}\n"
                "Call get_experiment_history for full prior code if needed. "
                "Do not summarize or ask for instructions — just act."
            )

    except KeyboardInterrupt:
        print(f"\n--- Interrupted at iteration {iteration} ---")
    except Exception as e:
        print(f"\n--- Agent error at iteration {iteration}: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        save_conversation_history(checkpointer, config, run_dir)
        print_final_report(args.iterations, iteration, start_time)


if __name__ == "__main__":
    main()
