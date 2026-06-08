#!/usr/bin/env bash
set -e
cd /workspace/grayscale-trad-agentic-loop

echo "Checking GPU..."
OUTPUT=$(uv run python grayscale_kernel/run_eval.py grayscale_kernel/submission.py -o /tmp/gpu_check.json --mode test 2>&1)
echo "$OUTPUT"

GPU_LINE=$(echo "$OUTPUT" | grep "GPU:" || true)
echo ""
echo "Detected: $GPU_LINE"

if echo "$OUTPUT" | grep -q "NVIDIA H100"; then
    echo ""
    echo "--- GPU is H100 — launching agent ---"
    echo ""
    uv run grayscale_kernel/agent.py --baseline grayscale_kernel/starting_point.py --iterations 25 2>&1 | tee /tmp/agent_run.log
else
    echo ""
    echo "--- GPU is NOT H100 — aborting ---"
    exit 1
fi
