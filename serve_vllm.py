#!/usr/bin/env python3
"""vLLM serving entrypoint — OpenAI-compatible /v1/chat/completions.

Runs the LoRA adapter produced by train_lora.py on a GPU machine:

  pip install -r requirements-gpu.txt
  python serve_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --adapter artifacts/lora-qwen25-05b --port 8000

vLLM itself exposes the OpenAI-compatible API; this wrapper only wires the
model/adapter/port config so the benchmark in bench/benchmark.py can target
either this server or the CPU fallback (src/denial_appeal/serve_stub.py)
with no client changes. Without vLLM installed it exits with instructions
instead of pretending to serve.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    try:
        from vllm.entrypoints.openai.api_server import run_server
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        print("vLLM is not installed. On a GPU machine: "
              "pip install -r requirements-gpu.txt\n"
              "CPU fallback with the same API: "
              "uvicorn denial_appeal.serve_stub:app --port 8000",
              file=sys.stderr)
        return 2
    # Delegate to vLLM's own OpenAI server arg handling.
    parser = make_arg_parser(FlexibleArgumentParser())
    vllm_args = parser.parse_args(
        ["--model", a.model, "--host", a.host, "--port", str(a.port)]
        + (["--enable-lora", "--lora-modules", f"denial-appeal={a.adapter}"] if a.adapter else []))
    import asyncio
    asyncio.run(run_server(vllm_args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
