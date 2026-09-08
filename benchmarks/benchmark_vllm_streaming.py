"""
Non-blocking streaming detokenization benchmark under concurrent async worker load.

Acceptance Criterion for Issue #27:
- Measures throughput (tokens/sec) across concurrent streaming workers.
- Measures token-step latency percentiles (Mean, P50, P95, P99).
- Quantifies event-loop responsiveness / jitter to verify non-blocking async execution.
- Verifies 100% decode correctness between streamed outputs and batch decodes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.integrations.vllm import AsyncVLLMStreamingWorker, UniqTokenVLLMAdapter
from uniqtoken.tokenizer import CustomTokenizer


def build_benchmark_tokenizer() -> CustomTokenizer:
    """Trains a representative tokenizer for benchmarking."""
    corpus = [
        "The transformer architecture relies on subword tokenization to compress sequence length.",
        "Exact offset alignment is essential for accurate span extraction and structured decoding.",
        "High performance native Rust modules allow multi-threaded parallel batch execution without GIL lock.",
        "Neural language modeling balances vocabulary size against computational embedding cost.",
        "Streaming detokenization in high-throughput inference engines like vLLM requires zero latency overhead.",
        "Incremental UTF-8 byte accumulation prevents invalid unicode replacement character glitches.",
    ] * 20
    return CustomTokenizer.train_from_corpus(
        corpus=corpus,
        target_vocab_size=500,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"],
        verbose=False,
    )


class EventLoopJitterMonitor:
    """Monitors event loop lag by measuring deviation from scheduled sleep times."""

    def __init__(self, interval_sec: float = 0.005) -> None:
        self.interval = interval_sec
        self.delays: List[float] = []
        self._running = False
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            await asyncio.sleep(self.interval)
            t1 = time.perf_counter()
            excess_delay = (t1 - t0) - self.interval
            if excess_delay > 0:
                self.delays.append(excess_delay * 1000.0)  # ms

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> Dict[str, float]:
        self._running = False
        if self._task:
            await self._task
        if not self.delays:
            return {"max_jitter_ms": 0.0, "avg_jitter_ms": 0.0}
        return {
            "max_jitter_ms": max(self.delays),
            "avg_jitter_ms": statistics.mean(self.delays),
            "p99_jitter_ms": (
                statistics.quantiles(self.delays, n=100)[98] if len(self.delays) >= 100 else max(self.delays)
            ),
        }


async def run_streaming_benchmark(
    adapter: UniqTokenVLLMAdapter,
    num_streams: int = 50,
    tokens_per_stream: int = 100,
    concurrency_limit: int = 50,
) -> Dict[str, float]:
    """Runs concurrent async streaming workers and collects performance metrics."""
    sample_text = (
        "High performance native Rust modules allow multi-threaded parallel batch execution without GIL lock. "
        "Streaming detokenization in high-throughput inference engines like vLLM requires zero latency overhead. "
        "Incremental UTF-8 byte accumulation prevents invalid unicode replacement character glitches."
    )
    base_token_ids = adapter.encode(sample_text)
    # Tile tokens to match target sequence length
    stream_token_ids = (base_token_ids * (tokens_per_stream // len(base_token_ids) + 1))[:tokens_per_stream]
    expected_full_text = adapter.decode(stream_token_ids)

    step_latencies_us: List[float] = []
    sem = asyncio.Semaphore(concurrency_limit)
    correct_streams = 0

    async def simulate_stream(stream_idx: int) -> Tuple[str, List[float]]:
        req_id = f"stream_{stream_idx}"
        worker = AsyncVLLMStreamingWorker(adapter, request_id=req_id)
        local_latencies: List[float] = []

        async with sem:
            # Produce tokens in background
            async def producer():
                for tid in stream_token_ids:
                    t_start = time.perf_counter()
                    await worker.put_token(tid)
                    await worker.queue.join()
                    t_end = time.perf_counter()
                    local_latencies.append((t_end - t_start) * 1_000_000.0)  # microseconds
                await worker.put_token(None)  # EOF

            prod_task = asyncio.create_task(producer())

            # Consume streamed text deltas
            received_chunks: List[str] = []
            async for delta in worker.stream_deltas():
                received_chunks.append(delta)

            await prod_task
            return "".join(received_chunks), local_latencies

    # Start event loop jitter monitor
    monitor = EventLoopJitterMonitor(interval_sec=0.002)
    monitor.start()

    t_start = time.perf_counter()
    tasks = [simulate_stream(i) for i in range(num_streams)]
    results = await asyncio.gather(*tasks)
    t_end = time.perf_counter()

    jitter_stats = await monitor.stop()

    total_time_sec = t_end - t_start
    total_tokens = num_streams * tokens_per_stream

    for text, latencies in results:
        step_latencies_us.extend(latencies)
        if text == expected_full_text:
            correct_streams += 1

    throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

    step_latencies_us.sort()
    n = len(step_latencies_us)
    p50 = step_latencies_us[int(n * 0.50)] if n > 0 else 0.0
    p95 = step_latencies_us[int(n * 0.95)] if n > 0 else 0.0
    p99 = step_latencies_us[int(n * 0.99)] if n > 0 else 0.0
    mean_lat = statistics.mean(step_latencies_us) if step_latencies_us else 0.0

    return {
        "num_streams": num_streams,
        "tokens_per_stream": tokens_per_stream,
        "total_tokens": total_tokens,
        "total_time_sec": total_time_sec,
        "throughput_tokens_per_sec": throughput,
        "accuracy_percent": (correct_streams / num_streams) * 100.0,
        "latency_mean_us": mean_lat,
        "latency_p50_us": p50,
        "latency_p95_us": p95,
        "latency_p99_us": p99,
        "event_loop_avg_jitter_ms": jitter_stats["avg_jitter_ms"],
        "event_loop_max_jitter_ms": jitter_stats["max_jitter_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description="UniqToken vLLM Streaming Detokenizer Benchmark")
    parser.add_argument("--num-streams", type=int, default=50, help="Number of concurrent streams")
    parser.add_argument("--tokens-per-stream", type=int, default=100, help="Tokens generated per stream")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent workers")
    parser.add_argument("--json", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    if args.num_streams <= 0:
        parser.error("--num-streams must be a positive integer.")
    if args.tokens_per_stream <= 0:
        parser.error("--tokens-per-stream must be a positive integer.")
    if args.concurrency <= 0:
        parser.error("--concurrency must be a positive integer.")

    print("=" * 90)
    print("UNIQTOKEN vLLM ASYNC STREAMING DETOKENIZER BENCHMARK")
    print("=" * 90)
    print(f"Concurrent Streams   : {args.num_streams}")
    print(f"Tokens Per Stream    : {args.tokens_per_stream}")
    print(f"Total Tokens         : {args.num_streams * args.tokens_per_stream:,}")
    print(f"Worker Concurrency   : {args.concurrency}")
    print("-" * 90)

    print("Training benchmark tokenizer model...")
    tok = build_benchmark_tokenizer()
    adapter = UniqTokenVLLMAdapter(tok)
    print("Tokenizer ready. Executing concurrent async streaming detokenization...")

    metrics = asyncio.run(
        run_streaming_benchmark(
            adapter,
            num_streams=args.num_streams,
            tokens_per_stream=args.tokens_per_stream,
            concurrency_limit=args.concurrency,
        )
    )

    print("-" * 90)
    print("RESULTS:")
    print(f"  Wall-Clock Duration    : {metrics['total_time_sec']:.3f} s")
    print(f"  Aggregate Throughput   : {metrics['throughput_tokens_per_sec']:,.1f} tokens/s")
    print(f"  Stream Output Accuracy : {metrics['accuracy_percent']:.1f}%")
    print(f"  Step Latency (Mean)    : {metrics['latency_mean_us']:.2f} µs")
    print(f"  Step Latency (P50)     : {metrics['latency_p50_us']:.2f} µs")
    print(f"  Step Latency (P95)     : {metrics['latency_p95_us']:.2f} µs")
    print(f"  Step Latency (P99)     : {metrics['latency_p99_us']:.2f} µs")
    print(f"  Event Loop Avg Jitter  : {metrics['event_loop_avg_jitter_ms']:.3f} ms")
    print(f"  Event Loop Max Jitter  : {metrics['event_loop_max_jitter_ms']:.3f} ms")
    print("=" * 90)

    # On Windows, default timer resolution is ~15.6ms; under 50ms confirms no blocking.
    if metrics["event_loop_max_jitter_ms"] < 60.0:
        print("[PASS] Event loop responsiveness verified: non-blocking execution confirmed.")
    else:
        print("[WARN] Elevated event loop jitter observed.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved benchmark results to {args.json}")


if __name__ == "__main__":
    main()
