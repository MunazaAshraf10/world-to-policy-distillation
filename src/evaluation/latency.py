from collections.abc import Callable

import numpy as np
import torch


@torch.no_grad()
def measure(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> dict[str, float]:
    """Wall clock latency of a CUDA callable in milliseconds, with peak allocated memory."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def parameter_count(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())
