import os
import sys
import time
import json
import uuid
import math
import socket
import psutil
import hashlib
import platform
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

numba_cuda: Any = None
numba_uint8: Any = None
CUDA_AVAILABLE = False

try:
    from numba import cuda as _nb_cuda
    from numba import uint8 as _nb_uint8
    if _nb_cuda.is_available():
        numba_cuda = _nb_cuda
        numba_uint8 = _nb_uint8
        CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

TILE_DIM = 256
MAX_PATTERN_LEN = 32
SHARED_BUFFER_SIZE = 288

console = Console()

class SharedMemoryKernelError(Exception):
    def __init__(self, message: str, error_code: int = 1400):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class VRAMAllocationError(SharedMemoryKernelError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1401)

class KernelParityError(SharedMemoryKernelError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1402)

@dataclass
class TileConfig:
    tile_dim: int
    max_pattern_len: int
    shared_buffer_bytes: int
    sequence_length: int
    grid_blocks: int

@dataclass
class DeviceCapability:
    device_name: str
    execution_backend: str
    cuda_active: bool
    compute_capability: str
    shared_mem_per_block_kb: float
    logical_cores: int
    available_ram_gb: float
    hostname: str

@dataclass
class SequenceBatch:
    batch_id: str
    sequence_array: np.ndarray
    pattern_array: np.ndarray
    sequence_len: int
    pattern_len: int

@dataclass
class KernelBenchmarkResult:
    batch_id: str
    backend_mode: str
    sequence_kb: float
    global_latency_ms: float
    shared_latency_ms: float
    latency_reduction_pct: float
    speedup_factor: float
    global_matches: int
    shared_matches: int
    parity_verified: bool
    shared_bandwidth_gbs: float
    timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class SharedMemoryAuditReport:
    session_id: str
    timestamp: str
    device_capability: DeviceCapability
    tile_config: TileConfig
    total_batches_executed: int
    total_megabases_scanned: float
    avg_global_latency_ms: float
    avg_shared_latency_ms: float
    mean_latency_reduction_pct: float
    peak_speedup_factor: float
    parity_pass_rate_pct: float
    cryptographic_checksum: str
    batch_ledger: List[KernelBenchmarkResult]

def build_cuda_kernels() -> Tuple[Any, Any]:
    if not CUDA_AVAILABLE or numba_cuda is None:
        return None, None

    @numba_cuda.jit
    def global_memory_kernel(sequence, pattern, seq_len, pat_len, out_flags):
        idx = numba_cuda.grid(1)
        if idx <= seq_len - pat_len:
            matched = 1
            for j in range(pat_len):
                if sequence[idx + j] != pattern[j]:
                    matched = 0
                    break
            out_flags[idx] = matched

    @numba_cuda.jit
    def shared_memory_tiled_kernel(sequence, pattern, seq_len, pat_len, out_flags):
        shared_tile = numba_cuda.shared.array(shape=SHARED_BUFFER_SIZE, dtype=numba_uint8)
        shared_pat = numba_cuda.shared.array(shape=MAX_PATTERN_LEN, dtype=numba_uint8)
        tx = numba_cuda.threadIdx.x
        block_start = numba_cuda.blockIdx.x * TILE_DIM
        global_idx = block_start + tx

        if tx < pat_len:
            shared_pat[tx] = pattern[tx]
        if global_idx < seq_len:
            shared_tile[tx] = sequence[global_idx]
        else:
            shared_tile[tx] = 0
        if tx < pat_len - 1:
            halo_idx = block_start + TILE_DIM + tx
            if halo_idx < seq_len:
                shared_tile[TILE_DIM + tx] = sequence[halo_idx]
            else:
                shared_tile[TILE_DIM + tx] = 0

        numba_cuda.syncthreads()

        if global_idx <= seq_len - pat_len:
            matched = 1
            for j in range(pat_len):
                if shared_tile[tx + j] != shared_pat[j]:
                    matched = 0
                    break
            out_flags[global_idx] = matched

    return global_memory_kernel, shared_memory_tiled_kernel

class CUDADeviceInspector:
    def inspect(self) -> DeviceCapability:
        mem = psutil.virtual_memory()
        avail_ram = mem.available / (1024 ** 3)
        cores = psutil.cpu_count(logical=True) or 2
        host = socket.gethostname()

        if CUDA_AVAILABLE and numba_cuda is not None:
            try:
                dev = numba_cuda.get_current_device()
                raw_name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
                cc = f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
                sram_kb = float(getattr(dev, "MAX_SHARED_MEMORY_PER_BLOCK", 49152)) / 1024.0
                return DeviceCapability(
                    device_name=raw_name,
                    execution_backend="CUDA_SHARED_SRAM",
                    cuda_active=True,
                    compute_capability=cc,
                    shared_mem_per_block_kb=sram_kb,
                    logical_cores=cores,
                    available_ram_gb=avail_ram,
                    hostname=host
                )
            except Exception:
                pass

        cpu_label = platform.processor() or f"{platform.machine()} Host CPU"
        return DeviceCapability(
            device_name=f"{cpu_label[:30]} (Tiled L1 Fallback)",
            execution_backend="CPU_TILED_L1_CACHE",
            cuda_active=False,
            compute_capability="EMULATED",
            shared_mem_per_block_kb=48.0,
            logical_cores=cores,
            available_ram_gb=avail_ram,
            hostname=host
        )

class TiledSequenceGenerator:
    def __init__(self, target_motif: str = "GATCGATCGATC"):
        self.rng = np.random.default_rng()
        self.bases = np.array([65, 67, 71, 84], dtype=np.uint8)
        self.pattern_array = np.frombuffer(target_motif.encode("ascii"), dtype=np.uint8).copy()

    def create_batch(self, sequence_len: int) -> SequenceBatch:
        seq = self.rng.choice(self.bases, size=sequence_len)
        pat_len = int(self.pattern_array.size)
        injections = int(self.rng.integers(4, 18))

        for _ in range(injections):
            pos = int(self.rng.integers(0, max(1, sequence_len - pat_len)))
            seq[pos:pos + pat_len] = self.pattern_array

        return SequenceBatch(
            batch_id=f"SHM_{uuid.uuid4().hex[:8].upper()}",
            sequence_array=np.ascontiguousarray(seq, dtype=np.uint8),
            pattern_array=np.ascontiguousarray(self.pattern_array, dtype=np.uint8),
            sequence_len=sequence_len,
            pattern_len=pat_len
        )

class HybridSharedMemoryExecutor:
    def __init__(self, device_cap: DeviceCapability):
        self.device_cap = device_cap
        self.global_kernel, self.shared_kernel = build_cuda_kernels()
        if self.device_cap.cuda_active and self.global_kernel is not None:
            self._warmup_cuda_kernels()

    def _warmup_cuda_kernels(self) -> None:
        dummy_seq = np.zeros(TILE_DIM * 2, dtype=np.uint8)
        dummy_pat = np.zeros(8, dtype=np.uint8)
        d_seq = numba_cuda.to_device(dummy_seq)
        d_pat = numba_cuda.to_device(dummy_pat)
        d_out = numba_cuda.to_device(np.zeros(TILE_DIM * 2, dtype=np.uint8))
        self.global_kernel[2, TILE_DIM](d_seq, d_pat, dummy_seq.size, dummy_pat.size, d_out)
        self.shared_kernel[2, TILE_DIM](d_seq, d_pat, dummy_seq.size, dummy_pat.size, d_out)
        numba_cuda.synchronize()

    def _run_cuda_pair(self, batch: SequenceBatch) -> Tuple[float, float, int, int]:
        d_seq = numba_cuda.to_device(batch.sequence_array)
        d_pat = numba_cuda.to_device(batch.pattern_array)
        d_out_global = numba_cuda.to_device(np.zeros(batch.sequence_len, dtype=np.uint8))
        d_out_shared = numba_cuda.to_device(np.zeros(batch.sequence_len, dtype=np.uint8))
        grid_blocks = math.ceil(batch.sequence_len / TILE_DIM)

        t0 = time.perf_counter()
        self.global_kernel[grid_blocks, TILE_DIM](d_seq, d_pat, batch.sequence_len, batch.pattern_len, d_out_global)
        numba_cuda.synchronize()
        global_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        self.shared_kernel[grid_blocks, TILE_DIM](d_seq, d_pat, batch.sequence_len, batch.pattern_len, d_out_shared)
        numba_cuda.synchronize()
        shared_ms = (time.perf_counter() - t1) * 1000.0

        global_hits = int(d_out_global.copy_to_host().sum())
        shared_hits = int(d_out_shared.copy_to_host().sum())
        return global_ms, shared_ms, global_hits, shared_hits

    def _run_cpu_tiled_pair(self, batch: SequenceBatch) -> Tuple[float, float, int, int]:
        seq = batch.sequence_array
        pat = batch.pattern_array
        pat_len = batch.pattern_len

        t0 = time.perf_counter()
        windows = np.lib.stride_tricks.sliding_window_view(seq, pat_len)
        global_hits = int(np.count_nonzero(np.all(windows == pat, axis=1)))
        global_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        limit = batch.sequence_len - pat_len + 1
        candidates = np.flatnonzero(seq[:limit] == pat[0])
        if candidates.size == 0:
            shared_hits = 0
        else:
            offsets = candidates[:, None] + np.arange(pat_len, dtype=np.int64)
            shared_hits = int(np.count_nonzero(np.all(seq[offsets] == pat, axis=1)))
        shared_ms = (time.perf_counter() - t1) * 1000.0

        return global_ms, shared_ms, global_hits, shared_hits

    def execute_batch(self, batch: SequenceBatch) -> KernelBenchmarkResult:
        if self.device_cap.cuda_active and self.global_kernel is not None:
            g_ms, s_ms, g_hits, s_hits = self._run_cuda_pair(batch)
        else:
            g_ms, s_ms, g_hits, s_hits = self._run_cpu_tiled_pair(batch)

        safe_g = max(g_ms, 0.0001)
        safe_s = max(s_ms, 0.0001)
        reduction_pct = ((safe_g - safe_s) / safe_g) * 100.0
        speedup = safe_g / safe_s
        seq_gb = batch.sequence_len / (1024 ** 3)
        bandwidth_gbs = seq_gb / (safe_s / 1000.0)

        return KernelBenchmarkResult(
            batch_id=batch.batch_id,
            backend_mode=self.device_cap.execution_backend,
            sequence_kb=batch.sequence_len / 1024.0,
            global_latency_ms=g_ms,
            shared_latency_ms=s_ms,
            latency_reduction_pct=reduction_pct,
            speedup_factor=speedup,
            global_matches=g_hits,
            shared_matches=s_hits,
            parity_verified=(g_hits == s_hits),
            shared_bandwidth_gbs=bandwidth_gbs
        )

class SharedMemoryUIRenderer:
    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="kernel_feed", ratio=2),
            Layout(name="side_col", ratio=1)
        )
        layout["side_col"].split_column(
            Layout(name="opt_metrics", ratio=1),
            Layout(name="hw_panel", ratio=1)
        )

        header = Text("GENEWEAVER INDUSTRY PIPELINE: CUDA SHARED MEMORY KERNEL (WEEK 4 DAY 1)", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(header, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("ALLOCATING TILED SHARED MEMORY BLOCKS...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update(self, layout: Layout, results: List[KernelBenchmarkResult], total: int, dev: DeviceCapability, tile_cfg: TileConfig) -> None:
        feed = Table(box=None, expand=True)
        feed.add_column("Batch ID", style="cyan")
        feed.add_column("Global VRAM", style="yellow", justify="right")
        feed.add_column("Shared SRAM", style="bold green", justify="right")
        feed.add_column("Latency Drop", style="magenta", justify="right")
        feed.add_column("Speedup", style="bright_cyan", justify="right")
        feed.add_column("Parity", style="white", justify="center")
        feed.add_column("Bandwidth", style="blue", justify="right")

        for r in results[-14:]:
            parity_tag = "[green]MATCH[/green]" if r.parity_verified else "[red]DRIFT[/red]"
            feed.add_row(
                r.batch_id,
                f"{r.global_latency_ms:.3f} ms",
                f"{r.shared_latency_ms:.3f} ms",
                f"{r.latency_reduction_pct:.1f}%",
                f"{r.speedup_factor:.2f}x",
                parity_tag,
                f"{r.shared_bandwidth_gbs:.2f} GB/s"
            )

        layout["kernel_feed"].update(Panel(feed, title="[bold green]Global vs Shared Memory Tiled Kernel Stream[/bold green]", border_style="green"))

        opt = Table(box=None, expand=True)
        opt.add_column("Optimization Metric", style="cyan")
        opt.add_column("Value", style="white", justify="right")
        count = len(results)
        avg_g = sum(r.global_latency_ms for r in results) / max(1, count)
        avg_s = sum(r.shared_latency_ms for r in results) / max(1, count)
        avg_red = sum(r.latency_reduction_pct for r in results) / max(1, count)
        peak_sp = max([r.speedup_factor for r in results], default=0.0)

        opt.add_row("Batches Benchmarked", f"{count} / {total}")
        opt.add_row("Avg Global Latency", f"{avg_g:.3f} ms")
        opt.add_row("Avg Shared Latency", f"[bold green]{avg_s:.3f} ms[/bold green]")
        opt.add_row("Mean Latency Drop", f"{avg_red:.1f}%")
        opt.add_row("Peak Kernel Speedup", f"[bold bright_cyan]{peak_sp:.2f}x[/bold bright_cyan]")
        layout["opt_metrics"].update(Panel(opt, title="[bold magenta]VRAM Latency Reduction[/bold magenta]", border_style="magenta"))

        hw = Table(box=None, expand=True)
        hw.add_column("Kernel Hardware", style="cyan")
        hw.add_column("Config", style="white", justify="right")
        hw.add_row("Backend", dev.execution_backend)
        hw.add_row("Tile / Block Dim", f"{tile_cfg.tile_dim} threads")
        hw.add_row("Shared Tile Buffer", f"{tile_cfg.shared_buffer_bytes} B/blk")
        hw.add_row("Grid Blocks", str(tile_cfg.grid_blocks))
        hw.add_row("Available RAM", f"{dev.available_ram_gb:.2f} GB")
        layout["hw_panel"].update(Panel(hw, title="[bold blue]Device & Tile Architecture[/bold blue]", border_style="blue"))

        layout["footer"].update(Panel(Text(f"EXECUTING SHARED MEMORY TILES... [{count}/{total}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class TelemetryVault:
    def __init__(self, output_dir: str = "reports/cuda_shared_memory"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_report(self, report: SharedMemoryAuditReport) -> str:
        path = os.path.join(self.output_dir, f"shared_mem_audit_{report.session_id}.json")
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=4)
        return path

class SharedMemoryOrchestrator:
    def __init__(self, batch_count: int = 24, sequence_len: int = 524288):
        self.batch_count = batch_count
        self.sequence_len = sequence_len
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.inspector = CUDADeviceInspector()
        self.device_cap = self.inspector.inspect()
        self.tile_config = TileConfig(
            tile_dim=TILE_DIM,
            max_pattern_len=MAX_PATTERN_LEN,
            shared_buffer_bytes=SHARED_BUFFER_SIZE + MAX_PATTERN_LEN,
            sequence_length=sequence_len,
            grid_blocks=math.ceil(sequence_len / TILE_DIM)
        )
        self.generator = TiledSequenceGenerator(target_motif="GATCGATCGATC")
        self.executor = HybridSharedMemoryExecutor(self.device_cap)
        self.renderer = SharedMemoryUIRenderer()
        self.vault = TelemetryVault()

    def run(self) -> None:
        if self.device_cap.available_ram_gb < 1.0:
            raise VRAMAllocationError("Insufficient host RAM for safe tiled buffer allocation.")

        layout = self.renderer.generate_layout()
        results: List[KernelBenchmarkResult] = []

        with Live(layout, refresh_per_second=10, screen=True):
            for _ in range(self.batch_count):
                batch = self.generator.create_batch(self.sequence_len)
                res = self.executor.execute_batch(batch)
                results.append(res)
                self.renderer.update(layout, results, self.batch_count, self.device_cap, self.tile_config)
                time.sleep(0.08)

            layout["footer"].update(Panel(Text("SHARED MEMORY KERNEL BENCHMARK COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

        avg_g = sum(r.global_latency_ms for r in results) / max(1, len(results))
        avg_s = sum(r.shared_latency_ms for r in results) / max(1, len(results))
        mean_red = sum(r.latency_reduction_pct for r in results) / max(1, len(results))
        peak_sp = max([r.speedup_factor for r in results], default=0.0)
        parity_pct = (sum(1 for r in results if r.parity_verified) / max(1, len(results))) * 100.0
        total_mb = (self.batch_count * self.sequence_len) / (1024 ** 2)
        digest = hashlib.sha256(f"{self.session_id}_{avg_g:.4f}_{avg_s:.4f}_{parity_pct:.2f}".encode("utf-8")).hexdigest()

        report = SharedMemoryAuditReport(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            device_capability=self.device_cap,
            tile_config=self.tile_config,
            total_batches_executed=len(results),
            total_megabases_scanned=total_mb,
            avg_global_latency_ms=avg_g,
            avg_shared_latency_ms=avg_s,
            mean_latency_reduction_pct=mean_red,
            peak_speedup_factor=peak_sp,
            parity_pass_rate_pct=parity_pct,
            cryptographic_checksum=digest,
            batch_ledger=results
        )

        saved_path = self.vault.save_report(report)
        console.print(f"\n[bold bright_green][+] Week 4 Day 1 Shared Memory Kernel Complete. Parity: {parity_pct:.1f}% | Peak Speedup: {peak_sp:.2f}x[/bold bright_green]")
        console.print(f"[bold cyan][i] Shared Memory Telemetry saved to: {saved_path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    app = SharedMemoryOrchestrator()
    app.run()