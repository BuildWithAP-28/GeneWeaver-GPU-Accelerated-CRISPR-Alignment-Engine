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
from typing import List, Dict, Any, Tuple

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
MAX_PAT_LEN = 32
SHARED_BUF = 288

console = Console()

class ProfilerError(Exception):
    def __init__(self, message: str, error_code: int = 1500):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class MemoryGuardError(ProfilerError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1501)

class ParityMismatchError(ProfilerError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1502)

@dataclass
class HardwareSnapshot:
    gpu_device: str
    backend_tier: str
    cuda_active: bool
    compute_capability: str
    logical_cores: int
    available_ram_gb: float
    hostname: str

@dataclass
class TierLatencySample:
    run_id: str
    sequence_kb: float
    cpu_baseline_ms: float
    global_vram_ms: float
    shared_sram_ms: float
    vram_latency_drop_pct: float
    global_vs_cpu_speedup: float
    shared_vs_cpu_speedup: float
    coalescing_efficiency_pct: float
    effective_bandwidth_gbs: float
    matches_detected: int
    parity_ok: bool
    timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class ProfilerAuditManifest:
    session_id: str
    timestamp: str
    hardware: HardwareSnapshot
    total_runs: int
    avg_cpu_baseline_ms: float
    avg_global_vram_ms: float
    avg_shared_sram_ms: float
    mean_vram_drop_pct: float
    mean_shared_speedup_x: float
    peak_shared_speedup_x: float
    mean_coalescing_eff_pct: float
    parity_pass_rate_pct: float
    target_100x_achieved: bool
    cryptographic_digest: str
    benchmark_samples: List[TierLatencySample]

def compile_benchmark_kernels() -> Tuple[Any, Any]:
    if not CUDA_AVAILABLE or numba_cuda is None:
        return None, None

    @numba_cuda.jit
    def global_vram_align_kernel(seq, pat, seq_len, pat_len, out_hits):
        idx = numba_cuda.grid(1)
        if idx <= seq_len - pat_len:
            ok = 1
            for j in range(pat_len):
                if seq[idx + j] != pat[j]:
                    ok = 0
                    break
            out_hits[idx] = ok

    @numba_cuda.jit
    def shared_sram_align_kernel(seq, pat, seq_len, pat_len, out_hits):
        tile = numba_cuda.shared.array(shape=SHARED_BUF, dtype=numba_uint8)
        s_pat = numba_cuda.shared.array(shape=MAX_PAT_LEN, dtype=numba_uint8)
        tx = numba_cuda.threadIdx.x
        block_offset = numba_cuda.blockIdx.x * TILE_DIM
        g_idx = block_offset + tx

        if tx < pat_len:
            s_pat[tx] = pat[tx]
        if g_idx < seq_len:
            tile[tx] = seq[g_idx]
        else:
            tile[tx] = 0
        if tx < pat_len - 1:
            halo_pos = block_offset + TILE_DIM + tx
            if halo_pos < seq_len:
                tile[TILE_DIM + tx] = seq[halo_pos]
            else:
                tile[TILE_DIM + tx] = 0

        numba_cuda.syncthreads()

        if g_idx <= seq_len - pat_len:
            ok = 1
            for j in range(pat_len):
                if tile[tx + j] != s_pat[j]:
                    ok = 0
                    break
            out_hits[g_idx] = ok

    return global_vram_align_kernel, shared_sram_align_kernel

class PurePythonBaselineEngine:
    def scan(self, seq_bytes: bytes, pat_bytes: bytes) -> Tuple[float, int]:
        seq_len = len(seq_bytes)
        pat_len = len(pat_bytes)
        limit = seq_len - pat_len + 1
        first_byte = pat_bytes[0]
        hits = 0

        t0 = time.perf_counter()
        for i in range(limit):
            if seq_bytes[i] == first_byte:
                matched = True
                for j in range(1, pat_len):
                    if seq_bytes[i + j] != pat_bytes[j]:
                        matched = False
                        break
                if matched:
                    hits += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, hits

class VRAMBenchmarkRunner:
    def __init__(self, target_motif: str = "GATCGATCGATC"):
        self.rng = np.random.default_rng()
        self.bases = np.array([65, 67, 71, 84], dtype=np.uint8)
        self.pat_bytes = target_motif.encode("ascii")
        self.pat_arr = np.ascontiguousarray(np.frombuffer(self.pat_bytes, dtype=np.uint8))
        self.cpu_engine = PurePythonBaselineEngine()
        self.hardware = self._capture_hardware()
        self.global_kernel, self.shared_kernel = compile_benchmark_kernels()
        if self.hardware.cuda_active and self.global_kernel is not None:
            self._warmup_gpu()

    def _capture_hardware(self) -> HardwareSnapshot:
        mem = psutil.virtual_memory()
        ram_gb = mem.available / (1024 ** 3)
        cores = psutil.cpu_count(logical=True) or 2
        host = socket.gethostname()

        if CUDA_AVAILABLE and numba_cuda is not None:
            try:
                dev = numba_cuda.get_current_device()
                name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
                cc = f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
                return HardwareSnapshot(
                    gpu_device=name,
                    backend_tier="CUDA_GLOBAL_VS_SHARED",
                    cuda_active=True,
                    compute_capability=cc,
                    logical_cores=cores,
                    available_ram_gb=ram_gb,
                    hostname=host
                )
            except Exception:
                pass

        cpu_name = platform.processor() or platform.machine()
        return HardwareSnapshot(
            gpu_device=f"{cpu_name[:28]} (Fallback)",
            backend_tier="CPU_VECTOR_FALLBACK",
            cuda_active=False,
            compute_capability="EMULATED",
            logical_cores=cores,
            available_ram_gb=ram_gb,
            hostname=host
        )

    def _warmup_gpu(self) -> None:
        d_s = numba_cuda.to_device(np.zeros(TILE_DIM * 2, dtype=np.uint8))
        d_p = numba_cuda.to_device(self.pat_arr)
        d_o = numba_cuda.to_device(np.zeros(TILE_DIM * 2, dtype=np.uint8))
        self.global_kernel[2, TILE_DIM](d_s, d_p, TILE_DIM * 2, self.pat_arr.size, d_o)
        self.shared_kernel[2, TILE_DIM](d_s, d_p, TILE_DIM * 2, self.pat_arr.size, d_o)
        numba_cuda.synchronize()

    def _build_sequence(self, seq_len: int) -> np.ndarray:
        seq = self.rng.choice(self.bases, size=seq_len)
        pat_len = int(self.pat_arr.size)
        for _ in range(int(self.rng.integers(5, 20))):
            idx = int(self.rng.integers(0, max(1, seq_len - pat_len)))
            seq[idx:idx + pat_len] = self.pat_arr
        return np.ascontiguousarray(seq, dtype=np.uint8)

    def _run_gpu_tiers(self, seq_arr: np.ndarray) -> Tuple[float, float, int, int]:
        seq_len = int(seq_arr.size)
        pat_len = int(self.pat_arr.size)
        blocks = math.ceil(seq_len / TILE_DIM)
        d_seq = numba_cuda.to_device(seq_arr)
        d_pat = numba_cuda.to_device(self.pat_arr)
        d_out_g = numba_cuda.to_device(np.zeros(seq_len, dtype=np.uint8))
        d_out_s = numba_cuda.to_device(np.zeros(seq_len, dtype=np.uint8))

        t0 = time.perf_counter()
        self.global_kernel[blocks, TILE_DIM](d_seq, d_pat, seq_len, pat_len, d_out_g)
        numba_cuda.synchronize()
        g_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        self.shared_kernel[blocks, TILE_DIM](d_seq, d_pat, seq_len, pat_len, d_out_s)
        numba_cuda.synchronize()
        s_ms = (time.perf_counter() - t1) * 1000.0

        g_hits = int(d_out_g.copy_to_host().sum())
        s_hits = int(d_out_s.copy_to_host().sum())
        return g_ms, s_ms, g_hits, s_hits

    def _run_fallback_tiers(self, seq_arr: np.ndarray) -> Tuple[float, float, int, int]:
        pat_len = int(self.pat_arr.size)
        t0 = time.perf_counter()
        win = np.lib.stride_tricks.sliding_window_view(seq_arr, pat_len)
        g_hits = int(np.count_nonzero(np.all(win == self.pat_arr, axis=1)))
        g_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        cands = np.flatnonzero(seq_arr[:seq_arr.size - pat_len + 1] == self.pat_arr[0])
        if cands.size == 0:
            s_hits = 0
        else:
            offs = cands[:, None] + np.arange(pat_len, dtype=np.int64)
            s_hits = int(np.count_nonzero(np.all(seq_arr[offs] == self.pat_arr, axis=1)))
        s_ms = (time.perf_counter() - t1) * 1000.0
        return g_ms, s_ms, g_hits, s_hits

    def execute_profile_run(self, seq_len: int) -> TierLatencySample:
        seq_arr = self._build_sequence(seq_len)
        cpu_ms, cpu_hits = self.cpu_engine.scan(seq_arr.tobytes(), self.pat_bytes)

        if self.hardware.cuda_active and self.global_kernel is not None:
            g_ms, s_ms, g_hits, s_hits = self._run_gpu_tiers(seq_arr)
        else:
            g_ms, s_ms, g_hits, s_hits = self._run_fallback_tiers(seq_arr)

        safe_g = max(g_ms, 0.0001)
        safe_s = max(s_ms, 0.0001)
        vram_drop = ((safe_g - safe_s) / safe_g) * 100.0
        pat_len = int(self.pat_arr.size)
        theoretical_coalesce = (1.0 - ((TILE_DIM + pat_len - 1) / float(TILE_DIM * pat_len))) * 100.0
        eff_bw = (seq_len / (1024 ** 3)) / (safe_s / 1000.0)

        return TierLatencySample(
            run_id=f"PRF_{uuid.uuid4().hex[:8].upper()}",
            sequence_kb=seq_len / 1024.0,
            cpu_baseline_ms=cpu_ms,
            global_vram_ms=g_ms,
            shared_sram_ms=s_ms,
            vram_latency_drop_pct=vram_drop,
            global_vs_cpu_speedup=cpu_ms / safe_g,
            shared_vs_cpu_speedup=cpu_ms / safe_s,
            coalescing_efficiency_pct=theoretical_coalesce,
            effective_bandwidth_gbs=eff_bw,
            matches_detected=s_hits,
            parity_ok=(cpu_hits == g_hits == s_hits)
        )

class LatencyProfilerUI:
    def create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="matrix_panel", ratio=2),
            Layout(name="right_col", ratio=1)
        )
        layout["right_col"].split_column(
            Layout(name="speedup_panel", ratio=1),
            Layout(name="device_panel", ratio=1)
        )

        title = Text("GENEWEAVER INDUSTRY PIPELINE: VRAM LATENCY & SPEEDUP PROFILER (WEEK 4 DAY 2)", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(title, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("INITIALIZING 3-TIER BENCHMARK MATRIX...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def render(self, layout: Layout, samples: List[TierLatencySample], total: int, hw: HardwareSnapshot) -> None:
        table = Table(box=None, expand=True)
        table.add_column("Run ID", style="cyan")
        table.add_column("Size", style="white", justify="right")
        table.add_column("CPU Base", style="red", justify="right")
        table.add_column("Global VRAM", style="yellow", justify="right")
        table.add_column("Shared SRAM", style="bold green", justify="right")
        table.add_column("VRAM Drop", style="magenta", justify="right")
        table.add_column("Speedup", style="bold bright_cyan", justify="right")
        table.add_column("Parity", style="white", justify="center")

        for s in samples[-14:]:
            badge = "[green]PASS[/green]" if s.parity_ok else "[red]FAIL[/red]"
            table.add_row(
                s.run_id,
                f"{s.sequence_kb:.0f} KB",
                f"{s.cpu_baseline_ms:.2f} ms",
                f"{s.global_vram_ms:.3f} ms",
                f"{s.shared_sram_ms:.3f} ms",
                f"{s.vram_latency_drop_pct:.1f}%",
                f"{s.shared_vs_cpu_speedup:.1f}x",
                badge
            )

        layout["matrix_panel"].update(Panel(table, title="[bold green]CPU Baseline vs Global VRAM vs Shared SRAM[/bold green]", border_style="green"))

        cnt = max(1, len(samples))
        avg_cpu = sum(s.cpu_baseline_ms for s in samples) / cnt
        avg_sram = sum(s.shared_sram_ms for s in samples) / cnt
        avg_drop = sum(s.vram_latency_drop_pct for s in samples) / cnt
        avg_sp = sum(s.shared_vs_cpu_speedup for s in samples) / cnt
        peak_sp = max([s.shared_vs_cpu_speedup for s in samples], default=0.0)
        goal_tag = "[bold green]EXCEEDED (>100x)[/bold green]" if peak_sp >= 100.0 else "[yellow]SCALING[/yellow]"

        sp_table = Table(box=None, expand=True)
        sp_table.add_column("Profiler Metric", style="cyan")
        sp_table.add_column("Value", style="white", justify="right")
        sp_table.add_row("Runs Completed", f"{len(samples)} / {total}")
        sp_table.add_row("Avg CPU / SRAM", f"{avg_cpu:.1f} / {avg_sram:.3f} ms")
        sp_table.add_row("Mean VRAM Drop", f"{avg_drop:.1f}%")
        sp_table.add_row("Mean Speedup", f"[bold bright_cyan]{avg_sp:.1f}x[/bold bright_cyan]")
        sp_table.add_row("Peak Speedup", f"[bold green]{peak_sp:.1f}x[/bold green]")
        sp_table.add_row("100x Benchmark", goal_tag)
        layout["speedup_panel"].update(Panel(sp_table, title="[bold magenta]Speedup & Coalescing Audit[/bold magenta]", border_style="magenta"))

        dev_table = Table(box=None, expand=True)
        dev_table.add_column("Hardware Spec", style="cyan")
        dev_table.add_column("State", style="white", justify="right")
        dev_table.add_row("Accelerator", hw.gpu_device[:22])
        dev_table.add_row("Backend Tier", hw.backend_tier)
        dev_table.add_row("Compute Cap", hw.compute_capability)
        dev_table.add_row("Coalescing Red.", f"{samples[-1].coalescing_efficiency_pct:.1f}%" if samples else "-")
        dev_table.add_row("Available RAM", f"{hw.available_ram_gb:.2f} GB")
        layout["device_panel"].update(Panel(dev_table, title="[bold blue]Device & Memory Profile[/bold blue]", border_style="blue"))

        layout["footer"].update(Panel(Text(f"BENCHMARKING MEMORY TIERS... [{len(samples)}/{total}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class TelemetryExporter:
    def __init__(self, output_dir: str = "reports/vram_latency_profiler"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, manifest: ProfilerAuditManifest) -> str:
        filepath = os.path.join(self.output_dir, f"vram_profile_{manifest.session_id}.json")
        with open(filepath, "w") as f:
            json.dump(asdict(manifest), f, indent=4)
        return filepath

class VRAMProfilerOrchestrator:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = VRAMBenchmarkRunner(target_motif="GATCGATCGATC")
        self.ui = LatencyProfilerUI()
        self.exporter = TelemetryExporter()
        self.workload_sizes = [65536, 131072, 262144, 524288] * 4

    def execute(self) -> None:
        if self.runner.hardware.available_ram_gb < 1.0:
            raise MemoryGuardError("Insufficient system RAM to execute profiler safely.")

        layout = self.ui.create_layout()
        samples: List[TierLatencySample] = []
        total_runs = len(self.workload_sizes)

        with Live(layout, refresh_per_second=10, screen=True):
            for seq_len in self.workload_sizes:
                sample = self.runner.execute_profile_run(seq_len)
                samples.append(sample)
                self.ui.render(layout, samples, total_runs, self.runner.hardware)
                time.sleep(0.06)

            layout["footer"].update(Panel(Text("3-TIER VRAM LATENCY PROFILING COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

        cnt = max(1, len(samples))
        avg_cpu = sum(s.cpu_baseline_ms for s in samples) / cnt
        avg_g = sum(s.global_vram_ms for s in samples) / cnt
        avg_s = sum(s.shared_sram_ms for s in samples) / cnt
        mean_drop = sum(s.vram_latency_drop_pct for s in samples) / cnt
        mean_sp = sum(s.shared_vs_cpu_speedup for s in samples) / cnt
        peak_sp = max([s.shared_vs_cpu_speedup for s in samples], default=0.0)
        mean_coal = sum(s.coalescing_efficiency_pct for s in samples) / cnt
        parity_pct = (sum(1 for s in samples if s.parity_ok) / cnt) * 100.0
        digest = hashlib.sha256(f"{self.session_id}_{avg_cpu:.3f}_{avg_s:.4f}_{peak_sp:.2f}".encode("utf-8")).hexdigest()

        manifest = ProfilerAuditManifest(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            hardware=self.runner.hardware,
            total_runs=len(samples),
            avg_cpu_baseline_ms=avg_cpu,
            avg_global_vram_ms=avg_g,
            avg_shared_sram_ms=avg_s,
            mean_vram_drop_pct=mean_drop,
            mean_shared_speedup_x=mean_sp,
            peak_shared_speedup_x=peak_sp,
            mean_coalescing_eff_pct=mean_coal,
            parity_pass_rate_pct=parity_pct,
            target_100x_achieved=(peak_sp >= 100.0),
            cryptographic_digest=digest,
            benchmark_samples=samples
        )

        out_path = self.exporter.save(manifest)
        console.print(f"\n[bold bright_green][+] Week 4 Day 2 Profiling Complete. Peak Speedup: {peak_sp:.1f}x | Parity: {parity_pct:.1f}%[/bold bright_green]")
        console.print(f"[bold cyan][i] Latency Profile JSON saved to: {out_path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    app = VRAMProfilerOrchestrator()
    app.execute()