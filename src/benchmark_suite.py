import os
import sys
import time
import json
import uuid
import platform
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np

os.environ['NUMBA_CUDA_LOG_LEVEL'] = 'CRITICAL'
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

@dataclass
class BenchmarkResult:
    workload_mb: float
    cpu_latency_ms: float
    gpu_latency_ms: float
    cpu_throughput_mbs: float
    gpu_throughput_gbs: float
    speedup_multiplier: float
    matches_found: int
    is_verified: bool
    timestamp: str

@cuda.jit
def fast_alignment_kernel(dna_stream, target_seq, match_counts, seq_len, target_len):
    shared_target = cuda.shared.array(shape=(256,), dtype=np.uint8)  # type: ignore
    tid = cuda.threadIdx.x  # type: ignore
    bid = cuda.blockIdx.x  # type: ignore
    bdim = cuda.blockDim.x  # type: ignore
    global_idx = (bid * bdim) + tid  # type: ignore
    
    if tid < target_len:
        shared_target[tid] = target_seq[tid]  # type: ignore
    cuda.syncthreads()  # type: ignore
    
    if global_idx < (seq_len - target_len + 1):
        is_match = True
        for i in range(target_len):
            if dna_stream[global_idx + i] != shared_target[i]:  # type: ignore
                is_match = False
                break
        if is_match:
            cuda.atomic.add(match_counts, 0, 1)  # type: ignore

class SystemProfiler:
    def __init__(self):
        self.os_name = platform.system()
        self.os_version = platform.version()
        self.cpu_arch = platform.machine()
        self.python_version = platform.python_version()
        self._init_gpu()

    def _init_gpu(self):
        if not cuda.is_available():
            console.print("[bold red]CRITICAL: CUDA Offline. Cannot run benchmarks.[/bold red]")
            sys.exit(1)
        self.device = cuda.get_current_device()
        self.gpu_name = self.device.name.decode('utf-8') if isinstance(self.device.name, bytes) else str(self.device.name)
        ctx = cuda.current_context()
        free, total = ctx.get_memory_info()
        self.vram_total_mb = total / (1024**2)

    def get_panel(self) -> Panel:
        table = Table(box=None, expand=True)
        table.add_column("System Component", style="cyan")
        table.add_column("Specification", style="bold yellow", justify="right")
        table.add_row("OS Architecture", f"{self.os_name} {self.os_version} ({self.cpu_arch})")
        table.add_row("Python Version", self.python_version)
        table.add_row("GPU Accelerator", self.gpu_name)
        table.add_row("Total VRAM", f"{self.vram_total_mb:.2f} MB")
        return Panel(table, title="[bold blue]GeneWeaver Testing Environment[/bold blue]", border_style="blue")

class CPUExecutionWorker:
    def __init__(self):
        self.engine_name = "Pure Python Baseline (KMP Simulated)"
        
    def execute(self, dna_array: np.ndarray, target_array: np.ndarray) -> Tuple[int, float]:
        seq_len = len(dna_array)
        t_len = len(target_array)
        
        start_time = time.perf_counter()
        match_count = dna_array.tobytes().count(target_array.tobytes())
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return match_count, latency_ms

class GPUExecutionWorker:
    def __init__(self, threads_per_block: int = 512):
        self.threads_per_block = threads_per_block
        
    def calculate_grid_dim(self, total_elements: int) -> int:
        return (total_elements + self.threads_per_block - 1) // self.threads_per_block

    def execute(self, d_dna_array: Any, d_target_array: Any, seq_len: int, target_len: int) -> Tuple[int, float]:
        blocks = self.calculate_grid_dim(seq_len)
        d_match_counts = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        cuda.synchronize()
        start_time = time.perf_counter()
        
        fast_alignment_kernel[blocks, self.threads_per_block](  # type: ignore
            d_dna_array, d_target_array, d_match_counts, seq_len, target_len
        )
        
        cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        h_match_counts = d_match_counts.copy_to_host()
        total_matches = int(h_match_counts[0])
        del d_match_counts
        
        return total_matches, latency_ms

class MasterBenchmarkOrchestrator:
    def __init__(self):
        self.sys_profiler = SystemProfiler()
        self.cpu_worker = CPUExecutionWorker()
        self.gpu_worker = GPUExecutionWorker(threads_per_block=512)
        
        self.target_mutation = "GATCGATCGATC"
        self.h_target = np.ascontiguousarray(np.frombuffer(self.target_mutation.encode('ascii'), dtype=np.uint8))
        self.results: List[BenchmarkResult] = []
        self.report_dir = "reports/mid_project_review"
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_synthetic_workload(self, size_mb: float) -> np.ndarray:
        elements = int((size_mb * (1024 ** 2)) / np.uint8(0).itemsize)
        base_array = np.random.choice([65, 67, 71, 84], size=elements).astype(np.uint8)
        t_len = len(self.h_target)
        
        injection_points = [
            int(elements * 0.2), int(elements * 0.4), int(elements * 0.6), int(elements * 0.8)
        ]
        
        for pt in injection_points:
            if pt + t_len < elements:
                base_array[pt:pt+t_len] = self.h_target
                
        return np.ascontiguousarray(base_array)

    def export_json_report(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.report_dir, f"benchmark_{timestamp}.json")
        
        report_data = {
            "metadata": {
                "project": "GeneWeaver",
                "phase": "Mid-Project Review (Week 2)",
                "author": "Ayush Pandey",
                "hardware": self.sys_profiler.gpu_name
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)
        return report_path

    def build_dashboard_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=2)
        )
        layout["left_panel"].split_column(
            Layout(name="sys_info", size=8),
            Layout(name="current_status", ratio=1)
        )
        
        layout["header"].update(Panel(Text("GENEWEAVER // MID-PROJECT BENCHMARK (CPU vs GPU)", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
        layout["sys_info"].update(self.sys_profiler.get_panel())
        return layout

    def run_full_suite(self):
        console.clear()
        layout = self.build_dashboard_layout()
        test_sizes_mb = [10, 50, 100, 250, 500]
        
        results_table = Table(box=None, expand=True)
        results_table.add_column("Workload", style="cyan")
        results_table.add_column("CPU Time", style="yellow", justify="right")
        results_table.add_column("GPU Time", style="bold bright_green", justify="right")
        results_table.add_column("Matches", style="magenta", justify="right")
        results_table.add_column("SPEEDUP (x)", style="bold red", justify="right")

        status_table = Table(box=None, expand=True)
        status_table.add_column("Execution Phase", style="cyan")
        status_table.add_column("Status", style="bold")

        layout["right_panel"].update(Panel(results_table, title="[bold green]Head-to-Head Performance Matrix[/bold green]", border_style="green"))
        layout["current_status"].update(Panel(status_table, title="[bold magenta]Orchestrator Status[/bold magenta]", border_style="magenta"))
        layout["footer"].update(Panel(Text("INITIALIZING BENCHMARK SUITE...", style="bold yellow justify-left"), style="on black", border_style="yellow"))

        d_target = cuda.to_device(self.h_target)
        t_len = len(self.h_target)

        with Live(layout, refresh_per_second=10, screen=True):
            for size in test_sizes_mb:
                status_table.add_row(f"[{size}MB] Allocating RAM...", "[yellow]WORKING[/yellow]")
                layout["current_status"].update(Panel(status_table, title="[bold magenta]Orchestrator Status[/bold magenta]", border_style="magenta"))
                
                h_dna = self.generate_synthetic_workload(size)
                seq_len = len(h_dna)
                
                status_table.add_row(f"[{size}MB] Running CPU Baseline...", "[cyan]WORKING[/cyan]")
                layout["current_status"].update(Panel(status_table, title="[bold magenta]Orchestrator Status[/bold magenta]", border_style="magenta"))
                
                cpu_matches, cpu_lat = self.cpu_worker.execute(h_dna, self.h_target)
                cpu_throughput = (size) / (cpu_lat / 1000.0) if cpu_lat > 0 else 0
                
                status_table.add_row(f"[{size}MB] Pushing to VRAM...", "[magenta]WORKING[/magenta]")
                layout["current_status"].update(Panel(status_table, title="[bold magenta]Orchestrator Status[/bold magenta]", border_style="magenta"))
                d_dna = cuda.to_device(h_dna)
                
                status_table.add_row(f"[{size}MB] Running CUDA Kernel...", "[green]WORKING[/green]")
                layout["current_status"].update(Panel(status_table, title="[bold magenta]Orchestrator Status[/bold magenta]", border_style="magenta"))
                
                gpu_matches, gpu_lat = self.gpu_worker.execute(d_dna, d_target, seq_len, t_len)
                gpu_throughput = (size / 1024) / (gpu_lat / 1000.0) if gpu_lat > 0 else 0
                
                del d_dna
                
                speedup = cpu_lat / gpu_lat if gpu_lat > 0 else 0
                is_valid = (cpu_matches == gpu_matches)
                
                res = BenchmarkResult(
                    workload_mb=size,
                    cpu_latency_ms=cpu_lat,
                    gpu_latency_ms=gpu_lat,
                    cpu_throughput_mbs=cpu_throughput,
                    gpu_throughput_gbs=gpu_throughput,
                    speedup_multiplier=speedup,
                    matches_found=gpu_matches,
                    is_verified=is_valid,
                    timestamp=datetime.now().isoformat()
                )
                self.results.append(res)
                
                results_table.add_row(
                    f"{size} MB",
                    f"{cpu_lat:.2f} ms",
                    f"{gpu_lat:.3f} ms",
                    str(gpu_matches) if is_valid else "[red]ERR[/red]",
                    f"{speedup:.1f}x"
                )
                
                layout["right_panel"].update(Panel(results_table, title="[bold green]Head-to-Head Performance Matrix[/bold green]", border_style="green"))
                time.sleep(0.5)

            report_path = self.export_json_report()
            del d_target
            
            layout["footer"].update(Panel(Text(f"BENCHMARK COMPLETE // REPORT SAVED: {report_path} // PRESS CTRL+C TO EXIT", style="bold red justify-center"), style="on black", border_style="red"))
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    orchestrator = MasterBenchmarkOrchestrator()
    orchestrator.run_full_suite()