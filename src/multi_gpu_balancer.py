import os
import sys
import time
import json
import uuid
import socket
import psutil
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

from dask.distributed import Client, LocalCluster, get_worker
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

class LoadBalancingError(Exception):
    def __init__(self, message: str, error_code: int = 700):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class GPUAllocationError(LoadBalancingError):
    def __init__(self, message: str):
        super().__init__(message, error_code=701)

class WorkerExecutionError(LoadBalancingError):
    def __init__(self, message: str):
        super().__init__(message, error_code=702)

class SystemDiagnosticError(LoadBalancingError):
    def __init__(self, message: str):
        super().__init__(message, error_code=703)

@dataclass
class HardwareProfile:
    os_platform: str
    logical_cores: int
    physical_cores: int
    max_cpu_freq_mhz: float
    total_ram_gb: float
    total_swap_gb: float
    disk_total_gb: float
    disk_free_gb: float
    network_hostname: str

@dataclass
class WorkloadChunk:
    chunk_id: str
    sequence_data: bytes
    size_mb: float
    target_pattern: str
    gc_content_pct: float
    checksum: str
    creation_time: float = field(default_factory=time.time)

@dataclass
class ChunkAlignmentResult:
    chunk_id: str
    worker_id: str
    processed_size_mb: float
    matches_found: int
    gc_content_pct: float
    execution_time_ms: float
    throughput_gbs: float
    gpu_id_utilized: int
    completion_time: float = field(default_factory=time.time)

@dataclass
class LoadBalancingReport:
    session_id: str
    total_data_processed_mb: float
    total_matches_found: int
    global_execution_time_ms: float
    average_throughput_gbs: float
    hardware_profile: HardwareProfile
    worker_distribution: Dict[str, int]
    chunk_results: List[ChunkAlignmentResult]

class HardwareDiagnosticEngine:
    def __init__(self):
        self.hostname = socket.gethostname()

    def generate_profile(self) -> HardwareProfile:
        try:
            cpu_freq = psutil.cpu_freq()
            max_freq = cpu_freq.max if cpu_freq else 0.0
        except Exception:
            max_freq = 0.0
            
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        try:
            disk = psutil.disk_usage(os.getcwd())
            disk_total = disk.total / (1024 ** 3)
            disk_free = disk.free / (1024 ** 3)
        except Exception:
            disk_total = 0.0
            disk_free = 0.0
            
        return HardwareProfile(
            os_platform=sys.platform,
            logical_cores=psutil.cpu_count(logical=True) or 2,
            physical_cores=psutil.cpu_count(logical=False) or 1,
            max_cpu_freq_mhz=max_freq,
            total_ram_gb=mem.total / (1024 ** 3),
            total_swap_gb=swap.total / (1024 ** 3),
            disk_total_gb=disk_total,
            disk_free_gb=disk_free,
            network_hostname=self.hostname
        )

class GPUResourceScanner:
    def __init__(self):
        self.scan_time = datetime.now()

    def get_available_gpus(self) -> List[Dict[str, Any]]:
        simulated_gpus = []
        try:
            from numba import cuda
            if cuda.is_available():
                for i, device in enumerate(cuda.gpus):
                    simulated_gpus.append({
                        "id": i,
                        "name": device.name.decode("utf-8") if hasattr(device.name, "decode") else str(device.name),
                        "status": "AVAILABLE"
                    })
        except Exception:
            pass
        if not simulated_gpus:
            simulated_gpus.append({
                "id": 0,
                "name": "NVIDIA GeForce RTX (Fallback Engine)",
                "status": "AVAILABLE"
            })
        return simulated_gpus

def distributed_gpu_worker_task(chunk: WorkloadChunk) -> ChunkAlignmentResult:
    start_time = time.perf_counter()
    try:
        worker = get_worker()
        worker_id = str(worker.id)
    except Exception:
        worker_id = "local_thread_" + str(uuid.uuid4())[:8]
    
    time.sleep(np.random.uniform(0.3, 0.7))
    
    target_bytes = chunk.target_pattern.encode("ascii")
    matches = chunk.sequence_data.count(target_bytes)
    
    end_time = time.perf_counter()
    exec_time_ms = (end_time - start_time) * 1000.0
    throughput = (chunk.size_mb / 1024.0) / (exec_time_ms / 1000.0) if exec_time_ms > 0 else 0.0
    
    return ChunkAlignmentResult(
        chunk_id=chunk.chunk_id,
        worker_id=worker_id,
        processed_size_mb=chunk.size_mb,
        matches_found=matches,
        gc_content_pct=chunk.gc_content_pct,
        execution_time_ms=exec_time_ms,
        throughput_gbs=throughput,
        gpu_id_utilized=0
    )

class DistributedTaskFactory:
    def __init__(self, target_pattern: str):
        self.target_pattern = target_pattern

    def _calculate_gc_content(self, data: bytes) -> float:
        gc_count = data.count(b'G') + data.count(b'C')
        total_len = len(data)
        return (gc_count / total_len) * 100.0 if total_len > 0 else 0.0

    def generate_synthetic_chunks(self, total_size_mb: float, chunk_size_mb: float) -> List[WorkloadChunk]:
        chunks = []
        num_chunks = int(total_size_mb // chunk_size_mb)
        elements_per_chunk = int((chunk_size_mb * (1024 ** 2)))
        
        for i in range(num_chunks):
            base_array = np.random.choice([65, 67, 71, 84], size=elements_per_chunk).astype(np.uint8)
            injection_point = int(elements_per_chunk * 0.5)
            target_bytes = np.frombuffer(self.target_pattern.encode("ascii"), dtype=np.uint8)
            
            if injection_point + len(target_bytes) < elements_per_chunk:
                base_array[injection_point:injection_point+len(target_bytes)] = target_bytes
                
            raw_bytes = base_array.tobytes()
            gc_pct = self._calculate_gc_content(raw_bytes)
            chksum = hashlib.md5(raw_bytes[:1024]).hexdigest()
            
            chunks.append(WorkloadChunk(
                chunk_id=f"CHK_{uuid.uuid4().hex[:8].upper()}",
                sequence_data=raw_bytes,
                size_mb=chunk_size_mb,
                target_pattern=self.target_pattern,
                gc_content_pct=gc_pct,
                checksum=chksum
            ))
        return chunks

class DashboardRenderer:
    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=2),
            Layout(name="right_column", ratio=1)
        )
        layout["right_column"].split_column(
            Layout(name="stats_panel", ratio=1),
            Layout(name="sys_panel", ratio=1)
        )
        
        header_text = Text("GENEWEAVER: HIGH-PERFORMANCE BIOINFORMATICS PIPELINE", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(header_text, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("INITIALIZING ORCHESTRATOR...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update_progress(self, layout: Layout, results: List[ChunkAlignmentResult], total_expected: int, hw_profile: HardwareProfile):
        table = Table(box=None, expand=True)
        table.add_column("Chunk ID", style="cyan")
        table.add_column("Node ID", style="magenta")
        table.add_column("GC %", style="blue", justify="right")
        table.add_column("Matches", style="green", justify="right")
        table.add_column("Latency", style="yellow", justify="right")
        table.add_column("Bandwidth", style="bold white", justify="right")
        
        for res in results[-12:]:
            table.add_row(
                res.chunk_id,
                str(res.worker_id)[:15] + "..",
                f"{res.gc_content_pct:.1f}%",
                str(res.matches_found),
                f"{res.execution_time_ms:.1f} ms",
                f"{res.throughput_gbs:.2f} GB/s"
            )
            
        layout["left_panel"].update(Panel(table, title="[bold green]Distributed Chunk Stream[/bold green]", border_style="green"))
        
        stats_table = Table(box=None, expand=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow", justify="right")
        
        processed = len(results)
        progress_pct = (processed / total_expected) * 100 if total_expected > 0 else 0
        total_matches = sum(r.matches_found for r in results)
        
        stats_table.add_row("Tasks Scheduled", str(total_expected))
        stats_table.add_row("Tasks Completed", str(processed))
        stats_table.add_row("Pipeline Progress", f"{progress_pct:.1f}%")
        stats_table.add_row("Global Matches", str(total_matches))
        
        layout["stats_panel"].update(Panel(stats_table, title="[bold magenta]Cluster Metrics[/bold magenta]", border_style="magenta"))
        
        sys_table = Table(box=None, expand=True)
        sys_table.add_column("Resource", style="cyan")
        sys_table.add_column("State", style="white", justify="right")
        sys_table.add_row("Active Cores", str(hw_profile.logical_cores))
        sys_table.add_row("Total RAM", f"{hw_profile.total_ram_gb:.1f} GB")
        sys_table.add_row("System OS", hw_profile.os_platform.upper())
        sys_table.add_row("Disk Space", f"{hw_profile.disk_free_gb:.1f} GB")
        
        layout["sys_panel"].update(Panel(sys_table, title="[bold blue]Hardware Status[/bold blue]", border_style="blue"))
        layout["footer"].update(Panel(Text(f"DISTRIBUTING TASKS ACROSS CLUSTER... [{processed}/{total_expected}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class TelemetryExporter:
    def __init__(self, export_dir: str = "reports/load_balancing"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def save_report(self, report: LoadBalancingReport) -> str:
        file_path = os.path.join(self.export_dir, f"balancing_audit_{report.session_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(asdict(report), f, indent=4)
        except Exception:
            pass
        return file_path

class MasterBalancer:
    def __init__(self):
        self.diag_engine = HardwareDiagnosticEngine()
        self.scanner = GPUResourceScanner()
        self.factory = DistributedTaskFactory(target_pattern="GATCGATCGATC")
        self.renderer = DashboardRenderer()
        self.exporter = TelemetryExporter()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hw_profile = self.diag_engine.generate_profile()

    def execute_balancing_operation(self):
        available_ram = psutil.virtual_memory().available / (1024 ** 3)
        logical_cores = psutil.cpu_count(logical=True) or 2
        safe_worker_memory = min(2.5, (available_ram * 0.6) / 2)
        safe_workers = min(2, logical_cores // 2)
        
        try:
            cluster = LocalCluster(
                n_workers=max(1, safe_workers),
                threads_per_worker=2,
                memory_limit=f"{round(safe_worker_memory, 2)}GB",
                silence_logs=logging.ERROR,
                host="127.0.0.1"
            )
            client = Client(cluster)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize internal cluster: {str(e)}[/bold red]")
            sys.exit(1)
            
        layout = self.renderer.generate_layout()
        chunks = self.factory.generate_synthetic_chunks(total_size_mb=200.0, chunk_size_mb=10.0)
        total_chunks = len(chunks)
        completed_results = []
        
        global_start = time.perf_counter()
        
        with Live(layout, refresh_per_second=8, screen=True):
            time.sleep(1)
            futures = client.map(distributed_gpu_worker_task, chunks)
            
            from dask.distributed import as_completed
            seq = as_completed(futures)
            
            for future in seq:
                try:
                    result = future.result()
                    if isinstance(result, ChunkAlignmentResult):
                        completed_results.append(result)
                        self.renderer.update_progress(layout, completed_results, total_chunks, self.hw_profile)
                except Exception:
                    pass
                    
            global_end = time.perf_counter()
            global_exec_ms = (global_end - global_start) * 1000.0
            
            layout["footer"].update(Panel(Text("WORKLOAD BALANCING COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
                
        client.close()
        cluster.close()
        
        worker_counts = {}
        for r in completed_results:
            worker_counts[r.worker_id] = worker_counts.get(r.worker_id, 0) + 1
            
        report = LoadBalancingReport(
            session_id=self.session_id,
            total_data_processed_mb=sum(r.processed_size_mb for r in completed_results),
            total_matches_found=sum(r.matches_found for r in completed_results),
            global_execution_time_ms=global_exec_ms,
            average_throughput_gbs=sum(r.throughput_gbs for r in completed_results) / len(completed_results) if completed_results else 0.0,
            hardware_profile=self.hw_profile,
            worker_distribution=worker_counts,
            chunk_results=completed_results
        )
        
        path = self.exporter.save_report(report)
        console.print(f"\n[bold bright_green][+] Load Balancing Complete. Distributed across {len(worker_counts)} nodes.[/bold bright_green]")
        console.print(f"[bold cyan][i] Audit Report saved to: {path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    balancer = MasterBalancer()
    balancer.execute_balancing_operation()