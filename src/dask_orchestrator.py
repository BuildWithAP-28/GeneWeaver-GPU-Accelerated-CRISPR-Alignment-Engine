import os
import sys
import json
import time
import socket
import psutil
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Set
from logging.handlers import RotatingFileHandler

from dask.distributed import Client, LocalCluster, performance_report
from dask.distributed import get_worker

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
from rich.live import Live

console = Console()

class DaskOrchestratorError(Exception):
    def __init__(self, message: str, code: int = 600):
        super().__init__(message)
        self.code = code
        self.timestamp = time.time()

class ClusterInitializationError(DaskOrchestratorError):
    def __init__(self, message: str):
        super().__init__(message, code=601)

class WorkerAllocationError(DaskOrchestratorError):
    def __init__(self, message: str, requested: int, available: int):
        super().__init__(message, code=602)
        self.requested = requested
        self.available = available

class DiagnosticPayloadError(DaskOrchestratorError):
    def __init__(self, message: str, task_id: str):
        super().__init__(message, code=603)
        self.task_id = task_id

@dataclass
class ClusterConfig:
    max_workers: int
    threads_per_worker: int
    memory_limit_gb: float
    dashboard_port: int
    temp_dir: str
    network_interface: str

@dataclass
class WorkerTelemetry:
    worker_id: str
    host: str
    cpu_percent: float
    memory_used_mb: float
    memory_limit_mb: float
    tasks_executing: int
    tasks_completed: int
    latency_ms: float

@dataclass
class ClusterHealthReport:
    timestamp: str
    status: str
    active_workers: int
    total_cpu_cores: int
    total_memory_gb: float
    dashboard_url: str
    worker_metrics: List[WorkerTelemetry] = field(default_factory=list)

def setup_dask_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "dask_orchestrator.log"
    logger = logging.getLogger("GeneWeaverDask")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"))
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    return logger

logger = setup_dask_logging()

class SystemResourceValidator:
    def __init__(self, required_memory_gb: float, required_cores: int):
        self.required_memory_gb = required_memory_gb
        self.required_cores = required_cores
        self.sys_memory = psutil.virtual_memory()
        self.sys_cores = psutil.cpu_count(logical=True) or 2

    def validate_resources(self) -> Tuple[bool, str]:
        available_memory_gb = self.sys_memory.available / (1024 ** 3)
        if available_memory_gb < self.required_memory_gb:
            return False, f"Insufficient Memory. Required: {self.required_memory_gb}GB, Available: {available_memory_gb:.2f}GB"
        if self.sys_cores < self.required_cores:
            return False, f"Insufficient CPU Cores. Required: {self.required_cores}, Available: {self.sys_cores}"
        return True, "System resources validated successfully."

    def get_system_snapshot(self) -> Dict[str, Any]:
        return {
            "total_memory_gb": self.sys_memory.total / (1024 ** 3),
            "available_memory_gb": self.sys_memory.available / (1024 ** 3),
            "cpu_cores_logical": self.sys_cores,
            "cpu_cores_physical": psutil.cpu_count(logical=False) or 1,
            "current_cpu_utilization": psutil.cpu_percent(interval=0.5)
        }

def diagnostic_matrix_multiplication(size: int) -> float:
    import numpy as np
    start_t = time.perf_counter()
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    _ = np.dot(matrix_a, matrix_b)
    return (time.perf_counter() - start_t) * 1000.0

class DaskDistributedOrchestrator:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.validator = SystemResourceValidator(
            required_memory_gb=self.config.max_workers * self.config.memory_limit_gb,
            required_cores=self.config.max_workers * self.config.threads_per_worker
        )
        self.cluster: Optional[LocalCluster] = None
        self.client: Optional[Client] = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_running = False

    def pre_flight_checks(self) -> None:
        is_valid, message = self.validator.validate_resources()
        if not is_valid:
            logger.error(message)
            raise ClusterInitializationError(message)
        logger.info(message)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)

    def initialize_cluster(self) -> None:
        self.pre_flight_checks()
        memory_limit_str = f"{self.config.memory_limit_gb}GB"
        try:
            cluster = LocalCluster(
                n_workers=self.config.max_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=memory_limit_str,
                dashboard_address=f":{self.config.dashboard_port}",
                local_directory=self.config.temp_dir,
                silence_logs=logging.ERROR,
                host="127.0.0.1"
            )
            self.cluster = cluster
            self.client = Client(cluster)
            self.is_running = True
            logger.info(f"Dask LocalCluster started at {cluster.dashboard_link}")
        except Exception as e:
            raise ClusterInitializationError(f"Failed to bind Dask cluster: {str(e)}")

    def scale_cluster(self, num_workers: int) -> None:
        cluster = self.cluster
        if not self.is_running or not cluster:
            raise ClusterInitializationError("Cluster is not running.")
        sys_cores = psutil.cpu_count(logical=True) or 2
        if num_workers > sys_cores:
            raise WorkerAllocationError("Requested workers exceed system cores.", num_workers, sys_cores)
        cluster.scale(num_workers)
        logger.info(f"Cluster scaling requested: {num_workers} workers.")

    def run_diagnostic_payload(self) -> Dict[str, Any]:
        if not self.client:
            raise ClusterInitializationError("Client not connected.")
        worker_info = self.client.scheduler_info().get("workers", {})
        if not worker_info:
            return {"status": "FAILED", "reason": "No active workers available for diagnostic payload."}
        payload_futures = []
        for _ in range(len(worker_info) * 2):
            future = self.client.submit(diagnostic_matrix_multiplication, 1000)
            payload_futures.append(future)
        
        gathered_data = self.client.gather(payload_futures)
        results = [float(x) for x in gathered_data] if isinstance(gathered_data, list) else []
        
        avg_latency = sum(results) / len(results) if results else 0.0
        return {
            "status": "SUCCESS",
            "tasks_executed": len(results),
            "average_task_latency_ms": avg_latency,
            "max_task_latency_ms": max(results) if results else 0.0,
            "min_task_latency_ms": min(results) if results else 0.0
        }

    def generate_health_report(self) -> ClusterHealthReport:
        client = self.client
        cluster = self.cluster
        if not client or not cluster:
            raise ClusterInitializationError("Cannot generate health report for offline cluster.")
        scheduler_info = client.scheduler_info()
        workers = scheduler_info.get("workers", {})
        telemetry_list = []
        total_mem = 0.0
        total_threads = 0
        for addr, w_info in workers.items():
            metrics = w_info.get("metrics", {})
            mem_limit_mb = w_info.get("memory_limit", 0) / (1024 ** 2)
            mem_used_mb = metrics.get("memory", 0) / (1024 ** 2)
            cpu_pct = metrics.get("cpu", 0.0)
            threads = w_info.get("nthreads", 0)
            total_mem += mem_limit_mb
            total_threads += threads
            wt = WorkerTelemetry(
                worker_id=w_info.get("name", addr),
                host=w_info.get("host", "unknown"),
                cpu_percent=cpu_pct,
                memory_used_mb=mem_used_mb,
                memory_limit_mb=mem_limit_mb,
                tasks_executing=metrics.get("executing", 0),
                tasks_completed=w_info.get("metrics", {}).get("task_counts", {}).get("memory", 0),
                latency_ms=w_info.get("latency", 0.0) * 1000.0
            )
            telemetry_list.append(wt)
        report = ClusterHealthReport(
            timestamp=datetime.now().isoformat(),
            status="HEALTHY" if len(telemetry_list) > 0 else "DEGRADED",
            active_workers=len(telemetry_list),
            total_cpu_cores=total_threads,
            total_memory_gb=total_mem / 1024.0,
            dashboard_url=cluster.dashboard_link or "Unknown",
            worker_metrics=telemetry_list
        )
        return report

    def export_telemetry_to_disk(self, report: ClusterHealthReport, export_dir: str = "reports/dask_telemetry") -> str:
        out_path = Path(export_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"cluster_health_{self.session_id}.json"
        with open(file_path, "w") as f:
            json.dump(asdict(report), f, indent=4)
        return str(file_path)

    def graceful_shutdown(self) -> None:
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        self.is_running = False
        logger.info("Dask Orchestrator safely terminated all worker processes.")

def render_dashboard_ui(orchestrator: DaskDistributedOrchestrator) -> None:
    report = orchestrator.generate_health_report()
    sys_snapshot = orchestrator.validator.get_system_snapshot()
    grid_table = Table(title="Dask Orchestrator Global State", title_style="bold cyan", expand=True)
    grid_table.add_column("Metric", style="bold yellow")
    grid_table.add_column("Value", style="bold white")
    grid_table.add_row("Session ID", orchestrator.session_id)
    grid_table.add_row("Cluster Status", f"[bold green]{report.status}[/bold green]")
    grid_table.add_row("Active Worker Nodes", str(report.active_workers))
    grid_table.add_row("Compute Cores Allocated", str(report.total_cpu_cores))
    grid_table.add_row("Total Worker Memory", f"{report.total_memory_gb:.2f} GB")
    grid_table.add_row("System CPU Utilization", f"{sys_snapshot['current_cpu_utilization']}%")
    grid_table.add_row("System Available RAM", f"{sys_snapshot['available_memory_gb']:.2f} GB")
    console.print(grid_table)
    console.print("")
    worker_table = Table(title="Worker Node Telemetry", title_style="bold green", expand=True)
    worker_table.add_column("Worker ID", style="cyan")
    worker_table.add_column("Host", style="magenta")
    worker_table.add_column("CPU %", justify="right", style="yellow")
    worker_table.add_column("RAM Usage", justify="right", style="red")
    worker_table.add_column("Tasks", justify="right", style="blue")
    worker_table.add_column("Latency", justify="right", style="green")
    for w in report.worker_metrics:
        worker_table.add_row(
            str(w.worker_id),
            w.host,
            f"{w.cpu_percent:.1f}%",
            f"{w.memory_used_mb:.1f} / {w.memory_limit_mb:.1f} MB",
            str(w.tasks_executing),
            f"{w.latency_ms:.2f} ms"
        )
    console.print(worker_table)
    console.print("")
    console.print(Panel(f"Live Dashboard URL: [bold blue][link={report.dashboard_url}]{report.dashboard_url}[/link][/bold blue]", border_style="cyan"))

def execute_dask_pipeline() -> None:
    # 1. Check actual system limits to prevent Out-Of-Memory crashes
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    logical_cores = psutil.cpu_count(logical=True) or 2
    
    # 2. Safely allocate only 60% of available RAM, capped at maximum 2.5GB per worker
    safe_worker_memory = min(2.5, (available_ram * 0.6) / 2)
    safe_workers = min(2, logical_cores // 2)
    
    config = ClusterConfig(
        max_workers=max(1, safe_workers), 
        threads_per_worker=2,
        memory_limit_gb=round(safe_worker_memory, 2),
        dashboard_port=8787,
        temp_dir="data/dask_spill",
        network_interface="127.0.0.1"
    )
    orchestrator = DaskDistributedOrchestrator(config)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan"),
            console=console
        ) as progress:
            t1 = progress.add_task("[cyan]Validating System Resources...", total=100)
            time.sleep(0.5)
            progress.update(t1, advance=100)
            t2 = progress.add_task("[cyan]Booting Dask LocalCluster...", total=100)
            orchestrator.initialize_cluster()
            progress.update(t2, advance=100)
            t3 = progress.add_task("[cyan]Connecting Scheduler Client...", total=100)
            time.sleep(0.5)
            progress.update(t3, advance=100)
            t4 = progress.add_task("[cyan]Executing Diagnostic Payload...", total=100)
            diag_res = orchestrator.run_diagnostic_payload()
            progress.update(t4, advance=100)
        console.clear()
        console.print(f"[bold bright_green]Diagnostic Payload Success! Processed {diag_res.get('tasks_executed', 0)} tasks at {diag_res.get('average_task_latency_ms', 0):.2f}ms latency.[/bold bright_green]\n")
        render_dashboard_ui(orchestrator)
        report = orchestrator.generate_health_report()
        export_path = orchestrator.export_telemetry_to_disk(report)
        console.print(f"[dim]Telemetry exported successfully to {export_path}[/dim]\n")
        console.print("[bold yellow]Dask Orchestrator is running in background. Press CTRL+C to terminate.[/bold yellow]")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[bold red]Shutdown signal received. Terminating cluster safely...[/bold red]")
        orchestrator.graceful_shutdown()
        sys.exit(0)
    except Exception as e:
        logger.exception("Critical orchestration failure.")
        orchestrator.graceful_shutdown()
        sys.exit(1)

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    console.print(Panel("[bold bright_magenta]OMNIAI GENEWEAVER // PHASE 3: DISTRIBUTED SCALING ORCHESTRATOR[/bold bright_magenta]", style="on black", border_style="bright_magenta"))
    execute_dask_pipeline()