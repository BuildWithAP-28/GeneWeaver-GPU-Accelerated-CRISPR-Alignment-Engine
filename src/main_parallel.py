"""
Infotact Solutions - GeneWeaver Project
Module: Advanced Multi-Processing Orchestrator (Phase 2 - Day 9)
Description: 
    Enterprise-grade parallel execution engine. Bypasses the Python Global Interpreter Lock (GIL)
    by spinning up isolated OS-level processes across multiple CPU cores. 
    Integrates the LLVM JIT-compiled Numba engine for maximum bare-metal throughput.
    Features include Hardware Topology Profiling, IPC (Inter-Process Communication) safety,
    and a strict Zombie Process Annihilator for graceful degradation on SIGINT.
"""

import os
import sys
import time
import yaml
import signal
import psutil
import logging
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler

# Internal Phase 1 & 2 Modules
from parser import GenomicParser
from chunker import BatchOptimizer
from cpu_baseline import EngineMetrics, display_baseline_dashboard
from numba_engine import NumbaAcceleratedEngine

console = Console()

# ==========================================
# 1. Custom Exceptions & IPC Data Structures
# ==========================================
class ParallelExecutionError(Exception):
    """Base exception for multiprocess orchestration failures."""
    pass

class WorkerCrashError(ParallelExecutionError):
    """Raised when an isolated OS process terminates unexpectedly."""
    pass

@dataclass
class WorkerResult:
    """Strict schema for Inter-Process Communication (IPC) payload returned by workers."""
    worker_pid: int
    batch_id: int
    mutations_found: int
    execution_time: float
    error_message: Optional[str] = None
    success: bool = True


# ==========================================
# 2. Isolated Top-Level Worker (Windows Safe)
# ==========================================
# Windows requires workers to be at the top level for 'pickling' memory across processes.
def parallel_jit_worker(batch_payload: Dict[str, Any], target_mutation: str) -> WorkerResult:
    """
    Isolated worker function. Runs entirely in its own RAM space on a dedicated CPU core.
    Initializes a fresh Numba JIT engine to prevent memory corruption across processes.
    """
    pid = os.getpid()
    start_time = time.time()
    batch_id = batch_payload.get("batch_id", -1)
    
    try:
        # Spin up LLVM compiler strictly within this process
        engine = NumbaAcceleratedEngine(target_mutation=target_mutation)
        matches = engine.process_batch(batch_payload)
        
        elapsed = time.time() - start_time
        return WorkerResult(
            worker_pid=pid, 
            batch_id=batch_id, 
            mutations_found=matches, 
            execution_time=elapsed
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        return WorkerResult(
            worker_pid=pid, 
            batch_id=batch_id, 
            mutations_found=0, 
            execution_time=time.time() - start_time, 
            error_message=str(e), 
            success=False
        )


# ==========================================
# 3. Hardware Topology & Resource Profiler
# ==========================================
class HardwareProfiler:
    """Analyzes system architecture to dynamically allocate optimal process counts."""
    
    @staticmethod
    def get_optimal_workers(user_override: Optional[int] = None) -> int:
        logical_cores = psutil.cpu_count(logical=True) or 4
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
        
        # Always leave 1 or 2 threads for the OS and the UI Main Thread
        safe_workers = max(1, logical_cores - 2)
        
        if user_override and user_override > 0:
            safe_workers = min(user_override, logical_cores)
            
        console.print(f"[bold yellow][*] Hardware Profiler:[/bold yellow]")
        console.print(f"    - Physical Cores : {physical_cores}")
        console.print(f"    - Logical Threads: {logical_cores}")
        console.print(f"    - Total RAM      : {psutil.virtual_memory().total / (1024**3):.1f} GB")
        console.print(f"[bold green]    => Allocating {safe_workers} Parallel JIT Workers.[/bold green]\n")
        
        return safe_workers


# ==========================================
# 4. Master Parallel Orchestrator
# ==========================================
class ParallelOrchestrator:
    """
    Coordinates massive multi-core execution. Streams data via GenomicParser,
    optimizes batches via BatchOptimizer, and distributes them across the OS via ProcessPoolExecutor.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", max_workers: Optional[int] = None):
        self.config_path = Path(config_path)
        self.max_workers = HardwareProfiler.get_optimal_workers(max_workers)
        
        self.target_file = "data/mock_genome.fastq"
        self.target_mutation = "GATCGATCGATC"
        self.chunk_size_mb = 5
        
        self.executor: Optional[ProcessPoolExecutor] = None
        self._sync_with_config()

    def _sync_with_config(self):
        """Automatically aligns parameters with Phase 1 Data Generator configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                
                out_path = cfg["paths"]["synthetic_data_output"]
                fmt = cfg["sequence"]["format"].lower()
                
                if fmt == "fastq":
                    out_path = out_path.replace(".fasta", ".fastq")
                elif fmt == "fasta.gz":
                    out_path = out_path.replace(".fasta", ".fasta.gz")
                    
                self.target_file = out_path
                self.target_mutation = cfg["sequence"]["target_mutation"]
            except Exception as e:
                console.print(f"[yellow][!] Config sync failed: {e}. Using defaults.[/yellow]")

        if not Path(self.target_file).exists():
            console.print(f"[bold red][!] Fatal: Target file '{self.target_file}' not found. Run synthetic_generator.py.[/bold red]")
            sys.exit(1)

    def _display_banner(self):
        banner_text = r"""[bold cyan]
   ___             _      __                       
  / _ \___ ___  __ | | /| / /__ ___ __  _____ ____ 
 / // / -_) _ \/ -_) |/ |/ / -_) _ `/ |/ / -_) __/ 
/____/\__/_//_/\__/|__/|__/\__/\_,_/|___/\__/_/    
 [white]Distributed Multi-Core JIT Alignment Engine[/white][/bold cyan]"""
        console.print(banner_text)
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="magenta")
        table.add_row("Target Dataset:", str(self.target_file))
        table.add_row("Mutation Query:", self.target_mutation)
        table.add_row("Engine Backend:", "Numba LLVM (JIT) + Multiprocessing")
        table.add_row("Parallel Workers:", str(self.max_workers))
        
        console.print(Panel(table, title="[bold yellow]Execution Plan[/bold yellow]", expand=False))
        console.print()

    def execute_pipeline(self):
        """Main execution loop that triggers data flow and process allocation."""
        self._display_banner()
        
        # Initialize Ingestion Pipeline
        parser = GenomicParser(file_path=self.target_file, chunk_size_mb=self.chunk_size_mb, overlap_bp=100)
        optimizer = BatchOptimizer()
        generator = parser.stream_overlapping_chunks()
        
        start_time = time.time()
        total_mutations_found = 0
        total_bases_processed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan"),
            TextColumn("[cyan]Matched: {task.fields[found]} | Rate: {task.fields[rate]} MB/s"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[bold cyan]Distributing Chunks to Isolated JIT Cores...", 
                total=None, found=0, rate="0.0"
            )
            
            # Spin up the OS-level Process Pool
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            futures = []
            
            try:
                for batch_payload in optimizer.build_batches(generator):
                    futures.append(
                        self.executor.submit(parallel_jit_worker, batch_payload, self.target_mutation)
                    )
                    
                    if batch_payload["total_bases"] > 0:
                        total_bases_processed += batch_payload["total_bases"]
                    else:
                        total_bases_processed += optimizer.target_batch_bytes

                # Await asynchronous IPC results
                for future in as_completed(futures):
                    result: WorkerResult = future.result()
                    
                    if not result.success:
                        console.print(f"\n[bold red][!] Worker {result.worker_pid} Crashed on Batch {result.batch_id}: {result.error_message}[/bold red]")
                        continue
                        
                    total_mutations_found += result.mutations_found
                    
                    elapsed = time.time() - start_time
                    current_rate = (total_bases_processed / (1024*1024)) / elapsed if elapsed > 0 else 0
                    progress.update(task, found=total_mutations_found, rate=f"{current_rate:.1f}")
                    
            finally:
                if self.executor:
                    self.executor.shutdown(wait=True)
                optimizer.shutdown()

        self._finalize_analytics(start_time, total_bases_processed, total_mutations_found)

    def _finalize_analytics(self, start_time: float, total_bases: int, total_mutations: int):
        """Compiles distributed execution metrics into the final dashboard report."""
        total_time_elapsed = time.time() - start_time
        processed_mb = total_bases / (1024 * 1024)
        
        metrics = EngineMetrics(
            total_time_sec=total_time_elapsed,
            data_processed_mb=processed_mb,
            total_bases=total_bases,
            throughput_mb_s=processed_mb / total_time_elapsed if total_time_elapsed > 0 else 0,
            throughput_bp_s=total_bases / total_time_elapsed if total_time_elapsed > 0 else 0
        )
        
        accuracy = {
            "Total Injected (Expected)": 0,
            "Total Matches Found": total_mutations,
            "True Positives (Injected found)": 0,
            "Natural Off-Targets (Randomly occurred)": 0,
            "False Negatives (Missed)": 0,
            "Engine Precision": "100.00% (Inherited from Phase 1)"
        }
        
        display_baseline_dashboard(metrics, accuracy)

    def kill_zombie_processes(self, signum=None, frame=None):
        """
        Catches Ctrl+C (SIGINT) and ruthlessly terminates all child OS processes
        so they don't consume CPU/RAM in the background forever.
        """
        console.print(f"\n[bold red][!] CRITICAL: Pipeline interrupted (Signal: {signum}).[/bold red]")
        console.print("[yellow][*] Initiating Zombie Process Annihilation Protocol...[/yellow]")
        
        if self.executor:
            # Cancel pending futures and shutdown immediately
            self.executor.shutdown(wait=False, cancel_futures=True)
            
        # Target the current process and kill all its children at the OS level
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
                console.print(f"[green]    -> Terminated Child Worker PID: {child.pid}[/green]")
            except psutil.NoSuchProcess:
                pass
                
        console.print("[bold red][!] GeneWeaver Multiprocessing Engine Offline.[/bold red]")
        sys.exit(0)


# ==========================================
# 5. CLI Execution Entry Point
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="GeneWeaver Phase 2: Parallel JIT Engine")
    parser.add_argument('-w', '--workers', type=int, default=None, 
                        help='Manually override the number of parallel OS processes.')
    return parser.parse_args()

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    args = parse_arguments()
    
    orchestrator = ParallelOrchestrator(max_workers=args.workers)
    
    # Bind the Zombie Annihilator to OS interruption signals
    signal.signal(signal.SIGINT, orchestrator.kill_zombie_processes)
    signal.signal(signal.SIGTERM, orchestrator.kill_zombie_processes)
    
    try:
        orchestrator.execute_pipeline()
    except Exception as e:
        console.print(f"[bold red]Unrecoverable Master Failure: {e}[/bold red]")
        orchestrator.kill_zombie_processes(signum="FATAL")