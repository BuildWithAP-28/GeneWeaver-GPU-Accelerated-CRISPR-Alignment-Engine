"""
Infotact Solutions - GeneWeaver Project
Module: Master Orchestrator & CLI Entry Point (Phase 1)
Description: 
    Enterprise-grade main execution script for the GeneWeaver Pipeline.
    Integrates Pre-Flight System Audits, advanced state-machine tracking, 
    strict type-hinted initialization, pipeline parallelism, and multi-signal 
    graceful degradation (SIGINT/SIGTERM) to protect OS memory.
"""

import os
import gc
import sys
import time
import signal
import psutil
import shutil
import logging
import argparse
import platform
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

# GeneWeaver Internal Modules
from parser import GenomicParser
from chunker import BatchOptimizer, SystemMonitor
from cpu_baseline import CPUBaselineEngine, EngineMetrics, display_baseline_dashboard

console = Console()
logger = logging.getLogger("GeneWeaverMain")

# ==========================================
# Application State Machine
# ==========================================
class PipelineState(Enum):
    """Tracks the exact state of the pipeline for safe interruption handling."""
    INITIALIZING = "INITIALIZING"
    PRE_FLIGHT_CHECKS = "PRE_FLIGHT_CHECKS"
    STREAMING_DATA = "STREAMING_DATA"
    ALIGNING_SEQUENCES = "ALIGNING_SEQUENCES"
    EXPORTING_REPORTS = "EXPORTING_REPORTS"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    COMPLETED = "COMPLETED"


# ==========================================
# Custom Orchestrator Exceptions
# ==========================================
class OrchestratorError(Exception):
    """Base exception for main orchestrator failures."""
    pass

class PreFlightCheckFailed(OrchestratorError):
    """Raised when the system does not meet hardware or software requirements."""
    pass


# ==========================================
# Pre-Flight System Auditor
# ==========================================
class SystemAuditor:
    """
    Executes a stringent series of hardware and software checks before 
    allowing the GeneWeaver pipeline to consume massive memory buffers.
    """
    def __init__(self, required_ram_mb: int = 1000, required_disk_mb: int = 500):
        self.required_ram = required_ram_mb
        self.required_disk = required_disk_mb

    def audit_environment(self) -> bool:
        """Runs the full diagnostic suite."""
        console.print("[bold cyan][*] Running Pre-Flight System Diagnostics...[/bold cyan]")
        self._check_python_version()
        self._check_available_ram()
        self._check_disk_space()
        console.print("[bold green][+] Pre-Flight Checks Passed. Hardware Environment Stable.[/bold green]\n")
        return True

    def _check_python_version(self):
        """Ensures modern Python features (like advanced typing) are supported."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("GeneWeaver requires Python 3.8 or higher.")
            raise PreFlightCheckFailed(f"Incompatible Python version: {platform.python_version()}")

    def _check_available_ram(self):
        """Verifies system RAM is sufficient for baseline pipeline startup."""
        available_ram = psutil.virtual_memory().available / (1024 * 1024)
        if available_ram < self.required_ram:
            logger.error(f"Insufficient RAM. Required: {self.required_ram}MB | Available: {available_ram:.1f}MB")
            raise PreFlightCheckFailed("System does not have enough free RAM to safely initialize buffers.")

    def _check_disk_space(self):
        """Verifies disk space for Out-of-Core memory spilling and audit reports."""
        disk_usage = shutil.disk_usage("/")
        free_space_mb = disk_usage.free / (1024 * 1024)
        if free_space_mb < self.required_disk:
            logger.error(f"Insufficient Disk Space for swapping. Required: {self.required_disk}MB")
            raise PreFlightCheckFailed("Not enough disk space for memory spilling.")


# ==========================================
# Master Pipeline Orchestrator
# ==========================================
class PipelineOrchestrator:
    """
    Central command unit for the GeneWeaver pipeline. 
    Strictly type-hinted to bypass dynamic attribute resolution errors.
    """
    
    def __init__(self, target_file: str, chunk_size_mb: int, overlap_bp: int, mutation: str):
        self.state = PipelineState.INITIALIZING
        
        self.target_file = target_file
        self.chunk_size_mb = chunk_size_mb
        self.overlap_bp = overlap_bp
        self.mutation = mutation
        
        self.system_monitor = SystemMonitor()
        self.auditor = SystemAuditor()
        
        # Strict Initialization to satisfy Pylance/MyPy
        if not Path(self.target_file).exists():
            console.print(f"[bold red][-] Fatal: Target dataset '{self.target_file}' not found on disk.[/bold red]")
            sys.exit(1)
            
        # Immediately instantiate modules to guarantee exact typing
        self.parser: GenomicParser = GenomicParser(
            file_path=self.target_file, 
            chunk_size_mb=self.chunk_size_mb, 
            overlap_bp=self.overlap_bp
        )
        self.optimizer: BatchOptimizer = BatchOptimizer()
        self.engine: CPUBaselineEngine = CPUBaselineEngine(target_mutation=self.mutation)

    def _display_banner(self):
        """Renders the Enterprise CLI Banner."""
        # Yahan 'r' add karna hai (Raw string)
        banner_text = r"""[bold green]
   ___             _      __                       
  / _ \___ ___  __ | | /| / /__ ___ __  _____ ____ 
 / // / -_) _ \/ -_) |/ |/ / -_) _ `/ |/ / -_) __/ 
/____/\__/_//_/\__/|__/|__/\__/\_,_/|___/\__/_/    
        [white]High-Performance CRISPR Alignment Engine[/white][/bold green]"""
        
        console.print(banner_text)
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="magenta")
        table.add_row("Target Dataset:", str(self.target_file))
        table.add_row("Mutation Query:", self.mutation)
        table.add_row("Hardware Backend:", f"{platform.system()} {platform.machine()} (Pure CPU)")
        table.add_row("VRAM Guard Cap:", "Active (Out-of-Core Spill Enabled)")
        
        console.print(Panel(table, title="[bold yellow]Pipeline Configuration[/bold yellow]", expand=False))
        console.print()

    def run(self):
        """Executes the ingestion, dynamic batching, and alignment algorithms."""
        self.state = PipelineState.PRE_FLIGHT_CHECKS
        self._display_banner()
        self.auditor.audit_environment()
        
        self.state = PipelineState.STREAMING_DATA
        generator = self.parser.stream_overlapping_chunks()
        start_time = time.time()
        
        total_mutations_found = 0
        total_bases_processed = 0
        
        self.state = PipelineState.ALIGNING_SEQUENCES
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan"),
            TextColumn("[cyan]Found: {task.fields[found]} | Rate: {task.fields[rate]} MB/s | RAM: {task.fields[ram]} MB"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[bold cyan]Executing GeneWeaver Multi-Threaded Pipeline...", 
                total=None, 
                found=0,
                rate="0.0",
                ram=f"{self.system_monitor.get_process_ram_mb():.1f}"
            )
            
            # Execute Pipeline with simulated parallelism for Phase 2 readiness
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for batch_payload in self.optimizer.build_batches(generator):
                    futures.append(executor.submit(self.engine.process_batch, batch_payload))
                    
                    if batch_payload["total_bases"] > 0:
                        total_bases_processed += batch_payload["total_bases"]
                    else:
                        total_bases_processed += self.optimizer.target_batch_bytes
                        
                for future in as_completed(futures):
                    total_mutations_found += future.result()
                    
                    elapsed = time.time() - start_time
                    current_rate = (total_bases_processed / (1024*1024)) / elapsed if elapsed > 0 else 0
                    current_ram = self.system_monitor.get_process_ram_mb()
                    
                    progress.update(
                        task, 
                        found=total_mutations_found, 
                        rate=f"{current_rate:.1f}",
                        ram=f"{current_ram:.1f}"
                    )

        self.state = PipelineState.EXPORTING_REPORTS
        total_time_elapsed = time.time() - start_time
        processed_mb = total_bases_processed / (1024 * 1024)
        
        metrics = EngineMetrics(
            total_time_sec=total_time_elapsed,
            data_processed_mb=processed_mb,
            total_bases=total_bases_processed,
            throughput_mb_s=processed_mb / total_time_elapsed if total_time_elapsed > 0 else 0,
            throughput_bp_s=total_bases_processed / total_time_elapsed if total_time_elapsed > 0 else 0
        )
        
        accuracy_report = self.engine.analytics.calculate_accuracy()
        self.engine.analytics.export_audit_report(metrics, accuracy_report)
        
        self.state = PipelineState.COMPLETED
        display_baseline_dashboard(metrics, accuracy_report)

    def graceful_shutdown(self, signum=None, frame=None):
        """
        Forces memory cleanup and disk flushing if interrupted by OS or User.
        Relies on the State Machine to know exactly what to clean up.
        """
        if self.state == PipelineState.SHUTTING_DOWN:
            return  # Prevent recursive shutdown loops
            
        self.state = PipelineState.SHUTTING_DOWN
        console.print(f"\n[bold red][!] CRITICAL: Pipeline interrupted (Signal: {signum}).[/bold red]")
        console.print("[yellow][*] Initiating graceful degradation and memory flush...[/yellow]")
        
        # 1. Clean up temporary disk spills if optimizer is instantiated
        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.shutdown()
            console.print("[green][+] Temporary out-of-core disk spills safely destroyed.[/green]")
            
        # 2. Force aggressive garbage collection to release OS RAM
        gc.collect()
        console.print(f"[green][+] System RAM forcibly flushed. Current usage: {psutil.Process(os.getpid()).memory_info().rss / (1024*1024):.1f} MB[/green]")
        
        console.print("[bold red][!] GeneWeaver Engine Offline.[/bold red]")
        sys.exit(0)


# ==========================================
# Advanced CLI Argument Parsing
# ==========================================
def parse_arguments():
    """Constructs a robust Command Line Interface synced with config/config.yaml."""
    import yaml
    
    # Defaults fallback
    default_file = "data/mock_genome.fastq"
    default_mutation = "GATCGATCGATC"
    
    # Sync automatically with config/config.yaml
    config_path = Path("config/config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            out_path = cfg["paths"]["synthetic_data_output"]
            fmt = cfg["sequence"]["format"].lower()
            if fmt == "fastq":
                out_path = out_path.replace(".fasta", ".fastq")
            elif fmt == "fasta.gz":
                out_path = out_path.replace(".fasta", ".fasta.gz")
            default_file = out_path
            default_mutation = cfg["sequence"]["target_mutation"]
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="GeneWeaver: High-Performance CRISPR Alignment Pipeline (Phase 1 Baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Designed by Ayush Pandey | Infotact Solutions"
    )
    
    io_group = parser.add_argument_group('Input/Output Settings')
    io_group.add_argument('-f', '--file', type=str, default=default_file, 
                          help='Absolute or relative path to the target DNA FASTA/FASTQ dataset.')
    
    bio_group = parser.add_argument_group('Biological Parameters')
    bio_group.add_argument('-m', '--mutation', type=str, default=default_mutation, 
                           help='The precise CRISPR off-target nucleotide sequence to hunt.')
                           
    mem_group = parser.add_argument_group('Memory Tuning (Advanced)')
    mem_group.add_argument('-c', '--chunk-mb', type=int, default=5, 
                           help='Max size (in MB) of internal sequence chunks before batching.')
    mem_group.add_argument('-o', '--overlap', type=int, default=100, 
                           help='Base pair overlap between chunks to prevent boundary target loss.')
    
    return parser.parse_args()


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    args = parse_arguments()
    
    orchestrator = PipelineOrchestrator(
        target_file=args.file,
        chunk_size_mb=args.chunk_mb,
        overlap_bp=args.overlap,
        mutation=args.mutation
    )
    
    # Bind robust termination signals (Ctrl+C and System Kill) to our graceful shutdown
    signal.signal(signal.SIGINT, orchestrator.graceful_shutdown)
    signal.signal(signal.SIGTERM, orchestrator.graceful_shutdown)
    
    try:
        orchestrator.run()
    except Exception as e:
        logger.exception("The pipeline encountered an unrecoverable crash.")
        orchestrator.graceful_shutdown(signum="FATAL_ERROR")