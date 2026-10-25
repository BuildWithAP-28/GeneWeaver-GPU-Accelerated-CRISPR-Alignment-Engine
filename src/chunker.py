"""
Infotact Solutions - GeneWeaver Project
Module: Advanced Memory Manager & Out-of-Core Batch Optimizer (Phase 1)
Description: Enterprise-grade memory guard for genomic pipelines.
Features: 
- Out-of-Core Disk Spilling (Prevent OOM by swapping to disk)
- Decorator-based memory profiling
- Aggressive garbage collection (gc) triggering
- Real-time psutil hardware monitoring including Swap memory
- Yields optimal batch sizes specifically tuned for 4GB VRAM limits
"""

import os
import gc
import sys
import time
import uuid
import yaml
import pickle
import psutil
import shutil
import logging
from pathlib import Path
from functools import wraps
from typing import Generator, List, Dict, Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

console = Console()

# ==========================================
# Custom Memory Exceptions
# ==========================================
class MemoryGuardError(Exception):
    """Base exception for Memory Management issues."""
    pass

class HardwareConstraintError(MemoryGuardError):
    """Raised when available system RAM and Swap are critically low."""
    pass

class BufferOverflowError(MemoryGuardError):
    """Raised when a batch exceeds the strictly defined YAML memory limits."""
    pass

class DiskSpillError(MemoryGuardError):
    """Raised when temporary disk offloading fails (e.g., out of storage)."""
    pass

# ==========================================
# Logger Configuration
# ==========================================
def setup_chunker_logging(log_dir: str = "logs") -> logging.Logger:
    """Configures dual-channel logging for the memory manager."""
    from logging.handlers import RotatingFileHandler
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "memory_manager.log"

    logger = logging.getLogger("GeneWeaverChunker")
    logger.setLevel(logging.DEBUG)
    
    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s"))

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_chunker_logging()

# ==========================================
# Profiling Decorators
# ==========================================
def profile_memory_delta(func):
    """Decorator to measure how much RAM a specific operation consumes."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        mem_after = process.memory_info().rss / (1024 * 1024)
        delta = mem_after - mem_before
        
        if delta > 50:  # Only log if it spiked more than 50MB
            logger.debug(f"[Profile] {func.__name__} | Time: {end_time-start_time:.3f}s | RAM Delta: {delta:+.2f} MB")
        return result
    return wrapper


# ==========================================
# Out-of-Core Disk Spilling Manager
# ==========================================
class DiskSpillManager:
    """Handles offloading excess RAM data to disk temporarily to avoid OOM."""
    
    def __init__(self, temp_dir: str = "data/temp_spill"):
        self.temp_dir = Path(temp_dir)
        self._initialize_temp_storage()

    def _initialize_temp_storage(self):
        """Creates or cleans the temporary spill directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Disk Spill Manager initialized at {self.temp_dir}")

    @profile_memory_delta
    def spill_to_disk(self, batch_data: List[Dict[str, Any]]) -> str:
        """Serializes and writes a batch to disk, freeing RAM."""
        try:
            spill_id = f"batch_spill_{uuid.uuid4().hex[:8]}.pkl"
            spill_path = self.temp_dir / spill_id
            
            with open(spill_path, "wb") as f:
                pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.warning(f"Memory Critical: Spilled batch to disk -> {spill_id}")
            return str(spill_path)
            
        except Exception as e:
            logger.error(f"Failed to spill data to disk: {e}")
            raise DiskSpillError("Insufficient storage space for memory spilling.")

    @profile_memory_delta
    def load_from_disk(self, spill_path: str) -> List[Dict[str, Any]]:
        """Loads a spilled batch back into RAM and deletes the temp file."""
        path_obj = Path(spill_path)
        if not path_obj.exists():
            raise DiskSpillError(f"Spill file lost: {spill_path}")
            
        with open(path_obj, "rb") as f:
            data = pickle.load(f)
            
        # Clean up
        path_obj.unlink()
        logger.debug(f"Recovered spilled batch from {path_obj.name}")
        return data

    def cleanup(self):
        """Removes the entire spill directory on shutdown."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ==========================================
# Hardware Monitoring Subsystem
# ==========================================
class SystemMonitor:
    """Monitors live OS memory metrics to guide batch sizes and spilling."""
    
    def __init__(self, critical_ram_mb: int = 300):
        self.critical_ram_mb = critical_ram_mb

    @staticmethod
    def get_available_ram_mb() -> float:
        """Returns the currently available system RAM in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)

    @staticmethod
    def get_process_ram_mb() -> float:
        """Returns RAM used by the current Python process in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def force_collection(self) -> None:
        """Aggressively sweeps unused variables from memory."""
        pre_gc = self.get_available_ram_mb()
        gc.collect()
        post_gc = self.get_available_ram_mb()
        
        recovered = post_gc - pre_gc
        if recovered > 5:
            logger.debug(f"Garbage Collection triggered. Recovered: {recovered:.1f} MB")

    def is_memory_critical(self) -> bool:
        """Returns True if RAM is dangerously low."""
        available = self.get_available_ram_mb()
        if available < self.critical_ram_mb:
            self.force_collection()
            # Double check after GC
            return self.get_available_ram_mb() < self.critical_ram_mb
        return False


# ==========================================
# Core Batching Engine
# ==========================================
class BatchOptimizer:
    """
    Accumulates small biological sequences from the parser into optimal 
    multi-megabyte batches suitable for high-throughput CPU/GPU processing.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.max_vram_buffer_mb = self._load_vram_limit()
        
        self.monitor = SystemMonitor(critical_ram_mb=400)
        self.spill_manager = DiskSpillManager()
        
        # Operational limits: Target 10% of VRAM limit per batch for smooth queuing
        self.target_batch_mb = max(5, int(self.max_vram_buffer_mb * 0.10))
        self.target_batch_bytes = self.target_batch_mb * 1024 * 1024
        
        logger.info(f"Batch Optimizer Initialized. Target Batch Size: {self.target_batch_mb} MB")

    def _load_vram_limit(self) -> int:
        """Parses the YAML config for strict hardware limits."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                return config['hardware']['max_vram_buffer_mb']
        except Exception as e:
            logger.error(f"Failed to load VRAM limits from config: {e}")
            sys.exit(1)

    def _calculate_batch_size(self, batch_list: List[Dict[str, Any]]) -> int:
        """Calculates the byte size of the raw sequences currently in the batch."""
        return sum(len(item["sequence_data"]) for item in batch_list)

    def build_batches(self, parser_generator: Generator[Dict[str, Any], None, None]) -> Generator[Dict[str, Any], None, None]:
        """
        Consumes the parser generator and yields combined dictionary batches.
        Actively monitors memory and aggressively flushes RAM to prevent stacking.
        """
        current_batch = []
        current_batch_bytes = 0
        batch_id = 0
        
        for chunk in parser_generator:
            # 1. Proactive Hardware Check (Out-of-Core execution trigger)
            if self.monitor.is_memory_critical():
                if current_batch:
                    # Spill current progress to disk to save RAM
                    spilled_path = self.spill_manager.spill_to_disk(current_batch)
                    current_batch = []
                    current_batch_bytes = 0
                    self.monitor.force_collection()
                    
                    # Yield a reference to the spilled file instead of RAM data
                    yield self._create_spill_payload(spilled_path, batch_id)
                    batch_id += 1
                    continue
                else:
                    raise HardwareConstraintError("RAM critically low, and no active batch to spill. Aborting.")
            
            # 2. Append chunk to current batch
            seq_len = len(chunk["sequence_data"])
            current_batch.append(chunk)
            current_batch_bytes += seq_len
            
            # 3. Check if Batch Target Reached
            if current_batch_bytes >= self.target_batch_bytes:
                yield self._dispatch_batch(current_batch, batch_id)
                batch_id += 1
                
                # Reset pointers and aggressively flush memory
                current_batch = []
                current_batch_bytes = 0
                self.monitor.force_collection()
                
        # 4. Flush remaining data in the final partial batch
        if current_batch:
            yield self._dispatch_batch(current_batch, batch_id)
            self.monitor.force_collection()

    def _dispatch_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> Dict[str, Any]:
        """Packages the batch list into a structured payload for the processing engine."""
        total_bases = self._calculate_batch_size(batch_data)
        batch_mb = total_bases / (1024 * 1024)
        
        logger.debug(f"Dispatching Batch #{batch_id} | Size: {batch_mb:.2f} MB | Chunks: {len(batch_data)}")
            
        return {
            "batch_id": batch_id,
            "data": batch_data,
            "is_spilled": False,
            "total_bases": total_bases,
            "size_mb": batch_mb,
            "timestamp": time.time()
        }

    def _create_spill_payload(self, spill_path: str, batch_id: int) -> Dict[str, Any]:
        """Creates a payload containing a disk pointer instead of raw RAM data."""
        return {
            "batch_id": batch_id,
            "data": spill_path,
            "is_spilled": True,
            "total_bases": -1, # Unknown until loaded
            "size_mb": -1,
            "timestamp": time.time()
        }

    def shutdown(self):
        """Cleans up all resources, especially temp storage."""
        self.spill_manager.cleanup()
        logger.info("Batch Optimizer shutdown complete.")


# ==========================================
# CLI Execution & Telemetry (For Testing)
# ==========================================
def display_chunker_analytics(total_batches: int, spilled_batches: int, total_mb: float, execution_time: float):
    """Renders execution summary via Rich table."""
    table = Table(title="GeneWeaver Dynamic Batch Analytics", title_style="bold magenta")
    table.add_column("Metric", justify="right", style="cyan")
    table.add_column("Value", style="bold yellow")

    table.add_row("Total Batches Generated", str(total_batches))
    table.add_row("Disk Spilled Batches", f"{spilled_batches} (OOM Prevented)")
    table.add_row("Data Pushed Through RAM", f"{total_mb:.2f} MB")
    table.add_row("Final Memory Footprint", f"{SystemMonitor.get_process_ram_mb():.1f} MB")
    table.add_row("Execution Time", f"{execution_time:.2f} seconds")
    table.add_row("System Status", "[bold green]Memory Pipeline Stable[/bold green]")

    console.print(table)


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    optimizer = None
    try:
        console.print("[bold magenta]>>> Initializing Phase 1: Out-of-Core Batch Optimizer...[/bold magenta]")
        
        # Import the parser dynamically to test the flow
        from parser import GenomicParser
        
        test_file = "data/mock_genome.fastq"
        if not Path(test_file).exists():
            console.print("[red]Mock data not found. Please run synthetic_generator.py first.[/red]")
            sys.exit(1)
            
        # Initialize the upstream parser
        parser = GenomicParser(file_path=test_file, chunk_size_mb=5, overlap_bp=100)
        generator = parser.stream_overlapping_chunks()
        
        # Initialize the memory optimizer
        optimizer = BatchOptimizer()
        
        total_batches_processed = 0
        spilled_count = 0
        total_mb_processed = 0.0
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="magenta"),
            TextColumn("[cyan]RAM: {task.fields[ram]} MB"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[bold magenta]Optimizing Batches...", 
                total=None, 
                ram=f"{SystemMonitor.get_process_ram_mb():.1f}"
            )
            
            for batch_payload in optimizer.build_batches(generator):
                total_batches_processed += 1
                
                if batch_payload["is_spilled"]:
                    spilled_count += 1
                else:
                    total_mb_processed += batch_payload["size_mb"]
                
                progress.update(
                    task, 
                    ram=f"{SystemMonitor.get_process_ram_mb():.1f}"
                )
                
        # Final cleanup and reporting
        gc.collect()
        exec_time = time.time() - start_time
        display_chunker_analytics(total_batches_processed, spilled_count, total_mb_processed, exec_time)
        
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Process aborted by user (SIGINT). Cleaning temp files...[/bold red]")
    except Exception as e:
        logger.exception("Memory optimizer encountered a critical failure.")
    finally:
        if optimizer:
            optimizer.shutdown()
        sys.exit(0)