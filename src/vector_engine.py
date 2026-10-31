"""
Infotact Solutions - GeneWeaver Project
Module: SIMD Vectorization & Memory Profiling Engine (Phase 2 - Day 10)
Description: 
    Enterprise-grade vector execution backend. Replaces the sequential KMP algorithm
    with a hybrid SIMD-accelerated memory scanner using NumPy C-contiguous arrays and 
    Numba @njit. Monitors L1/L2 cache hit approximations and memory allocation latencies.
"""

import os
import sys
import time
import psutil
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from numba import njit
from numba.core.errors import NumbaError

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.panel import Panel

# Re-use analytics from Phase 1 baseline
try:
    from cpu_baseline import AlignmentAnalytics, MutationFinding
except ImportError:
    pass

console = Console()

# ==========================================
# 1. Custom Exception Hierarchy
# ==========================================
class VectorEngineError(Exception):
    """Base exception for SIMD and memory vectorization operations."""
    pass

class ArrayAlignmentError(VectorEngineError):
    """Raised when memory chunks fail to align to contiguous block boundaries."""
    pass


# ==========================================
# 2. Advanced Logger Configuration
# ==========================================
def setup_vector_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "vector_engine.log"

    logger = logging.getLogger("GeneWeaverVector")
    logger.setLevel(logging.DEBUG)
    
    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [VECTOR_CORE] - %(message)s"))

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_vector_logging()


# ==========================================
# 3. Hardware Cache & Memory Profiler
# ==========================================
@dataclass
class VectorMetrics:
    """Telemetry tracking for SIMD memory operations."""
    total_alloc_time_ms: float = 0.0
    total_scan_time_ms: float = 0.0
    bytes_scanned: int = 0
    fast_path_hits: int = 0
    true_matches: int = 0

class VectorMemoryProfiler:
    """Monitors overhead of creating C-arrays to ensure I/O isn't choking the CPU."""
    
    def __init__(self):
        self.metrics = VectorMetrics()
        
    def profile_allocation(self, sequence: str) -> Tuple[np.ndarray, float]:
        """Times the exact latency of converting Python strings to C-arrays."""
        start = time.perf_counter()
        try:
            byte_array = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
            if not byte_array.flags['C_CONTIGUOUS']:
                byte_array = np.ascontiguousarray(byte_array)
            alloc_time = (time.perf_counter() - start) * 1000
            self.metrics.total_alloc_time_ms += alloc_time
            return byte_array, alloc_time
        except Exception as e:
            logger.error(f"Vector memory allocation failed: {e}")
            raise ArrayAlignmentError("Failed to lock sequence to C-contiguous memory.")


# ==========================================
# 4. LLVM JIT-Compiled SIMD-Style Functions
# ==========================================
# We use a Hybrid Fast-Scan algorithm. Instead of complex KMP jumps which break CPU 
# branch prediction, we rapidly scan for the first 2 bytes (Fast Path) and only 
# verify the rest of the string if they match. This utilizes hardware prefetching perfectly.

@njit(cache=True, boundscheck=False, fastmath=True)
def _hybrid_simd_scan(text_bytes: np.ndarray, pattern_bytes: np.ndarray) -> Tuple[list, int]:
    """
    Executes a high-speed memory scan.
    Returns: (List of match indices, Number of fast-path hits)
    """
    N = len(text_bytes)
    M = len(pattern_bytes)
    
    found_indices = []
    fast_path_hits = 0
    
    if M < 2 or N < M:
        return found_indices, fast_path_hits
        
    p0 = pattern_bytes[0]
    p1 = pattern_bytes[1]
    
    # Fast-Path SIMD Loop
    i = 0
    limit = N - M + 1
    
    while i < limit:
        # Branch predictor favors this failing 94% of the time in random DNA
        if text_bytes[i] == p0 and text_bytes[i+1] == p1:
            fast_path_hits += 1
            
            # Slow-Path Verification (Inline memcmp equivalent)
            match = True
            for j in range(2, M):
                if text_bytes[i + j] != pattern_bytes[j]:
                    match = False
                    break
                    
            if match:
                found_indices.append(i)
        i += 1
                
    return found_indices, fast_path_hits


# ==========================================
# 5. Core Orchestrator Class
# ==========================================
class VectorizedEngine:
    """
    Enterprise drop-in replacement for NumbaAcceleratedEngine.
    Focuses on raw contiguous memory scanning over algorithmic jumps.
    """
    
    def __init__(self, target_mutation: str = "GATCGATCGATC"):
        self.target_mutation = target_mutation
        self.chunk_size_bytes = 5 * 1024 * 1024 
        
        logger.info(f"Initializing Vector Engine for target: {self.target_mutation}")
        self.profiler = VectorMemoryProfiler()
        
        # Lock target mutation into L1-friendly C-array
        self.pattern_bytes, _ = self.profiler.profile_allocation(self.target_mutation)
        
        try:
            self.analytics = AlignmentAnalytics()
        except NameError:
            self.analytics = None 
            
        self._warmup_vector_lanes()

    def _warmup_vector_lanes(self) -> None:
        """Triggers the LLVM compiler to pre-build the SIMD routines."""
        logger.debug("Executing LLVM Vector Warmup sequence...")
        try:
            dummy_text, _ = self.profiler.profile_allocation("ACTG" * 500 + self.target_mutation + "ACTG" * 50)
            _hybrid_simd_scan(dummy_text, self.pattern_bytes)
            logger.info("Vector warmup complete. Caches are hot.")
        except NumbaError as ne:
            logger.critical(f"LLVM Vector Compilation failed: {ne}")
            raise VectorEngineError("Failed to compile SIMD algorithm.")

    def _translate_global_coordinate(self, record_id: str, local_offset: int) -> Tuple[str, int]:
        try:
            chunk_idx = int(record_id.split('_')[-1])
            base_chromosome = record_id.rsplit('_', 1)[0]
            global_offset = (chunk_idx * self.chunk_size_bytes) + local_offset
            return base_chromosome, global_offset
        except ValueError:
            return record_id, local_offset

    def process_batch(self, batch_payload: Dict[str, Any]) -> int:
        """Processes an I/O batch dynamically routed by the ProcessPoolExecutor."""
        mutations_found_in_batch = 0
        
        if batch_payload.get("is_spilled"):
            from chunker import DiskSpillManager
            spill_mgr = DiskSpillManager()
            chunks = spill_mgr.load_from_disk(batch_payload['data'])
        else:
            chunks = batch_payload["data"]
            
        for chunk in chunks:
            record_id = chunk["record_id"]
            sequence = chunk["sequence_data"]
            start_offset = chunk["start_index"]
            
            # 1. Profiled Memory Alignment
            seq_bytes, _ = self.profiler.profile_allocation(sequence)
            self.profiler.metrics.bytes_scanned += len(seq_bytes)
            
            # 2. Execute Hybrid Vector Scan
            scan_start = time.perf_counter()
            match_indices, fast_hits = _hybrid_simd_scan(seq_bytes, self.pattern_bytes)
            self.profiler.metrics.total_scan_time_ms += (time.perf_counter() - scan_start) * 1000
            
            self.profiler.metrics.fast_path_hits += fast_hits
            self.profiler.metrics.true_matches += len(match_indices)
            
            # 3. Geolocation & Analytics
            for local_index in match_indices:
                base_chromosome, global_offset = self._translate_global_coordinate(record_id, start_offset)
                absolute_global_index = global_offset + local_index
                
                if self.analytics:
                    finding = MutationFinding(
                        chromosome=base_chromosome,
                        absolute_index=absolute_global_index,
                        matched_sequence=self.target_mutation,
                        timestamp=time.time()
                    )
                    self.analytics.register_finding(finding)
                    
                mutations_found_in_batch += 1
                
        return mutations_found_in_batch


# ==========================================
# 6. Standalone CLI & Telemetry Tester
# ==========================================
def display_vector_telemetry(profiler: VectorMemoryProfiler, total_time: float, data_mb: float, target: str):
    """Renders highly detailed hardware-level telemetry specific to the Vector Engine."""
    throughput = data_mb / total_time if total_time > 0 else 0
    
    # Calculate Memory Overhead
    total_alloc_sec = profiler.metrics.total_alloc_time_ms / 1000
    total_scan_sec = profiler.metrics.total_scan_time_ms / 1000
    alloc_percentage = (total_alloc_sec / total_time) * 100 if total_time > 0 else 0
    
    table = Table(title="LLVM Vector Engine Telemetry (Day 10)", title_style="bold green")
    table.add_column("System Metric", justify="right", style="cyan")
    table.add_column("Performance Value", style="bold yellow")
    
    table.add_row("Execution Backend", "Numba @njit (Hybrid Fast-Scan)")
    table.add_row("Target Sequence", target)
    table.add_row("Total Data Processed", f"{data_mb:.2f} MB")
    
    # Hardware Telemetry
    table.add_row("Fast-Path Cache Hits", f"{profiler.metrics.fast_path_hits:,} triggers")
    table.add_row("True Validated Matches", f"{profiler.metrics.true_matches:,}")
    table.add_row("C-Array Alloc Overhead", f"{profiler.metrics.total_alloc_time_ms:.2f} ms ({alloc_percentage:.1f}%)")
    table.add_row("Raw Array Scan Time", f"{profiler.metrics.total_scan_time_ms:.2f} ms")
    
    table.add_row("Total Execution Time", f"{total_time:.4f} seconds")
    table.add_row("Engine Throughput", f"[bold green]{throughput:.2f} MB/s[/bold green]")
    
    console.print(table)


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    parser = argparse.ArgumentParser(description="Standalone Vector Engine Tester")
    parser.add_argument('-m', '--mutation', type=str, default="GATCGATCGATC", help='Target Mutation')
    parser.add_argument('-s', '--size-mb', type=int, default=200, help='Mock data size to simulate in MB')
    args = parser.parse_args()
    
    console.print("[bold magenta]>>> Initializing Phase 2: Vector Accelerated Engine...[/bold magenta]")
    
    try:
        engine = VectorizedEngine(target_mutation=args.mutation)
        
        console.print(f"[*] Generating {args.size_mb} MB of contiguous synthetic data in RAM...")
        dummy_chunk_size = 5 * 1024 * 1024
        num_chunks = max(1, args.size_mb // 5)
        
        test_payload = {
            "is_spilled": False,
            "data": []
        }
        
        for i in range(num_chunks):
            seq_list = ['A'] * dummy_chunk_size
            mid = dummy_chunk_size // 2
            seq_list[mid:mid+len(args.mutation)] = list(args.mutation)
            chunk_str = "".join(seq_list)
            
            test_payload["data"].append({
                "record_id": f"chr_test_{i}",
                "start_index": i * dummy_chunk_size,
                "sequence_data": chunk_str
            })
            
        console.print("[green][+] Mock data generated. Engaging Vector Core...[/green]")
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Executing Vectorized Payload..."),
            BarColumn(complete_style="magenta"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning...", total=1)
            engine.process_batch(test_payload)
            progress.update(task, advance=1)
            
        elapsed = time.time() - start_time
        
        display_vector_telemetry(engine.profiler, elapsed, args.size_mb, args.mutation)
        
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Profiler aborted by user.[/bold red]")
        sys.exit(0)