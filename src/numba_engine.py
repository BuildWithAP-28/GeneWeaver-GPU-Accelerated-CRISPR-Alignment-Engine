"""
Infotact Solutions - GeneWeaver Project
Module: Advanced Numba JIT Engine (Phase 2 - Day 1 / Day 8)
Description: 
    Enterprise-grade Just-In-Time (JIT) compiled sequence alignment engine.
    Bypasses the Python Global Interpreter Lock (GIL) by compiling the Knuth-Morris-Pratt (KMP) 
    algorithm into optimized LLVM machine code. Includes advanced memory alignment, 
    strict type safety, and standalone performance telemetry.
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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler

# Re-use analytics from Phase 1 baseline
try:
    from cpu_baseline import AlignmentAnalytics, MutationFinding, EngineMetrics
except ImportError:
    # Fallback if run completely isolated
    pass

console = Console()

# ==========================================
# 1. Custom Exception Hierarchy
# ==========================================
class JITEngineError(Exception):
    """Base exception for all LLVM/Numba related failures."""
    pass

class CompilationTimeoutError(JITEngineError):
    """Raised when LLVM compiler takes too long to warm up."""
    pass

class MemoryAlignmentError(JITEngineError):
    """Raised when sequence strings fail to convert to C-contiguous byte arrays."""
    pass


# ==========================================
# 2. Advanced Logger Configuration
# ==========================================
def setup_jit_logging(log_dir: str = "logs") -> logging.Logger:
    """Configures high-precision logging for the compiler engine."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "numba_engine.log"

    logger = logging.getLogger("GeneWeaverJIT")
    logger.setLevel(logging.DEBUG)
    
    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [JIT_CORE] - %(message)s"))

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_jit_logging()


# ==========================================
# 3. LLVM JIT-Compiled C-Level Functions
# ==========================================
# We use @njit (No-Python mode) with cache=True so it only compiles once and saves to disk.
# boundscheck=False makes it exponentially faster but requires strict memory alignment beforehand.

@njit(cache=True, boundscheck=False, fastmath=True)
def _compute_lps_numba(pattern_bytes: np.ndarray) -> np.ndarray:
    """
    Highly optimized Longest Proper Prefix which is also Suffix (LPS) calculator.
    Executes entirely in C-space.
    """
    M = len(pattern_bytes)
    lps = np.zeros(M, dtype=np.int32)
    length = 0
    i = 1
    
    while i < M:
        if pattern_bytes[i] == pattern_bytes[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


@njit(cache=True, boundscheck=False, fastmath=True)
def _kmp_search_numba(text_bytes: np.ndarray, pattern_bytes: np.ndarray, lps: np.ndarray) -> list:
    """
    The core string matching algorithm running at bare-metal speed.
    Takes uint8 arrays (ASCII byte representation) instead of Python string objects.
    """
    N = len(text_bytes)
    M = len(pattern_bytes)
    
    found_indices = []
    i = 0  # index for text_bytes
    j = 0  # index for pattern_bytes
    
    while (N - i) >= (M - j):
        if pattern_bytes[j] == text_bytes[i]:
            i += 1
            j += 1
            
        if j == M:
            # Match found! Record the starting index
            found_indices.append(i - j)
            j = lps[j - 1]
            
        elif i < N and pattern_bytes[j] != text_bytes[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
                
    return found_indices


# ==========================================
# 4. Memory Aligner (Safety Layer)
# ==========================================
class MemoryAligner:
    """
    Ensures that Python strings are properly cast into C-contiguous NumPy arrays
    before handing them over to the boundscheck-free LLVM compiler.
    Prevents Segmentation Faults (Segfaults).
    """
    
    @staticmethod
    def string_to_uint8(sequence: str) -> np.ndarray:
        """Converts a biological sequence string to a memory-safe byte array."""
        try:
            # encode('ascii') is fast; frombuffer is zero-copy
            byte_array = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
            
            # Ensure C-contiguous memory layout for JIT compiler safety
            if not byte_array.flags['C_CONTIGUOUS']:
                byte_array = np.ascontiguousarray(byte_array)
                
            return byte_array
        except Exception as e:
            logger.error(f"Memory Alignment failed for sequence block: {e}")
            raise MemoryAlignmentError("Failed to convert sequence to contiguous C-array.")


# ==========================================
# 5. Core Orchestrator Class
# ==========================================
class NumbaAcceleratedEngine:
    """
    Enterprise drop-in replacement for the CPUBaselineEngine.
    Manages the lifecycle of the LLVM compiler, coordinates chunk processing,
    and maps local array indices back to global biological chromosomes.
    """
    
    def __init__(self, target_mutation: str = "GATCGATCGATC"):
        self.target_mutation = target_mutation
        self.chunk_size_bytes = 5 * 1024 * 1024 
        
        # 1. Align Target Pattern in Memory
        logger.info(f"Initializing JIT Engine for target: {self.target_mutation}")
        self.pattern_bytes = MemoryAligner.string_to_uint8(self.target_mutation)
        
        # 2. Pre-compute LPS Array at C-level
        self.lps_array = _compute_lps_numba(self.pattern_bytes)
        
        # 3. Analytics Integration
        try:
            self.analytics = AlignmentAnalytics()
        except NameError:
            self.analytics = None # Fallback if cpu_baseline is not imported
            
        # 4. Compiler Warmup
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        """
        Forces the Numba LLVM compiler to compile the functions by feeding it 
        a dummy array during initialization. This prevents the first real data 
        chunk from suffering compilation latency (JIT lag).
        """
        logger.debug("Executing LLVM Compiler Warmup sequence...")
        start_time = time.time()
        
        try:
            dummy_text = MemoryAligner.string_to_uint8("ACTG" * 1000 + self.target_mutation + "ACTG" * 100)
            _kmp_search_numba(dummy_text, self.pattern_bytes, self.lps_array)
            
            elapsed = time.time() - start_time
            logger.info(f"JIT Warmup complete in {elapsed:.3f} seconds. Engine is hot.")
        except NumbaError as ne:
            logger.critical(f"LLVM Compilation failed: {ne}")
            raise JITEngineError("Failed to compile KMP algorithm.")
        except Exception as e:
            raise CompilationTimeoutError(f"Warmup interrupted: {e}")

    def _translate_global_coordinate(self, record_id: str, local_offset: int) -> Tuple[str, int]:
        """Maps local chunk index back to the exact genomic absolute index."""
        try:
            chunk_idx = int(record_id.split('_')[-1])
            base_chromosome = record_id.rsplit('_', 1)[0]
            global_offset = (chunk_idx * self.chunk_size_bytes) + local_offset
            return base_chromosome, global_offset
        except ValueError:
            return record_id, local_offset

    def process_batch(self, batch_payload: Dict[str, Any]) -> int:
        """
        Processes a fully constructed batch payload. Designed to be called 
        asynchronously by ProcessPoolExecutors.
        """
        mutations_found_in_batch = 0
        
        # Handle Disk-Spilled Out-of-Core data
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
            
            # 1. Strict Memory Alignment (Python str -> C-array)
            seq_bytes = MemoryAligner.string_to_uint8(sequence)
            
            # 2. Execute LLVM Compiled Code (Bypass GIL)
            match_indices = _kmp_search_numba(seq_bytes, self.pattern_bytes, self.lps_array)
            
            # 3. Post-Processing & Telemetry
            for local_index in match_indices:
                base_chromosome, global_offset = self._translate_global_coordinate(record_id, start_offset)
                absolute_global_index = global_offset + local_index
                
                # Only log if Analytics module is attached
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
def display_jit_telemetry(total_time: float, data_mb: float, target: str, matches: int):
    """Renders a cinematic summary table specifically for JIT profiling."""
    throughput = data_mb / total_time if total_time > 0 else 0
    
    table = Table(title="LLVM JIT Engine Telemetry (Phase 2)", title_style="bold green")
    table.add_column("System Metric", justify="right", style="cyan")
    table.add_column("Performance Value", style="bold yellow")
    
    table.add_row("Execution Backend", "Numba @njit (No-Python Mode)")
    table.add_row("Target Sequence", target)
    table.add_row("Total Data Processed", f"{data_mb:.2f} MB")
    table.add_row("Total Matches Found", str(matches))
    table.add_row("Total Execution Time", f"{total_time:.4f} seconds")
    table.add_row("Engine Throughput", f"[bold green]{throughput:.2f} MB/s[/bold green]")
    
    console.print(table)


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    parser = argparse.ArgumentParser(description="Standalone Numba JIT Engine Tester")
    parser.add_argument('-m', '--mutation', type=str, default="GATCGATCGATC", help='Target Mutation')
    parser.add_argument('-s', '--size-mb', type=int, default=150, help='Mock data size to simulate in MB')
    args = parser.parse_args()
    
    console.print("[bold magenta]>>> Initializing Phase 2: Numba Accelerated Engine...[/bold magenta]")
    
    try:
        # 1. Initialize Engine (Trigger Warmup)
        engine = NumbaAcceleratedEngine(target_mutation=args.mutation)
        
        # 2. Generate massive dummy payload in RAM to stress-test JIT
        console.print(f"[*] Generating {args.size_mb} MB of synthetic testing data in RAM...")
        dummy_chunk_size = 5 * 1024 * 1024
        num_chunks = max(1, args.size_mb // 5)
        
        # Create a large payload dictionary simulating the Chunker module output
        test_payload = {
            "is_spilled": False,
            "data": []
        }
        
        # Inject exact mutations at random spots to verify logic
        for i in range(num_chunks):
            # Create a 5MB string of 'A's with the mutation injected in the middle
            seq_list = ['A'] * dummy_chunk_size
            mid = dummy_chunk_size // 2
            seq_list[mid:mid+len(args.mutation)] = list(args.mutation)
            chunk_str = "".join(seq_list)
            
            test_payload["data"].append({
                "record_id": f"chr_test_{i}",
                "start_index": i * dummy_chunk_size,
                "sequence_data": chunk_str
            })
            
        console.print("[green][+] Mock data generated. Engaging JIT Core...[/green]")
        
        # 3. Execute & Profile
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="magenta"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[bold magenta]Executing JIT Compiled Payload...", total=1)
            
            total_matches = engine.process_batch(test_payload)
            
            progress.update(task, advance=1)
            
        elapsed = time.time() - start_time
        
        # 4. Show Stats
        display_jit_telemetry(elapsed, args.size_mb, args.mutation, total_matches)
        
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Profiler aborted by user.[/bold red]")
        sys.exit(0)
    except Exception as e:
        logger.exception("JIT Tester crashed unexpectedly.")
        sys.exit(1)