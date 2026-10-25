"""
Infotact Solutions - GeneWeaver Project
Module: CPU Baseline Engine (Phase 1)
Description: 
    Enterprise-grade pure Python genomic alignment engine utilizing the 
    Knuth-Morris-Pratt (KMP) pattern matching algorithm.
    
    This module serves as the performance baseline. It incorporates 
    pipeline parallelism, strict dataclass schemas, global coordinate 
    reconstruction, and automated CSV/JSON audit reporting.
"""

import os
import sys
import csv
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

console = Console()

# ==========================================
# Custom Exception Classes
# ==========================================
class EngineError(Exception):
    """Base exception for all execution engine errors."""
    pass

class AlignmentError(EngineError):
    """Raised when the KMP matcher encounters a fatal sequence error."""
    pass

class MetricsExportError(EngineError):
    """Raised when the automated reporting system fails to write to disk."""
    pass


# ==========================================
# Data Structures (Strict Typing)
# ==========================================
@dataclass
class MutationFinding:
    """Strict schema representing a single CRISPR off-target mutation find."""
    chromosome: str
    absolute_index: int
    matched_sequence: str
    timestamp: float

@dataclass
class EngineMetrics:
    """Strict schema for execution metrics and throughput calculations."""
    total_time_sec: float
    data_processed_mb: float
    total_bases: int
    throughput_mb_s: float
    throughput_bp_s: float


# ==========================================
# Logger Configuration
# ==========================================
def setup_baseline_logging(log_dir: str = "logs") -> logging.Logger:
    """Sets up dual-channel logging (Rich UI + Rotating Files)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "cpu_baseline.log"

    logger = logging.getLogger("GeneWeaverBaseline")
    logger.setLevel(logging.DEBUG)
    
    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s"))

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_baseline_logging()


# ==========================================
# Algorithmic Engine: Knuth-Morris-Pratt
# ==========================================
class KMPMatcher:
    """
    Implements the Knuth-Morris-Pratt (KMP) algorithm.
    Optimizes CPU search by avoiding redundant character comparisons 
    using the Longest Proper Prefix which is also Suffix (LPS) array.
    """
    
    def __init__(self, pattern: str):
        if not pattern:
            raise ValueError("KMP Pattern cannot be empty.")
            
        self.pattern = pattern.upper()
        self.M = len(self.pattern)
        self.lps = self._compute_lps_array()
        logger.debug(f"KMP Matcher initialized for pattern: {self.pattern} (Length: {self.M})")

    def _compute_lps_array(self) -> List[int]:
        """Calculates the LPS array to determine bypass distances on mismatches."""
        lps = [0] * self.M
        length = 0
        i = 1
        
        while i < self.M:
            if self.pattern[i] == self.pattern[length]:
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

    def search(self, text: str) -> List[int]:
        """
        Executes the O(N+M) string matching algorithm.
        Returns a list of local indices where the pattern starts.
        """
        N = len(text)
        found_indices = []
        i = 0  # text index
        j = 0  # pattern index
        
        while (N - i) >= (self.M - j):
            if self.pattern[j] == text[i]:
                i += 1
                j += 1
                
            if j == self.M:
                found_indices.append(i - j)
                j = self.lps[j - 1]
            elif i < N and self.pattern[j] != text[i]:
                if j != 0:
                    j = self.lps[j - 1]
                else:
                    i += 1
                    
        return found_indices


# ==========================================
# Ground Truth & Analytics System
# ==========================================
class AlignmentAnalytics:
    """
    Evaluates engine accuracy against ground truth datasets, handling 
    sliding window coordinate translation, and generates audit reports.
    """
    
    def __init__(self, ground_truth_path: str = "data/ground_truth_mutations.json"):
        self.truth_path = Path(ground_truth_path)
        self.expected_mutations = self._load_ground_truth()
        self.found_mutations: List[MutationFinding] = []
        
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if not self.truth_path.exists():
            logger.warning("Ground truth JSON missing. Running in Blind Mode (No Accuracy Metrics).")
            return []
            
        try:
            with open(self.truth_path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} ground truth coordinates for validation.")
                return data
        except Exception as e:
            logger.error(f"Failed to parse ground truth JSON: {e}")
            return []

    def register_finding(self, finding: MutationFinding) -> None:
        """Safely appends a new mutation finding to the registry."""
        self.found_mutations.append(finding)

    def calculate_accuracy(self) -> Dict[str, Any]:
        """Calculates Precision, True Positives, and Natural Occurrences."""
        if not self.expected_mutations:
            return {"status": "Blind Mode - No Ground Truth Available"}
            
        expected_set = {(m["chromosome"], m["absolute_index"]) for m in self.expected_mutations}
        found_set = {(f.chromosome, f.absolute_index) for f in self.found_mutations}
        
        true_positives = 0
        natural_off_targets = 0
        
        # We allow a +/- 20bp buffer to account for overlapping window shifts
        for found in found_set:
            is_match = False
            for exp in expected_set:
                if found[0] == exp[0] and abs(found[1] - exp[1]) <= 20:
                    true_positives += 1
                    is_match = True
                    break
            if not is_match:
                natural_off_targets += 1
                
        false_negatives = len(expected_set) - true_positives
        precision = true_positives / len(expected_set) if len(expected_set) > 0 else 0
        
        return {
            "Total Injected (Expected)": len(expected_set),
            "Total Matches Found": len(found_set),
            "True Positives (Injected found)": true_positives,
            "Natural Off-Targets (Randomly occurred)": natural_off_targets,
            "False Negatives (Missed)": false_negatives,
            "Engine Precision": f"{precision * 100:.2f}%"
        }

    def export_audit_report(self, metrics: EngineMetrics, accuracy: Dict[str, Any], output_dir: str = "reports"):
        """Exports a full compliance and performance report to CSV and JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = out_path / f"alignment_audit_{timestamp}.json"
        csv_file = out_path / f"findings_{timestamp}.csv"
        
        try:
            # Export JSON Summary
            report_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "engine": "Pure Python KMP (Baseline)",
                },
                "performance": asdict(metrics),
                "accuracy": accuracy
            }
            with open(json_file, "w") as jf:
                json.dump(report_data, jf, indent=4)
                
            # Export CSV Detailed Findings
            if self.found_mutations:
                with open(csv_file, "w", newline="") as cf:
                    writer = csv.DictWriter(cf, fieldnames=["chromosome", "absolute_index", "matched_sequence", "timestamp"])
                    writer.writeheader()
                    for finding in self.found_mutations:
                        writer.writerow(asdict(finding))
                        
            logger.info(f"Audit reports generated successfully in {output_dir}/")
        except Exception as e:
            raise MetricsExportError(f"Failed to export audit report: {e}")


# ==========================================
# Main CPU Execution Engine
# ==========================================
class CPUBaselineEngine:
    """
    Coordinates data ingestion, out-of-core batching, and parallel KMP alignment.
    Translates local sequence chunk coordinates back into global chromosome indices.
    """
    
    def __init__(self, target_mutation: str = "GATCGATCGATC"):
        self.target_mutation = target_mutation
        self.matcher = KMPMatcher(pattern=self.target_mutation)
        self.analytics = AlignmentAnalytics()
        self.chunk_size_bytes = 5 * 1024 * 1024 # Standard 5MB chunk from parser
        
    def _translate_global_coordinate(self, record_id: str, local_offset: int) -> Tuple[str, int]:
        """
        Parses FASTQ fragment headers (e.g., chr1_mock_..._3) to reconstruct 
        the exact absolute global coordinate across the entire genome.
        """
        try:
            chunk_idx = int(record_id.split('_')[-1])
            base_chromosome = record_id.rsplit('_', 1)[0]
            global_offset = (chunk_idx * self.chunk_size_bytes) + local_offset
            return base_chromosome, global_offset
        except ValueError:
            # Fallback if the header isn't chunked
            return record_id, local_offset

    def process_batch(self, batch_payload: Dict[str, Any]) -> int:
        """
        Processes a payload batch. Resolves disk-spilled data if necessary.
        """
        mutations_found_in_batch = 0
        
        # Resolve Out-of-Core data
        if batch_payload.get("is_spilled"):
            from chunker import DiskSpillManager
            spill_mgr = DiskSpillManager()
            chunks = spill_mgr.load_from_disk(batch_payload['data'])
        else:
            chunks = batch_payload["data"]
            
        # We simulate pipeline parallelism structure here for Phase 2 readiness
        for chunk in chunks:
            record_id = chunk["record_id"]
            sequence = chunk["sequence_data"]
            start_offset = chunk["start_index"]
            
            # Execute KMP Algorithm
            match_indices = self.matcher.search(sequence)
            
            for local_index in match_indices:
                base_chromosome, global_offset = self._translate_global_coordinate(record_id, start_offset)
                absolute_global_index = global_offset + local_index
                
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
# CLI Execution & Display formatting
# ==========================================
def display_baseline_dashboard(metrics: EngineMetrics, accuracy: Dict[str, Any]):
    """Renders the final performance and accuracy evaluation to the terminal."""
    
    # 1. Performance Table
    perf_table = Table(title="CPU Baseline Performance Analytics", title_style="bold cyan")
    perf_table.add_column("Metric", justify="right", style="cyan")
    perf_table.add_column("Value", style="magenta")
    
    perf_table.add_row("Execution Engine", "Pure Python KMP (GIL Active)")
    perf_table.add_row("Total Time", f"{metrics.total_time_sec:.2f} seconds")
    perf_table.add_row("Data Processed", f"{metrics.data_processed_mb:.2f} MB")
    perf_table.add_row("Throughput (MB/s)", f"{metrics.throughput_mb_s:.2f} MB/s")
    perf_table.add_row("Throughput (Bases/s)", f"{metrics.throughput_bp_s:,.0f} bp/s")
    
    console.print(perf_table)
    console.print("")
    
    # 2. Accuracy Table
    acc_table = Table(title="CRISPR Alignment Accuracy", title_style="bold green")
    acc_table.add_column("Metric", justify="right", style="cyan")
    acc_table.add_column("Result", style="bold yellow")
    
    for key, value in accuracy.items():
        if "Natural" in key:
            acc_table.add_row(key, f"[bold cyan]{value}[/bold cyan]")
        elif "False Neg" in key and value > 0:
            acc_table.add_row(key, f"[bold red]{value}[/bold red]")
        elif "Precision" in key:
            acc_table.add_row(key, f"[bold green]{value}[/bold green]")
        else:
            acc_table.add_row(key, str(value))
            
    console.print(acc_table)


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    try:
        console.print("[bold cyan]>>> Initializing Phase 1: CPU Baseline Engine (Enterprise)...[/bold cyan]")
        
        # Dynamically import ingestion pipeline dependencies
        from parser import GenomicParser
        from chunker import BatchOptimizer
        
        test_file = "data/mock_genome.fastq"
        if not Path(test_file).exists():
            console.print("[bold red][!] Mock data missing. Run synthetic_generator.py first.[/bold red]")
            sys.exit(1)
            
        # Initialize Core Pipeline
        parser = GenomicParser(file_path=test_file, chunk_size_mb=5, overlap_bp=100)
        optimizer = BatchOptimizer()
        engine = CPUBaselineEngine(target_mutation="GATCGATCGATC")
        
        generator = parser.stream_overlapping_chunks()
        
        start_time = time.time()
        total_mutations_found = 0
        total_bases_processed = 0
        
        # Terminal UI Logic
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan"),
            TextColumn("[cyan]Found: {task.fields[found]} | Rate: {task.fields[rate]} MB/s"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[bold cyan]Executing Sequence Alignment...", 
                total=None, 
                found=0,
                rate="0.0"
            )
            
            # Using ThreadPoolExecutor logic here allows the main thread to 
            # manage UI while worker threads execute KMP (even with GIL, it's structurally parallel)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for batch_payload in optimizer.build_batches(generator):
                    futures.append(executor.submit(engine.process_batch, batch_payload))
                    
                    # Estimate throughput for UI
                    if batch_payload["total_bases"] > 0:
                        total_bases_processed += batch_payload["total_bases"]
                    else:
                        total_bases_processed += optimizer.target_batch_bytes
                        
                # Collect results
                for future in as_completed(futures):
                    total_mutations_found += future.result()
                    
                    elapsed = time.time() - start_time
                    current_rate = (total_bases_processed / (1024*1024)) / elapsed if elapsed > 0 else 0
                    
                    progress.update(task, found=total_mutations_found, rate=f"{current_rate:.1f}")
                    
        # Metric Calculations
        total_time_elapsed = time.time() - start_time
        processed_mb = total_bases_processed / (1024 * 1024)
        
        metrics = EngineMetrics(
            total_time_sec=total_time_elapsed,
            data_processed_mb=processed_mb,
            total_bases=total_bases_processed,
            throughput_mb_s=processed_mb / total_time_elapsed if total_time_elapsed > 0 else 0,
            throughput_bp_s=total_bases_processed / total_time_elapsed if total_time_elapsed > 0 else 0
        )
        
        accuracy_report = engine.analytics.calculate_accuracy()
        
        # Generate JSON & CSV Audit Reports
        engine.analytics.export_audit_report(metrics, accuracy_report)
        
        optimizer.shutdown()
        display_baseline_dashboard(metrics, accuracy_report)
        
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Execution aborted by user (SIGINT).[/bold red]")
        sys.exit(0)
    except Exception as e:
        logger.exception("Baseline Engine encountered a critical operational failure.")
        sys.exit(1)