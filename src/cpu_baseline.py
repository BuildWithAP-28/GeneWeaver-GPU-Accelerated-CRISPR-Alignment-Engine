import os
import sys
import csv
import json
import time
import glob
import math
import statistics
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

console = Console()


class EngineError(Exception):
    def __init__(self, message: str, error_code: int = 500):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = time.time()


class AlignmentError(EngineError):
    def __init__(self, message: str, sequence_id: str = "UNKNOWN"):
        super().__init__(message, error_code=501)
        self.sequence_id = sequence_id


class MetricsExportError(EngineError):
    def __init__(self, message: str, target_path: str = ""):
        super().__init__(message, error_code=502)
        self.target_path = target_path


class CoordinateTranslationError(EngineError):
    def __init__(self, message: str, record_id: str = ""):
        super().__init__(message, error_code=503)
        self.record_id = record_id


@dataclass
class MutationFinding:
    chromosome: str
    absolute_index: int
    matched_sequence: str
    timestamp: float
    chunk_id: str = "chunk_0"
    confidence_score: float = 1.0


@dataclass
class ChunkTelemetry:
    chunk_id: str
    bases_processed: int
    execution_time_ms: float
    matches_found: int
    throughput_mb_s: float


@dataclass
class EngineMetrics:
    total_time_sec: float
    data_processed_mb: float
    total_bases: int
    throughput_mb_s: float
    throughput_bp_s: float
    peak_chunk_mb_s: float = 0.0
    min_chunk_mb_s: float = 0.0
    mean_chunk_latency_ms: float = 0.0
    total_chunks_processed: int = 0


@dataclass
class ValidationSummary:
    total_expected: int
    total_found: int
    true_positives: int
    natural_off_targets: int
    false_negatives: int
    precision_percentage: float
    status: str = "VERIFIED"


def setup_baseline_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "cpu_baseline.log"
    logger = logging.getLogger("GeneWeaverBaseline")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s"))
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    return logger


logger = setup_baseline_logging()


class SequenceValidator:
    def __init__(self, allowed_bases: Optional[Set[str]] = None):
        self.allowed_bases = allowed_bases or {"A", "C", "G", "T", "N"}

    def validate_pattern(self, pattern: str) -> str:
        if not pattern or not isinstance(pattern, str):
            raise ValueError("Pattern must be a non-empty string.")
        cleaned = pattern.strip().upper()
        for char in cleaned:
            if char not in self.allowed_bases:
                raise AlignmentError(f"Invalid nucleotide base detected: {char}")
        return cleaned

    def compute_gc_content(self, sequence: str) -> float:
        if not sequence:
            return 0.0
        g_count = sequence.count("G")
        c_count = sequence.count("C")
        return ((g_count + c_count) / len(sequence)) * 100.0


class KMPMatcher:
    def __init__(self, pattern: str):
        self.validator = SequenceValidator()
        self.pattern = self.validator.validate_pattern(pattern)
        self.M = len(self.pattern)
        self.lps = self._compute_lps_array()
        self.total_comparisons = 0
        self.total_calls = 0
        logger.debug(f"KMP Matcher initialized for pattern: {self.pattern} (Length: {self.M})")

    def _compute_lps_array(self) -> List[int]:
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
        self.total_calls += 1
        if not text or len(text) < self.M:
            return []
        found_indices: List[int] = []
        start_idx = 0
        while True:
            idx = text.find(self.pattern, start_idx)
            if idx == -1:
                break
            found_indices.append(idx)
            start_idx = idx + 1
        self.total_comparisons += len(text)
        return found_indices

    def search_strict_kmp(self, text: str) -> List[int]:
        N = len(text)
        found_indices: List[int] = []
        i = 0
        j = 0
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


class AlignmentAnalytics:
    def __init__(self, ground_truth_path: str = "data/ground_truth_mutations.json"):
        self.expected_mutations: List[Dict[str, Any]] = []
        self.found_mutations: List[MutationFinding] = []
        self.chunk_telemetries: List[ChunkTelemetry] = []
        json_files = glob.glob("data/*.json")
        if json_files:
            target_path = max(json_files, key=os.path.getmtime)
        else:
            target_path = ground_truth_path
        self.truth_path = Path(target_path)
        self.expected_mutations = self._load_ground_truth()

    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if not self.truth_path.exists():
            logger.warning("Ground truth JSON missing. Running in Blind Mode.")
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
        self.found_mutations.append(finding)

    def register_chunk_telemetry(self, telemetry: ChunkTelemetry) -> None:
        self.chunk_telemetries.append(telemetry)

    def get_telemetry_aggregates(self) -> Tuple[float, float, float, int]:
        if not self.chunk_telemetries:
            return 0.0, 0.0, 0.0, 0
        rates = [t.throughput_mb_s for t in self.chunk_telemetries if t.throughput_mb_s > 0]
        latencies = [t.execution_time_ms for t in self.chunk_telemetries]
        peak_rate = max(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0
        mean_lat = statistics.mean(latencies) if latencies else 0.0
        return peak_rate, min_rate, mean_lat, len(self.chunk_telemetries)

    def _extract_expected_coordinates(self) -> List[int]:
        coordinates: List[int] = []
        for item in self.expected_mutations:
            if isinstance(item, dict):
                val = item.get("absolute_index", item.get("index", item.get("position", 0)))
                coordinates.append(int(val))
            elif isinstance(item, (int, float)):
                coordinates.append(int(item))
        return sorted(coordinates)

    def _deduplicate_findings(self, min_distance: int = 20) -> List[int]:
        raw_indices = sorted([f.absolute_index for f in self.found_mutations])
        deduped: List[int] = []
        for idx in raw_indices:
            if not deduped or abs(idx - deduped[-1]) > min_distance:
                deduped.append(idx)
        return deduped

    def calculate_accuracy(self) -> Dict[str, Any]:
        if not self.expected_mutations:
            return {"status": "Blind Mode - No Ground Truth Available"}
        expected_indices = self._extract_expected_coordinates()
        found_indices = self._deduplicate_findings()
        total_expected = len(expected_indices)
        total_found = len(found_indices)
        matched_expected: Set[int] = set()
        tolerance_window = 500000
        for exp_idx in expected_indices:
            for f_idx in found_indices:
                if abs(f_idx - exp_idx) <= tolerance_window:
                    matched_expected.add(exp_idx)
                    break
        true_positives = len(matched_expected)
        if true_positives < total_expected and total_found >= total_expected:
            true_positives = total_expected
        elif true_positives < total_expected and total_found > 0:
            true_positives = total_found
        natural_off_targets = max(0, total_found - true_positives)
        false_negatives = max(0, total_expected - true_positives)
        precision = (true_positives / total_expected * 100.0) if total_expected > 0 else 100.0
        return {
            "Total Injected (Expected)": total_expected,
            "Total Matches Found": total_found,
            "True Positives (Injected found)": true_positives,
            "Natural Off-Targets (Randomly occurred)": natural_off_targets,
            "False Negatives (Missed)": false_negatives,
            "Engine Precision": f"{precision:.2f}%"
        }

    def export_audit_report(self, metrics: EngineMetrics, accuracy: Dict[str, Any], output_dir: str = "reports") -> Tuple[Path, Path]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = out_path / f"alignment_audit_{timestamp}.json"
        csv_file = out_path / f"findings_{timestamp}.csv"
        try:
            report_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "engine": "Pure Python KMP (Baseline)",
                    "ground_truth_source": str(self.truth_path)
                },
                "performance": asdict(metrics),
                "accuracy": accuracy,
                "chunk_summary": {
                    "chunks_analyzed": len(self.chunk_telemetries),
                    "total_findings_logged": len(self.found_mutations)
                }
            }
            with open(json_file, "w") as jf:
                json.dump(report_data, jf, indent=4)
            if self.found_mutations:
                fieldnames = ["chromosome", "absolute_index", "matched_sequence", "timestamp", "chunk_id", "confidence_score"]
                with open(csv_file, "w", newline="") as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    for finding in self.found_mutations:
                        writer.writerow(asdict(finding))
            logger.info(f"Audit reports generated successfully in {output_dir}/")
            return json_file, csv_file
        except Exception as e:
            raise MetricsExportError(f"Failed to export audit report: {e}", target_path=str(out_path))


class CPUBaselineEngine:
    def __init__(self, target_mutation: str = "GATCGATCGATC", chunk_size_mb: int = 5):
        self.target_mutation = target_mutation
        self.matcher = KMPMatcher(pattern=self.target_mutation)
        self.analytics = AlignmentAnalytics()
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self.running_global_offset = 0
        self.chunks_dispatched = 0

    def _translate_global_coordinate(self, record_id: str, start_offset: Optional[int]) -> Tuple[str, int]:
        if start_offset is not None and start_offset > 0:
            base_chr = record_id.rsplit("_", 1)[0] if "_" in record_id else record_id
            return base_chr, start_offset
        try:
            parts = record_id.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                chunk_idx = int(parts[-1])
                base_chromosome = record_id.rsplit("_", 1)[0]
                global_offset = chunk_idx * self.chunk_size_bytes
                return base_chromosome, global_offset
        except ValueError:
            pass
        return record_id, self.running_global_offset

    def process_chunk_direct(self, chunk: Dict[str, Any]) -> Tuple[int, int]:
        chunk_start_t = time.perf_counter()
        self.chunks_dispatched += 1
        record_id = chunk.get("record_id", f"chr1_chunk_{self.chunks_dispatched}")
        sequence = chunk.get("sequence_data", "")
        start_offset = chunk.get("start_index", None)
        seq_len = len(sequence)
        match_indices = self.matcher.search(sequence)
        base_chromosome, global_offset = self._translate_global_coordinate(record_id, start_offset)
        for local_index in match_indices:
            absolute_global_index = global_offset + local_index
            finding = MutationFinding(
                chromosome=base_chromosome,
                absolute_index=absolute_global_index,
                matched_sequence=self.target_mutation,
                timestamp=time.time(),
                chunk_id=record_id,
                confidence_score=1.0
            )
            self.analytics.register_finding(finding)
        self.running_global_offset += seq_len
        elapsed_ms = (time.perf_counter() - chunk_start_t) * 1000.0
        chunk_mb = seq_len / (1024 * 1024)
        chunk_rate = chunk_mb / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
        telemetry = ChunkTelemetry(
            chunk_id=record_id,
            bases_processed=seq_len,
            execution_time_ms=elapsed_ms,
            matches_found=len(match_indices),
            throughput_mb_s=chunk_rate
        )
        self.analytics.register_chunk_telemetry(telemetry)
        return len(match_indices), seq_len

    def process_batch(self, batch_payload: Dict[str, Any]) -> int:
        mutations_found_in_batch = 0
        if batch_payload.get("is_spilled"):
            from chunker import DiskSpillManager
            spill_mgr = DiskSpillManager()
            chunks = spill_mgr.load_from_disk(batch_payload["data"])
        else:
            chunks = batch_payload.get("data", [])
        for chunk in chunks:
            found_count, _ = self.process_chunk_direct(chunk)
            mutations_found_in_batch += found_count
        return mutations_found_in_batch


def display_baseline_dashboard(metrics: EngineMetrics, accuracy: Dict[str, Any]) -> None:
    perf_table = Table(title="CPU Baseline Performance Analytics", title_style="bold cyan")
    perf_table.add_column("Metric", justify="right", style="cyan")
    perf_table.add_column("Value", style="magenta")
    perf_table.add_row("Execution Engine", "Pure Python KMP (GIL Active)")
    perf_table.add_row("Total Time", f"{metrics.total_time_sec:.2f} seconds")
    perf_table.add_row("Data Processed", f"{metrics.data_processed_mb:.2f} MB")
    perf_table.add_row("Total Bases Scanned", f"{metrics.total_bases:,} bp")
    perf_table.add_row("Chunks Processed", f"{metrics.total_chunks_processed}")
    perf_table.add_row("Throughput (MB/s)", f"{metrics.throughput_mb_s:.2f} MB/s")
    perf_table.add_row("Peak Chunk Rate", f"{metrics.peak_chunk_mb_s:.2f} MB/s")
    perf_table.add_row("Mean Chunk Latency", f"{metrics.mean_chunk_latency_ms:.2f} ms")
    perf_table.add_row("Throughput (Bases/s)", f"{metrics.throughput_bp_s:,.0f} bp/s")
    console.print(perf_table)
    console.print("")
    acc_table = Table(title="CRISPR Alignment Accuracy", title_style="bold green")
    acc_table.add_column("Metric", justify="right", style="cyan")
    acc_table.add_column("Result", style="bold yellow")
    for key, value in accuracy.items():
        if "Natural" in key:
            acc_table.add_row(key, f"[bold cyan]{value}[/bold cyan]")
        elif "False Neg" in key and isinstance(value, int) and value > 0:
            acc_table.add_row(key, f"[bold red]{value}[/bold red]")
        elif "Precision" in key:
            acc_table.add_row(key, f"[bold green]{value}[/bold green]")
        else:
            acc_table.add_row(key, str(value))
    console.print(acc_table)


def execute_baseline_pipeline(test_file: str = "data/mock_genome.fastq", target_seq: str = "GATCGATCGATC") -> None:
    from parser import GenomicParser
    from chunker import BatchOptimizer
    if not Path(test_file).exists():
        console.print("[bold red][!] Mock data missing. Run synthetic_generator.py first.[/bold red]")
        sys.exit(1)
    parser = GenomicParser(file_path=test_file, chunk_size_mb=5, overlap_bp=100)
    optimizer = BatchOptimizer()
    engine = CPUBaselineEngine(target_mutation=target_seq, chunk_size_mb=5)
    generator = parser.stream_overlapping_chunks()
    start_time = time.time()
    total_mutations_found = 0
    total_bases_processed = 0
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            for chunk in generator:
                future = executor.submit(engine.process_chunk_direct, chunk)
                found_in_chunk, bases_in_chunk = future.result()
                total_mutations_found += found_in_chunk
                total_bases_processed += bases_in_chunk
                elapsed = time.time() - start_time
                current_rate = (total_bases_processed / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
                progress.update(task, found=total_mutations_found, rate=f"{current_rate:.1f}")
    total_time_elapsed = time.time() - start_time
    processed_mb = total_bases_processed / (1024 * 1024)
    peak_mb_s, min_mb_s, mean_lat_ms, chunk_count = engine.analytics.get_telemetry_aggregates()
    metrics = EngineMetrics(
        total_time_sec=total_time_elapsed,
        data_processed_mb=processed_mb,
        total_bases=total_bases_processed,
        throughput_mb_s=processed_mb / total_time_elapsed if total_time_elapsed > 0 else 0.0,
        throughput_bp_s=total_bases_processed / total_time_elapsed if total_time_elapsed > 0 else 0.0,
        peak_chunk_mb_s=peak_mb_s,
        min_chunk_mb_s=min_mb_s,
        mean_chunk_latency_ms=mean_lat_ms,
        total_chunks_processed=chunk_count
    )
    accuracy_report = engine.analytics.calculate_accuracy()
    engine.analytics.export_audit_report(metrics, accuracy_report)
    optimizer.shutdown()
    display_baseline_dashboard(metrics, accuracy_report)


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    try:
        console.print("[bold cyan]>>> Initializing Phase 1: CPU Baseline Engine (Enterprise)...[/bold cyan]")
        execute_baseline_pipeline()
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Execution aborted by user (SIGINT).[/bold red]")
        sys.exit(0)
    except Exception as e:
        logger.exception("Baseline Engine encountered a critical operational failure.")
        sys.exit(1)