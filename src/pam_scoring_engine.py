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
from dask.distributed import as_completed
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

class BiologicalScoringError(Exception):
    def __init__(self, message: str, error_code: int = 900):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class MatrixComputationError(BiologicalScoringError):
    def __init__(self, message: str):
        super().__init__(message, error_code=901)

class PAMValidationError(BiologicalScoringError):
    def __init__(self, message: str):
        super().__init__(message, error_code=902)

@dataclass
class HardwareProfile:
    os_platform: str
    logical_cores: int
    physical_cores: int
    max_cpu_freq_mhz: float
    total_ram_gb: float
    network_hostname: str

@dataclass
class SequenceTarget:
    target_id: str
    raw_sequence: bytes
    sequence_length: int
    pam_motif: str
    creation_timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class PAMScoreResult:
    target_id: str
    worker_id: str
    pam_found: bool
    biological_score: float
    gc_content_pct: float
    execution_time_ms: float
    gpu_offload_used: bool
    completion_timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class ScorerHealthReport:
    session_id: str
    total_sequences_scored: int
    viable_targets_found: int
    average_biological_score: float
    global_execution_time_ms: float
    hardware_profile: HardwareProfile
    worker_distribution: Dict[str, int]
    scoring_ledger: List[PAMScoreResult]

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
        
        return HardwareProfile(
            os_platform=sys.platform,
            logical_cores=psutil.cpu_count(logical=True) or 2,
            physical_cores=psutil.cpu_count(logical=False) or 1,
            max_cpu_freq_mhz=max_freq,
            total_ram_gb=mem.total / (1024 ** 3),
            network_hostname=self.hostname
        )

class PAMScoringMatrix:
    def __init__(self, ideal_gc_min: float = 40.0, ideal_gc_max: float = 60.0):
        self.ideal_gc_min = ideal_gc_min
        self.ideal_gc_max = ideal_gc_max

    def calculate_gc_content(self, sequence: bytes) -> float:
        gc_count = sequence.count(b'G') + sequence.count(b'C')
        total_len = len(sequence)
        return (gc_count / total_len) * 100.0 if total_len > 0 else 0.0

    def evaluate_target(self, sequence: bytes, motif: str) -> Tuple[bool, float, float]:
        motif_bytes = motif.encode("ascii")
        pam_index = sequence.find(motif_bytes)
        gc_pct = self.calculate_gc_content(sequence)
        
        if pam_index == -1:
            return False, 0.0, gc_pct
            
        base_score = 100.0
        
        if gc_pct < self.ideal_gc_min:
            base_score -= (self.ideal_gc_min - gc_pct) * 1.5
        elif gc_pct > self.ideal_gc_max:
            base_score -= (gc_pct - self.ideal_gc_max) * 1.5
            
        optimal_position = len(sequence) * 0.75
        distance_penalty = abs(pam_index - optimal_position) * 0.05
        base_score -= distance_penalty
        
        final_score = max(0.0, min(100.0, base_score))
        return True, final_score, gc_pct

class DistributedTaskFactory:
    def __init__(self, target_pam: str = "NGG"):
        self.target_pam = target_pam
        self.nucleotides = [65, 67, 71, 84]

    def generate_targets(self, count: int, seq_length: int = 256) -> List[SequenceTarget]:
        targets = []
        for _ in range(count):
            base_array = np.random.choice(self.nucleotides, size=seq_length).astype(np.uint8)
            
            inject_pam = np.random.choice([True, False], p=[0.7, 0.3])
            if inject_pam:
                injection_point = int(seq_length * np.random.uniform(0.5, 0.9))
                if injection_point + 3 < seq_length:
                    base_array[injection_point] = np.random.choice(self.nucleotides)
                    base_array[injection_point+1] = 71
                    base_array[injection_point+2] = 71
                    
            targets.append(SequenceTarget(
                target_id=f"SEQ_{uuid.uuid4().hex[:8].upper()}",
                raw_sequence=base_array.tobytes(),
                sequence_length=seq_length,
                pam_motif=self.target_pam
            ))
        return targets

def distributed_pam_scorer_task(target: SequenceTarget) -> PAMScoreResult:
    start_time = time.perf_counter()
    try:
        worker = get_worker()
        worker_id = str(worker.id)
    except Exception:
        worker_id = "local_score_node_" + str(uuid.uuid4())[:8]
        
    matrix = PAMScoringMatrix()
    
    time.sleep(np.random.uniform(0.05, 0.15))
    
    pam_found = False
    best_score = 0.0
    final_gc = 0.0
    
    raw_seq = target.raw_sequence
    for n in [b'A', b'C', b'G', b'T']:
        simulated_motif = n + b'GG'
        found, score, gc = matrix.evaluate_target(raw_seq, simulated_motif.decode("ascii"))
        if found and score > best_score:
            pam_found = True
            best_score = score
            final_gc = gc
            
    if not pam_found:
        final_gc = matrix.calculate_gc_content(raw_seq)
        
    end_time = time.perf_counter()
    exec_time_ms = (end_time - start_time) * 1000.0
    
    return PAMScoreResult(
        target_id=target.target_id,
        worker_id=worker_id,
        pam_found=pam_found,
        biological_score=best_score,
        gc_content_pct=final_gc,
        execution_time_ms=exec_time_ms,
        gpu_offload_used=False
    )

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
        
        header_text = Text("GENEWEAVER: HIGH-PERFORMANCE BIOINFORMATICS PIPELINE\nModule: Distributed PAM Scoring Engine", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(header_text, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("INITIALIZING SCORING MATRIX...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update_progress(self, layout: Layout, results: List[PAMScoreResult], total_expected: int, hw_profile: HardwareProfile):
        table = Table(box=None, expand=True)
        table.add_column("Sequence ID", style="cyan")
        table.add_column("Node ID", style="magenta")
        table.add_column("PAM Status", style="blue", justify="center")
        table.add_column("Viability Score", style="bold green", justify="right")
        table.add_column("GC %", style="yellow", justify="right")
        table.add_column("Latency", style="white", justify="right")
        
        for res in results[-12:]:
            status_str = "[green]FOUND[/green]" if res.pam_found else "[red]ABSENT[/red]"
            score_str = f"{res.biological_score:.2f}/100" if res.pam_found else "N/A"
            table.add_row(
                res.target_id,
                str(res.worker_id)[:15] + "..",
                status_str,
                score_str,
                f"{res.gc_content_pct:.1f}%",
                f"{res.execution_time_ms:.1f} ms"
            )
            
        layout["left_panel"].update(Panel(table, title="[bold green]Live Biological Evaluation[/bold green]", border_style="green"))
        
        stats_table = Table(box=None, expand=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow", justify="right")
        
        processed = len(results)
        progress_pct = (processed / total_expected) * 100 if total_expected > 0 else 0
        viable_count = sum(1 for r in results if r.pam_found and r.biological_score >= 70.0)
        avg_score = sum(r.biological_score for r in results if r.pam_found) / max(1, sum(1 for r in results if r.pam_found))
        
        stats_table.add_row("Targets Evaluated", str(processed))
        stats_table.add_row("Matrix Progress", f"{progress_pct:.1f}%")
        stats_table.add_row("Highly Viable Targets", f"[bold green]{viable_count}[/bold green]")
        stats_table.add_row("Avg Viability Score", f"{avg_score:.2f}")
        
        layout["stats_panel"].update(Panel(stats_table, title="[bold magenta]Scoring Analytics[/bold magenta]", border_style="magenta"))
        
        sys_table = Table(box=None, expand=True)
        sys_table.add_column("Resource", style="cyan")
        sys_table.add_column("State", style="white", justify="right")
        sys_table.add_row("Active Cores", str(hw_profile.logical_cores))
        sys_table.add_row("Total RAM", f"{hw_profile.total_ram_gb:.1f} GB")
        sys_table.add_row("System OS", hw_profile.os_platform.upper())
        
        layout["sys_panel"].update(Panel(sys_table, title="[bold blue]Hardware Limits[/bold blue]", border_style="blue"))
        layout["footer"].update(Panel(Text(f"EVALUATING SEQUENCES ACROSS CLUSTER... [{processed}/{total_expected}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class TelemetryExporter:
    def __init__(self, export_dir: str = "reports/pam_scoring"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def save_report(self, report: ScorerHealthReport) -> str:
        file_path = os.path.join(self.export_dir, f"scoring_audit_{report.session_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(asdict(report), f, indent=4)
        except Exception:
            pass
        return file_path

class MasterScoringOrchestrator:
    def __init__(self):
        self.diag_engine = HardwareDiagnosticEngine()
        self.factory = DistributedTaskFactory(target_pam="NGG")
        self.renderer = DashboardRenderer()
        self.exporter = TelemetryExporter()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hw_profile = self.diag_engine.generate_profile()

    def execute_scoring_pipeline(self):
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
            console.print(f"[bold red]Failed to initialize cluster: {str(e)}[/bold red]")
            sys.exit(1)
            
        layout = self.renderer.generate_layout()
        targets = self.factory.generate_targets(count=150, seq_length=512)
        total_targets = len(targets)
        completed_results = []
        
        global_start = time.perf_counter()
        
        with Live(layout, refresh_per_second=8, screen=True):
            time.sleep(1)
            futures = client.map(distributed_pam_scorer_task, targets)
            
            seq = as_completed(futures)
            
            for future in seq:
                try:
                    result = future.result()
                    if isinstance(result, PAMScoreResult):
                        completed_results.append(result)
                        self.renderer.update_progress(layout, completed_results, total_targets, self.hw_profile)
                except Exception:
                    pass
                    
            global_end = time.perf_counter()
            global_exec_ms = (global_end - global_start) * 1000.0
            
            layout["footer"].update(Panel(Text("BIOLOGICAL SCORING COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
                
        client.close()
        cluster.close()
        
        worker_counts = {}
        viable = 0
        total_score = 0.0
        for r in completed_results:
            worker_counts[r.worker_id] = worker_counts.get(r.worker_id, 0) + 1
            if r.pam_found:
                total_score += r.biological_score
                if r.biological_score >= 70.0:
                    viable += 1
                    
        report = ScorerHealthReport(
            session_id=self.session_id,
            total_sequences_scored=len(completed_results),
            viable_targets_found=viable,
            average_biological_score=total_score / max(1, sum(1 for r in completed_results if r.pam_found)),
            global_execution_time_ms=global_exec_ms,
            hardware_profile=self.hw_profile,
            worker_distribution=worker_counts,
            scoring_ledger=completed_results
        )
        
        path = self.exporter.save_report(report)
        console.print(f"\n[bold bright_green][+] Day 4 PAM Scoring Engine Completed. {viable} Highly Viable Targets Found.[/bold bright_green]")
        console.print(f"[bold cyan][i] Matrix Audit saved to: {path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    orchestrator = MasterScoringOrchestrator()
    orchestrator.execute_scoring_pipeline()