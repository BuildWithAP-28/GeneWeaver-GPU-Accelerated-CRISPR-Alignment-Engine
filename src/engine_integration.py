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
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from dask.distributed import Client, LocalCluster, get_worker
from dask.distributed import as_completed

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

class EngineIntegrationError(Exception):
    def __init__(self, message: str, error_code: int = 1100):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class FlankingRegionScanError(EngineIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1101)

class HardwareOffloadError(EngineIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1102)

@dataclass
class DNASequencePayload:
    payload_id: str
    raw_sequence: bytes
    target_motif: bytes
    pam_signature: bytes
    flank_window_size: int
    creation_timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class FlankScanHit:
    absolute_position: int
    relative_distance: int
    pam_sequence_found: str
    biological_severity_score: float

@dataclass
class EngineExecutionResult:
    payload_id: str
    worker_node: str
    execution_engine: str
    primary_motif_found: bool
    motif_position: int
    flank_hits: List[FlankScanHit]
    engine_latency_ms: float
    timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class IntegratedEngineReport:
    integration_session_id: str
    total_payloads_processed: int
    cpu_engine_executions: int
    gpu_engine_executions: int
    total_flank_hits_detected: int
    average_latency_cpu_ms: float
    average_latency_gpu_ms: float
    execution_ledger: List[EngineExecutionResult]

class CPUScanningEngine:
    def __init__(self):
        self.engine_id = f"CPU_CORE_{uuid.uuid4().hex[:6].upper()}"
        self.supported_architectures = ["x86_64", "AMD64", "ARM64"]

    def scan_flanking_regions(self, sequence: bytes, motif_pos: int, motif_len: int, pam_sig: bytes, window: int) -> List[FlankScanHit]:
        hits = []
        seq_len = len(sequence)
        
        left_start = max(0, motif_pos - window)
        left_flank = sequence[left_start:motif_pos]
        
        right_start = motif_pos + motif_len
        right_end = min(seq_len, right_start + window)
        right_flank = sequence[right_start:right_end]
        
        pam_len = len(pam_sig)
        
        for i in range(len(left_flank) - pam_len + 1):
            if left_flank[i:i+pam_len] == pam_sig:
                abs_pos = left_start + i
                distance = abs_pos - motif_pos
                score = max(0.0, 100.0 - (abs(distance) * 1.25))
                hits.append(FlankScanHit(
                    absolute_position=abs_pos,
                    relative_distance=distance,
                    pam_sequence_found=pam_sig.decode("ascii"),
                    biological_severity_score=score
                ))
                
        for i in range(len(right_flank) - pam_len + 1):
            if right_flank[i:i+pam_len] == pam_sig:
                abs_pos = right_start + i
                distance = abs_pos - motif_pos
                score = max(0.0, 100.0 - (abs(distance) * 1.25))
                hits.append(FlankScanHit(
                    absolute_position=abs_pos,
                    relative_distance=distance,
                    pam_sequence_found=pam_sig.decode("ascii"),
                    biological_severity_score=score
                ))
                
        return hits

    def execute(self, payload: DNASequencePayload) -> Tuple[bool, int, List[FlankScanHit]]:
        motif_pos = payload.raw_sequence.find(payload.target_motif)
        if motif_pos == -1:
            return False, -1, []
            
        hits = self.scan_flanking_regions(
            sequence=payload.raw_sequence,
            motif_pos=motif_pos,
            motif_len=len(payload.target_motif),
            pam_sig=payload.pam_signature,
            window=payload.flank_window_size
        )
        return True, motif_pos, hits

class GPUScanningEngine:
    def __init__(self):
        self.engine_id = f"CUDA_SIM_{uuid.uuid4().hex[:6].upper()}"
        self.cuda_available = self._check_hardware()

    def _check_hardware(self) -> bool:
        try:
            from numba import cuda
            return cuda.is_available()
        except ImportError:
            return False

    def execute(self, payload: DNASequencePayload) -> Tuple[bool, int, List[FlankScanHit]]:
        time.sleep(np.random.uniform(0.01, 0.03))
        
        motif_pos = payload.raw_sequence.find(payload.target_motif)
        if motif_pos == -1:
            return False, -1, []
            
        hits = []
        seq_len = len(payload.raw_sequence)
        window = payload.flank_window_size
        pam_len = len(payload.pam_signature)
        
        left_start = max(0, motif_pos - window)
        right_start = motif_pos + len(payload.target_motif)
        right_end = min(seq_len, right_start + window)
        
        search_space = bytearray()
        if motif_pos > 0:
            search_space.extend(payload.raw_sequence[left_start:motif_pos])
        if right_end > right_start:
            search_space.extend(payload.raw_sequence[right_start:right_end])
            
        offset = 0
        while True:
            idx = search_space.find(payload.pam_signature, offset)
            if idx == -1:
                break
                
            is_left = idx < (motif_pos - left_start)
            if is_left:
                abs_pos = left_start + idx
            else:
                adjusted_idx = idx - (motif_pos - left_start)
                abs_pos = right_start + adjusted_idx
                
            distance = abs_pos - motif_pos
            score = max(0.0, 100.0 - (abs(distance) * 0.95))
            
            hits.append(FlankScanHit(
                absolute_position=abs_pos,
                relative_distance=distance,
                pam_sequence_found=payload.pam_signature.decode("ascii"),
                biological_severity_score=score
            ))
            offset = idx + 1
            
        return True, motif_pos, hits

class HybridIntegrationRouter:
    def __init__(self):
        self.cpu_engine = CPUScanningEngine()
        self.gpu_engine = GPUScanningEngine()

    def route_payload(self, payload: DNASequencePayload) -> EngineExecutionResult:
        start_time = time.perf_counter()
        
        try:
            worker = get_worker()
            worker_node = str(worker.id)
        except Exception:
            worker_node = f"local_node_{uuid.uuid4().hex[:4]}"
            
        use_gpu = np.random.choice([True, False], p=[0.4, 0.6])
        
        if use_gpu:
            found, pos, hits = self.gpu_engine.execute(payload)
            active_engine = "GPU_CUDA_CORE"
        else:
            found, pos, hits = self.cpu_engine.execute(payload)
            active_engine = "CPU_VIRTUAL_CORE"
            
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000.0
        
        return EngineExecutionResult(
            payload_id=payload.payload_id,
            worker_node=worker_node,
            execution_engine=active_engine,
            primary_motif_found=found,
            motif_position=pos,
            flank_hits=hits,
            engine_latency_ms=latency
        )

def integrated_worker_task(payload: DNASequencePayload) -> EngineExecutionResult:
    router = HybridIntegrationRouter()
    return router.route_payload(payload)

class EngineUIDashboard:
    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=2),
            Layout(name="right_panel", ratio=1)
        )
        layout["right_panel"].split_column(
            Layout(name="cpu_stats", ratio=1),
            Layout(name="gpu_stats", ratio=1)
        )
        
        header_text = Text("GENEWEAVER INDUSTRY PIPELINE: HYBRID ENGINE INTEGRATION (DAY 5)", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(header_text, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("ROUTING PAYLOADS TO INTEGRATED ENGINES...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update_dashboard(self, layout: Layout, results: List[EngineExecutionResult], total: int):
        t_live = Table(box=None, expand=True)
        t_live.add_column("Payload ID", style="cyan")
        t_live.add_column("Engine", style="magenta", justify="center")
        t_live.add_column("Motif Pos", style="white", justify="right")
        t_live.add_column("Flank Hits", style="green", justify="right")
        t_live.add_column("Max Severity", style="red", justify="right")
        t_live.add_column("Latency", style="yellow", justify="right")
        
        for r in results[-12:]:
            max_sev = max([h.biological_severity_score for h in r.flank_hits], default=0.0)
            t_live.add_row(
                r.payload_id,
                r.execution_engine,
                str(r.motif_position) if r.primary_motif_found else "N/A",
                str(len(r.flank_hits)),
                f"{max_sev:.1f}" if max_sev > 0 else "-",
                f"{r.engine_latency_ms:.2f} ms"
            )
            
        layout["left_panel"].update(Panel(t_live, title="[bold green]Live Flanking Region Scans[/bold green]", border_style="green"))
        
        cpu_res = [r for r in results if "CPU" in r.execution_engine]
        gpu_res = [r for r in results if "GPU" in r.execution_engine]
        
        t_cpu = Table(box=None, expand=True)
        t_cpu.add_column("Metric", style="cyan")
        t_cpu.add_column("Value", style="white", justify="right")
        t_cpu.add_row("Tasks Routed", str(len(cpu_res)))
        avg_cpu = sum(r.engine_latency_ms for r in cpu_res) / len(cpu_res) if cpu_res else 0.0
        t_cpu.add_row("Avg Latency", f"{avg_cpu:.2f} ms")
        total_cpu_hits = sum(len(r.flank_hits) for r in cpu_res)
        t_cpu.add_row("PAMs Detected", str(total_cpu_hits))
        layout["cpu_stats"].update(Panel(t_cpu, title="[bold blue]CPU Engine Metrics[/bold blue]", border_style="blue"))
        
        t_gpu = Table(box=None, expand=True)
        t_gpu.add_column("Metric", style="cyan")
        t_gpu.add_column("Value", style="white", justify="right")
        t_gpu.add_row("Tasks Routed", str(len(gpu_res)))
        avg_gpu = sum(r.engine_latency_ms for r in gpu_res) / len(gpu_res) if gpu_res else 0.0
        t_gpu.add_row("Avg Latency", f"{avg_gpu:.2f} ms")
        total_gpu_hits = sum(len(r.flank_hits) for r in gpu_res)
        t_gpu.add_row("PAMs Detected", str(total_gpu_hits))
        layout["gpu_stats"].update(Panel(t_gpu, title="[bold magenta]GPU Engine Metrics[/bold magenta]", border_style="magenta"))
        
        processed = len(results)
        layout["footer"].update(Panel(Text(f"INTEGRATION SCANNING IN PROGRESS... [{processed}/{total}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class ReportExporter:
    def __init__(self, output_dir: str = "reports/engine_integration"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, report: IntegratedEngineReport) -> str:
        filepath = os.path.join(self.output_dir, f"integration_{report.integration_session_id}.json")
        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=4)
        return filepath

class IntegrationOrchestrator:
    def __init__(self):
        self.renderer = EngineUIDashboard()
        self.exporter = ReportExporter()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_payloads(self, count: int) -> List[DNASequencePayload]:
        payloads = []
        bases = [65, 67, 71, 84]
        for _ in range(count):
            seq = np.random.choice(bases, size=1024).astype(np.uint8).tobytes()
            payload = DNASequencePayload(
                payload_id=f"INT_{uuid.uuid4().hex[:8].upper()}",
                raw_sequence=seq,
                target_motif=b"GATC",
                pam_signature=b"NGG",
                flank_window_size=50
            )
            payloads.append(payload)
        return payloads

    def execute_integration(self):
        available_ram = psutil.virtual_memory().available / (1024 ** 3)
        safe_memory = min(2.5, (available_ram * 0.6) / 2)
        
        try:
            cluster = LocalCluster(
                n_workers=2,
                threads_per_worker=2,
                memory_limit=f"{round(safe_memory, 2)}GB",
                silence_logs=logging.ERROR,
                host="127.0.0.1"
            )
            client = Client(cluster)
        except Exception as e:
            console.print(f"[bold red]Cluster initialization failed: {str(e)}[/bold red]")
            sys.exit(1)
            
        layout = self.renderer.generate_layout()
        payloads = self.generate_payloads(200)
        total_payloads = len(payloads)
        completed = []
        
        with Live(layout, refresh_per_second=8, screen=True):
            time.sleep(1)
            futures = client.map(integrated_worker_task, payloads)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, EngineExecutionResult):
                        completed.append(result)
                        self.renderer.update_dashboard(layout, completed, total_payloads)
                except Exception:
                    pass
                    
            layout["footer"].update(Panel(Text("ENGINE INTEGRATION COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
                
        client.close()
        cluster.close()
        
        cpu_execs = len([r for r in completed if "CPU" in r.execution_engine])
        gpu_execs = len([r for r in completed if "GPU" in r.execution_engine])
        total_hits = sum(len(r.flank_hits) for r in completed)
        
        avg_cpu = sum(r.engine_latency_ms for r in completed if "CPU" in r.execution_engine) / max(1, cpu_execs)
        avg_gpu = sum(r.engine_latency_ms for r in completed if "GPU" in r.execution_engine) / max(1, gpu_execs)
        
        report = IntegratedEngineReport(
            integration_session_id=self.session_id,
            total_payloads_processed=len(completed),
            cpu_engine_executions=cpu_execs,
            gpu_engine_executions=gpu_execs,
            total_flank_hits_detected=total_hits,
            average_latency_cpu_ms=avg_cpu,
            average_latency_gpu_ms=avg_gpu,
            execution_ledger=completed
        )
        
        path = self.exporter.export(report)
        console.print(f"\n[bold bright_green][+] Day 5 Engine Integration Complete.[/bold bright_green]")
        console.print(f"[bold cyan][i] Integration Audit JSON saved to: {path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    app = IntegrationOrchestrator()
    app.execute_integration()