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
from typing import List, Dict, Any, Optional, Tuple, Set

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

class SchemaUpdateError(Exception):
    def __init__(self, message: str, error_code: int = 1200):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class StrictTypingViolation(SchemaUpdateError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1201)

class AggregationTimeoutError(SchemaUpdateError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1202)

@dataclass
class HardwareFootprint:
    node_id: str
    memory_usage_mb: float
    cpu_utilization_pct: float

@dataclass
class SequenceIdentifier:
    uid: str
    batch_id: str
    origin_source: str

@dataclass
class MutationFinding:
    mutation_id: str
    sequence_ref: SequenceIdentifier
    target_motif_detected: bool
    motif_absolute_position: int
    pam_distance: int
    severity_score: float
    flanking_gc_content: float
    worker_footprint: HardwareFootprint
    processing_latency_ms: float
    schema_version: str = "v2.1.0-strict"
    timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class AggregatedLedger:
    session_id: str
    total_findings_processed: int
    valid_schemas_enforced: int
    schema_violations: int
    average_severity_score: float
    maximum_severity_score: float
    worker_node_participation: Dict[str, int]
    compiled_mutations: List[MutationFinding]

class StrictSchemaValidator:
    def __init__(self, expected_version: str = "v2.1.0-strict"):
        self.expected_version = expected_version

    def validate_mutation_finding(self, finding: MutationFinding) -> bool:
        if not isinstance(finding.severity_score, float):
            return False
        if not isinstance(finding.pam_distance, int):
            return False
        if finding.severity_score < 0.0 or finding.severity_score > 100.0:
            return False
        if finding.schema_version != self.expected_version:
            return False
        if len(finding.mutation_id) < 8:
            return False
        return True

class GenomicDataTransformer:
    def __init__(self):
        self.bases = [b'A', b'C', b'G', b'T']

    def simulate_mutation_scan(self, raw_data: bytes, target: bytes) -> Tuple[bool, int, int, float, float]:
        pos = raw_data.find(target)
        if pos == -1:
            return False, -1, 0, 0.0, 0.0
            
        distance = int(np.random.normal(15, 5))
        severity = max(0.0, min(100.0, 100.0 - (abs(distance) * 2.5)))
        gc_pct = float(np.random.uniform(40.0, 60.0))
        
        return True, pos, distance, severity, gc_pct

def dask_schema_worker(payload_tuple: Tuple[str, str, bytes, bytes]) -> MutationFinding:
    start_time = time.perf_counter()
    uid, batch, raw_seq, motif = payload_tuple
    
    try:
        worker = get_worker()
        wid = str(worker.id)
        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=0.0)
    except Exception:
        wid = f"local_thread_{uuid.uuid4().hex[:6]}"
        mem = 0.0
        cpu = 0.0
        
    time.sleep(np.random.uniform(0.02, 0.08))
    
    transformer = GenomicDataTransformer()
    found, pos, dist, sev, gc = transformer.simulate_mutation_scan(raw_seq, motif)
    
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000.0
    
    return MutationFinding(
        mutation_id=f"MUT_{uuid.uuid4().hex[:8].upper()}",
        sequence_ref=SequenceIdentifier(uid=uid, batch_id=batch, origin_source="HDFS_STREAM"),
        target_motif_detected=found,
        motif_absolute_position=pos,
        pam_distance=dist,
        severity_score=sev,
        flanking_gc_content=gc,
        worker_footprint=HardwareFootprint(node_id=wid, memory_usage_mb=mem, cpu_utilization_pct=cpu),
        processing_latency_ms=latency
    )

class UIDashboardRenderer:
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
            Layout(name="schema_panel", ratio=1),
            Layout(name="agg_panel", ratio=1)
        )
        
        header_text = Text("GENEWEAVER INDUSTRY PIPELINE: SCHEMA ENFORCEMENT & AGGREGATOR (DAY 6)", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(header_text, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("AGGREGATING MUTATION FINDINGS INTO MAIN THREAD...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update_dashboard(self, layout: Layout, findings: List[MutationFinding], violations: int, total: int):
        t_live = Table(box=None, expand=True)
        t_live.add_column("Mutation ID", style="cyan")
        t_live.add_column("Node Source", style="magenta")
        t_live.add_column("PAM Dist", style="blue", justify="right")
        t_live.add_column("Severity Score", style="bold red", justify="right")
        t_live.add_column("Schema", style="green", justify="right")
        
        for f in findings[-15:]:
            schema_badge = f"[green]{f.schema_version}[/green]"
            score_str = f"{f.severity_score:.2f}" if f.target_motif_detected else "-"
            dist_str = f"{f.pam_distance} bp" if f.target_motif_detected else "-"
            
            t_live.add_row(
                f.mutation_id,
                str(f.worker_footprint.node_id)[:15] + "..",
                dist_str,
                score_str,
                schema_badge
            )
            
        layout["left_panel"].update(Panel(t_live, title="[bold green]Live Master Thread Aggregation[/bold green]", border_style="green"))
        
        t_schema = Table(box=None, expand=True)
        t_schema.add_column("Validation Rule", style="cyan")
        t_schema.add_column("Status", style="white", justify="right")
        t_schema.add_row("Strict Typing", "[green]ENFORCED[/green]")
        t_schema.add_row("Version Match", "v2.1.0-strict")
        t_schema.add_row("Valid Records", str(len(findings)))
        t_schema.add_row("Schema Violations", f"[red]{violations}[/red]" if violations > 0 else "[green]0[/green]")
        layout["schema_panel"].update(Panel(t_schema, title="[bold blue]Schema Enforcer Metrics[/bold blue]", border_style="blue"))
        
        t_agg = Table(box=None, expand=True)
        t_agg.add_column("Aggregation Stat", style="cyan")
        t_agg.add_column("Value", style="white", justify="right")
        
        viable_mutations = [f for f in findings if f.target_motif_detected]
        avg_sev = sum(f.severity_score for f in viable_mutations) / len(viable_mutations) if viable_mutations else 0.0
        max_sev = max([f.severity_score for f in viable_mutations], default=0.0)
        
        t_agg.add_row("Total Compiled", f"{len(findings)} / {total}")
        t_agg.add_row("Motifs Confirmed", str(len(viable_mutations)))
        t_agg.add_row("Global Avg Severity", f"{avg_sev:.2f}")
        t_agg.add_row("Peak Severity", f"{max_sev:.2f}")
        layout["agg_panel"].update(Panel(t_agg, title="[bold magenta]Main Thread Analytics[/bold magenta]", border_style="magenta"))
        
        processed = len(findings) + violations
        layout["footer"].update(Panel(Text(f"VALIDATING STRICT DATACLASSES... [{processed}/{total}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class ReportExporter:
    def __init__(self, output_dir: str = "reports/schema_aggregation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, ledger: AggregatedLedger) -> str:
        filepath = os.path.join(self.output_dir, f"ledger_{ledger.session_id}.json")
        with open(filepath, "w") as f:
            json.dump(asdict(ledger), f, indent=4)
        return filepath

class MainThreadOrchestrator:
    def __init__(self):
        self.renderer = UIDashboardRenderer()
        self.exporter = ReportExporter()
        self.validator = StrictSchemaValidator()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def prepare_payloads(self, count: int) -> List[Tuple[str, str, bytes, bytes]]:
        payloads = []
        bases = [65, 67, 71, 84]
        for _ in range(count):
            uid = f"SEQ_{uuid.uuid4().hex[:6].upper()}"
            batch = "BATCH_v2.1"
            seq = np.random.choice(bases, size=512).astype(np.uint8).tobytes()
            target = b"GATC"
            
            if np.random.random() > 0.4:
                inject_pos = int(np.random.uniform(10, 500))
                seq = seq[:inject_pos] + target + seq[inject_pos+4:]
                
            payloads.append((uid, batch, seq, target))
        return payloads

    def execute_aggregation(self):
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
            console.print(f"[bold red]Cluster init failed: {str(e)}[/bold red]")
            sys.exit(1)
            
        layout = self.renderer.generate_layout()
        payloads = self.prepare_payloads(250)
        total_payloads = len(payloads)
        
        valid_findings = []
        schema_violations = 0
        
        with Live(layout, refresh_per_second=10, screen=True):
            time.sleep(1)
            futures = client.map(dask_schema_worker, payloads)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, MutationFinding):
                        if self.validator.validate_mutation_finding(result):
                            valid_findings.append(result)
                        else:
                            schema_violations += 1
                        
                        self.renderer.update_dashboard(layout, valid_findings, schema_violations, total_payloads)
                except Exception:
                    pass
                    
            layout["footer"].update(Panel(Text("SCHEMA AGGREGATION COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
                
        client.close()
        cluster.close()
        
        worker_counts = {}
        viable = [f for f in valid_findings if f.target_motif_detected]
        avg_sev = sum(f.severity_score for f in viable) / len(viable) if viable else 0.0
        max_sev = max([f.severity_score for f in viable], default=0.0)
        
        for f in valid_findings:
            node = f.worker_footprint.node_id
            worker_counts[node] = worker_counts.get(node, 0) + 1
            
        ledger = AggregatedLedger(
            session_id=self.session_id,
            total_findings_processed=len(valid_findings) + schema_violations,
            valid_schemas_enforced=len(valid_findings),
            schema_violations=schema_violations,
            average_severity_score=avg_sev,
            maximum_severity_score=max_sev,
            worker_node_participation=worker_counts,
            compiled_mutations=valid_findings
        )
        
        path = self.exporter.export(ledger)
        console.print(f"\n[bold bright_green][+] Day 6 Pipeline & Schema Update Complete. All findings aggregated to Main Thread.[/bold bright_green]")
        console.print(f"[bold cyan][i] Aggregation Ledger JSON saved to: {path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    app = MainThreadOrchestrator()
    app.execute_aggregation()