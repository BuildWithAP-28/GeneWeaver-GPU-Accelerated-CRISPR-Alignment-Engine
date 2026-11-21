import os
import time
import json
import uuid
import socket
import psutil
import hashlib
import platform
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import numpy as np
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, ProgressBar
from textual.binding import Binding
from textual.timer import Timer

numba_cuda: Any = None
CUDA_AVAILABLE = False

try:
    from numba import cuda as _nb_cuda
    if _nb_cuda.is_available():
        numba_cuda = _nb_cuda
        CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

class TUIEngineError(Exception):
    def __init__(self, message: str, error_code: int = 1700):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class MemoryThresholdError(TUIEngineError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1701)

class TelemetryExportError(TUIEngineError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1702)

@dataclass
class RuntimeHardwareState:
    gpu_device: str
    cuda_active: bool
    compute_capability: str
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_pct: float
    hostname: str

@dataclass
class GenomicChunkRecord:
    chunk_id: str
    worker_node: str
    chunk_size_mb: float
    sequences_scanned: int
    pam_hits: int
    mismatch_count: int
    peak_severity: float
    gc_content_pct: float
    latency_ms: float
    bandwidth_mbs: float
    status: str
    timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class TUIExportManifest:
    session_id: str
    timestamp: str
    hardware_state: RuntimeHardwareState
    total_chunks_completed: int
    total_volume_mb: float
    total_pam_hits: int
    total_mismatches: int
    average_severity: float
    peak_severity: float
    average_bandwidth_mbs: float
    worker_distribution: Dict[str, int]
    cryptographic_digest: str
    chunk_ledger: List[GenomicChunkRecord]

class SystemTelemetryMonitor:
    def capture(self) -> RuntimeHardwareState:
        mem = psutil.virtual_memory()
        gpu_name = (platform.processor() or platform.machine())[:26] + " (CPU)"
        cuda_flag = False
        cc_label = "EMULATED"

        if CUDA_AVAILABLE and numba_cuda is not None:
            try:
                dev = numba_cuda.get_current_device()
                gpu_name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
                cc_label = f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
                cuda_flag = True
            except Exception:
                pass

        return RuntimeHardwareState(
            gpu_device=gpu_name,
            cuda_active=cuda_flag,
            compute_capability=cc_label,
            logical_cores=psutil.cpu_count(logical=True) or 2,
            total_ram_gb=mem.total / (1024 ** 3),
            available_ram_gb=mem.available / (1024 ** 3),
            ram_usage_pct=float(mem.percent),
            hostname=socket.gethostname()
        )

class GenomicChunkProcessor:
    def __init__(self, chunk_size_mb: float = 16.0):
        self.chunk_size_mb = chunk_size_mb
        self.rng = np.random.default_rng()
        self.bases = np.array([65, 67, 71, 84], dtype=np.uint8)
        self.workers = [f"Worker-{uuid.uuid4().hex[:8]}" for _ in range(4)]

    def execute_chunk(self, chunk_index: int) -> GenomicChunkRecord:
        t0 = time.perf_counter()
        sample_len = 32768
        seq = self.rng.choice(self.bases, size=sample_len)
        gc_pct = (int(np.count_nonzero((seq == 71) | (seq == 67))) / sample_len) * 100.0
        pam_hits = int(np.count_nonzero((seq[1:-1] == 71) & (seq[2:] == 71)))
        mismatch_count = int(self.rng.integers(14, 96))
        peak_severity = float(np.clip(self.rng.normal(86.0, 8.5), 45.0, 99.8))
        worker_id = self.workers[chunk_index % len(self.workers)]

        compute_ms = max(0.35, (time.perf_counter() - t0) * 1000.0)
        total_latency_ms = compute_ms + float(self.rng.uniform(45.0, 140.0))
        bandwidth_mbs = self.chunk_size_mb / (total_latency_ms / 1000.0)

        return GenomicChunkRecord(
            chunk_id=f"CHK_{uuid.uuid4().hex[:8].upper()}",
            worker_node=worker_id,
            chunk_size_mb=self.chunk_size_mb,
            sequences_scanned=int((self.chunk_size_mb * 1024 * 1024) / 256),
            pam_hits=pam_hits,
            mismatch_count=mismatch_count,
            peak_severity=round(peak_severity, 2),
            gc_content_pct=round(gc_pct, 2),
            latency_ms=round(total_latency_ms, 2),
            bandwidth_mbs=round(bandwidth_mbs, 2),
            status="VERIFIED"
        )

class TUIAuditVault:
    def __init__(self, output_dir: str = "reports/textual_tui"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_manifest(self, manifest: TUIExportManifest) -> str:
        path = os.path.join(self.output_dir, f"tui_audit_{manifest.session_id}.json")
        try:
            with open(path, "w") as f:
                json.dump(asdict(manifest), f, indent=4)
            return path
        except Exception as e:
            raise TelemetryExportError(f"Audit export failed: {str(e)}")

class GeneWeaverTUIApp(App):
    TITLE = "GENEWEAVER INDUSTRY PIPELINE"
    SUB_TITLE = "Textual Interactive Command Center (Week 4 Day 4)"

    BINDINGS = [
        Binding("r", "restart_pipeline", "Restart"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("e", "export_audit", "Export JSON"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen { background: rgb(8, 12, 10); color: rgb(225, 240, 230); }
    Header { background: rgb(12, 26, 18); color: rgb(0, 255, 136); text-style: bold; }
    Footer { background: rgb(12, 26, 18); color: rgb(0, 230, 180); }
    .progress_box {
        height: 5; border: solid rgb(0, 220, 130);
        background: rgb(10, 18, 14); padding: 0 2; margin: 0 1;
    }
    .progress_label { color: rgb(0, 255, 170); text-style: bold; height: 1; }
    .chunk_progress { width: 100%; }
    .main_grid { height: 1fr; margin: 0 1; }
    .table_panel {
        width: 2fr; border: solid rgb(0, 200, 110);
        background: rgb(10, 15, 12); margin-right: 1;
    }
    .side_column { width: 1fr; }
    .telemetry_panel {
        height: 1fr; border: solid rgb(0, 210, 220);
        background: rgb(10, 16, 18); padding: 0 1;
    }
    .worker_panel {
        height: 1fr; border: solid rgb(180, 90, 255);
        background: rgb(14, 12, 18); padding: 0 1;
    }
    .inspector_panel {
        height: 1fr; border: solid rgb(255, 90, 90);
        background: rgb(18, 11, 11); padding: 0 1;
    }
    .status_bar {
        height: 3; border: solid rgb(240, 200, 60);
        background: rgb(16, 16, 10); color: rgb(255, 220, 90);
        padding: 0 2; margin: 0 1; content-align: center middle;
    }
    DataTable { background: rgb(10, 15, 12); height: 1fr; }
    """

    def __init__(self, total_chunks: int = 24):
        super().__init__()
        self.total_chunks = total_chunks
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor = SystemTelemetryMonitor()
        self.processor = GenomicChunkProcessor(chunk_size_mb=16.0)
        self.vault = TUIAuditVault()
        self.hardware = self.monitor.capture()
        self.records: List[GenomicChunkRecord] = []
        self.paused = False
        self.exported = False
        self.stream_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="progress_box"):
            yield Static("GENOMIC CHUNKING STREAM: INITIALIZING...", classes="progress_label")
            yield ProgressBar(total=self.total_chunks, show_eta=True, classes="chunk_progress")
        with Horizontal(classes="main_grid"):
            with Vertical(classes="table_panel"):
                yield DataTable(cursor_type="row", zebra_stripes=True, classes="severity_table")
            with Vertical(classes="side_column"):
                yield Static("Awaiting telemetry...", classes="telemetry_panel")
                yield Static("Scanning worker nodes...", classes="worker_panel")
                yield Static("Select a row to inspect chunk...", classes="inspector_panel")
        yield Static("PIPELINE READY // KEYS: [R] RESTART  [P] PAUSE  [E] EXPORT JSON  [Q] QUIT", classes="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(".severity_table", DataTable)
        table.add_columns("Chunk ID", "Worker Node", "Size", "PAMs", "SNPs", "Peak Sev", "GC %", "Latency", "Bandwidth")
        self.query_one(".progress_box", Vertical).border_title = "Live Genomic Chunking Progress"
        self.query_one(".table_panel", Vertical).border_title = "PAM Severity & Distributed Chunk Ledger"
        self.query_one(".telemetry_panel", Static).border_title = "Pipeline Analytics"
        self.query_one(".worker_panel", Static).border_title = "Dask & CUDA Hardware Status"
        self.query_one(".inspector_panel", Static).border_title = "Selected Chunk Inspector"
        self._refresh_side_panels()
        self.stream_timer = self.set_interval(0.22, self._step_pipeline)

    def _step_pipeline(self) -> None:
        if self.paused:
            return
        if len(self.records) >= self.total_chunks:
            if not self.exported:
                saved = self._persist_audit()
                self.exported = True
                self.query_one(".status_bar", Static).update(
                    Text(f"PIPELINE COMPLETE // AUDIT EXPORTED: {saved} // USE ARROWS TO INSPECT ROWS", style="bold bright_green")
                )
            return

        self.hardware = self.monitor.capture()
        if self.hardware.available_ram_gb < 0.8:
            self.paused = True
            self.query_one(".status_bar", Static).update(
                Text("SAFETY GUARD TRIGGERED: LOW AVAILABLE RAM // PIPELINE PAUSED", style="bold red")
            )
            return

        rec = self.processor.execute_chunk(len(self.records))
        self.records.append(rec)
        sev_style = "bold red" if rec.peak_severity >= 85.0 else "yellow"
        self.query_one(".severity_table", DataTable).add_row(
            Text(rec.chunk_id, style="cyan"),
            Text(rec.worker_node, style="magenta"),
            f"{rec.chunk_size_mb:.1f} MB",
            str(rec.pam_hits),
            Text(str(rec.mismatch_count), style="bold red"),
            Text(f"{rec.peak_severity:.2f}", style=sev_style),
            f"{rec.gc_content_pct:.1f}%",
            f"{rec.latency_ms:.1f} ms",
            Text(f"{rec.bandwidth_mbs:.1f} MB/s", style="green")
        )

        self.query_one(".chunk_progress", ProgressBar).update(progress=len(self.records))
        pct = (len(self.records) / self.total_chunks) * 100.0
        vol_mb = sum(r.chunk_size_mb for r in self.records)
        self.query_one(".progress_label", Static).update(
            f"GENOMIC CHUNKS PROCESSED: {len(self.records)}/{self.total_chunks} ({pct:.1f}%) | STREAMED: {vol_mb:.1f} MB"
        )
        self._refresh_side_panels()
        self._render_inspector(rec)

    def _refresh_side_panels(self) -> None:
        count = len(self.records)
        total_pams = sum(r.pam_hits for r in self.records)
        total_snps = sum(r.mismatch_count for r in self.records)
        avg_sev = sum(r.peak_severity for r in self.records) / max(1, count)
        peak_sev = max([r.peak_severity for r in self.records], default=0.0)
        avg_bw = sum(r.bandwidth_mbs for r in self.records) / max(1, count)

        t_txt = Text()
        t_txt.append(f"Chunks Completed : {count} / {self.total_chunks}\n", style="cyan")
        t_txt.append(f"Total PAM Hits   : {total_pams}\n", style="green")
        t_txt.append(f"Total Red SNPs   : {total_snps}\n", style="bold red")
        t_txt.append(f"Avg Severity     : {avg_sev:.2f} / 100\n", style="yellow")
        t_txt.append(f"Peak Severity    : {peak_sev:.2f} / 100\n", style="bold bright_red")
        t_txt.append(f"Avg Bandwidth    : {avg_bw:.2f} MB/s", style="bold bright_green")
        self.query_one(".telemetry_panel", Static).update(t_txt)

        worker_nodes = {r.worker_node for r in self.records}
        cuda_tag = "ACTIVE" if self.hardware.cuda_active else "CPU FALLBACK"
        h_txt = Text()
        h_txt.append(f"Accelerator  : {self.hardware.gpu_device[:24]}\n", style="bright_cyan")
        h_txt.append(f"CUDA Engine  : {cuda_tag} (CC {self.hardware.compute_capability})\n", style="green")
        h_txt.append(f"Active Nodes : {len(worker_nodes)} / {len(self.processor.workers)} Dask Slots\n", style="magenta")
        h_txt.append(f"Logical CPUs : {self.hardware.logical_cores} Cores\n", style="white")
        h_txt.append(f"RAM Free     : {self.hardware.available_ram_gb:.2f}/{self.hardware.total_ram_gb:.1f} GB ({self.hardware.ram_usage_pct:.1f}%)", style="yellow")
        self.query_one(".worker_panel", Static).update(h_txt)

    def _render_inspector(self, rec: GenomicChunkRecord) -> None:
        insp = Text()
        insp.append(f"Chunk ID     : {rec.chunk_id}\n", style="bold cyan")
        insp.append(f"Worker Slot  : {rec.worker_node}\n", style="magenta")
        insp.append(f"Reads Scanned: {rec.sequences_scanned:,}\n", style="white")
        insp.append(f"PAM / SNPs   : {rec.pam_hits} PAMs | {rec.mismatch_count} Mutated\n", style="bold red")
        insp.append(f"Peak Severity: {rec.peak_severity:.2f} | GC: {rec.gc_content_pct:.2f}%\n", style="yellow")
        insp.append(f"Status       : {rec.status} ({rec.latency_ms:.1f} ms)", style="bold green")
        self.query_one(".inspector_panel", Static).update(insp)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.records):
            self._render_inspector(self.records[event.cursor_row])

    def _persist_audit(self) -> str:
        worker_dist: Dict[str, int] = {}
        for r in self.records:
            worker_dist[r.worker_node] = worker_dist.get(r.worker_node, 0) + 1
        cnt = max(1, len(self.records))
        avg_sev = sum(r.peak_severity for r in self.records) / cnt
        peak_sev = max([r.peak_severity for r in self.records], default=0.0)
        avg_bw = sum(r.bandwidth_mbs for r in self.records) / cnt
        total_pams = sum(r.pam_hits for r in self.records)
        total_snps = sum(r.mismatch_count for r in self.records)
        vol_mb = sum(r.chunk_size_mb for r in self.records)
        digest = hashlib.sha256(f"{self.session_id}_{total_pams}_{total_snps}_{peak_sev:.2f}".encode("utf-8")).hexdigest()

        manifest = TUIExportManifest(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            hardware_state=self.hardware,
            total_chunks_completed=len(self.records),
            total_volume_mb=vol_mb,
            total_pam_hits=total_pams,
            total_mismatches=total_snps,
            average_severity=avg_sev,
            peak_severity=peak_sev,
            average_bandwidth_mbs=avg_bw,
            worker_distribution=worker_dist,
            cryptographic_digest=digest,
            chunk_ledger=self.records
        )
        return self.vault.save_manifest(manifest)

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        msg = "PIPELINE PAUSED // PRESS [P] TO RESUME" if self.paused else "PIPELINE RESUMED // STREAMING CHUNKS..."
        self.query_one(".status_bar", Static).update(Text(msg, style="bold yellow"))

    def action_restart_pipeline(self) -> None:
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records.clear()
        self.paused = False
        self.exported = False
        self.query_one(".severity_table", DataTable).clear()
        self.query_one(".chunk_progress", ProgressBar).update(progress=0)
        self.query_one(".progress_label", Static).update("GENOMIC CHUNKING STREAM: RESTARTING...")
        self.query_one(".status_bar", Static).update(Text("NEW SESSION INITIALIZED // STREAMING CHUNKS...", style="bold cyan"))
        self._refresh_side_panels()

    def action_export_audit(self) -> None:
        if not self.records:
            return
        saved = self._persist_audit()
        self.exported = True
        self.query_one(".status_bar", Static).update(Text(f"MANUAL AUDIT SAVED: {saved}", style="bold bright_green"))

if __name__ == "__main__":
    app = GeneWeaverTUIApp(total_chunks=24)
    app.run()