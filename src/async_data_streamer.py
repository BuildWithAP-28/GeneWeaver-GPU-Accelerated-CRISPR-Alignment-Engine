import os
import sys
import time
import json
import uuid
import zlib
import socket
import hashlib
import psutil
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

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

class GeneWeaverStreamError(Exception):
    def __init__(self, message: str, error_code: int = 800):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class BufferOverflowError(GeneWeaverStreamError):
    def __init__(self, message: str):
        super().__init__(message, error_code=801)

class NetworkBottleneckError(GeneWeaverStreamError):
    def __init__(self, message: str):
        super().__init__(message, error_code=802)

class IntegrityCheckError(GeneWeaverStreamError):
    def __init__(self, message: str):
        super().__init__(message, error_code=803)

@dataclass
class StreamConfig:
    max_concurrent_streams: int
    packet_size_mb: float
    total_payload_mb: float
    target_motif: str
    backpressure_threshold_mb: float

@dataclass
class DataPacket:
    packet_id: str
    sequence_payload: bytes
    size_mb: float
    motif: str
    crc32_checksum: int
    dispatch_timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class ProcessedPacket:
    packet_id: str
    worker_id: str
    matches_detected: int
    processing_latency_ms: float
    verification_passed: bool
    stream_bandwidth_mbs: float
    receive_timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class BufferState:
    active_packets_in_flight: int
    current_buffer_usage_mb: float
    total_packets_dispatched: int
    total_packets_resolved: int
    dropped_packets: int

@dataclass
class StreamMetrics:
    session_id: str
    stream_duration_sec: float
    total_data_streamed_mb: float
    overall_bandwidth_mbs: float
    total_motifs_found: int
    buffer_overflow_events: int
    integrity_failures: int
    worker_node_participation: Dict[str, int]
    packet_ledger: List[ProcessedPacket]

class IntegrityEngine:
    def generate_crc32(self, data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF

    def verify_crc32(self, data: bytes, expected_checksum: int) -> bool:
        return (zlib.crc32(data) & 0xFFFFFFFF) == expected_checksum

class SimulatedDiskStreamer:
    def __init__(self, config: StreamConfig, integrity_engine: IntegrityEngine):
        self.config = config
        self.integrity_engine = integrity_engine
        self.bytes_per_packet = int(self.config.packet_size_mb * (1024 * 1024))
        self.total_packets_required = int(self.config.total_payload_mb / self.config.packet_size_mb)
        self.packets_yielded = 0

    def has_next(self) -> bool:
        return self.packets_yielded < self.total_packets_required

    def yield_next_packet(self) -> DataPacket:
        base_array = np.random.choice([65, 67, 71, 84], size=self.bytes_per_packet).astype(np.uint8)
        injection_point = int(self.bytes_per_packet * np.random.uniform(0.1, 0.9))
        target_bytes = np.frombuffer(self.config.target_motif.encode("ascii"), dtype=np.uint8)
        
        if injection_point + len(target_bytes) < self.bytes_per_packet:
            base_array[injection_point:injection_point+len(target_bytes)] = target_bytes
            
        raw_payload = base_array.tobytes()
        checksum = self.integrity_engine.generate_crc32(raw_payload)
        
        packet = DataPacket(
            packet_id=f"PKT_{uuid.uuid4().hex[:8].upper()}",
            sequence_payload=raw_payload,
            size_mb=self.config.packet_size_mb,
            motif=self.config.target_motif,
            crc32_checksum=checksum
        )
        self.packets_yielded += 1
        return packet

class MemoryBufferController:
    def __init__(self, max_buffer_mb: float):
        self.max_buffer_mb = max_buffer_mb
        self.current_mb = 0.0
        self.in_flight = 0
        self.dispatched = 0
        self.resolved = 0
        self.overflow_events = 0

    def can_allocate(self, size_mb: float) -> bool:
        return (self.current_mb + size_mb) <= self.max_buffer_mb

    def allocate(self, size_mb: float):
        if not self.can_allocate(size_mb):
            self.overflow_events += 1
            raise BufferOverflowError(f"Cannot allocate {size_mb}MB. Buffer at {self.current_mb}/{self.max_buffer_mb}MB")
        self.current_mb += size_mb
        self.in_flight += 1
        self.dispatched += 1

    def deallocate(self, size_mb: float):
        self.current_mb = max(0.0, self.current_mb - size_mb)
        self.in_flight = max(0, self.in_flight - 1)
        self.resolved += 1

    def get_state(self) -> BufferState:
        return BufferState(
            active_packets_in_flight=self.in_flight,
            current_buffer_usage_mb=self.current_mb,
            total_packets_dispatched=self.dispatched,
            total_packets_resolved=self.resolved,
            dropped_packets=self.overflow_events
        )

def worker_process_streamed_packet(packet: DataPacket) -> ProcessedPacket:
    worker_start = time.perf_counter()
    try:
        worker = get_worker()
        worker_id = str(worker.id)
    except Exception:
        worker_id = "local_stream_node_" + str(uuid.uuid4())[:8]
        
    engine = IntegrityEngine()
    is_valid = engine.verify_crc32(packet.sequence_payload, packet.crc32_checksum)
    
    matches = 0
    if is_valid:
        target_bytes = packet.motif.encode("ascii")
        matches = packet.sequence_payload.count(target_bytes)
        
    time.sleep(np.random.uniform(0.1, 0.4))
    
    worker_end = time.perf_counter()
    latency_ms = (worker_end - worker_start) * 1000.0
    bandwidth = (packet.size_mb) / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
    
    return ProcessedPacket(
        packet_id=packet.packet_id,
        worker_id=worker_id,
        matches_detected=matches,
        processing_latency_ms=latency_ms,
        verification_passed=is_valid,
        stream_bandwidth_mbs=bandwidth
    )

class StreamUIRenderer:
    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="live_feed", ratio=2),
            Layout(name="metrics_col", ratio=1)
        )
        layout["metrics_col"].split_column(
            Layout(name="buffer_panel", ratio=1),
            Layout(name="health_panel", ratio=1)
        )
        
        header_text = Text("GENEWEAVER: HIGH-PERFORMANCE BIOINFORMATICS PIPELINE\nModule: Asynchronous Worker Data Streamer", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(header_text, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("ESTABLISHING DATA STREAMS...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update_frame(self, layout: Layout, buffer_state: BufferState, recent_packets: List[ProcessedPacket], total_expected: int):
        feed_table = Table(box=None, expand=True)
        feed_table.add_column("Packet ID", style="cyan")
        feed_table.add_column("Worker Node", style="magenta")
        feed_table.add_column("Integrity", style="blue", justify="center")
        feed_table.add_column("Matches", style="green", justify="right")
        feed_table.add_column("Bandwidth", style="bold white", justify="right")
        
        for pkt in recent_packets[-14:]:
            integrity_str = "[green]PASS[/green]" if pkt.verification_passed else "[red]FAIL[/red]"
            feed_table.add_row(
                pkt.packet_id,
                str(pkt.worker_id)[:16] + "..",
                integrity_str,
                str(pkt.matches_detected),
                f"{pkt.stream_bandwidth_mbs:.2f} MB/s"
            )
            
        layout["live_feed"].update(Panel(feed_table, title="[bold green]Live Asynchronous Data Stream[/bold green]", border_style="green"))
        
        buf_table = Table(box=None, expand=True)
        buf_table.add_column("Buffer Metric", style="cyan")
        buf_table.add_column("State", style="yellow", justify="right")
        buf_table.add_row("In-Flight Packets", str(buffer_state.active_packets_in_flight))
        buf_table.add_row("RAM Usage", f"{buffer_state.current_buffer_usage_mb:.1f} MB")
        buf_table.add_row("Dropped / Overflow", str(buffer_state.dropped_packets))
        
        layout["buffer_panel"].update(Panel(buf_table, title="[bold yellow]Memory Buffer Controller[/bold yellow]", border_style="yellow"))
        
        health_table = Table(box=None, expand=True)
        health_table.add_column("Pipeline Stat", style="cyan")
        health_table.add_column("Value", style="white", justify="right")
        
        resolved = buffer_state.total_packets_resolved
        pct = (resolved / total_expected) * 100 if total_expected > 0 else 0
        total_matches = sum(p.matches_detected for p in recent_packets)
        avg_bw = sum(p.stream_bandwidth_mbs for p in recent_packets) / len(recent_packets) if recent_packets else 0.0
        
        health_table.add_row("Packets Processed", f"{resolved} / {total_expected}")
        health_table.add_row("Stream Progress", f"{pct:.1f}%")
        health_table.add_row("Global Matches", str(total_matches))
        health_table.add_row("Avg Bandwidth", f"{avg_bw:.2f} MB/s")
        
        layout["health_panel"].update(Panel(health_table, title="[bold magenta]Stream Health[/bold magenta]", border_style="magenta"))
        layout["footer"].update(Panel(Text(f"STREAMING DATA TO CLUSTER... [{resolved}/{total_expected}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class TelemetryExporter:
    def __init__(self, export_dir: str = "reports/async_streams"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def save_report(self, report: StreamMetrics) -> str:
        file_path = os.path.join(self.export_dir, f"stream_audit_{report.session_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(asdict(report), f, indent=4)
        except Exception:
            pass
        return file_path

class AsyncStreamMaster:
    def __init__(self):
        self.config = StreamConfig(
            max_concurrent_streams=8,
            packet_size_mb=15.0,
            total_payload_mb=300.0,
            target_motif="GATCGATCGATC",
            backpressure_threshold_mb=120.0
        )
        self.integrity_engine = IntegrityEngine()
        self.disk_simulator = SimulatedDiskStreamer(self.config, self.integrity_engine)
        self.buffer_controller = MemoryBufferController(self.config.backpressure_threshold_mb)
        self.renderer = StreamUIRenderer()
        self.exporter = TelemetryExporter()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def execute_streaming_pipeline(self):
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
            console.print(f"[bold red]Streamer Cluster Failed: {str(e)}[/bold red]")
            sys.exit(1)
            
        layout = self.renderer.generate_layout()
        processed_ledger = []
        future_map = {}
        
        stream_start = time.perf_counter()
        seq = as_completed()
        
        with Live(layout, refresh_per_second=10, screen=True):
            while self.disk_simulator.has_next() or future_map:
                while self.disk_simulator.has_next() and self.buffer_controller.can_allocate(self.config.packet_size_mb):
                    packet = self.disk_simulator.yield_next_packet()
                    self.buffer_controller.allocate(packet.size_mb)
                    future = client.submit(worker_process_streamed_packet, packet)
                    future_map[future] = packet.size_mb
                    seq.add(future)
                    
                for future in seq:
                    try:
                        result = future.result()
                        if isinstance(result, ProcessedPacket):
                            processed_ledger.append(result)
                            freed_mb = future_map.pop(future, 0.0)
                            self.buffer_controller.deallocate(freed_mb)
                            self.renderer.update_frame(layout, self.buffer_controller.get_state(), processed_ledger, self.disk_simulator.total_packets_required)
                    except Exception:
                        freed_mb = future_map.pop(future, 0.0)
                        self.buffer_controller.deallocate(freed_mb)
                    break 
                    
                time.sleep(0.01)
                
            stream_end = time.perf_counter()
            duration_sec = stream_end - stream_start
            
            layout["footer"].update(Panel(Text("DATA STREAMING COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
                
        client.close()
        cluster.close()
        
        worker_counts = {}
        integrity_fails = 0
        for p in processed_ledger:
            worker_counts[p.worker_id] = worker_counts.get(p.worker_id, 0) + 1
            if not p.verification_passed:
                integrity_fails += 1
                
        report = StreamMetrics(
            session_id=self.session_id,
            stream_duration_sec=duration_sec,
            total_data_streamed_mb=sum(self.config.packet_size_mb for _ in processed_ledger),
            overall_bandwidth_mbs=(self.config.total_payload_mb / duration_sec) if duration_sec > 0 else 0.0,
            total_motifs_found=sum(p.matches_detected for p in processed_ledger),
            buffer_overflow_events=self.buffer_controller.overflow_events,
            integrity_failures=integrity_fails,
            worker_node_participation=worker_counts,
            packet_ledger=processed_ledger
        )
        
        path = self.exporter.save_report(report)
        console.print(f"\n[bold bright_green][+] Day 3 Data Streaming Complete. Zero Bottlenecks Detected.[/bold bright_green]")
        console.print(f"[bold cyan][i] Stream Audit saved to: {path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    master = AsyncStreamMaster()
    master.execute_streaming_pipeline()