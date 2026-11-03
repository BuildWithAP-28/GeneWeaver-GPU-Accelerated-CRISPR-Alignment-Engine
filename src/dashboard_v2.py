import os
import sys
import time
import json
import math
import random
import psutil
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.align import Align

console = Console()

@dataclass
class LiveMetrics:
    elapsed_time_sec: float = 0.0
    current_mbps: float = 0.0
    total_mb_processed: float = 0.0
    simd_fast_path_hits: int = 0
    simd_true_matches: int = 0
    active_workers: int = 0
    c_array_overhead_ms: float = 0.0
    ema_mbps: float = 0.0

@dataclass
class SystemAlert:
    timestamp: str
    level: str
    message: str

class AlertManager:
    def __init__(self, max_alerts: int = 5):
        self.alerts: deque = deque(maxlen=max_alerts)
        
    def add_alert(self, level: str, message: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.alerts.append(SystemAlert(timestamp=now, level=level, message=message))
        
    def check_system_health(self, cpu_usage: float, ram_usage: float, throughput: float) -> None:
        if cpu_usage > 95.0:
            self.add_alert("CRITICAL", f"CPU Thermal Throttling Risk ({cpu_usage}%)")
        if ram_usage > 90.0:
            self.add_alert("WARNING", f"High Memory Pressure ({ram_usage}%)")
        if throughput < 10.0 and throughput > 0.0:
            self.add_alert("INFO", "I/O Bottleneck Detected")

class TelemetryExporter:
    def __init__(self, export_dir: str = "reports/telemetry"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history: List[Dict[str, Any]] = []

    def record_snapshot(self, metrics: LiveMetrics, cpu: float, ram: float) -> None:
        snapshot = {
            "timestamp": time.time(),
            "elapsed_sec": round(metrics.elapsed_time_sec, 2),
            "throughput_mbps": round(metrics.current_mbps, 2),
            "fast_path_hits": metrics.simd_fast_path_hits,
            "c_array_overhead": round(metrics.c_array_overhead_ms, 2),
            "cpu_percent": cpu,
            "ram_percent": ram
        }
        self.history.append(snapshot)

    def export_to_json(self) -> str:
        filepath = os.path.join(self.export_dir, f"session_{self.session_id}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(self.history, f, indent=4)
            return filepath
        except Exception:
            return ""

class TerminalGraphEngine:
    def __init__(self, max_data_points: int = 50, smoothing_factor: float = 0.2):
        self.max_points = max_data_points
        self.timestamps: deque = deque(maxlen=max_data_points)
        self.mbps_history: deque = deque(maxlen=max_data_points)
        self.start_time = time.time()
        self.alpha = smoothing_factor
        self.current_ema = 0.0
        
        for _ in range(max_data_points):
            self.timestamps.append(0)
            self.mbps_history.append(0.0)

    def update_data(self, current_mbps: float) -> float:
        elapsed = time.time() - self.start_time
        if self.current_ema == 0.0:
            self.current_ema = current_mbps
        else:
            self.current_ema = (self.alpha * current_mbps) + ((1 - self.alpha) * self.current_ema)
            
        self.timestamps.append(elapsed)
        self.mbps_history.append(self.current_ema)
        return self.current_ema

    def render_ansi_graph(self, width: int, height: int) -> Text:
        if hasattr(plt, 'clf'):
            plt.clf()
        elif hasattr(plt, 'clear_plot'):
            plt.clear_plot()  # type: ignore
        elif hasattr(plt, 'clear_figure'):
            plt.clear_figure()
            
        if hasattr(plt, 'plotsize'):
            plt.plotsize(width, height)
        elif hasattr(plt, 'plot_size'):
            plt.plot_size(width, height)
            
        if hasattr(plt, 'theme'):
            plt.theme("dark")
            
        x_data = list(self.timestamps)
        y_data = list(self.mbps_history)
        
        plt.plot(x_data, y_data, marker="fhd", color="green")
        
        if hasattr(plt, 'title'):
            plt.title("Live Pipeline Throughput (EMA Adjusted MB/s)")
            plt.xlabel("Elapsed Time (s)")
            plt.ylabel("Speed (MB/s)")
            
        max_y = max(y_data) if y_data else 50
        if hasattr(plt, 'ylim'):
            plt.ylim(0, max_y * 1.2 if max_y > 0 else 50)
            
        ansi_str = plt.build() if hasattr(plt, 'build') else ""
        return Text.from_ansi(ansi_str)

class DashboardLayout:
    def __init__(self):
        self.layout = Layout()
        self.graph_engine = TerminalGraphEngine()
        self.alert_manager = AlertManager()
        self.exporter = TelemetryExporter()
        self.init_grid()
        
    def init_grid(self) -> None:
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="left_sidebar", ratio=1),
            Layout(name="right_graph", ratio=2)
        )
        self.layout["left_sidebar"].split_column(
            Layout(name="sys_monitor", ratio=2),
            Layout(name="vector_telemetry", ratio=2),
            Layout(name="alerts_panel", ratio=1)
        )

    def generate_header(self) -> Panel:
        header_text = Text("OMNIAI LOGISTICS & DATA COMMAND CENTER // GENEWEAVER PIPELINE", style="bold bright_green justify-center")
        return Panel(header_text, style="on black", border_style="bright_green")

    def generate_footer(self, status: str = "ACTIVE") -> Panel:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_table = Table.grid(expand=True)
        footer_table.add_column(justify="left", ratio=1)
        footer_table.add_column(justify="center", ratio=1)
        footer_table.add_column(justify="right", ratio=1)
        
        footer_table.add_row(
            Text(f"System Status: {status}", style="bold bright_green"),
            Text("Engine: SIMD Vector + IPC", style="bold cyan"),
            Text(f"Timestamp: {now}", style="dim bright_green")
        )
        return Panel(footer_table, style="on black", border_style="bright_green")

    def generate_sys_monitor(self, metrics: LiveMetrics, cpu: float, ram: float) -> Panel:
        table = Table(box=None, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold bright_green", justify="right")
        
        table.add_row("Logical Cores", str(psutil.cpu_count()))
        table.add_row("CPU Utilization", f"{cpu}%")
        table.add_row("RAM Utilization", f"{ram}%")
        table.add_row("Active IPC Workers", str(metrics.active_workers))
        table.add_row("Execution Time", f"{metrics.elapsed_time_sec:.1f}s")
        table.add_row("Total Processed", f"{metrics.total_mb_processed:.1f} MB")
        
        return Panel(
            Align.center(table, vertical="middle"), 
            title="[bold]Hardware Monitor[/bold]", 
            border_style="cyan"
        )

    def generate_vector_telemetry(self, metrics: LiveMetrics) -> Panel:
        table = Table(box=None, expand=True)
        table.add_column("Vector Metric", style="cyan")
        table.add_column("Count/Latency", style="bold bright_green", justify="right")
        
        table.add_row("Fast-Path Cache Hits", f"{metrics.simd_fast_path_hits:,}")
        table.add_row("True Memory Matches", f"{metrics.simd_true_matches:,}")
        table.add_row("C-Array Align Overhead", f"{metrics.c_array_overhead_ms:.2f} ms")
        table.add_row("Raw Throughput", f"{metrics.current_mbps:.2f} MB/s")
        table.add_row("EMA Adjusted Speed", f"{metrics.ema_mbps:.2f} MB/s")
        
        return Panel(
            Align.center(table, vertical="middle"), 
            title="[bold]SIMD Vector Core[/bold]", 
            border_style="magenta"
        )

    def generate_alerts_panel(self) -> Panel:
        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Time", style="dim cyan", width=10)
        table.add_column("Level", width=10)
        table.add_column("Message", style="bright_green")
        
        if not self.alert_manager.alerts:
            table.add_row("", "", "System operating at optimal parameters.")
        else:
            for alert in reversed(self.alert_manager.alerts):
                level_style = "bold red" if alert.level == "CRITICAL" else "bold yellow" if alert.level == "WARNING" else "bold cyan"
                table.add_row(alert.timestamp, f"[{level_style}]{alert.level}[/{level_style}]", alert.message)
                
        return Panel(table, title="[bold]Live System Alerts[/bold]", border_style="red")

    def generate_graph(self, metrics: LiveMetrics, width: int = 60, height: int = 15) -> Panel:
        metrics.ema_mbps = self.graph_engine.update_data(metrics.current_mbps)
        graph_text = self.graph_engine.render_ansi_graph(width, height)
        return Panel(graph_text, title="[bold]Pipeline Telemetry Graph[/bold]", border_style="bright_green")

    def update_layout(self, metrics: LiveMetrics) -> Layout:
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        self.alert_manager.check_system_health(cpu_usage, ram_usage, metrics.current_mbps)
        self.exporter.record_snapshot(metrics, cpu_usage, ram_usage)
        
        self.layout["header"].update(self.generate_header())
        self.layout["footer"].update(self.generate_footer())
        self.layout["sys_monitor"].update(self.generate_sys_monitor(metrics, cpu_usage, ram_usage))
        self.layout["vector_telemetry"].update(self.generate_vector_telemetry(metrics))
        self.layout["alerts_panel"].update(self.generate_alerts_panel())
        
        term_width = console.size.width or 120
        term_height = console.size.height or 30
        
        plot_w = max(40, int(term_width * 0.6) - 6)
        plot_h = max(10, term_height - 10)
        
        self.layout["right_graph"].update(self.generate_graph(metrics, width=plot_w, height=plot_h))
        return self.layout

def run_telemetry_simulator() -> None:
    dashboard = DashboardLayout()
    metrics = LiveMetrics(active_workers=psutil.cpu_count(logical=True) or 4)
    start_time = time.time()
    
    console.clear()
    
    with Live(dashboard.update_layout(metrics), refresh_per_second=4, screen=True) as live:
        try:
            for step in range(1, 400):
                time.sleep(0.25)
                metrics.elapsed_time_sec = time.time() - start_time
                
                if step < 15:
                    base_speed = 5.0
                else:
                    base_speed = 46.0
                    
                noise = random.uniform(-6.0, 6.0)
                metrics.current_mbps = max(0.0, base_speed + noise)
                metrics.total_mb_processed += (metrics.current_mbps * 0.25)
                metrics.simd_fast_path_hits += random.randint(5, 25)
                
                if random.random() > 0.8:
                    metrics.simd_true_matches += 1
                    
                metrics.c_array_overhead_ms = random.uniform(1.0, 5.0)
                
                if step == 50:
                    dashboard.alert_manager.add_alert("INFO", "Vector JIT Warmup Complete.")
                
                live.update(dashboard.update_layout(metrics))
                
        except KeyboardInterrupt:
            pass
            
    export_path = dashboard.exporter.export_to_json()
    console.print(f"\n[bold bright_green][+] Telemetry session disconnected. Logs saved to: {export_path}[/bold bright_green]")

if __name__ == "__main__":
    run_telemetry_simulator()