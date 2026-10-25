"""
Infotact Solutions - GeneWeaver Project
Module: Terminal UI Command Center (Phase 1 - Day 7)
Description: 
    Enterprise-grade Textual user interface for the GeneWeaver Pipeline.
    Features real-time hardware telemetry (RAM/CPU), asynchronous subprocess 
    execution to prevent UI blocking, and a sleek dark-green/black hacker aesthetic.
"""

import os
import sys
import psutil
import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Button, RichLog, Label, Rule
from textual.reactive import reactive
from textual.worker import Worker, WorkerState

console = Console()

# ==========================================
# CSS Styling (Dark Mode & Green Accents)
# ==========================================
DASHBOARD_CSS = """
Screen {
    background: #0a0a0a;
    color: #00ff00;
}

Header {
    background: #002200;
    color: #00ff00;
    text-style: bold;
}

Footer {
    background: #001100;
    color: #00aa00;
}

#sidebar {
    width: 30%;
    height: 100%;
    dock: left;
    border-right: vkey #00ff00;
    background: #050505;
    padding: 1;
}

#main_area {
    width: 70%;
    height: 100%;
    padding: 1;
}

.panel_title {
    text-align: center;
    text-style: bold;
    color: #00ff00;
    padding-bottom: 1;
}

.stat_label {
    color: #00aa00;
    margin-top: 1;
}

.stat_value {
    color: #ffffff;
    text-style: bold;
}

Rule {
    color: #005500;
    margin-top: 1;
    margin-bottom: 1;
}

Button {
    width: 100%;
    margin-top: 2;
    background: #004400;
    color: #ffffff;
    border: tall #00ff00;
}

Button:hover {
    background: #00aa00;
    color: #000000;
}

Button.-active {
    background: #00ff00;
    color: #000000;
}

#stop_btn {
    background: #440000;
    border: tall #ff0000;
    display: none;
}

#stop_btn:hover {
    background: #aa0000;
}

RichLog {
    background: #000000;
    color: #00ff00;
    border: solid #005500;
    height: 1fr;
}
"""

# ==========================================
# Custom UI Widgets
# ==========================================
class HardwareMonitor(Static):
    """Widget to display real-time CPU and RAM telemetry."""
    
    cpu_percent = reactive(0.0)
    ram_percent = reactive(0.0)
    ram_used = reactive(0.0)

    def compose(self) -> ComposeResult:
        yield Label("SYSTEM TELEMETRY", classes="panel_title")
        yield Rule()
        
        yield Label("CPU Utilization:", classes="stat_label")
        yield Label("0.0%", id="cpu_val", classes="stat_value")
        
        yield Label("RAM Utilization:", classes="stat_label")
        yield Label("0.0%", id="ram_val", classes="stat_value")
        
        yield Label("RAM Used (GB):", classes="stat_label")
        yield Label("0.0 GB", id="ram_used_val", classes="stat_value")
        
        yield Rule()
        yield Label("VRAM Guard:", classes="stat_label")
        yield Label("ACTIVE (3000 MB Cap)", classes="stat_value")

    def on_mount(self) -> None:
        """Starts the background timer to update stats every second."""
        self.update_stats()
        self.set_interval(1.0, self.update_stats)

    def update_stats(self) -> None:
        """Fetches live metrics from psutil."""
        self.cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        self.ram_percent = mem.percent
        self.ram_used = mem.used / (1024 ** 3)

    def watch_cpu_percent(self, cpu_percent: float) -> None:
        self.query_one("#cpu_val", Label).update(f"{cpu_percent:.1f}%")

    def watch_ram_percent(self, ram_percent: float) -> None:
        self.query_one("#ram_val", Label).update(f"{ram_percent:.1f}%")

    def watch_ram_used(self, ram_used: float) -> None:
        self.query_one("#ram_used_val", Label).update(f"{ram_used:.2f} GB")


class ControlPanel(Static):
    """Widget containing action buttons for the pipeline."""
    def compose(self) -> ComposeResult:
        yield Label("COMMAND CENTER", classes="panel_title")
        yield Button("START ALIGNMENT PIPELINE", id="start_btn")
        yield Button("ABORT PROCESS", id="stop_btn")


# ==========================================
# Main Dashboard Application
# ==========================================
class GeneWeaverDashboard(App):
    """
    Master Orchestrator App. Coordinates the layout, intercepts button presses,
    and manages the asynchronous execution of the GeneWeaver backend.
    """
    
    CSS = DASHBOARD_CSS
    TITLE = "GeneWeaver Command Center"
    SUB_TITLE = "High-Performance CRISPR Alignment (Phase 1)"
    
    BINDINGS = [
        ("q", "quit", "Quit Dashboard"),
        ("r", "run_pipeline", "Run Pipeline"),
        ("c", "clear_log", "Clear Console")
    ]

    def __init__(self):
        super().__init__()
        self.process: Optional[asyncio.subprocess.Process] = None
        self._manual_abort: bool = False

    def compose(self) -> ComposeResult:
        """Constructs the TUI layout."""
        yield Header()
        
        with Horizontal():
            # Left Sidebar: Controls & Telemetry
            with VerticalScroll(id="sidebar"):
                yield ControlPanel()
                yield HardwareMonitor()
                
            # Right Main Area: Live Execution Logs
            with Vertical(id="main_area"):
                yield Label("LIVE EXECUTION LOGS", classes="panel_title")
                yield RichLog(id="console_log", highlight=True, markup=True)
                
        yield Footer()

    def on_mount(self) -> None:
        """Fires when the app is first rendered."""
        log = self.query_one("#console_log", RichLog)
        log.write("[bold green]System Initialized. Awaiting commands...[/bold green]")
        log.write("Press [bold white]'START ALIGNMENT PIPELINE'[/bold white] or [bold white]'R'[/bold white] to begin.")

    def action_clear_log(self) -> None:
        """Clears the RichLog console."""
        self.query_one("#console_log", RichLog).clear()

    def action_run_pipeline(self) -> None:
        """Triggers the pipeline via keyboard shortcut."""
        self.post_message(Button.Pressed(self.query_one("#start_btn", Button)))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handles button click events."""
        if event.button.id == "start_btn":
            await self.start_pipeline()
        elif event.button.id == "stop_btn":
            await self.stop_pipeline()

    async def start_pipeline(self) -> None:
        """Initiates the GeneWeaver backend asynchronously."""
        if self.process is not None and self.process.returncode is None:
            return  # Already running
            
        self._manual_abort = False
        log = self.query_one("#console_log", RichLog)
        start_btn = self.query_one("#start_btn", Button)
        stop_btn = self.query_one("#stop_btn", Button)
        
        self.action_clear_log()
        log.write("[bold cyan]>>> Initializing Subprocess: python src/main.py...[/bold cyan]")
        
        # Toggle buttons
        start_btn.display = False
        stop_btn.display = True
        
        # Determine correct python executable
        python_exec = sys.executable
        main_script = str(Path("src/main.py").absolute())
        
        try:
            # Launch the backend as a subprocess so it doesn't freeze the UI
            self.process = await asyncio.create_subprocess_exec(
                python_exec, "-u", main_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd()
            )
            
            # Spin up a worker to read the output stream without blocking asyncio event loop
            self.run_worker(self.stream_output(), name="log_streamer")
            
        except Exception as e:
            log.write(f"[bold red]Failed to start pipeline: {e}[/bold red]")
            self.reset_buttons()

    async def stream_output(self) -> None:
        """Reads stdout from the subprocess and pipes it to the RichLog."""
        log = self.query_one("#console_log", RichLog)
        
        if not self.process or not self.process.stdout:
            return
            
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                    
                # Decode and clean up terminal escape codes for the log
                decoded_line = line.decode('utf-8', errors='replace').rstrip()
                log.write(decoded_line)
                
            await self.process.wait()
            
            if self._manual_abort:
                log.write("\n[bold yellow][!] Pipeline aborted by user.[/bold yellow]")
            elif self.process.returncode == 0:
                log.write("\n[bold green][+] Pipeline execution completed successfully.[/bold green]")
            else:
                log.write(f"\n[bold red][-] Pipeline exited with error code {self.process.returncode}.[/bold red]")
                
        except asyncio.CancelledError:
            pass
        finally:
            self.reset_buttons()

    async def stop_pipeline(self) -> None:
        """Gracefully terminates the running pipeline."""
        if self.process and self.process.returncode is None:
            self._manual_abort = True
            log = self.query_one("#console_log", RichLog)
            log.write("\n[bold red]>>> Sending termination signal to pipeline...[/bold red]")
            
            # Send SIGTERM
            self.process.terminate()
            
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                log.write("[bold red]Pipeline did not respond. Forcing kill...[/bold red]")
                self.process.kill()
                
            self.reset_buttons()

    def reset_buttons(self) -> None:
        """Restores the UI state after execution finishes or aborts."""
        self.query_one("#start_btn", Button).display = True
        self.query_one("#stop_btn", Button).display = False


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    try:
        import textual
    except ImportError:
        console.print("[bold red][!] Textual library is missing.[/bold red]")
        console.print("Please install it using: [cyan]pip install textual[/cyan]")
        sys.exit(1)
        
    app = GeneWeaverDashboard()
    app.run()