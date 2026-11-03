import os
import sys
import time
import uuid
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

os.environ['NUMBA_CUDA_LOG_LEVEL'] = 'CRITICAL'
from numba import cuda
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

console = Console()

class CUDAAllocationError(MemoryError):
    pass

class CUDADeviceError(RuntimeError):
    pass

@dataclass
class HardwareSpecs:
    device_id: int = 0
    name: str = ""
    compute_capability: Tuple[int, int] = (0, 0)
    total_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    max_threads_per_block: int = 0
    max_block_dim_x: int = 0
    max_grid_dim_x: int = 0
    warp_size: int = 32
    shared_memory_per_block: int = 0
    multiprocessor_count: int = 0
    l2_cache_size: int = 0

class CUDAHardwareProfiler:
    def __init__(self):
        self.specs = HardwareSpecs()
        self.active_context = None
        self._initialize_device()

    def _initialize_device(self):
        if not cuda.is_available():
            console.print("[bold red]CRITICAL: CUDA is not available on this system.[/bold red]")
            sys.exit(1)
        try:
            device = cuda.get_current_device()
            self.specs.device_id = device.id
            self.specs.name = device.name.decode('utf-8') if isinstance(device.name, bytes) else str(device.name)
            self.specs.compute_capability = device.compute_capability
            self.active_context = cuda.current_context()
            free_mem, total_mem = self.active_context.get_memory_info()
            self.specs.total_vram_mb = total_mem / (1024 ** 2)
            self.specs.free_vram_mb = free_mem / (1024 ** 2)
            self.specs.max_threads_per_block = getattr(device, 'MAX_THREADS_PER_BLOCK', 1024)
            self.specs.max_block_dim_x = getattr(device, 'MAX_BLOCK_DIM_X', 1024)
            self.specs.max_grid_dim_x = getattr(device, 'MAX_GRID_DIM_X', 2147483647)
            self.specs.warp_size = getattr(device, 'WARP_SIZE', 32)
            self.specs.shared_memory_per_block = getattr(device, 'MAX_SHARED_MEMORY_PER_BLOCK', 49152)
            self.specs.multiprocessor_count = getattr(device, 'MULTIPROCESSOR_COUNT', 1)
            self.specs.l2_cache_size = getattr(device, 'L2_CACHE_SIZE', 0)
        except Exception as e:
            raise CUDADeviceError(f"Hardware initialization failed: {e}")

    def get_diagnostic_panel(self) -> Panel:
        table = Table(box=None, expand=True)
        table.add_column("Hardware Property", style="cyan")
        table.add_column("Specification", style="bold yellow", justify="right")
        table.add_row("GPU Model", self.specs.name)
        table.add_row("Compute Capability", f"{self.specs.compute_capability[0]}.{self.specs.compute_capability[1]}")
        table.add_row("Total VRAM", f"{self.specs.total_vram_mb:.2f} MB")
        table.add_row("Available VRAM", f"{self.specs.free_vram_mb:.2f} MB")
        table.add_row("Multiprocessors (SMs)", str(self.specs.multiprocessor_count))
        table.add_row("Max Threads / Block", str(self.specs.max_threads_per_block))
        table.add_row("Shared Memory / Block", f"{self.specs.shared_memory_per_block / 1024:.1f} KB")
        table.add_row("L2 Cache", f"{self.specs.l2_cache_size / (1024**2):.2f} MB")
        table.add_row("Warp Size", str(self.specs.warp_size))
        return Panel(table, title="[bold green]CUDA Architecture Diagnostics[/bold green]", border_style="green")

@dataclass
class MemoryBlock:
    block_id: str
    size_mb: float
    is_pinned: bool
    allocation_time: float
    host_array: Optional[np.ndarray] = None
    device_ptr: Any = None
    state: str = "ALLOCATED"

class PinnedMemoryAllocator:
    def __init__(self, max_budget_mb: float):
        self.max_budget_mb = max_budget_mb
        self.current_usage_mb = 0.0
        self.active_blocks: Dict[str, MemoryBlock] = {}

    def allocate_pinned(self, shape: tuple, dtype: np.dtype) -> MemoryBlock:
        bytes_req = np.prod(shape) * np.dtype(dtype).itemsize
        mb_req = bytes_req / (1024 ** 2)
        if self.current_usage_mb + mb_req > self.max_budget_mb:
            raise CUDAAllocationError(f"Pinned budget exceeded. Req: {mb_req:.2f}MB, Avail: {self.max_budget_mb - self.current_usage_mb:.2f}MB")
        
        start_t = time.perf_counter()
        pinned_arr = cuda.pinned_array(shape, dtype=dtype) # type: ignore
        alloc_t = (time.perf_counter() - start_t) * 1000
        block_id = str(uuid.uuid4())[:8]
        
        block = MemoryBlock(
            block_id=block_id,
            size_mb=float(mb_req), 
            is_pinned=True,
            allocation_time=alloc_t,
            host_array=pinned_arr
        )
        self.active_blocks[block_id] = block
        self.current_usage_mb += mb_req
        return block

    def free_pinned(self, block_id: str) -> None:
        if block_id in self.active_blocks:
            block = self.active_blocks.pop(block_id)
            self.current_usage_mb -= block.size_mb
            if block.host_array is not None:
                del block.host_array

class CUDAStreamOrchestrator:
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.streams = [cuda.stream() for _ in range(num_streams)]
        self.stream_status = {i: "IDLE" for i in range(num_streams)}
        self.current_stream_idx = 0

    def get_next_stream(self) -> Tuple[int, Any]:
        idx = self.current_stream_idx
        stream = self.streams[idx]
        self.stream_status[idx] = "BUSY"
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        return idx, stream

    def mark_stream_idle(self, idx: int) -> None:
        self.stream_status[idx] = "IDLE"

    def synchronize_all(self) -> None:
        cuda.synchronize()
        for i in range(self.num_streams):
            self.stream_status[i] = "IDLE"

    def get_stream_panel(self) -> Panel:
        table = Table(box=None, expand=True)
        table.add_column("Stream ID", style="cyan")
        table.add_column("Status", style="bold")
        for i in range(self.num_streams):
            status = self.stream_status[i]
            color = "green" if status == "IDLE" else "yellow"
            table.add_row(f"Stream-{i}", f"[{color}]{status}[/{color}]")
        return Panel(table, title="[bold magenta]Concurrent PCIe Streams[/bold magenta]", border_style="magenta")

class VRAMFragmentationMonitor:
    def __init__(self, total_vram: float):
        self.total_vram = total_vram
        self.allocated_vram = 0.0
        self.peak_vram = 0.0
        self.fragment_count = 0
        self.history: List[float] = []

    def register_allocation(self, size_mb: float) -> None:
        self.allocated_vram += size_mb
        self.fragment_count += 1
        if self.allocated_vram > self.peak_vram:
            self.peak_vram = self.allocated_vram
        self.history.append(self.allocated_vram)

    def register_deallocation(self, size_mb: float) -> None:
        self.allocated_vram = max(0.0, self.allocated_vram - size_mb)
        self.fragment_count = max(0, self.fragment_count - 1)
        self.history.append(self.allocated_vram)

    def get_monitor_panel(self) -> Panel:
        table = Table(box=None, expand=True)
        table.add_column("Memory Metric", style="cyan")
        table.add_column("Utilization", style="bold yellow", justify="right")
        table.add_row("Current VRAM Usage", f"{self.allocated_vram:.2f} MB")
        table.add_row("Peak VRAM Usage", f"{self.peak_vram:.2f} MB")
        table.add_row("Active Data Fragments", str(self.fragment_count))
        utilization = (self.allocated_vram / self.total_vram) * 100 if self.total_vram > 0 else 0
        table.add_row("VRAM Utilization", f"{utilization:.1f}%")
        return Panel(table, title="[bold blue]VRAM Fragmentation Engine[/bold blue]", border_style="blue")

class TelemetryExporter:
    def __init__(self, export_dir: str = "reports/cuda_telemetry"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records: List[Dict[str, Any]] = []

    def log_transfer(self, op_type: str, stream_id: int, size_mb: float, latency_ms: float, bw_gbs: float):
        self.records.append({
            "timestamp": time.time(),
            "operation": op_type,
            "stream_id": stream_id,
            "size_mb": round(size_mb, 2),
            "latency_ms": round(latency_ms, 2),
            "bandwidth_gbs": round(bw_gbs, 2)
        })

    def save(self):
        path = os.path.join(self.export_dir, f"pcie_session_{self.session_id}.json")
        try:
            with open(path, "w") as f:
                json.dump(self.records, f, indent=4)
        except Exception:
            pass

class EnterpriseVRAMStreamer:
    def __init__(self, profiler: CUDAHardwareProfiler):
        self.profiler = profiler
        self.safe_budget_mb = self.profiler.specs.free_vram_mb * 0.70
        self.host_allocator = PinnedMemoryAllocator(max_budget_mb=self.safe_budget_mb)
        self.orchestrator = CUDAStreamOrchestrator(num_streams=4)
        self.frag_monitor = VRAMFragmentationMonitor(total_vram=self.profiler.specs.total_vram_mb)
        self.exporter = TelemetryExporter()

    def upload_async(self, block: MemoryBlock, stream: Any) -> Tuple[Any, float]:
        start_t = time.perf_counter()
        d_array = cuda.to_device(block.host_array, stream=stream)
        lat = (time.perf_counter() - start_t) * 1000
        self.frag_monitor.register_allocation(block.size_mb)
        block.state = "TRANSFERRED_H2D"
        return d_array, lat

    def download_async(self, d_array: Any, block: MemoryBlock, stream: Any) -> float:
        start_t = time.perf_counter()
        d_array.copy_to_host(block.host_array, stream=stream)
        lat = (time.perf_counter() - start_t) * 1000
        block.state = "TRANSFERRED_D2H"
        return lat

    def free_device_memory(self, d_array: Any, size_mb: float):
        del d_array
        self.frag_monitor.register_deallocation(size_mb)

@cuda.jit
def dummy_warmup_kernel(d_array):
    idx = cuda.grid(1) # type: ignore
    if idx < d_array.size:
        d_array[idx] = d_array[idx] + 1

def run_pcie_stress_test():
    console.clear()
    profiler = CUDAHardwareProfiler()
    streamer = EnterpriseVRAMStreamer(profiler)
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )
    layout["left"].split_column(
        Layout(name="diag", ratio=1),
        Layout(name="streams", ratio=1)
    )
    layout["right"].split_column(
        Layout(name="frag", ratio=1),
        Layout(name="results", ratio=2)
    )
    
    layout["header"].update(Panel(Text("GENEWEAVER PHASE 2: CUDA VRAM MANAGER & PCIE PROFILER", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
    layout["diag"].update(profiler.get_diagnostic_panel())
    layout["streams"].update(streamer.orchestrator.get_stream_panel())
    layout["frag"].update(streamer.frag_monitor.get_monitor_panel())
    
    results_table = Table(box=None, expand=True)
    results_table.add_column("Operation", style="cyan")
    results_table.add_column("Chunk Size", style="yellow")
    results_table.add_column("Latency", style="magenta")
    results_table.add_column("Bandwidth", style="bold bright_green")
    layout["results"].update(Panel(results_table, title="[bold green]PCIe Telemetry[/bold green]", border_style="green"))

    chunk_size_mb = 120
    num_chunks = 4
    dtype = np.uint8
    elements = int((chunk_size_mb * (1024**2)) / dtype(0).itemsize)
    
    threads_per_block = profiler.specs.max_threads_per_block
    blocks_per_grid = (elements + threads_per_block - 1) // threads_per_block
    
    with Live(layout, refresh_per_second=10, screen=True):
        blocks = []
        for i in range(num_chunks):
            time.sleep(0.4)
            block = streamer.host_allocator.allocate_pinned((elements,), dtype) # type: ignore
            block.host_array[:] = np.random.randint(0, 255, size=elements, dtype=dtype) # type: ignore
            blocks.append(block)
            layout["frag"].update(streamer.frag_monitor.get_monitor_panel())
        
        d_arrays = []
        for i in range(num_chunks):
            time.sleep(0.3)
            idx, stream = streamer.orchestrator.get_next_stream()
            layout["streams"].update(streamer.orchestrator.get_stream_panel())
            
            d_arr, upload_lat = streamer.upload_async(blocks[i], stream)
            d_arrays.append(d_arr)
            bw = chunk_size_mb / (upload_lat / 1000) / 1024
            streamer.exporter.log_transfer("H2D", idx, chunk_size_mb, upload_lat, bw)
            
            results_table.add_row(f"H2D Upload (Stream {idx})", f"{chunk_size_mb} MB", f"{upload_lat:.2f} ms", f"{bw:.2f} GB/s")
            layout["results"].update(Panel(results_table, title="[bold green]PCIe Telemetry[/bold green]", border_style="green"))
            layout["frag"].update(streamer.frag_monitor.get_monitor_panel())
            
            dummy_warmup_kernel[blocks_per_grid, threads_per_block, stream](d_arr) # type: ignore
        
        streamer.orchestrator.synchronize_all()
        layout["streams"].update(streamer.orchestrator.get_stream_panel())
        
        for i in range(num_chunks):
            time.sleep(0.3)
            idx, stream = streamer.orchestrator.get_next_stream()
            layout["streams"].update(streamer.orchestrator.get_stream_panel())
            
            download_lat = streamer.download_async(d_arrays[i], blocks[i], stream)
            bw = chunk_size_mb / (download_lat / 1000) / 1024
            streamer.exporter.log_transfer("D2H", idx, chunk_size_mb, download_lat, bw)
            
            results_table.add_row(f"D2H Download (Stream {idx})", f"{chunk_size_mb} MB", f"{download_lat:.2f} ms", f"{bw:.2f} GB/s")
            layout["results"].update(Panel(results_table, title="[bold green]PCIe Telemetry[/bold green]", border_style="green"))
            
        streamer.orchestrator.synchronize_all()
        layout["streams"].update(streamer.orchestrator.get_stream_panel())
        
        for i in range(num_chunks):
            time.sleep(0.2)
            streamer.free_device_memory(d_arrays[i], chunk_size_mb)
            streamer.host_allocator.free_pinned(blocks[i].block_id)
            layout["frag"].update(streamer.frag_monitor.get_monitor_panel())
            
        layout["header"].update(Panel(Text("STRESS TEST COMPLETE // PRESS CTRL+C TO EXIT", style="bold red justify-center"), style="on black", border_style="red"))
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    run_pcie_stress_test()