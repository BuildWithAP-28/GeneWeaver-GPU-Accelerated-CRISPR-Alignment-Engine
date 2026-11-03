# GeneWeaver-GPU-Accelerated-CRISPR-Alignment-Engine
GeneWeaver tackles a massive computational bottleneck in bioinformatics: searching the 3.2 billion base pairs of the human genome for CRISPR off-target mutations. Standard Python string matching takes days, traditionally forcing reliance on C++ clusters.   

🧬 Phase 1: Core Engine & Memory Optimization

Running brute-force genomic alignments usually requires massive compute clusters. For this project, the immediate bottleneck is hardware: specifically, getting this to run without crashing on a laptop GPU with only 4GB of VRAM (RTX 3050 Mobile).

To make this work, Phase 1 isn't just about writing a sequence matcher; it's about extreme memory paranoia. If we send a 3.2 billion base-pair string to the GPU all at once, the system will OOM (Out of Memory) crash instantly.

Here are the 8 architecture decisions baked into Phase 1 to keep the script lean, fast, and stable on limited hardware:

1. Ultra-Strict Dynamic Memory Chunking

We can't rely on blind chunking. The script uses psutil and strict byte-size math to calculate exact chunk sizes dynamically. We enforce a hard 3GB VRAM buffer cap, leaving ~1GB untouched so Windows and your display drivers don't freeze during execution. We also heavily abuse Python's gc.collect() to force-clear RAM the second a chunk is processed.

2. Synthetic Genome Generator

Downloading the massive 3GB+ human genome every time you want to test a bug fix is terrible for iteration speed. I built a synthetic_generator.py utility that spits out custom 100MB+ .fasta files with randomized A, C, T, G strings and injects specific target mutations so we can test the pipeline offline.

3. Multi-Format Ingestion

Genomic data is messy. The parser leverages BioPython to seamlessly handle raw .fasta, .fastq, and compressed .fasta.gz files out of the box, so we don't have to manually unzip massive datasets before running the tool.

4. SHA-256 Checksums

There's nothing worse than running a script for 20 minutes only to realize the source file was truncated during download. The pipeline runs a quick SHA-256 hash check on load to validate file integrity before spending a single CPU cycle on alignment.

5. Config-Driven Limits (config.yaml)

No magic numbers in the codebase. Everything from regex patterns to the max_vram_buffer_mb: 3000 limit is centralized in a YAML file.

6. Persistent Audit Logging

We need to know exactly what the baseline pure-Python speed is so we can prove the Numba/CUDA speedup later. The logging module tracks execution times, chunk processing speeds, and GC events, dumping them into a local .log file.

7. Custom CSS TUI (Textual)

Command-line tools don't have to look like they were built in 1995. We're using the Textual library to build a responsive, dark-mode terminal UI. Custom CSS enforces a high-contrast matrix-green and black aesthetic specifically for monitoring live VRAM telemetry and progress bars.

8. Graceful SIGINT Handling (Ctrl+C Safety)

If you hit Ctrl+C while the script is holding 3GB of genomic data in memory, standard Python will just dump an ugly traceback and sometimes leave memory hanging. We catch KeyboardInterrupts properly to instantly flush the buffers, clear VRAM, and restore the terminal state cleanly.

🚀 Key Mid-Project Metrics (Achieved)

Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)

PCIe Bandwidth (Host-to-Device): ~9.60 GB/s sustained throughput

Performance Speedup: 25x Faster than Pure Python Baseline

500MB Workload Latency: CPU (~1253 ms) vs GPU (~50 ms)

Accuracy: 100% Cross-Validated with CPU byte-counting algorithms

🛠️ Tech Stack

Core Logic: Python, NumPy (v2.x)

GPU Acceleration: Numba (@cuda.jit), CUDA Toolkit

Telemetry & UI: Rich (Advanced TUI, Live Dashboards)

Memory Management: Pinned Memory (Page-locked), L1 Cache (Shared Memory)

📅 Project Roadmap & Development Timeline

🟢 Week 1: Foundations & CPU Baseline (100% Complete)

Established the core data pipelines, synthetic genome generation, and pure Python benchmarking.

Day 1-4: Project setup, architectural planning, and environment configuration (Conda).

Day 5-7: Implemented synthetic DNA generation and established the CPU baseline using simulated KMP/Sliding window algorithms (Initial speed: ~3 MB/s to 46 MB/s).

🟢 Week 2: CUDA GPU Acceleration & Benchmarking (100% Complete)

Transitioned from CPU architecture to High-Performance Computing (HPC) using NVIDIA CUDA.

Day 8: Numba JIT CPU optimization and syntax warmups.

Day 9: Multi-processing Orchestrator implementation.

Day 10: SIMD Vectorization & Memory Profiling (C-contiguous arrays).

Day 11: Advanced Telemetry UI (dashboard_v2.py) for live monitoring.

Day 12: CUDA Environment Setup & Host-to-Device Memory Bridge (cuda_bridge.py). Successfully managed 4GB VRAM allocation limits via chunking and pinned memory.

Day 13: The @cuda.jit Kernel (cuda_engine.py). Implemented ultra-fast DNA alignment using Shared Memory (L1 Cache) and Atomic Operations. Achieved ~9.6 GB/s throughput.

Day 14 (Mid-Review): CPU vs GPU Master Benchmarking (benchmark_suite.py). Orchestrated head-to-head testing proving a 25x compute speedup for massive datasets.

🟡 Week 3: Dask Integration & Multi-Node Scaling (Upcoming)

Focus: Scaling beyond the 4GB VRAM limit to process 10GB+ genome files using Dask distributed computing.

⚪ Week 4: Advanced VRAM Optimization & Final Polish (Upcoming)

Focus: Extreme memory optimizations, handling multi-GB FASTA files, and final project deployment.

💻 Quickstart (Running the Benchmark)

To witness the head-to-head performance of the CPU vs GPU, run the Master Benchmark Orchestrator:

# Ensure you are in the project environment with CUDA toolkit installed
python src/benchmark_suite.py


This will launch a live TUI dashboard comparing CPU and GPU latencies across workloads ranging from 10MB to 500MB, outputting a final JSON telemetry report.
