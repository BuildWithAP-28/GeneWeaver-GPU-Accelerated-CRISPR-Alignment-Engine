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
