"""
Infotact Solutions - GeneWeaver Project
Module: Synthetic Genome Generator (Phase 1)
Description: Enterprise-grade synthetic data generator for testing CRISPR alignment.
Features: 
- strict RAM management (psutil/gc)
- Biological GC content targeting
- Ground truth mutation tracking
- Multi-format export (FASTA, FASTQ, GZ)
- SHA-256 data integrity checksums
"""

import os
import gc
import sys
import json
import time
import gzip
import yaml
import random
import psutil
import hashlib
import logging
import argparse
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

# Initialize rich console for terminal UI
console = Console()

# ==========================================
# Custom Exception Classes
# ==========================================
class GeneWeaverError(Exception):
    """Base exception for GeneWeaver pipeline."""
    pass

class ConfigValidationError(GeneWeaverError):
    """Raised when YAML configuration schema is invalid."""
    pass

class MemoryConstraintError(GeneWeaverError):
    """Raised when available system VRAM/RAM is critically low."""
    pass

class DataExportError(GeneWeaverError):
    """Raised when IO operations fail during synthesis."""
    pass

# ==========================================
# Advanced Logger Setup
# ==========================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Sets up dual logging: Rich UI for terminal, Rotating files for audit."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "synthetic_generation.log"

    logger = logging.getLogger("GeneWeaverLogger")
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # Terminal Handler (Rich)
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(logging.INFO)
    rich_formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(rich_formatter)

    # File Handler (Rotating log files, max 5MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ==========================================
# Configuration Management & Validation
# ==========================================
class ConfigManager:
    """Handles loading, strict type validation, and retrieval of YAML settings."""
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_schema()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at {self.config_path}")
            sys.exit(1)
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing failed: {e}")
            sys.exit(1)

    def _validate_schema(self):
        """Ensures all expected keys and data types exist in the config."""
        required_schema = {
            'hardware': {'max_vram_buffer_mb': int},
            'paths': {'synthetic_data_output': str, 'mutation_ground_truth': str},
            'sequence': {'target_mutation': str, 'gc_content_target': float, 'mock_file_size_mb': int, 'format': str}
        }
        for section, keys in required_schema.items():
            if section not in self.config:
                raise ConfigValidationError(f"Missing section '{section}' in config.")
            for key, expected_type in keys.items():
                if key not in self.config[section]:
                    raise ConfigValidationError(f"Missing key '{key}' in section '{section}'.")
                if not isinstance(self.config[section][key], expected_type):
                    raise ConfigValidationError(f"Invalid type for '{key}'. Expected {expected_type.__name__}.")

# ==========================================
# Core Generator Engine
# ==========================================
class GenomeGenerator:
    """Engine for generating biologically accurate, memory-safe FASTA/FASTQ datasets."""
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        
        # Hardware Limits
        self.max_vram_mb = self.config['hardware']['max_vram_buffer_mb']
        
        # Paths
        out_path = self.config['paths']['synthetic_data_output']
        self.output_format = self.config['sequence']['format'].lower()
        
        # Adjust file extension based on format preference
        if self.output_format == "fastq":
            out_path = out_path.replace(".fasta", ".fastq")
        elif self.output_format == "fasta.gz":
            out_path = out_path.replace(".fasta", ".fasta.gz")
            
        self.fasta_out = out_path
        self.truth_out = self.config['paths']['mutation_ground_truth']
        
        # Sequence Parameters
        self.target_mutation = self.config['sequence']['target_mutation']
        self.gc_content = self.config['sequence']['gc_content_target']
        self.file_size_mb = self.config['sequence']['mock_file_size_mb']
        
        # Internal sizing logic
        self.bytes_per_mb = 1024 * 1024
        self.target_bytes = self.file_size_mb * self.bytes_per_mb
        # Chunk size is strictly 5MB to ensure safe memory GC pooling on 4GB VRAM systems
        self.chunk_size_bytes = 5 * self.bytes_per_mb 

    def _check_system_memory(self):
        """Proactively checks system RAM before massive array allocations."""
        available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_ram_mb < 500:
            logger.warning(f"Low system memory detected ({available_ram_mb:.0f} MB). Forcing GC.")
            gc.collect()
            if psutil.virtual_memory().available / (1024 * 1024) < 300:
                raise MemoryConstraintError("System out of safe memory bounds. Aborting to prevent crash.")

    def _generate_dna_chunk(self) -> str:
        """Generates a DNA string matching the target GC content ratio."""
        gc_w = self.gc_content / 2
        at_w = (1 - self.gc_content) / 2
        bases, weights = ['A', 'T', 'G', 'C'], [at_w, at_w, gc_w, gc_w]
        return "".join(random.choices(bases, weights=weights, k=self.chunk_size_bytes))

    def _generate_fastq_qualities(self, length: int) -> str:
        """Generates mock Phred quality scores for FASTQ format."""
        # ASCII range 33 to 73 represents standard illumina quality scores
        return "".join(chr(random.randint(33, 73)) for _ in range(length))

    def _calculate_sha256(self, filepath: str) -> str:
        """Generates a SHA-256 checksum for the generated file."""
        sha256_hash = hashlib.sha256()
        logger.info("[bold cyan]Calculating SHA-256 data checksum...[/bold cyan]")
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate(self):
        """Main orchestrator for generating and exporting the data."""
        Path(self.fasta_out).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing GeneWeaver Synthetic Generator...")
        logger.info(f"Target Size: {self.file_size_mb} MB | Format: {self.output_format.upper()}")
        logger.debug(f"Hardware Buffer Limit: {self.max_vram_mb} MB")
        
        total_chunks = max(1, self.target_bytes // self.chunk_size_bytes)
        ground_truth_records = []
        current_offset = 0
        start_time = time.time()
        
        # Handle file opener based on compression requirement
        open_func = gzip.open if self.output_format == "fasta.gz" else open
        mode = "wt" if self.output_format == "fasta.gz" else "w"

        try:
            with open_func(self.fasta_out, mode) as out_file:
                # Write appropriate Header
                header_id = f"chr1_mock_infotact_size_{self.file_size_mb}MB"
                if "fasta" in self.output_format:
                    out_file.write(f">{header_id}\n") # type: ignore
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(complete_style="green", finished_style="bold green"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("[bold cyan]Synthesizing Sequence...", total=total_chunks)
                    
                    for chunk_idx in range(total_chunks):
                        self._check_system_memory()
                        
                        chunk = self._generate_dna_chunk()
                        
                        # Mutation Injection (15% chance per 5MB chunk)
                        if random.random() > 0.85:
                            insert_pos = random.randint(0, len(chunk) - len(self.target_mutation))
                            chunk = chunk[:insert_pos] + self.target_mutation + chunk[insert_pos + len(self.target_mutation):]
                            
                            ground_truth_records.append({
                                "chromosome": header_id,
                                "mutation": self.target_mutation,
                                "chunk_id": chunk_idx,
                                "absolute_index": current_offset + insert_pos
                            })
                        
                        # Format Export Logic
                        if self.output_format == "fastq":
                            out_file.write(f"@{header_id}_{chunk_idx}\n") # type: ignore
                            out_file.write(chunk + "\n+\n") # type: ignore
                            out_file.write(self._generate_fastq_qualities(len(chunk)) + "\n") # type: ignore
                        else:
                            # Standard FASTA line wrap (typically 80 chars, but keeping continuous for chunk speed)
                            out_file.write(chunk + "\n") # type: ignore
                            
                        current_offset += len(chunk)
                        progress.advance(task)
                        
                        # Aggressive GC collection per chunk for 4GB systems
                        del chunk
                        gc.collect()
                        
        except IOError as e:
            logger.error(f"Failed to write genomic data to disk: {e}")
            raise DataExportError("Disk IO Write Failure.")

        # Export Ground Truth JSON
        try:
            with open(self.truth_out, "w") as json_file:
                json.dump(ground_truth_records, json_file, indent=4)
        except IOError as e:
            logger.error(f"Failed to write ground truth JSON: {e}")

        # Post-Processing
        exec_time = time.time() - start_time
        checksum = self._calculate_sha256(self.fasta_out)
        
        # Save checksum to file
        with open(f"{self.fasta_out}.sha256", "w") as sha_file:
            sha_file.write(f"{checksum}  {Path(self.fasta_out).name}\n")
            
        self._display_summary(exec_time, len(ground_truth_records), checksum)

    def _display_summary(self, exec_time: float, mutation_count: int, checksum: str):
        """Renders an enterprise-grade telemetry table to the terminal."""
        table = Table(title="GeneWeaver Synthesis Telemetry", title_style="bold green")
        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Total Data Size", f"{self.file_size_mb} MB")
        table.add_row("Output Format", self.output_format.upper())
        table.add_row("Output File", str(self.fasta_out))
        table.add_row("Mutations Injected", str(mutation_count))
        table.add_row("Ground Truth Mapping", str(self.truth_out))
        table.add_row("SHA-256 Checksum", f"{checksum[:16]}...{checksum[-8:]}")
        table.add_row("Execution Time", f"{exec_time:.2f} seconds")
        table.add_row("Average IO Speed", f"{(self.file_size_mb / exec_time):.2f} MB/s")

        console.print(table)
        logger.info("Pipeline Execution Completed Successfully.")

# ==========================================
# Entry Point & CLI
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Infotact GeneWeaver: Synthetic Genome Data Generator")
    parser.add_argument('--size', type=int, help='Override the mock file size in MB', default=None)
    parser.add_argument('--format', type=str, choices=['fasta', 'fastq', 'fasta.gz'], help='Override output format', default=None)
    parser.add_argument('--mutation', type=str, help='Override target mutation sequence', default=None)
    return parser.parse_args()

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    try:
        args = parse_args()
        config_mgr = ConfigManager()
        
        # Apply CLI overrides to Config
        if args.size:
            config_mgr.config['sequence']['mock_file_size_mb'] = args.size
        if args.format:
            config_mgr.config['sequence']['format'] = args.format
        if args.mutation:
            config_mgr.config['sequence']['target_mutation'] = args.mutation
            
        generator = GenomeGenerator(config_mgr)
        generator.generate()
        
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Process interrupted by user (SIGINT). Cleaning memory buffer...[/bold red]")
        gc.collect()
        sys.exit(0)
    except Exception as e:
        logger.exception("An unexpected critical error occurred during pipeline execution.")
        sys.exit(1)