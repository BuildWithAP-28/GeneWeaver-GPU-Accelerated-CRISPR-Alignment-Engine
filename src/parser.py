"""
Infotact Solutions - GeneWeaver Project
Module: Advanced Genomic Data Parser (Phase 1)
Description: Enterprise-grade ingestion pipeline for biological sequence data.
Features: 
- Zero-copy memory-mapped (mmap) SHA-256 checksum validation
- BioPython lazy loading (SeqIO) for memory safety on massive files
- Overlapping Sliding Window algorithm to prevent boundary mutation drops
- Biological sequence sanitization (IUPAC nucleotide validation)
- Multi-format support (.fasta, .fastq, .gz compressed)
"""

import os
import re
import sys
import mmap
import gzip
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

console = Console()

# ==========================================
# Custom Exception Hierarchy
# ==========================================
class ParserError(Exception):
    """Base exception for Genomic Parser operations."""
    pass

class FormatDetectionError(ParserError):
    """Raised when the biological file format cannot be determined."""
    pass

class ChecksumMismatchError(ParserError):
    """Raised when data integrity validation fails."""
    pass

class SequenceSanitizationError(ParserError):
    """Raised when sequence contains excessive invalid nucleotides."""
    pass


# ==========================================
# Logger Configuration
# ==========================================
def setup_parser_logging(log_dir: str = "logs") -> logging.Logger:
    """Configures dual-channel logging for the ingestion pipeline."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "ingestion_pipeline.log"

    logger = logging.getLogger("GeneWeaverParser")
    logger.setLevel(logging.DEBUG)
    
    if not logger.hasHandlers():
        # Terminal Handler (Rich)
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

        # File Handler
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s"))

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_parser_logging()

# ==========================================
# Core Parser Engine
# ==========================================
class GenomicParser:
    """
    Robust ingestion engine designed to feed massive FASTA/FASTQ datasets 
    into memory-constrained environments (e.g., 4GB VRAM limits).
    """
    
    # Valid IUPAC nucleotide characters for DNA
    VALID_NUCLEOTIDES = set("ACTGactgNn")

    def __init__(self, file_path: str, chunk_size_mb: int = 5, overlap_bp: int = 100):
        """
        Initializes the parser with specific memory and algorithmic constraints.
        
        Args:
            file_path (str): Path to the target genomic file.
            chunk_size_mb (int): Max RAM buffer per yielded sequence chunk.
            overlap_bp (int): Number of base pairs to overlap between chunks 
                              to prevent CRISPR target splitting.
        """
        self.file_path = Path(file_path)
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self.overlap_bp = overlap_bp
        
        if not self.file_path.exists():
            logger.error(f"Target dataset not found: {self.file_path}")
            raise FileNotFoundError(f"Missing file: {self.file_path}")
            
        self.format, self.is_compressed = self._detect_format()
        self.file_size_mb = self.file_path.stat().st_size / (1024 * 1024)

    def _detect_format(self) -> Tuple[str, bool]:
        """Dynamically identifies sequence format and compression status."""
        name = self.file_path.name.lower()
        is_gz = name.endswith(".gz")
        
        if "fastq" in name or "fq" in name:
            fmt = "fastq"
        elif "fasta" in name or "fa" in name or "fna" in name:
            fmt = "fasta"
        else:
            logger.error(f"Unrecognized sequence extension in {name}")
            raise FormatDetectionError("Supported formats: .fasta, .fastq (and .gz variants)")
            
        logger.debug(f"Detected Format: {fmt.upper()} | Compressed: {is_gz}")
        return fmt, is_gz

    def verify_checksum(self) -> bool:
        """
        Calculates SHA-256 hash using memory-mapped files (mmap) for zero-copy 
        extreme IO performance, comparing it against the tracker file.
        """
        hash_file = Path(f"{self.file_path}.sha256")
        
        if not hash_file.exists():
            console.print("[yellow][!] Warning: No .sha256 tracker found. Skipping integrity validation.[/yellow]")
            return True

        logger.info("Initiating zero-copy SHA-256 integrity validation...")
        
        try:
            with open(hash_file, "r") as f:
                expected_hash = f.read().split()[0].strip()

            sha256_hash = hashlib.sha256()
            
            with open(self.file_path, "rb") as f:
                # Use mmap for files larger than 0 bytes to bypass OS buffer cache
                if self.file_path.stat().st_size > 0:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        # Process in 1MB chunks to keep CPU cache hot
                        for offset in range(0, mm.size(), 1024*1024):
                            sha256_hash.update(mm[offset:offset + 1024*1024])
                else:
                    sha256_hash.update(f.read())
                    
            actual_hash = sha256_hash.hexdigest()

            if actual_hash == expected_hash:
                console.print("[bold green][+] Data Integrity Verified: Checksums match perfectly.[/bold green]")
                return True
            else:
                logger.error(f"Checksum mismatch! Expected: {expected_hash} | Actual: {actual_hash}")
                raise ChecksumMismatchError("Data corruption detected during ingestion.")
                
        except Exception as e:
            logger.error(f"Hashing failed: {e}")
            raise

    def sanitize_sequence(self, sequence: str) -> str:
        """
        Cleans the biological string. Converts to uppercase and alerts if 
        excessive garbage characters are found.
        """
        seq = sequence.upper()
        
        # Calculate percentage of unknown 'N' bases
        n_count = seq.count('N')
        if len(seq) > 0 and (n_count / len(seq)) > 0.3:
            logger.warning(f"High concentration of unknown 'N' bases detected ({n_count}).")
            
        # Regex to strip anything that is not a standard nucleotide
        clean_seq = re.sub(r'[^ACTGN]', '', seq)
        
        if len(clean_seq) < len(seq) * 0.5:
            raise SequenceSanitizationError("Sequence contains >50% invalid biological characters.")
            
        return clean_seq

    def stream_overlapping_chunks(self) -> Generator[Dict[str, Any], None, None]:
        """
        The core HPC generator. Yields sequence data in fixed-size chunks 
        with a specified base-pair overlap. 
        
        Why? A CRISPR mutation might be 20 base pairs long. If chunk 1 ends at 
        base 10, and chunk 2 starts at base 11, the mutation is split and invisible 
        to the GPU. The overlap window prevents this biological data loss.
        """
        open_func = gzip.open if self.is_compressed else open
        mode = "rt" if self.is_compressed else "r"

        logger.info(f"Opening data stream with {self.chunk_size_bytes/(1024*1024):.1f}MB chunks and {self.overlap_bp}bp overlap...")

        try:
            with open_func(self.file_path, mode) as handle:
                # BioPython SeqIO handles the complex format parsing safely
                for record in SeqIO.parse(handle, self.format):
                    full_seq = str(record.seq)
                    seq_len = len(full_seq)
                    
                    if seq_len == 0:
                        continue
                        
                    # Clean the full sequence once before chunking
                    clean_seq = self.sanitize_sequence(full_seq)
                    
                    # Sliding Window Chunking Logic
                    start_idx = 0
                    chunk_id = 0
                    
                    while start_idx < seq_len:
                        end_idx = min(start_idx + self.chunk_size_bytes, seq_len)
                        
                        yield {
                            "record_id": record.id,
                            "chunk_id": chunk_id,
                            "start_index": start_idx,
                            "end_index": end_idx,
                            "sequence_data": clean_seq[start_idx:end_idx],
                            "is_last_chunk": end_idx == seq_len
                        }
                        
                        # Advance the window, shifting back by overlap_bp
                        start_idx = end_idx - self.overlap_bp
                        chunk_id += 1
                        
                        # Safety break if we are at the end
                        if end_idx == seq_len:
                            break
                            
        except Exception as e:
            logger.error(f"Fatal error during sequence streaming: {e}")
            sys.exit(1)

# ==========================================
# CLI Execution & Telemetry
# ==========================================
def display_summary(total_records: int, total_chunks: int, total_bp: int, file_size: float):
    """Renders execution summary via Rich table."""
    table = Table(title="GeneWeaver Ingestion Summary", title_style="bold green")
    table.add_column("Metric", justify="right", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("File Size Processed", f"{file_size:.2f} MB")
    table.add_row("Total Sequence Records", str(total_records))
    table.add_row("Total Overlapping Chunks", str(total_chunks))
    table.add_row("Total Base Pairs (bp)", f"{total_bp:,}")
    table.add_row("Status", "[bold green]Ready for GPU Allocation[/bold green]")

    console.print(table)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    parser_args = argparse.ArgumentParser(description="Infotact GeneWeaver: Genomic Ingestion Pipeline")
    parser_args.add_argument('--file', type=str, default="data/mock_genome.fastq", help='Path to target dataset')
    parser_args.add_argument('--chunk-mb', type=int, default=5, help='Size of string chunks in MB')
    parser_args.add_argument('--overlap', type=int, default=100, help='Base pair overlap between chunks')
    args = parser_args.parse_args()

    try:
        console.print("[bold cyan]>>> Initializing Phase 1: Ingestion Pipeline...[/bold cyan]")
        
        # Initialize Parser Engine
        ingestion_engine = GenomicParser(
            file_path=args.file,
            chunk_size_mb=args.chunk_mb,
            overlap_bp=args.overlap
        )
        
        # Execute Integrity Check
        ingestion_engine.verify_checksum()
        
        # Execute Streaming with Progress Bar
        total_records = 0
        total_chunks = 0
        total_bp = 0
        current_record_id = None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green"),
            TextColumn("[cyan]{task.fields[status]}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[bold cyan]Streaming DNA sequences...", total=None, status="Initializing...")
            
            for chunk_data in ingestion_engine.stream_overlapping_chunks():
                total_chunks += 1
                total_bp += len(chunk_data["sequence_data"])
                
                # Count unique biological records (e.g., chromosomes)
                if chunk_data["record_id"] != current_record_id:
                    total_records += 1
                    current_record_id = chunk_data["record_id"]
                    
                progress.update(
                    task, 
                    status=f"Streaming Chunk #{chunk_data['chunk_id']} from {current_record_id}"
                )
                
        # Display Final Analytics
        display_summary(total_records, total_chunks, total_bp, ingestion_engine.file_size_mb)
        
    except KeyboardInterrupt:
        console.print("\n[bold red][!] Pipeline aborted by user (SIGINT).[/bold red]")
        sys.exit(0)
    except Exception as e:
        logger.exception("Ingestion Pipeline encountered a critical failure.")
        sys.exit(1)