import os
import sys
import time
import json
import uuid
import socket
import psutil
import hashlib
import platform
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

BioSeq: Any = None
BIOPYTHON_ACTIVE = False

try:
    from Bio.Seq import Seq as _BioSeq
    BioSeq = _BioSeq
    BIOPYTHON_ACTIVE = True
except Exception:
    BIOPYTHON_ACTIVE = False

console = Console()

class MismatchEngineError(Exception):
    def __init__(self, message: str, error_code: int = 1600):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()

class StrandLengthMismatchError(MismatchEngineError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1601)

class InvalidNucleotideError(MismatchEngineError):
    def __init__(self, message: str):
        super().__init__(message, error_code=1602)

@dataclass
class SystemProfile:
    os_name: str
    hostname: str
    biopython_active: bool
    logical_cores: int
    available_ram_gb: float

@dataclass
class MutatedBaseToken:
    position: int
    ref_base: str
    query_base: str
    mutation_type: str
    pam_distance: int
    severity_impact: float

@dataclass
class StrandDiffRecord:
    record_id: str
    reference_seq: str
    query_seq: str
    complement_preview: str
    total_bases: int
    mismatch_count: int
    identity_pct: float
    transition_count: int
    transversion_count: int
    nearest_pam_index: int
    biological_severity: float
    gc_delta_pct: float
    processing_latency_ms: float
    mutated_tokens: List[MutatedBaseToken]
    timestamp: float = field(default_factory=time.perf_counter)

@dataclass
class MismatchAuditManifest:
    session_id: str
    timestamp: str
    system_profile: SystemProfile
    total_strands_analyzed: int
    total_bases_compared: int
    total_mismatches_found: int
    mean_identity_pct: float
    total_transitions: int
    total_transversions: int
    peak_severity_score: float
    avg_latency_ms: float
    cryptographic_digest: str
    diff_records: List[StrandDiffRecord]

class EnvironmentInspector:
    def capture(self) -> SystemProfile:
        mem = psutil.virtual_memory()
        return SystemProfile(
            os_name=platform.system(),
            hostname=socket.gethostname(),
            biopython_active=BIOPYTHON_ACTIVE,
            logical_cores=psutil.cpu_count(logical=True) or 2,
            available_ram_gb=mem.available / (1024 ** 3)
        )

class MutationClassifier:
    TRANSITION_PAIRS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    COMPLEMENT_TABLE = str.maketrans("ACGT", "TGCA")

    def classify_substitution(self, ref_base: str, query_base: str) -> str:
        if (ref_base, query_base) in self.TRANSITION_PAIRS:
            return "TRANSITION"
        return "TRANSVERSION"

    def calculate_gc_pct(self, seq: str) -> float:
        if not seq:
            return 0.0
        gc_count = seq.count("G") + seq.count("C")
        return (gc_count / len(seq)) * 100.0

    def compute_complement(self, seq: str) -> str:
        if BIOPYTHON_ACTIVE and BioSeq is not None:
            return str(BioSeq(seq).complement())
        return seq.translate(self.COMPLEMENT_TABLE)

    def find_pam_positions(self, seq: str) -> List[int]:
        positions: List[int] = []
        seq_len = len(seq)
        for i in range(seq_len - 2):
            if seq[i + 1] == "G" and seq[i + 2] == "G":
                positions.append(i)
        return positions

    def score_mutation(self, pos: int, mut_type: str, pam_positions: List[int]) -> Tuple[int, float]:
        if not pam_positions:
            base = 42.0 if mut_type == "TRANSVERSION" else 30.0
            return -1, base
        nearest_dist = min(abs(pos - p) for p in pam_positions)
        proximity_score = max(0.0, 100.0 - (nearest_dist * 3.2))
        type_weight = 1.0 if mut_type == "TRANSVERSION" else 0.85
        return int(nearest_dist), round(min(100.0, proximity_score * type_weight), 2)

class DNAMismatchDiffEngine:
    def __init__(self):
        self.classifier = MutationClassifier()
        self.valid_ascii = {65, 67, 71, 84}

    def compare_strands(self, reference_seq: str, query_seq: str) -> StrandDiffRecord:
        t0 = time.perf_counter()
        if len(reference_seq) != len(query_seq):
            raise StrandLengthMismatchError("Reference and query DNA strands must have equal length.")

        ref_arr = np.frombuffer(reference_seq.encode("ascii"), dtype=np.uint8)
        qry_arr = np.frombuffer(query_seq.encode("ascii"), dtype=np.uint8)

        if not np.all(np.isin(ref_arr, [65, 67, 71, 84])) or not np.all(np.isin(qry_arr, [65, 67, 71, 84])):
            raise InvalidNucleotideError("Non-canonical nucleotide detected outside A, C, G, T.")

        mismatch_indices = np.flatnonzero(ref_arr != qry_arr)
        pam_positions = self.classifier.find_pam_positions(reference_seq)
        tokens: List[MutatedBaseToken] = []
        transitions = 0
        transversions = 0

        for raw_idx in mismatch_indices:
            pos = int(raw_idx)
            rb = reference_seq[pos]
            qb = query_seq[pos]
            m_type = self.classifier.classify_substitution(rb, qb)
            if m_type == "TRANSITION":
                transitions += 1
            else:
                transversions += 1
            pam_dist, impact = self.classifier.score_mutation(pos, m_type, pam_positions)
            tokens.append(MutatedBaseToken(
                position=pos,
                ref_base=rb,
                query_base=qb,
                mutation_type=m_type,
                pam_distance=pam_dist,
                severity_impact=impact
            ))

        total_len = len(reference_seq)
        m_count = len(tokens)
        identity = ((total_len - m_count) / total_len) * 100.0 if total_len > 0 else 0.0
        gc_delta = self.classifier.calculate_gc_pct(query_seq) - self.classifier.calculate_gc_pct(reference_seq)
        peak_sev = max([t.severity_impact for t in tokens], default=0.0)
        nearest_pam = pam_positions[0] if pam_positions else -1
        comp_preview = self.classifier.compute_complement(query_seq[:32])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return StrandDiffRecord(
            record_id=f"DIF_{uuid.uuid4().hex[:8].upper()}",
            reference_seq=reference_seq,
            query_seq=query_seq,
            complement_preview=comp_preview,
            total_bases=total_len,
            mismatch_count=m_count,
            identity_pct=identity,
            transition_count=transitions,
            transversion_count=transversions,
            nearest_pam_index=nearest_pam,
            biological_severity=peak_sev,
            gc_delta_pct=gc_delta,
            processing_latency_ms=latency_ms,
            mutated_tokens=tokens
        )

    def render_highlighted_alignment(self, record: StrandDiffRecord, window: int = 64) -> Text:
        ref = record.reference_seq[:window]
        qry = record.query_seq[:window]
        ref_line = Text(f"{record.record_id} REF | ", style="bold cyan")
        mid_line = Text("             | ", style="dim white")
        qry_line = Text("         QRY | ", style="bold magenta")

        for r_base, q_base in zip(ref, qry):
            if r_base == q_base:
                ref_line.append(r_base, style="white")
                mid_line.append("|", style="dim green")
                qry_line.append(q_base, style="green")
            else:
                ref_line.append(r_base, style="bold yellow")
                mid_line.append("X", style="bold bright_red")
                qry_line.append(q_base, style="bold white on red")

        block = Text()
        block.append_text(ref_line)
        block.append("\n")
        block.append_text(mid_line)
        block.append("\n")
        block.append_text(qry_line)
        return block

class SyntheticStrandFactory:
    def __init__(self, strand_length: int = 128):
        self.strand_length = strand_length
        self.rng = np.random.default_rng()
        self.bases = np.array(["A", "C", "G", "T"])

    def generate_pair(self) -> Tuple[str, str]:
        ref_chars = self.rng.choice(self.bases, size=self.strand_length)
        pam_pos = int(self.rng.integers(12, self.strand_length - 12))
        ref_chars[pam_pos + 1] = "G"
        ref_chars[pam_pos + 2] = "G"
        qry_chars = ref_chars.copy()

        mutation_count = int(self.rng.integers(3, 12))
        mut_positions = self.rng.choice(self.strand_length, size=mutation_count, replace=False)
        for pos in mut_positions:
            original = qry_chars[pos]
            alternatives = [b for b in ("A", "C", "G", "T") if b != original]
            qry_chars[pos] = self.rng.choice(alternatives)

        return "".join(ref_chars.tolist()), "".join(qry_chars.tolist())

class MismatchVisualizerUI:
    def __init__(self, diff_engine: DNAMismatchDiffEngine):
        self.diff_engine = diff_engine

    def create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="left_col", ratio=2),
            Layout(name="right_col", ratio=1)
        )
        layout["left_col"].split_column(
            Layout(name="visual_panel", ratio=1),
            Layout(name="ledger_panel", ratio=1)
        )
        layout["right_col"].split_column(
            Layout(name="bio_panel", ratio=1),
            Layout(name="env_panel", ratio=1)
        )

        title = Text("GENEWEAVER INDUSTRY PIPELINE: DNA MISMATCH VISUAL DIFF ENGINE (WEEK 4 DAY 3)", style="bold bright_cyan justify-center")
        layout["header"].update(Panel(title, style="on black", border_style="bright_cyan"))
        layout["footer"].update(Panel(Text("INITIALIZING BASE-BY-BASE MISMATCH SCANNER...", style="bold yellow justify-center"), style="on black", border_style="yellow"))
        return layout

    def update(self, layout: Layout, records: List[StrandDiffRecord], total: int, profile: SystemProfile) -> None:
        visual_stack = Text()
        recent_visuals = records[-2:]
        for idx, rec in enumerate(recent_visuals):
            visual_stack.append_text(self.diff_engine.render_highlighted_alignment(rec, window=64))
            if idx < len(recent_visuals) - 1:
                visual_stack.append("\n\n")

        layout["visual_panel"].update(Panel(visual_stack, title="[bold red]Visual DNA Mismatch Alignment (Mutated Base Pairs in Red)[/bold red]", border_style="red"))

        table = Table(box=None, expand=True)
        table.add_column("Record ID", style="cyan")
        table.add_column("Mismatches", style="bold red", justify="right")
        table.add_column("Identity", style="green", justify="right")
        table.add_column("Ti / Tv", style="magenta", justify="center")
        table.add_column("GC Delta", style="blue", justify="right")
        table.add_column("Severity", style="yellow", justify="right")
        table.add_column("Latency", style="white", justify="right")

        for rec in records[-8:]:
            table.add_row(
                rec.record_id,
                str(rec.mismatch_count),
                f"{rec.identity_pct:.2f}%",
                f"{rec.transition_count}/{rec.transversion_count}",
                f"{rec.gc_delta_pct:+.2f}%",
                f"{rec.biological_severity:.1f}",
                f"{rec.processing_latency_ms:.3f} ms"
            )

        layout["ledger_panel"].update(Panel(table, title="[bold green]Base-by-Base SNP Telemetry Ledger[/bold green]", border_style="green"))

        cnt = max(1, len(records))
        tot_muts = sum(r.mismatch_count for r in records)
        tot_ti = sum(r.transition_count for r in records)
        tot_tv = sum(r.transversion_count for r in records)
        mean_id = sum(r.identity_pct for r in records) / cnt
        peak_sev = max([r.biological_severity for r in records], default=0.0)

        bio = Table(box=None, expand=True)
        bio.add_column("Biological Metric", style="cyan")
        bio.add_column("Value", style="white", justify="right")
        bio.add_row("Strands Diffed", f"{len(records)} / {total}")
        bio.add_row("Total Red SNPs", f"[bold red]{tot_muts}[/bold red]")
        bio.add_row("Mean Identity", f"[green]{mean_id:.2f}%[/green]")
        bio.add_row("Transitions (Ti)", str(tot_ti))
        bio.add_row("Transversions (Tv)", str(tot_tv))
        bio.add_row("Peak PAM Severity", f"[bold yellow]{peak_sev:.2f}[/bold yellow]")
        layout["bio_panel"].update(Panel(bio, title="[bold magenta]Mutation Classification Metrics[/bold magenta]", border_style="magenta"))

        env = Table(box=None, expand=True)
        env.add_column("Engine Status", style="cyan")
        env.add_column("State", style="white", justify="right")
        bio_tag = "[green]ACTIVE[/green]" if profile.biopython_active else "[yellow]FALLBACK[/yellow]"
        env.add_row("BioPython Core", bio_tag)
        env.add_row("Diff Backend", "NUMPY_VECTORIZED")
        env.add_row("Logical Cores", str(profile.logical_cores))
        env.add_row("Available RAM", f"{profile.available_ram_gb:.2f} GB")
        env.add_row("Host Node", profile.hostname.upper())
        layout["env_panel"].update(Panel(env, title="[bold blue]Bioinformatics Runtime[/bold blue]", border_style="blue"))

        layout["footer"].update(Panel(Text(f"HIGHLIGHTING MUTATED BASE PAIRS... [{len(records)}/{total}]", style="bold yellow justify-center"), style="on black", border_style="yellow"))

class AuditVault:
    def __init__(self, output_dir: str = "reports/dna_mismatch_highlighter"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_manifest(self, manifest: MismatchAuditManifest) -> str:
        path = os.path.join(self.output_dir, f"mismatch_audit_{manifest.session_id}.json")
        with open(path, "w") as f:
            json.dump(asdict(manifest), f, indent=4)
        return path

class MismatchHighlighterOrchestrator:
    def __init__(self, total_strands: int = 30, strand_length: int = 128):
        self.total_strands = total_strands
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.inspector = EnvironmentInspector()
        self.profile = self.inspector.capture()
        self.diff_engine = DNAMismatchDiffEngine()
        self.factory = SyntheticStrandFactory(strand_length=strand_length)
        self.ui = MismatchVisualizerUI(self.diff_engine)
        self.vault = AuditVault()

    def execute(self) -> None:
        layout = self.ui.create_layout()
        records: List[StrandDiffRecord] = []

        with Live(layout, refresh_per_second=10, screen=True):
            for _ in range(self.total_strands):
                ref_seq, qry_seq = self.factory.generate_pair()
                record = self.diff_engine.compare_strands(ref_seq, qry_seq)
                records.append(record)
                self.ui.update(layout, records, self.total_strands, self.profile)
                time.sleep(0.12)

            layout["footer"].update(Panel(Text("DNA MISMATCH HIGHLIGHTING COMPLETE // PRESS CTRL+C TO EXIT", style="bold bright_green justify-center"), style="on black", border_style="bright_green"))
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

        cnt = max(1, len(records))
        tot_bases = sum(r.total_bases for r in records)
        tot_muts = sum(r.mismatch_count for r in records)
        tot_ti = sum(r.transition_count for r in records)
        tot_tv = sum(r.transversion_count for r in records)
        mean_id = sum(r.identity_pct for r in records) / cnt
        peak_sev = max([r.biological_severity for r in records], default=0.0)
        avg_lat = sum(r.processing_latency_ms for r in records) / cnt
        digest = hashlib.sha256(f"{self.session_id}_{tot_muts}_{mean_id:.3f}_{peak_sev:.2f}".encode("utf-8")).hexdigest()

        manifest = MismatchAuditManifest(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            system_profile=self.profile,
            total_strands_analyzed=len(records),
            total_bases_compared=tot_bases,
            total_mismatches_found=tot_muts,
            mean_identity_pct=mean_id,
            total_transitions=tot_ti,
            total_transversions=tot_tv,
            peak_severity_score=peak_sev,
            avg_latency_ms=avg_lat,
            cryptographic_digest=digest,
            diff_records=records
        )

        saved_path = self.vault.export_manifest(manifest)
        console.print(f"\n[bold bright_green][+] Week 4 Day 3 DNA Mismatch Highlighter Complete. Total Mutated Bases: {tot_muts}[/bold bright_green]")
        console.print(f"[bold cyan][i] Mismatch Diff Audit saved to: {saved_path}[/i][/bold cyan]")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    app = MismatchHighlighterOrchestrator()
    app.execute()