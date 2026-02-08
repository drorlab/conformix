"""
run_benchmark.py

Main entry point for the ConforMix benchmarking suite.
Runs ConforMix on benchmark datasets and saves results for evaluation.

Usage:
    python -m benchmarks.run_benchmark --config benchmarks/configs/domainmotion.yaml
    python -m benchmarks.run_benchmark --config benchmarks/configs/domainmotion.yaml --proteins P0205 P69441
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    
    dataset_name: str
    csv_path: Path
    output_dir: Path
    num_twist_targets: int = 5
    samples_per_target: int = 2
    structured_regions_only: bool = True
    subset_residues: Optional[str] = None
    twist_strength: float = 15.0
    timeout_seconds: int = 3600
    skip_existing: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BenchmarkConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            dataset_name=data["dataset_name"],
            csv_path=Path(data["csv_path"]),
            output_dir=Path(data["output_dir"]),
            num_twist_targets=data.get("num_twist_targets", 5),
            samples_per_target=data.get("samples_per_target", 2),
            structured_regions_only=data.get("structured_regions_only", True),
            subset_residues=data.get("subset_residues"),
            twist_strength=data.get("twist_strength", 15.0),
            timeout_seconds=data.get("timeout_seconds", 3600),
            skip_existing=data.get("skip_existing", True),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "csv_path": str(self.csv_path),
            "output_dir": str(self.output_dir),
            "num_twist_targets": self.num_twist_targets,
            "samples_per_target": self.samples_per_target,
            "structured_regions_only": self.structured_regions_only,
            "subset_residues": self.subset_residues,
            "twist_strength": self.twist_strength,
            "timeout_seconds": self.timeout_seconds,
            "skip_existing": self.skip_existing,
        }


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single protein."""
    
    system_id: str
    pdb1: str
    pdb2: str
    ground_truth_rmsd: float
    success: bool
    runtime_seconds: float
    output_dir: Path
    error_message: Optional[str] = None
    num_samples_generated: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "system_id": self.system_id,
            "pdb1": self.pdb1,
            "pdb2": self.pdb2,
            "ground_truth_rmsd": self.ground_truth_rmsd,
            "success": self.success,
            "runtime_seconds": self.runtime_seconds,
            "output_dir": str(self.output_dir),
            "error_message": self.error_message,
            "num_samples_generated": self.num_samples_generated,
        }


def fetch_sequence_from_pdb(pdb_id: str, chain_id: str, cache_dir: Path) -> Optional[Path]:
    """
    Fetch sequence from RCSB PDB and save as FASTA file.
    
    Args:
        pdb_id: PDB ID (e.g., '1AKE')
        chain_id: Chain ID (e.g., 'A')
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Path to FASTA file, or None if fetch failed
    """
    import urllib.request
    import urllib.error
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = cache_dir / f"{pdb_id}_{chain_id}.fasta"
    
    if fasta_path.exists():
        logger.debug(f"Using cached FASTA: {fasta_path}")
        return fasta_path
    
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
    
    try:
        logger.info(f"Fetching sequence for {pdb_id}_{chain_id} from RCSB...")
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
        
        lines = content.strip().split("\n")
        target_header = None
        target_sequence = []
        current_header = None
        current_sequence = []
        
        for line in lines:
            if line.startswith(">"):
                if current_header and chain_id in current_header:
                    target_header = current_header
                    target_sequence = current_sequence
                current_header = line
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_header and chain_id in current_header:
            target_header = current_header
            target_sequence = current_sequence
        
        if target_header and target_sequence:
            with open(fasta_path, "w") as f:
                f.write(f">{pdb_id}_{chain_id}\n")
                f.write("".join(target_sequence) + "\n")
            logger.info(f"Saved FASTA to: {fasta_path}")
            return fasta_path
        else:
            logger.warning(f"Chain {chain_id} not found in {pdb_id}")
            return None
            
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch {pdb_id}: {e}")
        return None


def run_conformix_boltz(
    fasta_path: Path,
    output_dir: Path,
    config: BenchmarkConfig,
) -> tuple[bool, float, str]:
    """
    Run ConforMix-Boltz on a single protein.
    
    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory to save outputs
        config: Benchmark configuration
        
    Returns:
        Tuple of (success, runtime_seconds, error_message)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "boltz.run_conformixrmsd_boltz",
        "--fasta_path", str(fasta_path),
        "--out_dir", str(output_dir),
        "--num_twist_targets", str(config.num_twist_targets),
        "--samples_per_target", str(config.samples_per_target),
        "--twist_strength", str(config.twist_strength),
    ]
    
    if config.structured_regions_only:
        cmd.append("--structured_regions_only")
    
    if config.subset_residues:
        cmd.extend(["--subset_residues", config.subset_residues])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            cwd=Path(__file__).parent.parent,
        )
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            return True, runtime, ""
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return False, runtime, error_msg
            
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        return False, runtime, f"Timeout after {config.timeout_seconds}s"
    except Exception as e:
        runtime = time.time() - start_time
        return False, runtime, str(e)


def count_samples(output_dir: Path) -> int:
    """Count the number of sample files generated."""
    patterns = ["*.cif", "*.pdb", "*.xtc"]
    count = 0
    for pattern in patterns:
        count += len(list(output_dir.glob(f"**/{pattern}")))
    return count


def run_benchmark(
    config: BenchmarkConfig,
    proteins: Optional[list[str]] = None,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """
    Run the benchmark on a dataset.
    
    Args:
        config: Benchmark configuration
        proteins: Optional list of specific protein IDs to run
        dry_run: If True, only print what would be done
        
    Returns:
        List of BenchmarkResult objects
    """
    logger.info(f"Loading dataset from {config.csv_path}")
    df = pd.read_csv(config.csv_path)
    
    if proteins:
        df = df[df["system_id"].isin(proteins) | 
                df["pdb1"].str.split("_").str[0].isin(proteins)]
        logger.info(f"Filtered to {len(df)} proteins")
    
    logger.info(f"Running benchmark on {len(df)} proteins from {config.dataset_name}")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    cache_dir = config.output_dir / ".cache"
    results = []
    
    for idx, row in df.iterrows():
        system_id = row["system_id"]
        pdb1 = row["pdb1"]
        pdb2 = row["pdb2"]
        ground_truth_rmsd = row.get("RMSD", 0.0)
        
        pdb_id, chain_id = pdb1.split("_") if "_" in pdb1 else (pdb1, "A")
        
        protein_output_dir = config.output_dir / system_id
        
        if config.skip_existing and (protein_output_dir / "result.json").exists():
            logger.info(f"Skipping {system_id} (already exists)")
            with open(protein_output_dir / "result.json") as f:
                cached = json.load(f)
            results.append(BenchmarkResult(
                system_id=cached["system_id"],
                pdb1=cached["pdb1"],
                pdb2=cached["pdb2"],
                ground_truth_rmsd=cached["ground_truth_rmsd"],
                success=cached["success"],
                runtime_seconds=cached["runtime_seconds"],
                output_dir=Path(cached["output_dir"]),
                error_message=cached.get("error_message"),
                num_samples_generated=cached.get("num_samples_generated", 0),
            ))
            continue
        
        logger.info(f"[{idx+1}/{len(df)}] Processing {system_id} ({pdb1} -> {pdb2})")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would process {system_id}")
            continue
        
        fasta_path = fetch_sequence_from_pdb(pdb_id, chain_id, cache_dir)
        
        if fasta_path is None:
            result = BenchmarkResult(
                system_id=system_id,
                pdb1=pdb1,
                pdb2=pdb2,
                ground_truth_rmsd=ground_truth_rmsd,
                success=False,
                runtime_seconds=0.0,
                output_dir=protein_output_dir,
                error_message="Failed to fetch sequence from PDB",
            )
        else:
            success, runtime, error = run_conformix_boltz(
                fasta_path, protein_output_dir, config
            )
            
            num_samples = count_samples(protein_output_dir) if success else 0
            
            result = BenchmarkResult(
                system_id=system_id,
                pdb1=pdb1,
                pdb2=pdb2,
                ground_truth_rmsd=ground_truth_rmsd,
                success=success,
                runtime_seconds=runtime,
                output_dir=protein_output_dir,
                error_message=error if not success else None,
                num_samples_generated=num_samples,
            )
        
        with open(protein_output_dir / "result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"  Status: {'SUCCESS' if result.success else 'FAILED'} "
                   f"(Runtime: {result.runtime_seconds:.1f}s, "
                   f"Total: {success_count}/{len(results)} successful)")
    
    with open(config.output_dir / "all_results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    logger.info(f"Benchmark complete. Results saved to {config.output_dir}")
    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run ConforMix benchmarks on standard datasets"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to benchmark configuration YAML file"
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        help="Optional: specific protein IDs to run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    config = BenchmarkConfig.from_yaml(args.config)
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    results = run_benchmark(
        config=config,
        proteins=args.proteins,
        dry_run=args.dry_run,
    )
    
    if not args.dry_run:
        success_count = sum(1 for r in results if r.success)
        total_runtime = sum(r.runtime_seconds for r in results)
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Dataset:    {config.dataset_name}")
        print(f"Proteins:   {len(results)}")
        print(f"Successful: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
        print(f"Runtime:    {total_runtime:.1f}s ({total_runtime/60:.1f}min)")
        print(f"Output:     {config.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
