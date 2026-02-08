"""
evaluate_metrics.py

Compute evaluation metrics for ConforMix benchmark results.
Calculates RMSD, TM-score, and conformational coverage.

Usage:
    python -m benchmarks.evaluate_metrics --results benchmark_results/domainmotion/all_results.json
"""

import argparse
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass 
class MetricsResult:
    """Metrics for a single protein benchmark."""
    
    system_id: str
    pdb1: str
    pdb2: str
    ground_truth_rmsd: float
    
    min_rmsd_to_alt: Optional[float] = None
    max_rmsd_to_alt: Optional[float] = None
    mean_rmsd_to_alt: Optional[float] = None
    
    best_tm_score: Optional[float] = None
    mean_tm_score: Optional[float] = None
    
    conformational_coverage: Optional[float] = None
    rmsd_diversity: Optional[float] = None
    
    num_samples: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "system_id": self.system_id,
            "pdb1": self.pdb1,
            "pdb2": self.pdb2,
            "ground_truth_rmsd": self.ground_truth_rmsd,
            "min_rmsd_to_alt": self.min_rmsd_to_alt,
            "max_rmsd_to_alt": self.max_rmsd_to_alt,
            "mean_rmsd_to_alt": self.mean_rmsd_to_alt,
            "best_tm_score": self.best_tm_score,
            "mean_tm_score": self.mean_tm_score,
            "conformational_coverage": self.conformational_coverage,
            "rmsd_diversity": self.rmsd_diversity,
            "num_samples": self.num_samples,
            "error_message": self.error_message,
        }


def fetch_pdb_structure(pdb_id: str, chain_id: str, cache_dir: Path) -> Optional[Path]:
    """
    Fetch PDB structure from RCSB.
    
    Args:
        pdb_id: PDB ID (e.g., '1AKE')
        chain_id: Chain ID (e.g., 'A')
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Path to PDB file, or None if fetch failed
    """
    import urllib.request
    import urllib.error
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f"{pdb_id}.pdb"
    
    if pdb_path.exists():
        return pdb_path
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        logger.info(f"Fetching structure {pdb_id} from RCSB...")
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        
        with open(pdb_path, "wb") as f:
            f.write(content)
        
        return pdb_path
        
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch {pdb_id}: {e}")
        return None


def compute_rmsd_mdtraj(
    sample_path: Path,
    reference_path: Path,
    atom_selection: str = "backbone",
) -> Optional[float]:
    """
    Compute RMSD between sample and reference using MDTraj.
    
    Args:
        sample_path: Path to sample structure
        reference_path: Path to reference structure
        atom_selection: Atom selection for RMSD
        
    Returns:
        RMSD in Angstroms, or None if computation failed
    """
    try:
        import mdtraj as md
        
        sample = md.load(str(sample_path))
        reference = md.load(str(reference_path))
        
        if atom_selection == "backbone":
            sample_atoms = sample.topology.select("backbone")
            ref_atoms = reference.topology.select("backbone")
        elif atom_selection == "ca":
            sample_atoms = sample.topology.select("name CA")
            ref_atoms = reference.topology.select("name CA")
        else:
            sample_atoms = sample.topology.select("all")
            ref_atoms = reference.topology.select("all")
        
        min_atoms = min(len(sample_atoms), len(ref_atoms))
        if min_atoms == 0:
            return None
        
        sample_coords = sample.xyz[0, sample_atoms[:min_atoms]]
        ref_coords = reference.xyz[0, ref_atoms[:min_atoms]]
        
        sample_centered = sample_coords - sample_coords.mean(axis=0)
        ref_centered = ref_coords - ref_coords.mean(axis=0)
        
        correlation_matrix = np.dot(sample_centered.T, ref_centered)
        U, S, Vt = np.linalg.svd(correlation_matrix)
        rotation = np.dot(Vt.T, U.T)
        
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = np.dot(Vt.T, U.T)
        
        sample_aligned = np.dot(sample_centered, rotation)
        rmsd = np.sqrt(np.mean(np.sum((sample_aligned - ref_centered) ** 2, axis=1)))
        
        return float(rmsd * 10)
        
    except Exception as e:
        logger.warning(f"RMSD computation failed: {e}")
        return None


def compute_rmsd_simple(
    coords1: np.ndarray,
    coords2: np.ndarray,
) -> float:
    """
    Compute RMSD between two coordinate arrays after optimal alignment.
    
    Args:
        coords1: First set of coordinates (N, 3)
        coords2: Second set of coordinates (N, 3)
        
    Returns:
        RMSD in same units as input
    """
    centered1 = coords1 - coords1.mean(axis=0)
    centered2 = coords2 - coords2.mean(axis=0)
    
    correlation = np.dot(centered1.T, centered2)
    U, S, Vt = np.linalg.svd(correlation)
    rotation = np.dot(Vt.T, U.T)
    
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(Vt.T, U.T)
    
    aligned1 = np.dot(centered1, rotation)
    rmsd = np.sqrt(np.mean(np.sum((aligned1 - centered2) ** 2, axis=1)))
    
    return float(rmsd)


def extract_ca_coords_from_pdb(pdb_path: Path) -> Optional[np.ndarray]:
    """
    Extract CA coordinates from a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        Array of CA coordinates (N, 3), or None if failed
    """
    coords = []
    
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        
        if coords:
            return np.array(coords)
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract CA coords from {pdb_path}: {e}")
        return None


def extract_ca_coords_from_cif(cif_path: Path) -> Optional[np.ndarray]:
    """
    Extract CA coordinates from a CIF file.
    
    Args:
        cif_path: Path to CIF file
        
    Returns:
        Array of CA coordinates (N, 3), or None if failed
    """
    coords = []
    
    try:
        with open(cif_path) as f:
            in_atom_site = False
            columns = {}
            
            for line in f:
                if line.startswith("_atom_site."):
                    column_name = line.split(".")[1].strip()
                    columns[column_name] = len(columns)
                    in_atom_site = True
                elif in_atom_site and line.startswith("ATOM"):
                    parts = line.split()
                    if len(parts) > max(columns.values()):
                        atom_name_idx = columns.get("label_atom_id", columns.get("auth_atom_id", 3))
                        if parts[atom_name_idx] == "CA":
                            x_idx = columns.get("Cartn_x", 10)
                            y_idx = columns.get("Cartn_y", 11)
                            z_idx = columns.get("Cartn_z", 12)
                            x = float(parts[x_idx])
                            y = float(parts[y_idx])
                            z = float(parts[z_idx])
                            coords.append([x, y, z])
                elif in_atom_site and line.strip() == "#":
                    in_atom_site = False
        
        if coords:
            return np.array(coords)
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract CA coords from {cif_path}: {e}")
        return None


def find_sample_files(output_dir: Path) -> list[Path]:
    """Find all sample structure files in an output directory."""
    samples = []
    
    for pattern in ["*.cif", "*.pdb"]:
        samples.extend(output_dir.glob(f"**/{pattern}"))
    
    samples = [s for s in samples if "reference" not in s.name.lower()]
    
    return sorted(samples)


def compute_metrics(
    benchmark_result: dict,
    cache_dir: Path,
) -> MetricsResult:
    """
    Compute metrics for a single protein benchmark result.
    
    Args:
        benchmark_result: Dictionary with benchmark result data
        cache_dir: Directory for caching downloaded structures
        
    Returns:
        MetricsResult with computed metrics
    """
    system_id = benchmark_result["system_id"]
    pdb1 = benchmark_result["pdb1"]
    pdb2 = benchmark_result["pdb2"]
    ground_truth_rmsd = benchmark_result["ground_truth_rmsd"]
    output_dir = Path(benchmark_result["output_dir"])
    
    if not benchmark_result["success"]:
        return MetricsResult(
            system_id=system_id,
            pdb1=pdb1,
            pdb2=pdb2,
            ground_truth_rmsd=ground_truth_rmsd,
            error_message=benchmark_result.get("error_message", "Benchmark failed"),
        )
    
    pdb2_id, chain2_id = pdb2.split("_") if "_" in pdb2 else (pdb2, "A")
    alt_pdb_path = fetch_pdb_structure(pdb2_id, chain2_id, cache_dir)
    
    if alt_pdb_path is None:
        return MetricsResult(
            system_id=system_id,
            pdb1=pdb1,
            pdb2=pdb2,
            ground_truth_rmsd=ground_truth_rmsd,
            error_message=f"Failed to fetch alternate structure {pdb2}",
        )
    
    alt_coords = extract_ca_coords_from_pdb(alt_pdb_path)
    if alt_coords is None:
        return MetricsResult(
            system_id=system_id,
            pdb1=pdb1,
            pdb2=pdb2,
            ground_truth_rmsd=ground_truth_rmsd,
            error_message=f"Failed to extract coords from {pdb2}",
        )
    
    sample_files = find_sample_files(output_dir)
    if not sample_files:
        return MetricsResult(
            system_id=system_id,
            pdb1=pdb1,
            pdb2=pdb2,
            ground_truth_rmsd=ground_truth_rmsd,
            num_samples=0,
            error_message="No sample files found",
        )
    
    rmsds_to_alt = []
    sample_coords_list = []
    
    for sample_path in sample_files:
        if sample_path.suffix == ".cif":
            sample_coords = extract_ca_coords_from_cif(sample_path)
        else:
            sample_coords = extract_ca_coords_from_pdb(sample_path)
        
        if sample_coords is None:
            continue
        
        sample_coords_list.append(sample_coords)
        
        min_len = min(len(sample_coords), len(alt_coords))
        if min_len > 0:
            rmsd = compute_rmsd_simple(
                sample_coords[:min_len],
                alt_coords[:min_len]
            )
            rmsds_to_alt.append(rmsd)
    
    pairwise_rmsds = []
    for i in range(len(sample_coords_list)):
        for j in range(i + 1, len(sample_coords_list)):
            coords_i = sample_coords_list[i]
            coords_j = sample_coords_list[j]
            min_len = min(len(coords_i), len(coords_j))
            if min_len > 0:
                rmsd = compute_rmsd_simple(coords_i[:min_len], coords_j[:min_len])
                pairwise_rmsds.append(rmsd)
    
    rmsd_diversity = np.mean(pairwise_rmsds) if pairwise_rmsds else None
    
    if rmsds_to_alt:
        conformational_coverage = min(rmsds_to_alt) / ground_truth_rmsd if ground_truth_rmsd > 0 else None
        conformational_coverage = min(1.0, 1.0 - abs(1.0 - conformational_coverage)) if conformational_coverage else None
    else:
        conformational_coverage = None
    
    return MetricsResult(
        system_id=system_id,
        pdb1=pdb1,
        pdb2=pdb2,
        ground_truth_rmsd=ground_truth_rmsd,
        min_rmsd_to_alt=min(rmsds_to_alt) if rmsds_to_alt else None,
        max_rmsd_to_alt=max(rmsds_to_alt) if rmsds_to_alt else None,
        mean_rmsd_to_alt=np.mean(rmsds_to_alt) if rmsds_to_alt else None,
        conformational_coverage=conformational_coverage,
        rmsd_diversity=rmsd_diversity,
        num_samples=len(sample_files),
    )


def compute_all_metrics(
    results_path: Path,
    output_path: Optional[Path] = None,
) -> list[MetricsResult]:
    """
    Compute metrics for all benchmark results.
    
    Args:
        results_path: Path to all_results.json
        output_path: Optional path to save metrics JSON
        
    Returns:
        List of MetricsResult objects
    """
    logger.info(f"Loading results from {results_path}")
    
    with open(results_path) as f:
        results = json.load(f)
    
    cache_dir = results_path.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    
    metrics = []
    
    for i, result in enumerate(results):
        logger.info(f"[{i+1}/{len(results)}] Computing metrics for {result['system_id']}")
        m = compute_metrics(result, cache_dir)
        metrics.append(m)
        
        if m.error_message:
            logger.warning(f"  Error: {m.error_message}")
        else:
            logger.info(f"  RMSD to alt: {m.min_rmsd_to_alt:.2f}Å (min), "
                       f"{m.mean_rmsd_to_alt:.2f}Å (mean)")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)
        logger.info(f"Saved metrics to {output_path}")
    
    return metrics


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Compute metrics for ConforMix benchmark results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to all_results.json from benchmark run"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save metrics JSON (default: metrics.json in same directory)"
    )
    
    args = parser.parse_args()
    
    output_path = args.output or args.results.parent / "metrics.json"
    
    metrics = compute_all_metrics(args.results, output_path)
    
    successful = [m for m in metrics if m.error_message is None]
    
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Total proteins:     {len(metrics)}")
    print(f"Successful:         {len(successful)}")
    
    if successful:
        rmsds = [m.min_rmsd_to_alt for m in successful if m.min_rmsd_to_alt is not None]
        if rmsds:
            print(f"Min RMSD to alt:    {np.mean(rmsds):.2f}Å ± {np.std(rmsds):.2f}Å")
        
        coverages = [m.conformational_coverage for m in successful if m.conformational_coverage is not None]
        if coverages:
            print(f"Coverage:           {np.mean(coverages)*100:.1f}% ± {np.std(coverages)*100:.1f}%")
    
    print(f"Output:             {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
