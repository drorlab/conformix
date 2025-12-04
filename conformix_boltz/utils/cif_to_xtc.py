import os
import logging
import json
import mdtraj
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import argparse
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_cif_files(parent_dir: Union[str, Path]) -> List[Path]:
    """Find all CIF files within subdirectories of the specified parent directory."""
    parent_dir = Path(parent_dir)
    cif_files = []
    
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith('.cif'):
                cif_files.append(Path(root) / file)
    
    logger.info(f"Found {len(cif_files)} CIF files in {parent_dir} and its subdirectories")
    return cif_files


def find_plddt_files(cif_files: List[Path]) -> Dict[Path, Path]:
    """Find associated pLDDT NPZ files for each CIF file."""
    plddt_files = {}
    
    for cif_path in cif_files:
        cif_dir = cif_path.parent
        cif_name = cif_path.stem
        
        # Extract model number if present
        model_num = None
        protein_name = cif_name
        if "_model_" in cif_name:
            parts = cif_name.split("_model_")
            protein_name = parts[0]
            model_num = parts[1]
        
        # Look for corresponding pLDDT file
        plddt_pattern = f"plddt_{protein_name}_model_{model_num}.npz" if model_num else f"plddt_{cif_name}.npz"
        potential_file = cif_dir / plddt_pattern
        
        if potential_file.exists():
            plddt_files[cif_path] = potential_file
    
    logger.info(f"Found {len(plddt_files)} pLDDT files for {len(cif_files)} CIF files")
    return plddt_files


def load_plddt_data(plddt_path: Path) -> Optional[np.ndarray]:
    """Load pLDDT data from an NPZ file."""
    try:
        data = np.load(plddt_path)
        key = list(data.keys())[0]
        return data[key]
    except Exception as e:
        logger.warning(f"Error loading pLDDT file {plddt_path}: {e}")
        return None


def extract_beta_value(cif_path: Path) -> Optional[float]:
    """Extract beta (twisting target) value from CIF file path."""
    path_str = str(cif_path)
    
    # Look for beta in directory structure
    for part in path_str.split(os.sep):
        if "beta_" in part:
            import re
            beta_match = re.search(r'beta_(-?\d+\.?\d*)', part)
            if beta_match:
                return float(beta_match.group(1))
    
    # Try to check confidence file
    cif_dir = cif_path.parent
    cif_name = cif_path.stem
    
    confidence_patterns = [
        f"confidence_semisweet_model_{cif_name.split('_model_')[-1]}.json" if "_model_" in cif_name else None,
        f"confidence_{cif_name}.json"
    ]
    
    for pattern in confidence_patterns:
        if not pattern:
            continue
            
        potential_file = cif_dir / pattern
        if potential_file.exists():
            try:
                with open(potential_file, 'r') as f:
                    data = json.load(f)
                    if 'beta' in data:
                        return float(data['beta'])
            except:
                pass
    
    return None


def filter_by_plddt_quality(
    cif_files: List[Path], 
    plddt_files: Dict[Path, Path], 
    window_size: int = 10, 
    decrease_threshold: float = 0.05
) -> List[Path]:
    """Filter CIF files based on pLDDT quality comparison to untwisted (beta=0) references."""
    # Collect beta values for all files
    beta_values = {}
    for cif_path in cif_files:
        beta = extract_beta_value(cif_path)
        if beta is not None:
            beta_values[cif_path] = beta
    
    # Skip if not enough beta values found
    if len(beta_values) < len(cif_files) / 2:
        logger.warning(f"Beta values found for only {len(beta_values)} out of {len(cif_files)} CIF files")
        return cif_files
    
    # Identify untwisted reference structures (beta=0)
    untwisted_cifs = [path for path, beta in beta_values.items() if beta == 0]
    
    if not untwisted_cifs:
        logger.warning("No untwisted (beta=0) structures found, skipping pLDDT quality filtering")
        return cif_files
    
    # Load pLDDT data for untwisted references
    reference_plddts = []
    for cif_path in untwisted_cifs:
        if cif_path in plddt_files:
            plddt_data = load_plddt_data(plddt_files[cif_path])
            if plddt_data is not None:
                reference_plddts.append(plddt_data)
    
    if not reference_plddts:
        logger.warning("No pLDDT data found for reference structures, skipping pLDDT quality filtering")
        return cif_files
    
    # Check if all reference arrays have the same length
    ref_lengths = set(len(arr) for arr in reference_plddts)
    if len(ref_lengths) > 1:
        logger.warning(f"Reference pLDDT arrays have inconsistent lengths: {ref_lengths}")
        most_common_length = max(ref_lengths, key=lambda l: sum(1 for arr in reference_plddts if len(arr) == l))
        logger.info(f"Using most common length: {most_common_length}")
        reference_plddts = [arr for arr in reference_plddts if len(arr) == most_common_length]
    
    # Calculate minimum reference pLDDT
    min_reference_plddt = np.min(reference_plddts, axis=0)
    
    # Filter twisted structures based on pLDDT quality
    filtered_cifs = []
    rejected_count = 0
    
    for cif_path in cif_files:
        # Skip untwisted references
        if cif_path in untwisted_cifs:
            filtered_cifs.append(cif_path)
            continue
        
        # Skip if no beta value or pLDDT data
        if cif_path not in beta_values or cif_path not in plddt_files:
            filtered_cifs.append(cif_path)
            continue
        
        # Load pLDDT data
        plddt_data = load_plddt_data(plddt_files[cif_path])
        if plddt_data is None or len(plddt_data) != len(min_reference_plddt):
            filtered_cifs.append(cif_path)
            continue
        
        # Check for significant pLDDT decreases using sliding window
        has_quality_issue = False
        for i in range(len(plddt_data) - window_size + 1):
            ref_window_mean = np.mean(min_reference_plddt[i:i+window_size])
            struct_window_mean = np.mean(plddt_data[i:i+window_size])
            
            decrease_amount = ref_window_mean - struct_window_mean
            if decrease_amount > decrease_threshold:
                has_quality_issue = True
                break
        
        # Keep structure if no quality issues
        if not has_quality_issue:
            filtered_cifs.append(cif_path)
        else:
            rejected_count += 1
    
    logger.info(f"pLDDT quality filtering: rejected {rejected_count} structures")
    return filtered_cifs


def load_cif_structures(cif_files: List[Path]) -> List[mdtraj.Trajectory]:
    """Load all CIF files into MDTraj structures."""
    structures = []
    for cif_path in tqdm(cif_files, desc="Loading structures"):
        try:
            structure = mdtraj.load(str(cif_path))
            structures.append(structure)
        except Exception as e:
            logger.error(f"Error loading {cif_path}: {e}")
    
    logger.info(f"Successfully loaded {len(structures)} out of {len(cif_files)} CIF files")
    return structures


def combine_structures(structures: List[mdtraj.Trajectory]) -> Optional[mdtraj.Trajectory]:
    """Combine all structures into a single multi-frame trajectory."""
    if not structures:
        logger.error("No structures to combine")
        return None
    
    template = structures[0]
    combined_xyz = []
    
    for structure in structures:
        if structure.n_atoms != template.n_atoms:
            logger.warning(f"Structure with {structure.n_atoms} atoms doesn't match template with {template.n_atoms} atoms. Skipping.")
            continue
        combined_xyz.append(structure.xyz[0])
    
    if not combined_xyz:
        logger.error("No valid structures to combine")
        return None
    
    combined_traj = mdtraj.Trajectory(
        xyz=np.array(combined_xyz),
        topology=template.topology
    )
    
    logger.info(f"Combined {len(combined_xyz)} structures into a single trajectory")
    return combined_traj


def filter_unphysical_traj(
    traj: mdtraj.Trajectory,
    max_ca_seq_distance: float = 4.5,
    max_cn_seq_distance: float = 2.0,
    clash_distance: float = 0.5,
    strict: bool = False,
) -> mdtraj.Trajectory:
    """Filter out 'unphysical' frames from a samples trajectory with proper chain handling."""
    print(f"\nStarting filtering with {traj.n_frames} total frames")
    
    # Group residues by chain
    chain_residues = {}
    for residue in traj.topology.residues:
        chain_id = residue.chain.index
        if chain_id not in chain_residues:
            chain_residues[chain_id] = []
        chain_residues[chain_id].append(residue)
    
    # Create pairs only for sequential residues within the same chain
    seq_contiguous_resid_pairs = []
    for chain_id, residues in chain_residues.items():
        sorted_residues = sorted(residues, key=lambda r: r.resSeq)
        for i in range(len(sorted_residues) - 1):
            if sorted_residues[i+1].resSeq - sorted_residues[i].resSeq == 1:
                seq_contiguous_resid_pairs.append((sorted_residues[i].index, sorted_residues[i+1].index))
    
    # CA-CA distance check
    if not seq_contiguous_resid_pairs:
        logger.warning("No sequential residue pairs found within chains. CA-CA distance check will be skipped.")
        frames_match_ca_seq_distance = np.ones(traj.n_frames, dtype=bool)
    else:
        seq_contiguous_resid_pairs = np.array(seq_contiguous_resid_pairs)
        ca_seq_distances, _ = mdtraj.compute_contacts(
            traj, scheme="ca", contacts=seq_contiguous_resid_pairs, periodic=False
        )
        ca_seq_distances = mdtraj.utils.in_units_of(ca_seq_distances, "nanometers", "angstrom")
        frames_match_ca_seq_distance = np.all(ca_seq_distances < max_ca_seq_distance, axis=1)
    
    n_pass_ca = np.sum(frames_match_ca_seq_distance)
    n_fail_ca = traj.n_frames - n_pass_ca
    print(f"CA-CA distance check: {n_pass_ca} pass, {n_fail_ca} fail ({n_fail_ca/traj.n_frames*100:.1f}% filtered)")

    # C-N distance check
    cn_atom_pair_indices = []
    for resid_i, resid_j in seq_contiguous_resid_pairs:
        residue_i = traj.topology.residue(resid_i)
        residue_j = traj.topology.residue(resid_j)
        
        try:
            c_i = list(residue_i.atoms_by_name("C"))
            n_j = list(residue_j.atoms_by_name("N"))
        except:
            c_i = [atom for atom in residue_i.atoms if atom.name == "C"]
            n_j = [atom for atom in residue_j.atoms if atom.name == "N"]
            
        if c_i and n_j:
            cn_atom_pair_indices.append((c_i[0].index, n_j[0].index))

    frames_match_cn_seq_distance = np.ones(traj.n_frames, dtype=bool)
    if cn_atom_pair_indices:
        cn_seq_distances = mdtraj.compute_distances(traj, cn_atom_pair_indices, periodic=False)
        cn_seq_distances = mdtraj.utils.in_units_of(cn_seq_distances, "nanometers", "angstrom")
        frames_match_cn_seq_distance = np.all(cn_seq_distances < max_cn_seq_distance, axis=1)
    
    n_pass_cn = np.sum(frames_match_cn_seq_distance)
    n_fail_cn = traj.n_frames - n_pass_cn
    print(f"C-N distance check: {n_pass_cn} pass, {n_fail_cn} fail ({n_fail_cn/traj.n_frames*100:.1f}% filtered)")

    # Clash check
    rest_distances, _ = mdtraj.compute_contacts(traj, periodic=False)
    frames_non_clash = np.all(
        mdtraj.utils.in_units_of(rest_distances, "nanometers", "angstrom") > clash_distance,
        axis=1,
    )
    
    n_pass_clash = np.sum(frames_non_clash)
    n_fail_clash = traj.n_frames - n_pass_clash
    print(f"Clash check: {n_pass_clash} pass, {n_fail_clash} fail ({n_fail_clash/traj.n_frames*100:.1f}% filtered)")
    
    # Combine all filters
    matches_all = frames_match_ca_seq_distance & frames_match_cn_seq_distance & frames_non_clash
    n_pass_all = np.sum(matches_all)
    n_fail_all = traj.n_frames - n_pass_all
    
    print(f"\nCombined filters: {n_pass_all} pass, {n_fail_all} fail ({n_fail_all/traj.n_frames*100:.1f}% filtered)")
    
    if strict and not np.any(matches_all):
        raise ValueError("All frames were filtered out as unphysical")
    
    filtered_traj = traj.slice(np.where(matches_all)[0], copy=True)
    logger.info(f"Filtered {traj.n_frames} frames down to {filtered_traj.n_frames}")
    return filtered_traj


def save_trajectory_in_chunks(
    structures: List[mdtraj.Trajectory],
    output_pdb: Path,
    output_xtc: Path,
    chunk_size: int = 100,
    filter_physicality: bool = True,
    superpose: bool = True
) -> int:
    """Save structures to XTC file in chunks using streaming write to avoid memory issues."""
    from mdtraj.formats import XTCTrajectoryFile
    
    if not structures:
        logger.error("No structures to save")
        return 0
    
    # Save topology from first structure
    structures[0][0].save_pdb(str(output_pdb))
    logger.info(f"Saved topology to {output_pdb}")
    
    # Get reference for superposition
    reference = structures[0][0] if superpose else None
    
    # Get alignment indices for superposition
    align_indices = None
    if superpose:
        align_indices = [atom.index for atom in structures[0].topology.atoms if atom.name == 'CA']
        if not align_indices:
            logger.warning("No CA atoms found for superposition. Using all atoms.")
            align_indices = None
    
    total_frames_saved = 0
    
    # Open XTC file for writing
    with XTCTrajectoryFile(str(output_xtc), 'w') as xtc_file:
        # Process in chunks
        for chunk_start in tqdm(range(0, len(structures), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(structures))
            chunk_structures = structures[chunk_start:chunk_end]
            
            # Combine chunk into single trajectory
            chunk_traj = combine_structures(chunk_structures)
            if chunk_traj is None:
                logger.warning(f"Failed to combine chunk {chunk_start}-{chunk_end}")
                continue
            
            # Filter if requested
            if filter_physicality and chunk_traj.n_frames > 0:
                try:
                    num_frames_before = chunk_traj.n_frames
                    chunk_traj = filter_unphysical_traj(chunk_traj, strict=False)
                    logger.info(f"Chunk {chunk_start}-{chunk_end}: Filtered {num_frames_before} frames down to {chunk_traj.n_frames}")
                    
                    if chunk_traj.n_frames == 0:
                        logger.warning(f"All frames in chunk {chunk_start}-{chunk_end} were filtered out")
                        del chunk_traj
                        del chunk_structures
                        continue
                except Exception as e:
                    logger.warning(f"Filtering failed for chunk {chunk_start}-{chunk_end}: {e}")
            
            # Superpose if requested
            if superpose and chunk_traj.n_frames > 0 and reference is not None: 
                try:
                    if align_indices:
                        chunk_traj.superpose(reference=reference, frame=0, atom_indices=align_indices)
                    else:
                        chunk_traj.superpose(reference=reference, frame=0)
                except Exception as e:
                    logger.warning(f"Superposition failed for chunk {chunk_start}-{chunk_end}: {e}")
            
            # Write frames directly to XTC file
            if chunk_traj.n_frames > 0:
                for frame_idx in range(chunk_traj.n_frames):
                    xtc_file.write(
                        xyz=chunk_traj.xyz[frame_idx],
                        time=total_frames_saved + frame_idx,
                        step=total_frames_saved + frame_idx,
                        box=chunk_traj.unitcell_lengths[frame_idx] if chunk_traj.unitcell_lengths is not None else None
                    )
                
                total_frames_saved += chunk_traj.n_frames
                logger.info(f"Wrote chunk {chunk_start}-{chunk_end}: {chunk_traj.n_frames} frames (total: {total_frames_saved})")
            
            # Clean up to free memory
            del chunk_traj
            del chunk_structures
    
    logger.info(f"Total frames saved: {total_frames_saved}")
    return total_frames_saved


def process_all_cifs_to_single_output(
    parent_dir: Union[str, Path],
    output_dir: Union[str, Path],
    filter_physicality: bool = True,
    filter_region_plddt: bool = False,
    plddt_window_size: int = 10,
    plddt_decrease_threshold: float = 0.05,
    chunk_size: int = 100
) -> bool:
    """Process all CIF files in subdirectories and create a single PDB+XTC output."""
    try:
        # Set output paths
        output_dir = Path(output_dir)
        output_pdb = output_dir / "topology.pdb"
        output_xtc = output_dir / "samples.xtc"
        
        # Find all CIF files
        cif_files = find_cif_files(parent_dir)
        if not cif_files:
            logger.error("No CIF files found")
            return False
        
        # Apply pLDDT quality filtering if requested
        if filter_region_plddt:
            logger.info(f"Applying pLDDT quality filtering (window: {plddt_window_size}, threshold: {plddt_decrease_threshold*100}%)")
            
            plddt_files = find_plddt_files(cif_files)
            
            if plddt_files:
                cif_files = filter_by_plddt_quality(
                    cif_files, 
                    plddt_files, 
                    window_size=plddt_window_size,
                    decrease_threshold=plddt_decrease_threshold
                )
                
                if not cif_files:
                    logger.error("No CIF files passed pLDDT quality filtering")
                    return False
            else:
                logger.warning("No pLDDT files found, skipping pLDDT quality filtering")
        
        # Load all structures
        structures = load_cif_structures(cif_files)
        if not structures:
            logger.error("No structures loaded")
            return False
        
        # Ensure output directory exists
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save in chunks
        total_frames = save_trajectory_in_chunks(
            structures=structures,
            output_pdb=output_pdb,
            output_xtc=output_xtc,
            chunk_size=chunk_size,
            filter_physicality=filter_physicality,
            superpose=True
        )
        
        if total_frames == 0:
            logger.error("No frames were saved")
            return False
        
        logger.info(f"Successfully saved {output_pdb} and {output_xtc}")
        logger.info(f"Combined trajectory has {total_frames} frames")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing CIF files: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CIF files to a single PDB+XTC output")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Parent directory containing subdirectories with CIF files")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="Directory to save output files")
    parser.add_argument("--filter_physicality", action="store_true",
                        help="Enable filtering of unphysical samples (CA-CA distance, C-N distance, clashes)")
    parser.add_argument("--filter_region_plddt", action="store_true",
                       help="Filter structures with decreased regional pLDDT compared to untwisted (beta=0) references")
    parser.add_argument("--plddt_window_size", type=int, default=10,
                       help="Size of sliding window for regional pLDDT analysis (default: 10)")
    parser.add_argument("--plddt_decrease_threshold", type=float, default=0.2,
                       help="Threshold for flagging regional pLDDT decrease (default: 0.2)")
    parser.add_argument("--chunk_size", type=int, default=100,
                       help="Number of structures to process at once (default: 100)")
    
    args = parser.parse_args()
    
    process_all_cifs_to_single_output(
        parent_dir=args.input_dir,
        output_dir=args.out_dir,
        filter_physicality=args.filter_physicality,
        filter_region_plddt=args.filter_region_plddt,
        plddt_window_size=args.plddt_window_size,
        plddt_decrease_threshold=args.plddt_decrease_threshold,
        chunk_size=args.chunk_size
    )