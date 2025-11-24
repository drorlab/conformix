import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
import re

import click
import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from boltz.model.sampling.fn_utils import PCA

from boltz.model.loss.diffusion import weighted_rigid_align
from Bio.PDB import MMCIFParser, PDBParser, Polypeptide
from typing import Union, List, Optional, Literal
import pymol
from pymol import cmd
import numpy as np
import glob

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1. ## changed to 1 for ease of noising injections, originally 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = False
 

@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            elif constraint_z and twist_rmsd:
                raise ValueError("Both constraint_z and twist_rmsd cannot be True at the same time.")
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Parse input data
    records: list[Record] = []
    for path in tqdm(data):
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            target = parse_yaml(path, ccd)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Keep record
        records.append(target.record)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")

def get_secondary_structure_masks(structure_file, twist_residue1=None, twist_residue2=None, exclude_residues=None):
    """
    Create masks where only beta sheets and helices are marked as 1.
    Uses PyMOL's secondary structure assignments.
    
    Parameters:
    structure_file: Path to CIF/PDB file from untwisted output
    twist_residue1: Start residue number of twist region (optional)
    twist_residue2: End residue number of twist region (optional)
    exclude_residues: Residues to exclude from ss_atom_mask_region (optional)
                     Can be provided as:
                     - String with comma-separated values: "100,50-55,105-110"
                     - List of values: [100, "50-55", [105, 110]]
                     - Single integer: 100
    
    Returns:
    atom_coords: Tensor of atom coordinates
    ss_atom_mask: Tensor of atom mask (1 for ss, 0 otherwise)
    ss_atom_mask_region: Tensor of atom mask (1 for ss within twist region and not excluded, 0 otherwise)
                         If twist_residues=None, this is the same as ss_atom_mask but with excluded residues set to 0.
    """
    # Determine file type and parse accordingly
    if structure_file.endswith(".cif"):
        parser = MMCIFParser()
    elif structure_file.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Unsupported file format. Please provide a .cif or .pdb file.")
    structure = parser.get_structure("structure", structure_file)
    
    # Build atom info list
    atom_coords = []
    atom_info = []  # (model_id, chain_id, res_id, res_name, atom_name, atom_serial)
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':  # Skip hetero-atoms and waters
                    continue
                res_id = res.id[1]
                for atom in res:
                    atom_coords.append(atom.coord)
                    atom_info.append((model.id, chain.id, res_id, res.resname, atom.name, atom.serial_number))

    # Initialize PyMOL in quiet mode if not already running
    if not hasattr(pymol, 'pymol_launched'):
        pymol.finish_launching(['pymol', '-qc'])
        pymol.pymol_launched = True
    
    # Load the structure if file provided
    cmd.load(structure_file, 'protein_for_ss')
    target = 'protein_for_ss'
    
    # Make sure secondary structure is calculated
    cmd.dss(target)
    
    # Get all atoms and residues
    all_atoms = cmd.get_model(f'{target} and all')
    
    # Create selections for atoms in sheets or helices (ss s or ss h in PyMOL)
    cmd.select('ss_atoms', f'{target} and (ss h or ss s)')
    ss_atoms = cmd.get_model('ss_atoms')
    
    # Create atom mask
    atom_mask = []
    # Create a set for faster lookups
    ss_atom_ids = {(a.chain, a.resi, a.name) for a in ss_atoms.atom}
    
    # Prepare for region mask
    region_mask = []
    # Process region parameters
    region_defined = twist_residue1 is not None and twist_residue2 is not None
    if region_defined:
        print(f"Twisting on RMSD of region between residues {twist_residue1} and {twist_residue2}")
        min_res = min(twist_residue1, twist_residue2)
        max_res = max(twist_residue1, twist_residue2)
    else:
        print("Twisting on RMSD of all residues")
    
    # Process exclude_residues to create a set of all excluded residue numbers
    exclude_set = set()
    if exclude_residues is not None:
        # Handle different input formats for exclude_residues
        if isinstance(exclude_residues, str):
            # String format: "100,50-55,105-110"
            exclude_items = [item.strip() for item in exclude_residues.split(',')]
        elif isinstance(exclude_residues, list):
            # List format: [100, "50-55", [105, 110]]
            exclude_items = exclude_residues
        else:
            # Single value: 100
            exclude_items = [exclude_residues]
            
        for item in exclude_items:
            if isinstance(item, int):
                # Single residue number
                exclude_set.add(item)
            elif isinstance(item, str):
                if "-" in item:
                    # Range specified as "start-end"
                    parts = item.split("-")
                    if len(parts) == 2:
                        try:
                            start, end = int(parts[0]), int(parts[1])
                            exclude_set.update(range(start, end + 1))  # +1 to include end
                        except ValueError:
                            print(f"Warning: Could not parse range '{item}', skipping")
                else:
                    # Try to parse as a single integer
                    try:
                        exclude_set.add(int(item))
                    except ValueError:
                        print(f"Warning: Could not parse '{item}' as an integer, skipping")
            elif isinstance(item, list) and len(item) == 2:
                # Range specified as [start, end]
                try:
                    start, end = int(item[0]), int(item[1])
                    exclude_set.update(range(start, end + 1))  # +1 to include end
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse range {item}, skipping")
            else:
                print(f"Warning: Unrecognized format in exclude_residues: {item}, skipping")
    
    if exclude_set:
        print(f"Excluding residues: {sorted(exclude_set)}")
    
    for atom in all_atoms.atom:
        # Check if in secondary structure
        in_ss = (atom.chain, atom.resi, atom.name) in ss_atom_ids
        atom_mask.append(1 if in_ss else 0)

        # Get residue number
        try:
            res_num = int(atom.resi)
            
            # Determine if this atom should be included in region mask
            if region_defined:
                # When region is defined, only include if within region AND not excluded
                in_region = min_res <= res_num <= max_res and res_num not in exclude_set
            else:
                # When no region defined, include everything except excluded residues
                in_region = res_num not in exclude_set
        except ValueError:
            # Handle case where resi might contain insertion codes or non-integer values
            raise ValueError(f"Error: Non-integer residue identifier '{atom.resi}' encountered.")
            
        region_mask.append(1 if in_region else 0)
    
    # Clean up
    if structure_file:
        cmd.delete('protein_for_ss')
    cmd.delete('ss_atoms')
    
    atom_coords = torch.tensor(np.array(atom_coords))
    ss_atom_mask = torch.tensor(atom_mask, dtype=int)
    region_mask = torch.tensor(region_mask, dtype=int)
    
    # Apply both masks: secondary structure AND region (with exclusions)
    ss_atom_mask_region = ss_atom_mask * region_mask

    return atom_coords, ss_atom_mask, ss_atom_mask_region

def get_region_index_ranges(cif_file, chain_id1, res_id1, chain_id2, res_id2, flank_size):
    """
    Get atom index ranges for two regions in a protein structure and a descriptive string.
    
    Args:
        cif_file: Path to the CIF file
        chain_id1: Chain identifier for first region
        res_id1: ID of the first central residue
        chain_id2: Chain identifier for second region
        res_id2: ID of the second central residue
        flank_size: Number of residues to include on each side
        
    Returns:
        region1_range: Tuple of (min, max) atom indices for region 1
        region2_range: Tuple of (min, max) atom indices for region 2
        desc_string: String in format "Nid_1-Nid_2" where N are residue types (e.g., "ALA12-GLY34")
    """
    # Parse CIF file
    parser = MMCIFParser()
    structure = parser.get_structure("structure", cif_file)
    
    # Build atom info list
    atom_coords = []
    atom_info = []  # (model_id, chain_id, res_id, res_name, atom_name, atom_serial)
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':  # Skip hetero-atoms and waters
                    continue
                res_id = res.id[1]
                for atom in res:
                    atom_coords.append(atom.coord)
                    atom_info.append((model.id, chain.id, res_id, res.resname, atom.name, atom.serial_number))
    
    # Get residue IDs for each chain
    chain1_res_ids = sorted(list(set([res_id for _, c_id, res_id, _, _, _ in atom_info if c_id == chain_id1])))
    chain2_res_ids = sorted(list(set([res_id for _, c_id, res_id, _, _, _ in atom_info if c_id == chain_id2])))
    
    # Find central residue indices
    try:
        central_idx1 = chain1_res_ids.index(res_id1)
        central_idx2 = chain2_res_ids.index(res_id2)
    except ValueError:
        # Return empty ranges if residues not found
        return (0, 0), (0, 0)
    
    # Calculate region residue ranges
    start_idx1 = max(0, central_idx1 - flank_size)
    end_idx1 = min(len(chain1_res_ids) - 1, central_idx1 + flank_size)
    start_idx2 = max(0, central_idx2 - flank_size)
    end_idx2 = min(len(chain2_res_ids) - 1, central_idx2 + flank_size)
    
    region1_res_ids = set(chain1_res_ids[start_idx1:end_idx1 + 1])
    region2_res_ids = set(chain2_res_ids[start_idx2:end_idx2 + 1])
    
    # Get atom indices for regions
    region1_indices = [i for i, (_, c_id, res_id, _, _, _) in enumerate(atom_info) 
                       if c_id == chain_id1 and res_id in region1_res_ids]
    region2_indices = [i for i, (_, c_id, res_id, _, _, _) in enumerate(atom_info) 
                       if c_id == chain_id2 and res_id in region2_res_ids]
    
    # Get index ranges
    region1_range = (min(region1_indices) if region1_indices else 0, 
                    max(region1_indices) if region1_indices else 0)
    region2_range = (min(region2_indices) if region2_indices else 0, 
                    max(region2_indices) if region2_indices else 0)
    
    # Get residue types for description string
    res1_type = None
    res2_type = None
    
    # Find residue types by looking through the structure again
    for model in structure:
        for chain in model:
            if chain.id == chain_id1:
                for res in chain:
                    if res.id[1] == res_id1 and res.id[0] == ' ':
                        res1_type = res.resname
            if chain.id == chain_id2:
                for res in chain:
                    if res.id[1] == res_id2 and res.id[0] == ' ':
                        res2_type = res.resname
    
    # Function to convert three-letter to one-letter amino acid codes
    def get_one_letter(three_letter_code):
        try:
            # Convert three-letter to index
            index = Polypeptide.three_to_index(three_letter_code)
            # Convert index to one-letter
            return Polypeptide.index_to_one(index)
        except (ValueError, KeyError):
            # If conversion fails, return "X"
            return "X"
    
    # Convert three-letter codes to one-letter codes
    res1_one_letter = get_one_letter(res1_type) if res1_type else "X"
    res2_one_letter = get_one_letter(res2_type) if res2_type else "X"
    
    # Create description string with one-letter codes
    desc_string = f"{res1_one_letter}{res_id1}-{res2_one_letter}{res_id2}"
    
    return region1_range, region2_range, desc_string

def get_residue_atoms_mask(structure_file, residue_ranges, chain_id="A"):
    """
    Get atom coordinates and a mask indicating whether atoms belong to specified residue ranges.
    
    Args:
        structure_file: Path to the CIF/PDB file
        residue_ranges: List of [start, end] ranges for residues of interest
                        (e.g., [[1,20], [40,42]] selects residues 1-20 and 40-42)
        chain_id: Chain identifier (default is "A")
        
    Returns:
        atom_coords: Numpy array of atom coordinates
        atom_mask: Numpy array mask (1 for atoms in specified residues, 0 otherwise)
    """
    import numpy as np
    
    # Determine file type and parse accordingly
    if structure_file.endswith(".cif"):
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser()
    elif structure_file.endswith(".pdb"):
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format. Please provide a .cif or .pdb file.")
    
    structure = parser.get_structure("structure", structure_file)
    
    # Create a set of all residue IDs in our ranges for faster lookups
    target_residues = set()
    for start, end in residue_ranges:
        target_residues.update(range(start, end + 1))  # +1 to include the end residue
    
    # Build atom info list and coordinates
    atom_coords = []
    atom_mask = []  # 1 for atoms in specified residues, 0 otherwise
    
    for model in structure:
        for chain in model:
            # Skip chains that don't match our target chain_id
            if chain.id != chain_id:
                continue
                
            for res in chain:
                if res.id[0] != ' ':  # Skip hetero-atoms and waters
                    continue
                
                res_id = res.id[1]
                # Check if this residue is in our target ranges
                is_target_residue = res_id in target_residues
                
                for atom in res:
                    atom_coords.append(atom.coord)
                    # Mark this atom in the mask if it belongs to a residue in our ranges
                    atom_mask.append(1 if is_target_residue else 0)
    
    # Convert to numpy arrays
    atom_coords = torch.tensor(np.array(atom_coords))
    atom_mask = torch.tensor(atom_mask, dtype=int)
    
    return atom_coords, atom_mask

@click.group()
def cli() -> None:
    """Boltz1."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--input_cif", type=click.Path(exists=True),
    help="Path to the input structure in CIF format. Should be previous output of Boltz-1.",
    required=True
)
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option("--twist_residue1", type=int,
    help="ID of the central residue for the first twisting region.",
    default=None
)
@click.option("--twist_residue2", type=int,
    help="ID of the central residue for the first twisting region.",
    default=None
)
@click.option("--exclude_residues", type=str,
    help="Residues to exclude from the twisting region. Comma-separated list of residue numbers or ranges. "
         "Examples: '100' (single residue), '50-55' (range), '100,50-55,105-110' (multiple specifications).",
    default=None
)
@click.option("--flank_size", type=int,
    help="Number of residues to include on each side of the central residue in twisting region.",
    default=5
)
@click.option("--chain_id1", type=str,
    help="Chain identifier for the first region.",
    default="A"
)
@click.option("--chain_id2", type=str,
    help="Chain identifier for the second region.",
    default="A"
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz.",
    default="~/.boltz",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default to 5.",
    default=5,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
    default="greedy",
)
@click.option(
    "--save_intermediates",
    is_flag=True,
    help="Whether to save intermediate structures during diffusion. Default is False.",
)
@click.option(
    "--inject_step",
    type=int,
    help="The step at which to inject coordinates during diffusion. Default is None.",
    default=None,
)
@click.option(
    "--inject_coords",
    type=click.Path(exists=True),
    help="Path to the file containing coordinates to inject during diffusion. Default is None.",
    default=None,
)
@click.option(
    "--inject_step_from_filename",
    is_flag=True,
    help="Whether to extract the injection step from the filename. Default is False.",
    default=False,
)
@click.option(
    "--alpha_values",
    help="Comma-separated list of alpha values for twist_fn variations.",
    default="15.0",
    callback=lambda ctx, param, value: [float(x) for x in value.split(',')] if isinstance(value, str) else ([float(value)] if isinstance(value, (int, float)) else value)
)
@click.option(
    "--beta_values",
    help="Comma-separated list of beta values for twist_fn variations.",
    default="1.0",
    callback=lambda ctx, param, value: [float(x) for x in value.split(',')] if isinstance(value, str) else ([float(value)] if isinstance(value, (int, float)) else value)
)
@click.option(
    "--tstart_step",
    type=str,
    default="200",
    help="Start steps for twist function, comma separated"
)
@click.option(
    "--tstop_step",
    type=str,
    default="0",
    help="Stop steps for twist function, comma separated"
)
@click.option(
    "--fbhw_width",
    type=float,
    default="0",
    help="flat bottom harmonic width, default 0 (no flat bottom)"
)
@click.option(
    "--constraint_z",
    default=False,
    is_flag=True,
    type=bool,
    help="apply z type twist"
)
@click.option(
    "--twist_rmsd",
    default=False,
    is_flag=True,
    type=bool,
    help="twisting using RMSD"
)
@click.option(
    "--twist_rmsd_full_sequence",
    is_flag=True,
    help="twisting using RMSD on full sequence, not just secondary structure",
    default=False,
)
@click.option(
    "--avoid",
    default=False,
    is_flag=True,
    type=bool,
    help="bias away from beta instead of towards"
)
@click.option(
    "--ess_threshold",
    type=float,
    default=1/3,
    help="effective sample size threshold in range [0,1], controls TDS"
)
def predict(
    data: str,
    out_dir: str,
    input_cif: str,
    twist_residue1: int = None,
    twist_residue2: int = None,
    exclude_residues: Optional[List[Union[int, List[int], str]]] = None,
    flank_size: int = 5,
    chain_id1: str = "A",
    chain_id2: str = "A",
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    save_intermediates: bool = False,
    inject_step: Optional[int] = None,
    inject_coords: Optional[str] = None,
    inject_step_from_filename: bool = False,
    twisted_sample: bool = True,
    alpha_values: Union[float, List[float]] = 15.0,
    beta_values: Union[float, List[float]] = 1.0,
    tstart_step: str = "200",
    tstop_step: str = "0",
    constraint_z: bool = False,
    twist_rmsd: bool = False,
    twist_rmsd_full_sequence: bool = False,
    fbhw_width: bool = 0.0,
    avoid: bool = False,
    ess_threshold: float = 1/3,
    model_module: Boltz1 = None,
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    ## JKARA DDR GRADIENT SEARCHING
    # torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    ## JKARA: Injection preparation
    # Ensure that only one of inject_step and inject_step_from_filename is defined
    assert not (inject_step and inject_step_from_filename), "Specify either inject_step or inject_step_from_filename, not both."

    assert not (twist_rmsd and twist_rmsd_full_sequence), "twist_rmsd and twist_rmsd_full_sequence are mutually exclusive. Please choose one."

    # Extract injection step from inject_coords file name if using inject_step_from_filename
    if inject_step_from_filename:
        inject_step = int(re.search(r'step_(\d+)', inject_coords).group(1))

    # Ensure injection step is within [0,200], i.e. number of diffusion steps
    if inject_step:
        assert 0 <= inject_step <= 200, f"Injection step read as {inject_step}, must be between 0 and 200"
    # Ensure that if inject_step is defined, inject_coords is also defined (and vice versa)
    if (inject_step is None) != (inject_coords is None):
        raise ValueError("Either both inject_step and inject_coords must be defined, or neither.")

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
 
    ## JKARA: Add injection step subfolder to output directory
    if inject_step is not None:
        # Extract injection source from inject_coords file name
        inject_source = Path(inject_coords).stem.split('_')[0]
        out_dir = out_dir / f"inject_from_{inject_source}" / f"inject_from_{inject_source}_step_{inject_step:03d}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return desc_string 

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "twisted_sample": twisted_sample
    }
    if not model_module:
        model_module: Boltz1 = Boltz1.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(BoltzDiffusionParams()),
            ema=False,
            conformix=True
        )

    model_module.confidence_module.use_s_diffusion = False
    model_module.accumulate_token_repr = False

    model_module.eval()

    # Parse alpha and beta values
    alpha_values = [float(alpha_values)] if isinstance(alpha_values, (int, float)) else [float(v) for v in alpha_values]    
    beta_values = [float(beta_values)] if isinstance(beta_values, (int, float)) else [float(v) for v in beta_values]

    # Parse tstart and tstop values
    tstart_values = [int(x) for x in tstart_step.split(",")]
    tstop_values = [int(x) for x in tstop_step.split(",")]

    # Parse twist_region1 and twist_region2
    if twist_rmsd:
        untwisted_coords, ss_atom_mask, ss_atom_mask_region = get_secondary_structure_masks(input_cif, twist_residue1, twist_residue2, exclude_residues)
        # TEMP
        # if twist_residue1==1 and twist_residue2==1:
        #     untwisted_coords, ss_atom_mask_region = get_residue_atoms_mask(input_cif, [[3,31], [40,67], [75,108], [120, 141], [170,202], [207,241], [250,271]])
        #     print("Twisting B2AR TM helices")
        # END TEMP
        desc_string = "rmsd"
        if twist_residue1 and twist_residue2 and twist_residue1 != twist_residue2:
            desc_string = f"rmsd_{twist_residue1}-{twist_residue2}"
        existing_runs = glob.glob(str(out_dir / "predictions" / f"{desc_string}" / "run*"))
        run_num = len(existing_runs)
        desc_string = f"{desc_string}/run{run_num:02d}"
        twist_region1, twist_region2 = (0, 0), (0, 0)
    elif twist_rmsd_full_sequence:
        untwisted_coords, ss_atom_mask, ss_atom_mask_region = get_secondary_structure_masks(input_cif, twist_residue1, twist_residue2, exclude_residues)
        ss_atom_mask_region = torch.ones_like(ss_atom_mask_region)
        desc_string = "rmsd_full_sequence"
        existing_runs = glob.glob(str(out_dir / "predictions" / f"{desc_string}" / "run*"))
        run_num = len(existing_runs)
        desc_string = f"{desc_string}/run{run_num:02d}"
        twist_region1, twist_region2 = (0, 0), (0, 0)
    else:
        twist_region1, twist_region2, desc_string = get_region_index_ranges(input_cif, chain_id1, twist_residue1, chain_id2, twist_residue2, flank_size)

    # Move model to the correct device
    device = torch.device('cuda' if accelerator == 'gpu' else accelerator)
    model_module.to(device)

    for batch in tqdm(data_module.predict_dataloader()):
        # Run through the model once so we can collect the inputs to the sample function
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float32):
            out = model_module(
                batch,
                recycling_steps=recycling_steps,
                num_sampling_steps=sampling_steps,
                diffusion_samples=1,
                run_confidence_sequentially=True,
                twisted_sample=False,
                save_diff_inputs=True
            )

            # Collect inputs to the sample function
            sample_inputs = {
                "s_trunk": out["s_trunk"],
                "z_trunk": out["z_trunk"],
                "s_inputs": out["s_inputs"],
                "feats": out["feats"],
                "relative_position_encoding": out["relative_position_encoding"],
                "num_sampling_steps": sampling_steps,
                "atom_mask": out["atom_mask"],
                "multiplicity": diffusion_samples,
                "train_accumulate_token_repr": False,
            }

        # Define twist_fn with parameters alpha and beta
        def twist_fn(alpha, beta, tstart_step, tstop_step, fbhw_width, constraint_z, twist_rmsd, avoid):
            def inner_twist_fn(xt, x0_hat, return_grad=True, t=None, atom_mask=None):
                def log_bias_potential(atom_pos: torch.Tensor):
                    # atom_diff = torch.mean(atom_pos[:, 1940:1990], dim=-2) - torch.mean(atom_pos[:, 990:1040], dim=-2)
                    # atom_diff = torch.mean(atom_pos[:, 408:495], dim=-2) - torch.mean(atom_pos[:, 408+730:495+730], dim=-2) #semisweet SER58-SER58 region
                    # atom_diff = torch.mean(atom_pos[:, 598:694], dim=-2) - torch.mean(atom_pos[:, 598+730:694+730], dim=-2) #semisweet ASN83-ASN83 region

                    atom_diff = torch.mean(atom_pos[:, twist_region1[0]:twist_region1[1]], dim=-2) - torch.mean(atom_pos[:, twist_region2[0]:twist_region2[1]], dim=-2)
                    atom_dist = torch.norm(atom_diff, dim=-1)
                    target_diff = beta  # Adjust target distance based on alpha and beta

                    within_fb = torch.abs(atom_dist - target_diff) < fbhw_width / 2.0

                    if avoid:
                        potential = -((atom_dist - target_diff)**2 - (fbhw_width / 2.0)**2)
                        return torch.where(within_fb, potential, torch.zeros_like(atom_dist))

                    potential = torch.minimum((atom_dist - (target_diff - fbhw_width / 2.0))**2, (atom_dist - (target_diff + fbhw_width / 2.0))**2)
                    return torch.where(within_fb, torch.zeros_like(atom_dist), potential)

                def bias_potential_z(atom_pos: torch.Tensor):
                    protein_mean = torch.mean(atom_pos[:, :2300], axis=-2, keepdim=True)
                    atom_pos = atom_pos - protein_mean

                    # compute lig pos wrt protein
                    lig_pos = torch.mean(atom_pos[:, 2320:2348], axis=-2)

                    # compute protein principal axis
                    P = atom_pos.shape[0]
                    particle_z_coords = torch.zeros(P, device=atom_pos.device)

                    for particle in range(P):
                        pca = PCA(n_components=3)
                        pca.fit(atom_pos[particle, :2300])

                        long_axis = principal_axes = pca.components_[0]

                        # project ligand position onto that axis
                        z_coord = torch.dot(lig_pos[particle], long_axis)
                        z_sign = torch.sign(torch.dot(atom_pos[particle, 0], long_axis)) # define orientation such that positive is towards atom 0

                        z_coord *= z_sign

                        particle_z_coords[particle] += z_coord

                    return (particle_z_coords - beta)**2

                def log_bias_potential_rmsd(atom_pos: torch.Tensor, atom_mask: torch.Tensor):
                    batch_size = atom_pos.shape[0]
                    padded_atom_size = atom_pos.shape[1]

                    ss_mask_region = torch.nn.functional.pad(
                        ss_atom_mask_region, 
                        (0, padded_atom_size - ss_atom_mask_region.shape[0]), 
                        value=0
                    ).to(atom_pos.device)
                    untwisted_pos = torch.nn.functional.pad(
                        untwisted_coords, 
                        (0, 0, 0, padded_atom_size - untwisted_coords.shape[0]), 
                        value=0
                    ).to(atom_pos.device)


                    atom_pos_aligned = weighted_rigid_align(atom_pos, untwisted_pos, atom_mask, ss_mask_region, keep_gradients=True)


                    # Compute RMSD between aligned atom_pos and untwisted_pos
                    mse_loss = ((atom_pos_aligned - untwisted_pos) ** 2).sum(dim=-1)
                    rmsd = torch.sqrt(
                        torch.sum(mse_loss * ss_mask_region, dim=-1)
                        / torch.sum(ss_mask_region, dim=-1)
                    )

                    return (rmsd - beta)**2

                if not constraint_z and not twist_rmsd and not twist_rmsd_full_sequence:
                    log_potential_xt_batch = log_bias_potential(x0_hat)
                elif constraint_z and not twist_rmsd and not twist_rmsd_full_sequence:
                    log_potential_xt_batch = bias_potential_z(x0_hat)
                elif not constraint_z and (twist_rmsd or twist_rmsd_full_sequence):
                    #print("RMSD twist")
                    log_potential_xt_batch = log_bias_potential_rmsd(x0_hat, atom_mask)
                else:
                    raise ValueError("Both constraint_z and twist_rmsd cannot be True at the same time.")

                # convert it to an unnormalized probability
                log_potential_xt_batch *= -1

                if return_grad:
                    if t is not None and tstart_step >= t >= tstop_step:
                        # note: we wind up going in the negative direction of this gradient
                        # ie we are finding the minimum of this potential/unnormalized logprob
                        grad_log_potential_xt_batch = torch.autograd.grad(
                            log_potential_xt_batch,
                            xt,
                            grad_outputs=torch.ones_like(log_potential_xt_batch),
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                    else:
                        grad_log_potential_xt_batch = torch.zeros_like(xt,
                                                                       device=xt.device,
                                                                       )
                    # param for sweeping
                    if t:
                        if alpha > 0:
                            #factor = t * alpha
                            factor = alpha
                        if alpha < 0:
                            factor = np.abs(alpha) * 200 * (1 + np.cos(np.pi * (np.log(1 + 4 * (230 - t)) / np.log(10)))) / 2
                        if alpha == 0:
                            factor = 0

                        grad_log_potential_xt_batch *= factor

                    return log_potential_xt_batch.to(model_module.device).detach(), grad_log_potential_xt_batch.to(model_module.device).detach()
                else:
                    return log_potential_xt_batch.to(model_module.device).detach()

            return inner_twist_fn


        def sample_step(sample_inputs, twist_fn_, pdistogram):
            try:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float32):
                    sample_out = model_module.structure_module.sample_twisted(
                        **sample_inputs,
                        twist_fn=twist_fn_,
                        ess_threshold=ess_threshold
                    )

                    # Compute confidence scores
                    sample_out.update(
                        model_module.confidence_module(
                            s_inputs=sample_inputs["s_inputs"].detach(),
                            s=sample_inputs["s_trunk"].detach(),
                            z=sample_inputs["z_trunk"].detach(),
                            s_diffusion=(
                                sample_out["diff_token_repr"]
                                if model_module.confidence_module.use_s_diffusion
                                else None
                            ),
                            x_pred=sample_out["sample_atom_coords"].detach(),
                            feats=sample_inputs["feats"],
                            pred_distogram_logits=pdistogram.detach(),
                            multiplicity=diffusion_samples,
                            run_sequentially=True,
                        )
                    )

                pred_dict = {"exception": False}
                pred_dict["masks"] = batch["atom_pad_mask"]
                pred_dict["coords"] = sample_out["sample_atom_coords"]
                pred_dict["confidence_score"] = (
                    4 * sample_out["complex_plddt"] +
                    (sample_out["iptm"] if not torch.allclose(sample_out["iptm"], torch.zeros_like(sample_out["iptm"])) else sample_out["ptm"])
                ) / 5

                for key in [
                    "ptm",
                    "iptm",
                    "ligand_iptm",
                    "protein_iptm",
                    "pair_chains_iptm",
                    "complex_plddt",
                    "complex_iplddt",
                    "complex_pde",
                    "complex_ipde",
                    "plddt",
                    "pae",
                    "pde",
                    "ess_trace",
                    "xt_trace",
                    "grad_log_potential_xt_trace",
                    "logp_y_given_x0_trace",
                    "log_w_trace"
                ]:
                    pred_dict[key] = sample_out[key]
                return pred_dict

            except RuntimeError as e:  # catch out of memory exceptions
                if "out of memory" in str(e):
                    print("| WARNING: ran out of memory, skipping batch")
                    torch.cuda.empty_cache()
                    #gc.collect()
                    return {"exception": True}
                else:
                    raise {"exception": True}

        # Rerun sample_twisted function with variations
        for i in range(len(alpha_values)):
            for j in range(len(beta_values)):
                for tstart in tstart_values:
                    for tstop in tstop_values:
                        print(f"Running with input structure {input_cif}")
                        print(f"Running variation {alpha_values[i]}, {beta_values[j]}, tstart {tstart}, tstop {tstop}")

                        # Create dictionary with added information about twisting
                        input_dict = {
                            "desc_string": desc_string,
                            "twist_residue1": twist_residue1,
                            "twist_residue2": twist_residue2,
                            "chain_id1": chain_id1,
                            "chain_id2": chain_id2,
                            "flank_size": flank_size,
                            "twist_region1": twist_region1,
                            "twist_region2": twist_region2,
                            "alpha": alpha_values[i],
                            "beta": beta_values[j],
                            "tstart": tstart,
                            "tstop": tstop,
                            "fbhw_width": fbhw_width,
                            "constraint_z": constraint_z,
                            "twist_rmsd": twist_rmsd,
                            "exclude_residues": exclude_residues,
                            "avoid": avoid,
                        }

                        sample_out = sample_step(
                            sample_inputs,
                            twist_fn(alpha_values[i],
                                    beta_values[j],
                                    tstart,
                                    tstop,
                                    fbhw_width,
                                    constraint_z,
                                    twist_rmsd,
                                    avoid),
                            out["pdistogram"]
                        )

                        # full_output_dir = out_dir / "predictions" / desc_string /f"variation_alpha_{alpha_values[i]}_beta_{beta_values[j]}_tstart_{tstart}_tstop_{tstop}"
                        full_output_dir = out_dir / "predictions" / desc_string / f"variation_alpha_{alpha_values[i]}_beta_{beta_values[j]}"

                        pred_writer = BoltzWriter(
                            data_dir=processed.targets_dir,
                            output_dir=full_output_dir,
                            output_format=output_format,
                        )

                        # mask should be the same for all particles, take particle 0
                        traj_mask = out["atom_mask"][0, :].cpu().bool()

                        def mask_tensor(tensor, mask):
                            """
                            Args:
                                tensor: tensor of shape (a, b, c, d)
                                mask: boolean tensor of shape (c,)
                            Returns:
                                tensor of shape (a, b, number_of_True_in_mask, d)
                            """
                            # Expand mask to match tensor dimensions
                            # None/newaxis adds a dimension
                            expanded_mask = mask[None, None, :, None]

                            # Broadcast the mask to all other dimensions
                            expanded_mask = expanded_mask.expand(tensor.shape)

                            # Use boolean indexing along the c dimension
                            return tensor[expanded_mask].reshape(tensor.shape[0], tensor.shape[1], -1, tensor.shape[3])

                        # write out trajectories for particle 0 only
                        # shape of tensors: nframes x nparticles x natoms x 3
                        # tensor_to_netcdf(full_output_dir / 'intermediates_trace.nc',
                        #                  mask_tensor(sample_out['xt_trace'], traj_mask)[:, 0, :, :],
                        #                  ess_trace=sample_out['ess_trace'],
                        #                  logp_y_given_x0_trace=sample_out['logp_y_given_x0_trace'][:,0], # only record particle 0
                        #                  logsumexp_w_trace=torch.logsumexp(sample_out['log_w_trace'], dim=-1)
                        #                  )
                        # tensor_to_netcdf(full_output_dir / 'intermediates_twist_deltas_trace.nc',
                        #                  mask_tensor(sample_out['grad_log_potential_xt_trace'], traj_mask)[:, 0, :, :]) 

                        if twist_rmsd or twist_rmsd_full_sequence:
                            print('writing RMSD')
                            atom_pos = sample_out["coords"]
                            atom_mask = sample_out["masks"]
                            padded_atom_size = atom_pos.shape[1]

                            ss_mask_region = torch.nn.functional.pad(
                                ss_atom_mask_region, 
                                (0, padded_atom_size - ss_atom_mask_region.shape[0]), 
                                value=0
                            ).to(atom_pos.device)
                            untwisted_pos = torch.nn.functional.pad(
                                untwisted_coords, 
                                (0, 0, 0, padded_atom_size - untwisted_coords.shape[0]), 
                                value=0
                            ).to(atom_pos.device)

                            atom_pos_aligned = weighted_rigid_align(atom_pos, untwisted_pos, atom_mask, ss_mask_region, keep_gradients=True)

                            # Compute RMSD between aligned atom_pos and untwisted_pos
                            mse_loss = ((atom_pos_aligned - untwisted_pos) ** 2).sum(dim=-1)
                            rmsd = torch.sqrt(
                                torch.sum(mse_loss * ss_mask_region, dim=-1)
                                / torch.sum(ss_mask_region, dim=-1)
                            )
                            rmsd = rmsd.cpu().numpy().tolist()

                            input_dict["input_cif"] = input_cif

                        pred_writer.write_on_batch_end(
                            trainer=None,
                            pl_module=None,
                            prediction=sample_out,
                            batch_indices=None,
                            batch=batch,
                            batch_idx=None,
                            dataloader_idx=None,
                            input_dict=input_dict,
                            rmsd=rmsd if twist_rmsd else None,
                        )


    return desc_string


if __name__ == "__main__":
    cli()
