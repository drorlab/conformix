import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
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

import os
cwd = os.getcwd()
import shutil
import yaml
from Bio import SeqIO
from io import StringIO

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
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


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

def pdb_to_fasta(pdb_id, fasta_dir, chains=None, override=False):
    """
    Fetch a PDB structure by ID and extract sequences in FASTA format.
    
    Args:
        pdb_id (str): The 4-character PDB ID to fetch
        fasta_dir (str): Directory where the FASTA file will be saved
        chains (list, optional): List of chain IDs to extract (default: all chains)
        
    Returns:
        str: Path to the saved FASTA file
    """
    # Setup
    os.makedirs(fasta_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    
    # Handle chain filter
    if chains:
        chains = [c.upper() for c in chains]
        print(f"Extracting only chains: {', '.join(chains)}")
    
    # Fetch PDB data
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to retrieve PDB {pdb_id}. Status code: {response.status}")
            fasta_content = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        raise ValueError(f"Failed to retrieve PDB {pdb_id}. Error: {e}")
    
    # Parse sequences
    records = list(SeqIO.parse(StringIO(fasta_content), "fasta"))
    if not records:
        raise ValueError(f"No sequences found for PDB ID {pdb_id}")
    
    # Prepare fasta filename
    chain_str = f"_{'_'.join(chains)}" if chains else ""
    fasta_file = os.path.join(fasta_dir, f"{pdb_id}{chain_str}.fasta")
    if os.path.exists(fasta_file) and not override:
        print(f"FASTA file {fasta_file} already exists. Skipping download.")
        return fasta_file
    
    # Process and write sequences
    found_chains = set()
    with open(fasta_file, "w") as f:
        for record in records:
            parts = record.description.split('|')
            if len(parts) >= 2:
                chain_part = parts[1].replace('Chains', '').replace('Chain', '').strip()
                for chain_id in [c.strip()[0].upper() for c in chain_part.split(',') if c.strip()]:
                    if chains and chain_id not in chains:
                        continue
                    
                    found_chains.add(chain_id)
                    f.write(f">{chain_id}|protein|\n{str(record.seq)}\n")
    
    # Report results
    if chains:
        missing_chains = set(chains) - found_chains
        if missing_chains:
            print(f"Warning: Could not find chains {', '.join(missing_chains)} in PDB {pdb_id}")
        if not found_chains:
            print(f"No specified chains were found in PDB {pdb_id}. File contains no sequences.")
            return fasta_file
    
    print(f"FASTA file saved as {fasta_file} with chains: {', '.join(found_chains)}")
    return fasta_file

import os
import urllib.request
from io import StringIO
from Bio import SeqIO

def uniprot_to_fasta(uniprot_id, fasta_dir, override=False):
    """
    Fetch a protein sequence by UniProt ID and save it as FASTA.
    
    Args:
        uniprot_id (str): The UniProt ID to fetch (e.g., 'P12345')
        fasta_dir (str): Directory where the FASTA file will be saved
        
    Returns:
        str: Path to the saved FASTA file
    """
    # Setup
    os.makedirs(fasta_dir, exist_ok=True)
    uniprot_id = uniprot_id.upper()  # UniProt IDs are conventionally uppercase
    
    # Fetch data from UniProt
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to retrieve UniProt {uniprot_id}. Status code: {response.status}")
            fasta_content = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        raise ValueError(f"Failed to retrieve UniProt {uniprot_id}. Error: {e}")
    
    # Parse sequence
    records = list(SeqIO.parse(StringIO(fasta_content), "fasta"))
    if not records:
        raise ValueError(f"No sequence found for UniProt ID {uniprot_id}")
    
    # Prepare output file
    fasta_file = os.path.join(fasta_dir, f"{uniprot_id}.fasta")
    if os.path.exists(fasta_file) and not override:
        print(f"FASTA file {fasta_file} already exists. Skipping download.")
        return fasta_file
    
    # Write sequence in desired format
    with open(fasta_file, "w") as f:
        record = records[0]  # UniProt returns only one sequence per ID
        f.write(f">A|protein|\n{str(record.seq)}\n")
    
    print(f"FASTA file saved as {fasta_file}")
    return fasta_file

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
    
            # Create backup and prepare path conversion
            backup_path = path.parent / f"{path.stem}_no-msa{path.suffix}"
            shutil.copy2(path, backup_path)
            
            def get_abs_path(p):
                p_str = str(p)
                return p_str if os.path.isabs(p_str) else os.path.abspath(os.path.join(cwd, p_str))
            
            # Get target chains with MSA info
            target_chains = [c for c in target.record.chains if c.mol_type == prot_id]
            
            if path.suffix in (".fa", ".fas", ".fasta"):
                # Extract FASTA chain IDs
                with path.open("r") as f:
                    fasta_content = f.readlines()
                
                fasta_chain_ids = [line.strip().split("|")[0][1:] for line in fasta_content 
                                if line.startswith(">") and "|" in line]
                
                # Map chains by position
                chain_mapping = {}
                for i, fasta_id in enumerate(fasta_chain_ids):
                    if i < len(target_chains) and target_chains[i].msa_id != -1:
                        chain_mapping[fasta_id] = get_abs_path(target_chains[i].msa_id)
                
                # Update headers
                new_content = []
                updated_chains = set()
                
                for line in fasta_content:
                    if line.startswith(">"):
                        parts = line.strip().split("|")
                        if len(parts) >= 2 and parts[0][1:] in chain_mapping:
                            chain_id = parts[0][1:]
                            msa_path = chain_mapping[chain_id]
                            parts = parts[:2] + [msa_path]  # Replace or add MSA path
                            new_content.append("{}|{}|{}\n".format(*parts))
                            updated_chains.add(chain_id)
                        else:
                            new_content.append(line)
                    else:
                        new_content.append(line)
                
                if updated_chains:
                    with path.open("w") as f:
                        f.writelines(new_content)
                    click.echo(f"Updated {len(updated_chains)} chain headers in {path}")
                
            # TODO: check if this works, has not been tested for YAML 
            elif path.suffix in (".yml", ".yaml"):                
                # Load YAML
                with path.open("r") as f:
                    yaml_content = yaml.safe_load(f)
                
                # Map entities by ID
                updated = 0
                if "sequences" in yaml_content:
                    entity_map = {str(c.entity_id): c for c in target_chains}
                    
                    for seq in yaml_content["sequences"]:
                        for entity_type, entity_data in seq.items():
                            if entity_type == "protein" and "id" in entity_data:
                                entity_id = str(entity_data["id"])
                                if entity_id in entity_map and entity_map[entity_id].msa_id != -1:
                                    entity_data["msa"] = get_abs_path(entity_map[entity_id].msa_id)
                                    updated += 1
                    
                    if updated:
                        with path.open("w") as f:
                            yaml.dump(yaml_content, f, default_flow_style=False)
                        click.echo(f"Updated {updated} protein entities in YAML")

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


def load_model(
        cache=Path("~/.cache"),
        checkpoint=None,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        write_full_pae: bool = True,
        write_full_pde: bool = True,
    ) -> Boltz1:

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
        conformix=True,
    )
    model_module.eval()

    return model_module


@click.group()
def cli() -> None:
    """Boltz1."""
    return


@cli.command()
@click.argument("data", required=True)
@click.option(
    "--chains",
    type=str,
    help="The chains to extract from the PDB file (comma-separated list like 'A,B,C' or single chain 'A'). Default is all chains.",
    default=None,
)
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
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
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
# @click.option(
#     "--write_full_pae",
#     type=bool,
#     is_flag=True,
#     help="Whether to dump the pae into a npz file. Default is True.",
# )
# @click.option(
#     "--write_full_pde",
#     type=bool,
#     is_flag=True,
#     help="Whether to dump the pde into a npz file. Default is False.",
# )
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
# @click.option(
#     "--use_msa_server",
#     is_flag=True,
#     help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
# )
@click.option(
    "--single_sequence_mode",
    is_flag=True,
    help="Whether to use single sequence mode, i.e. without MSA. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url.",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Options are 'greedy' and 'complete'",
    default="greedy",
)
def predict(
    data: str,
    out_dir: str,
    chains: Optional[str] = None,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    write_full_pae: bool = True,
    write_full_pde: bool = True,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = True,
    single_sequence_mode: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    model_module: Boltz1 = None,
) -> None:
    """
    Run predictions with Boltz-1.

    DATA: Specify one of: 
        (1) path to a FASTA/YAML file 
        OR
        (2) UniProt ID from which to extract a FASTA file
        OR
        (3) PDB ID from which to extract a FASTA file.
    """
    # Process input data as either a FASTA/YAML file or a PDB ID
    if os.path.exists(data) and os.path.isfile(data):
        click.echo(f"Processing data file: {data}")
        if chains:
            raise click.UsageError("Chains should not be specified when using FASTA/YAML mode.")
    elif data.isalnum() and len(data) == 4:
        pdb_id = data
        click.echo(f"Processing PDB ID: {pdb_id}")
        if chains:
            chains = chains.split(",")
        fasta_dir = os.path.dirname(out_dir)
        data = pdb_to_fasta(pdb_id, fasta_dir, chains=chains, override=override)
    elif data[0].isalpha() and data.isalnum() and len(data) > 6:
        uniprot_id = data
        click.echo(f"Processing UniProt ID: {uniprot_id}")
        if chains:
            raise click.UsageError("Chains should not be specified when using UniProt mode.")
        data = uniprot_to_fasta(uniprot_id, out_dir, override=override)
    else:
        raise click.UsageError(f"Invalid input: {data}. Must be either an existing file path or a valid PDB ID.")
    # keep path to input
    data_path = Path(data)

    # If single sequence mode, set use MSA server to False
    # Warning: single sequence mode is not recommended for most cases
    if single_sequence_mode: #TODO: implement single-sequence mode
        use_msa_server = False
        click.echo("Running in single sequence mode. MSA generation will be skipped.")

    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return data_path

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
    }
    if not model_module:
        model_module: Boltz1 = Boltz1.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(BoltzDiffusionParams()),
            ema=False,
        )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )

    return data_path


if __name__ == "__main__":
    cli()
