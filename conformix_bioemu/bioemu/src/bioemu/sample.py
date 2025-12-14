# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import os
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import stackprinter
from huggingface_hub import hf_hub_download
from tqdm import tqdm

stackprinter.set_excepthook(style="darkbg2")

import torch
import yaml
from torch_geometric.data.batch import Batch

from .chemgraph import ChemGraph
from .convert_chemgraph import save_pdb_and_xtc
from .get_embeds import get_colabfold_embeds
from .models import DiGConditionalScoreModel
from .sde_lib import SDE
from .seq_io import parse_sequence, write_fasta
from .utils import count_samples_in_output_dir, format_npz_samples_filename

from Bio.PDB import PDBParser, MMCIFParser # jkara
import pymol
from pymol import cmd

logger = logging.getLogger(__name__)

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
SupportedDenoisersLiteral = Literal["dpm", "heun", "first_order", "first_order_tds"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))


def maybe_download_checkpoint(
    *,
    model_name: str | None,
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> tuple[str, str]:
    """If ckpt_path and model config_path are specified, return them, else download named model from huggingface.
    Returns:
        tuple[str, str]: path to checkpoint, path to model config
    """
    if ckpt_path is not None:
        assert model_config_path is not None, "Must provide model_config_path if ckpt_path is set."
        return str(ckpt_path), str(model_config_path)
    assert model_name is not None
    assert (
        model_config_path is None
    ), f"Named model {model_name} comes with its own config. Do not provide model_config_path."
    ckpt_path = hf_hub_download(
        repo_id="microsoft/bioemu", filename=f"checkpoints/{model_name}/checkpoint.ckpt"
    )
    model_config_path = hf_hub_download(
        repo_id="microsoft/bioemu", filename=f"checkpoints/{model_name}/config.yaml"
    )
    return str(ckpt_path), str(model_config_path)


@torch.no_grad()
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    untwisted_input: str = None,
    batch_size_100: int = 10,
    batch_size: int | None = None,
    model_name: str | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config_path: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
    beta: float = 0.0, # used for denoisers with guidance
    c0: int | None = None, # center of twist region 0
    c1: int | None = None, # center of twist region 1
    twist_rmsd: bool = False, # twist RMSD (for TDS)
    twist_k: float = 0.0, # twist strength (for TDS)
    extra_twist_k: float = 1.0, # extra multiplier applying only to the gradient, not the resampling (for TDS)
    enable_guidance: bool = True, # if false, only do resampling (for TDS)
    rmsd_all_residues: bool = False, # whether to compute RMSD for twisting based on all residues instead of just those with secondary structure
    resample_start: int = 20, # what timestep to start resampling at
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples, or a path to a .fasta file, or a path to an .a3m file with MSAs.
            If it is not an a3m file, then colabfold will be used to generate an MSA and embedding.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        model_name: Name of pretrained model to use. The model will be retrieved from huggingface. If not set,
           this defaults to `bioemu-v1.0`. If this is set, you do not need to provide `ckpt_path` or `model_config_path`.
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
           Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config_path` not specified. Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config_path: Path to the denoiser config, defining the denoising process.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server. If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )

    assert os.path.isfile(ckpt_path), f"Checkpoint {ckpt_path} not found"
    assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # User may have provided an MSA file instead of a sequence. This will be used for embeddings.
    msa_file = sequence if str(sequence).endswith(".a3m") else None

    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    # Parse FASTA or A3M file if sequence is a file path. Extract the actual sequence.
    sequence = parse_sequence(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])

    if denoiser_config_path is None:
        assert (
            denoiser_type in SUPPORTED_DENOISERS
        ), f"denoiser_type must be one of {SUPPORTED_DENOISERS}"
        denoiser_config_path = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"

    with open(denoiser_config_path) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )

    if batch_size is None:
        batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    if batch_size == 0:
        logger.warning(f"Sequence {sequence} may be too long. Attempting with batch_size = 1.")
        batch_size = 1
    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    # Twist RMSD preparation
    if twist_rmsd:
        assert untwisted_input is not None, "untwisted_input must be provided for twist RMSD calculation."
        untwisted_coords, ss_mask = get_secondary_structure_masks(untwisted_input)

        if rmsd_all_residues:
            ss_mask = torch.ones_like(ss_mask)
    else:
        untwisted_coords = None
        ss_mask = None

    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    for seed in tqdm(
        range(existing_num_samples, num_samples, batch_size), desc="Sampling batches..."
    ):
        n = min(batch_size, num_samples - seed)
        npz_path = output_dir / format_npz_samples_filename(seed, n)
        if npz_path.exists():
            raise ValueError(
                f"Not sure why {npz_path} already exists when so far only {existing_num_samples} samples have been generated."
            )
        logger.info(f"Sampling {seed=}")
        batch = generate_batch(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=min(batch_size, n),
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
            beta=beta,
            c0=c0,
            c1=c1,
            twist_rmsd=twist_rmsd,
            ss_mask=ss_mask,
            untwisted_coords=untwisted_coords,
            twist_k=twist_k,
            extra_twist_k=extra_twist_k,
            enable_guidance=enable_guidance,
            resample_start=resample_start
        )
        batch = {k: v.cpu().detach().numpy() for k, v in batch.items()}
        print(batch.keys())
        np.savez(npz_path, **batch, sequence=sequence)
        print('***', np.max(batch['pos']) - np.min(batch['pos']))

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    if set(sequences) != {sequence}:
        raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")
    data = [np.load(f) for f in samples_files]
    positions = torch.tensor(np.concatenate([f["pos"] for f in data]))
    node_orientations = torch.tensor(
        np.concatenate([f["node_orientations"] for f in data])
    )
    if all(['twist_dist_final' in f for f in data]):
        twist_dist_final = torch.tensor(
            np.concatenate([f["twist_dist_final"] for f in data])
        )
    else:
        twist_dist_final = None
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "samples.xtc",
        sequence=sequence,
        filter_samples=filter_samples,
        twist_dist_final=twist_dist_final
    )
    logger.info(f"Completed. Your samples are in {output_dir}.")


def get_secondary_structure_masks(structure_file):
    """
    Create masks where only beta sheets and helices are marked as 1.
    Uses PyMOL's secondary structure assignments.
    
    Parameters:
    structure_file: Path to CIF/PDB file
    
    Returns:
    residue_coords: Tensor of alpha carbon coordinates (one per residue)
    ss_residue_mask: Tensor of residue mask (1 for ss, 0 otherwise)
    """
    # Determine file type and parse accordingly
    if structure_file.endswith(".cif"):
        parser = MMCIFParser()
    elif structure_file.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Unsupported file format. Please provide a .cif or .pdb file.")
    structure = parser.get_structure("structure", structure_file)
    
    # Build residue info list (using alpha carbons)
    residue_coords = []
    residue_info = []  # (model_id, chain_id, res_id, res_name)
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':  # Skip hetero-atoms and waters
                    continue
                res_id = res.id[1]
                # Look for alpha carbon in this residue
                if 'CA' in res:
                    ca_atom = res['CA']
                    residue_coords.append(ca_atom.coord)
                    residue_info.append((model.id, chain.id, res_id, res.resname))

    # Initialize PyMOL in quiet mode if not already running
    if not hasattr(pymol, 'pymol_launched'):
        pymol.finish_launching(['pymol', '-qc'])
        pymol.pymol_launched = True
    
    # Load the structure
    cmd.load(structure_file, 'protein_for_ss')
    target = 'protein_for_ss'
    
    # Make sure secondary structure is calculated
    cmd.dss(target)
    
    # Get all residues (represented by alpha carbons)
    all_residues = cmd.get_model(f'{target} and name CA')
    
    # Create selections for residues in sheets or helices (ss s or ss h in PyMOL)
    cmd.select('ss_residues', f'{target} and (ss h or ss s) and name CA')
    ss_residues = cmd.get_model('ss_residues')

    # Create residue mask
    residue_mask = []
    # Create a set for faster lookups of residues in secondary structure
    ss_residue_ids = {(a.chain, a.resi) for a in ss_residues.atom}
    
    for ca_atom in all_residues.atom:
        # Check if in secondary structure
        in_ss = (ca_atom.chain, ca_atom.resi) in ss_residue_ids
        residue_mask.append(1 if in_ss else 0)
    
    # Clean up
    cmd.delete('protein_for_ss')
    cmd.delete('ss_residues')
    
    residue_coords = torch.tensor(np.array(residue_coords))
    ss_residue_mask = torch.tensor(residue_mask, dtype=int)

    # Convert to nanometers for BioEmu
    residue_coords = residue_coords / 10.0

    return residue_coords, ss_residue_mask

def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    cache_embeds_dir: str | Path | None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    beta: float | None = None,
    c0: int | None = None, # center of twist region 0
    c1: int | None = None, # center of twist region 1
    twist_rmsd: bool = False, # twist RMSD (for heun_guided_tds)
    ss_mask: torch.Tensor = None, # use secondary structure mask (for heun_guided_tds)
    untwisted_coords: torch.Tensor = False, # use untwisted coordinates (for heun_guided_tds)
    twist_k: float = 0.0, # twist strength (for heun_guided_tds)
    extra_twist_k: float = 1.0, # extra twist strength not for resampling (for heun_guided_tds)
    enable_guidance: bool = True,
    resample_start: int = 20, # what timestep to start resampling at
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples, using GPU if available.

    Args:
        score_model: Score model.
        sequence: Amino acid sequence.
        sdes: SDEs defining corruption process. Keys should be 'node_orientations' and 'pos'.
        embeddings_file: Path to embeddings file.
        batch_size: Batch size.
        seed: Random seed.
        msa_file: Optional path to an MSA A3M file.
        msa_host_url: MSA server URL for colabfold.
    """

    torch.manual_seed(seed)
    n = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    single_embeds = np.load(single_embeds_file)
    pair_embeds = np.load(pair_embeds_file)
    assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
    assert single_embeds.shape[0] == n
    assert len(single_embeds.shape) == 2
    _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

    single_embeds, pair_embeds = torch.from_numpy(single_embeds), torch.from_numpy(pair_embeds)
    pair_embeds = pair_embeds.view(n**2, n_pair_feats)

    edge_index = torch.cat(
        [
            torch.arange(n).repeat_interleave(n).view(1, n**2),
            torch.arange(n).repeat(n).view(1, n**2),
        ],
        dim=0,
    )
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))
    log_importance_weight = torch.tensor([float("nan")])
    twist_dist_final = torch.tensor([float("nan")])

    chemgraph = ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
        log_importance_weight=log_importance_weight,
        twist_dist_final=twist_dist_final,
    )
    context_batch = Batch.from_data_list([chemgraph for _ in range(batch_size)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampled_chemgraph_batch = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
        beta=beta,
        c0=c0,
        c1=c1,
        twist_k=twist_k,
        extra_twist_k=extra_twist_k,
        twist_rmsd=twist_rmsd,
        ss_mask=ss_mask,
        untwisted_coords=untwisted_coords,
        sequence=sequence,
        enable_guidance=enable_guidance,
        resample_start=resample_start,
    )
    assert isinstance(sampled_chemgraph_batch, Batch)
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")

    return_dict = {"pos": pos, "node_orientations": node_orientations}

    if 'log_importance_weight' in sampled_chemgraphs[0]:
        return_dict['log_importance_weight'] = torch.stack([x.log_importance_weight for x in sampled_chemgraphs]).to("cpu")
        return_dict['twist_dist_final'] = torch.stack([x.twist_dist_final for x in sampled_chemgraphs]).to("cpu")

    return return_dict



if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.DEBUG)

    fire.Fire(main)
