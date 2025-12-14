# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any

import mdtraj
from pdb2pqr import io
from pdb2pqr.biomolecule import Biomolecule
from pdb2pqr.io import print_biomolecule_atoms as pdb2pqr_print_biomolecule_atoms
from pdb2pqr.main import build_main_parser
from pdb2pqr.main import check_files as pdb2pqr_check_files
from pdb2pqr.main import check_options as pdb2pqr_check_options
from pdb2pqr.main import non_trivial as pdb2pqr_non_trivial
from pdb2pqr.main import print_pdb as pdb2pqr_print_pdb
from pdb2pqr.main import setup_molecule as pdb2pqr_setup_molecule


class PDB2PQRInscriptParser:
    """Parse arguments to PDB2PQR inside a python script."""

    def __init__(self, **kwargs):

        removed_args = ["clean"]
        parser = build_main_parser()

        args = {}
        for action in parser._actions[1:]:
            if action.dest.lower() in removed_args:
                continue
            args[action.dest] = action.default

        for key, val in kwargs.items():
            if key not in args:
                raise RuntimeError(f"PDB2PQR option not valid or not part of this code: {key}")

        args.update(kwargs)

        for key, val in args.items():
            setattr(self, key, val)


def get_protonation_state_from_pdb2pqr(
    args: PDB2PQRInscriptParser,
) -> tuple[dict[str, Any], Biomolecule, bool]:
    """Function following the PDB2PQR main command line script

    Args:
        args: Arguments to be passed to PDB2PQR

    Returns:
        Tuple of (dictionary describing protonation states and pKa's,
        PDB2PQR biomolecule object,
        bool if in crystallographic information file format (CIF, required by PDB2PQR))
    """

    # Check and transform input arguments
    pdb2pqr_check_files(args)
    pdb2pqr_check_options(args)
    # Load topology files
    definition = io.get_definitions()
    # Load molecule
    pdblist, is_cif = io.get_molecule(args.input_path)  # type: ignore[attr-defined]

    # Set up molecule
    biomolecule, definition, ligand = pdb2pqr_setup_molecule(pdblist, definition, args.ligand)  # type: ignore[attr-defined]

    # set termini states for biomolecule chains
    biomolecule.set_termini(args.neutraln, args.neutralc)  # type: ignore[attr-defined]
    biomolecule.update_bonds()

    results = pdb2pqr_non_trivial(
        args=args,
        biomolecule=biomolecule,
        ligand=ligand,
        definition=definition,
        is_cif=is_cif,
    )

    return results, biomolecule, is_cif


@contextmanager
def silence_root_logger():
    root_logger = logging.getLogger()
    previous_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        root_logger.setLevel(previous_level)


def get_propka_protonation(traj: mdtraj.Trajectory, pH: float) -> list[mdtraj.Trajectory]:
    """Convenience function to apply PDB2PQR to a PDB file from within
    a python script.

    This function writes a PDB file output.

    Args:
        traj: trajectory containing heavy-atom protein structure to be protonated
        pH: pH to protonate

    Returns:
        List of Trajectory objects with hydrogens
    """
    out = []
    for frame in traj:
        with TemporaryDirectory() as tmp:
            pdbfile = os.path.join(tmp, "in.pdb")
            outfile = os.path.join(tmp, "out.pdb")
            frame.save_pdb(pdbfile)

            #  ensure compat with PDB2PQR
            pdb2pqr_args = PDB2PQRInscriptParser(
                ff="AMBER",
                ffout="AMBER",
                pka_method="propka",
                ph=pH,
                pdb_output=outfile,
                input_path=pdbfile,
                keep_chain=True,
            )
            with silence_root_logger():
                results, biomolecule, is_cif = get_protonation_state_from_pdb2pqr(pdb2pqr_args)
                pdb2pqr_print_pdb(
                    args=pdb2pqr_args,
                    pdb_lines=pdb2pqr_print_biomolecule_atoms(
                        biomolecule.atoms, chainflag=pdb2pqr_args.keep_chain, pdbfile=True  # type: ignore[attr-defined]
                    ),
                    header_lines=results["header"],
                    missing_lines=results["missed_residues"],
                    is_cif=is_cif,
                )
            out.append(mdtraj.load_pdb(outfile))

    return out
