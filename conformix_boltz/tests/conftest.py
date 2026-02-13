"""Pytest configuration and fixtures for ConforMix-Boltz tests."""
import pytest
import numpy as np
import tempfile
from pathlib import Path
import mdtraj as md


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def example_fasta_file(temp_dir):
    """Create an example FASTA file."""
    fasta_path = temp_dir / "test_protein.fasta"
    fasta_content = """>test_protein
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
"""
    fasta_path.write_text(fasta_content)
    return fasta_path


@pytest.fixture
def short_fasta_file(temp_dir):
    """Create a short FASTA file for quick tests."""
    fasta_path = temp_dir / "short_protein.fasta"
    fasta_content = """>short_protein
MKTAYIAKQR
"""
    fasta_path.write_text(fasta_content)
    return fasta_path


@pytest.fixture
def example_trajectory():
    """Create an example MDTraj trajectory for testing."""
    # Create a simple 3-residue trajectory
    n_frames = 10
    n_atoms = 15  # 5 atoms per residue (N, CA, C, O, CB)
    
    # Create topology
    top = md.Topology()
    chain = top.add_chain()
    
    for i in range(3):
        residue = top.add_residue(f"RES{i}", chain)
        for atom_name in ["N", "CA", "C", "O", "CB"]:
            top.add_atom(atom_name, md.element.carbon, residue)
    
    # Create coordinates
    xyz = np.random.randn(n_frames, n_atoms, 3) * 0.1 + np.array([0, 0, 0])
    
    traj = md.Trajectory(xyz=xyz, topology=top)
    return traj


@pytest.fixture
def example_cif_file(temp_dir):
    """Create a minimal example CIF file."""
    cif_path = temp_dir / "test_structure.cif"
    # Minimal CIF content
    cif_content = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM   1    N  N   MET A   1   10.0   20.0   30.0
ATOM   2    C  CA  MET A   1   11.0   21.0   31.0
ATOM   3    C  C   MET A   1   12.0   22.0   32.0
ATOM   4    O  O   MET A   1   13.0   23.0   33.0
"""
    cif_path.write_text(cif_content)
    return cif_path

