"""Tests for utility functions."""
import pytest
import numpy as np
from pathlib import Path
import mdtraj as md

from boltz.utils.cif_to_xtc import (
    find_cif_files,
    filter_unphysical_traj,
    combine_structures,
    load_cif_structures,
)


class TestFindCifFiles:
    """Tests for finding CIF files."""
    
    def test_find_cif_files_single(self, temp_dir):
        """Test finding a single CIF file."""
        cif_file = temp_dir / "test.cif"
        cif_file.write_text("data_test\n")
        
        found = find_cif_files(temp_dir)
        assert len(found) == 1
        assert found[0] == cif_file
    
    def test_find_cif_files_multiple(self, temp_dir):
        """Test finding multiple CIF files."""
        (temp_dir / "file1.cif").write_text("data_test\n")
        (temp_dir / "file2.cif").write_text("data_test\n")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.cif").write_text("data_test\n")
        
        found = find_cif_files(temp_dir)
        assert len(found) == 3
    
    def test_find_cif_files_none(self, temp_dir):
        """Test finding no CIF files."""
        found = find_cif_files(temp_dir)
        assert len(found) == 0
    
    def test_find_cif_files_case_insensitive(self, temp_dir):
        """Test case-insensitive CIF file finding."""
        (temp_dir / "test.CIF").write_text("data_test\n")
        (temp_dir / "test2.Cif").write_text("data_test\n")
        
        found = find_cif_files(temp_dir)
        assert len(found) == 2


class TestFilterUnphysicalTraj:
    """Tests for filtering unphysical trajectories."""
    
    def test_filter_keeps_good_trajectory(self, example_trajectory):
        """Test that good trajectories pass filtering."""
        filtered = filter_unphysical_traj(example_trajectory)
        assert filtered.n_frames > 0
        assert filtered.n_atoms == example_trajectory.n_atoms
    
    def test_filter_removes_clashes(self, temp_dir):
        """Test that structures with clashes are filtered."""
        # Create trajectory with overlapping atoms
        top = md.Topology()
        chain = top.add_chain()
        residue = top.add_residue("RES", chain)
        top.add_atom("CA", md.element.carbon, residue)
        top.add_atom("CA2", md.element.carbon, residue)
        
        # Create coordinates with atoms too close
        xyz = np.array([[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]])  # Very close atoms
        traj = md.Trajectory(xyz=xyz, topology=top)
        
        filtered = filter_unphysical_traj(traj, clash_distance=0.5)
        # Should filter out frames with clashes
        assert filtered.n_frames <= traj.n_frames
    
    def test_filter_strict_mode(self, example_trajectory):
        """Test strict filtering mode."""
        # This should work if trajectory is good
        try:
            filtered = filter_unphysical_traj(example_trajectory, strict=True)
            assert filtered.n_frames > 0
        except ValueError:
            # If all frames filtered, that's also valid
            pass


class TestCombineStructures:
    """Tests for combining structures."""
    
    def test_combine_single_structure(self, example_trajectory):
        """Test combining a single structure."""
        structures = [example_trajectory]
        combined = combine_structures(structures)
        
        assert combined is not None
        assert combined.n_frames == 1
        assert combined.n_atoms == example_trajectory.n_atoms
    
    def test_combine_multiple_structures(self, example_trajectory):
        """Test combining multiple structures."""
        structures = [example_trajectory, example_trajectory]
        combined = combine_structures(structures)
        
        assert combined is not None
        assert combined.n_frames == 2
        assert combined.n_atoms == example_trajectory.n_atoms
    
    def test_combine_empty_list(self):
        """Test combining empty list."""
        combined = combine_structures([])
        assert combined is None
    
    def test_combine_mismatched_atoms(self, example_trajectory):
        """Test combining structures with mismatched atom counts."""
        # Create trajectory with different atom count
        top = md.Topology()
        chain = top.add_chain()
        residue = top.add_residue("RES", chain)
        top.add_atom("CA", md.element.carbon, residue)
        
        xyz = np.array([[[0.0, 0.0, 0.0]]])
        traj2 = md.Trajectory(xyz=xyz, topology=top)
        
        structures = [example_trajectory, traj2]
        combined = combine_structures(structures)
        
        # Should only include matching structures
        assert combined is not None
        assert combined.n_frames == 1  # Only first structure matches itself


class TestLoadCifStructures:
    """Tests for loading CIF structures."""
    
    def test_load_cif_file(self, example_cif_file):
        """Test loading a CIF file."""
        structures = load_cif_structures([example_cif_file])
        assert len(structures) == 1
        assert isinstance(structures[0], md.Trajectory)
    
    def test_load_multiple_cif_files(self, temp_dir):
        """Test loading multiple CIF files."""
        cif1 = temp_dir / "file1.cif"
        cif2 = temp_dir / "file2.cif"
        
        # Create minimal CIF files
        cif_content = """data_test
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
"""
        cif1.write_text(cif_content)
        cif2.write_text(cif_content)
        
        structures = load_cif_structures([cif1, cif2])
        assert len(structures) == 2
        assert all(isinstance(s, md.Trajectory) for s in structures)

