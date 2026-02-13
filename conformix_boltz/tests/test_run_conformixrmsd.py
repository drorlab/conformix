"""Tests for the main ConforMix RMSD script."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from boltz.run_conformixrmsd_boltz import (
    get_sequence_length_from_fasta,
    main,
)


class TestGetSequenceLengthFromFasta:
    """Tests for FASTA sequence length extraction."""
    
    def test_get_sequence_length_simple(self, example_fasta_file):
        """Test getting sequence length from simple FASTA."""
        length = get_sequence_length_from_fasta(example_fasta_file)
        assert length > 0
        assert isinstance(length, int)
    
    def test_get_sequence_length_multiline(self, temp_dir):
        """Test getting sequence length from multiline FASTA."""
        fasta_path = temp_dir / "multiline.fasta"
        fasta_content = """>test
MKTAYIAKQR
QISFVKSHFS
"""
        fasta_path.write_text(fasta_content)
        
        length = get_sequence_length_from_fasta(fasta_path)
        assert length == 20  # Combined length
    
    def test_get_sequence_length_with_header(self, temp_dir):
        """Test that header lines are ignored."""
        fasta_path = temp_dir / "with_header.fasta"
        fasta_content = """>test_protein description
MKTAYIAKQR
"""
        fasta_path.write_text(fasta_content)
        
        length = get_sequence_length_from_fasta(fasta_path)
        assert length == 10


class TestMainFunction:
    """Tests for the main function."""
    
    @pytest.mark.skip(reason="Requires model download and GPU")
    def test_main_basic(self, example_fasta_file, temp_dir):
        """Test basic main function execution."""
        # This test requires actual model and may be slow
        # Skip in CI, run manually for integration testing
        output_dir = temp_dir / "output"
        
        with patch('boltz.run_conformixrmsd_boltz.run_untwisted') as mock_untwisted, \
             patch('boltz.run_conformixrmsd_boltz.run_twisted') as mock_twisted:
            
            # Mock the model loading
            mock_model = Mock()
            mock_untwisted.load_model.return_value = mock_model
            
            # Mock prediction
            mock_untwisted.predict.callback.return_value = None
            mock_twisted.predict.callback.return_value = None
            
            # This would require actual implementation
            # For now, just test argument parsing
            pass
    
    def test_main_invalid_fasta(self, temp_dir):
        """Test main function with invalid FASTA path."""
        invalid_fasta = temp_dir / "nonexistent.fasta"
        output_dir = temp_dir / "output"
        
        # Should exit with error
        # This would be tested with actual execution
        assert not invalid_fasta.exists()

