"""
test_benchmark.py

Unit tests for the ConforMix benchmarking suite.

Run with:
    pytest benchmarks/tests/test_benchmark.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from benchmarks.run_benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
    fetch_sequence_from_pdb,
    count_samples,
)
from benchmarks.evaluate_metrics import (
    MetricsResult,
    compute_rmsd_simple,
    extract_ca_coords_from_pdb,
    compute_metrics,
)
from benchmarks.generate_report import (
    generate_markdown_report,
    generate_html_report,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig class."""
    
    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        yaml_content = """
dataset_name: test_dataset
csv_path: datasets/test.csv
output_dir: outputs/test
num_twist_targets: 3
samples_per_target: 2
structured_regions_only: true
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)
        
        config = BenchmarkConfig.from_yaml(yaml_path)
        
        assert config.dataset_name == "test_dataset"
        assert config.num_twist_targets == 3
        assert config.samples_per_target == 2
        assert config.structured_regions_only is True
    
    def test_to_dict(self):
        """Test config serialization."""
        config = BenchmarkConfig(
            dataset_name="test",
            csv_path=Path("test.csv"),
            output_dir=Path("output"),
        )
        
        d = config.to_dict()
        
        assert d["dataset_name"] == "test"
        assert d["csv_path"] == "test.csv"
        assert "num_twist_targets" in d


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""
    
    def test_to_dict(self):
        """Test result serialization."""
        result = BenchmarkResult(
            system_id="test_system",
            pdb1="1ABC_A",
            pdb2="2XYZ_A",
            ground_truth_rmsd=5.0,
            success=True,
            runtime_seconds=120.5,
            output_dir=Path("/output/test"),
            num_samples_generated=10,
        )
        
        d = result.to_dict()
        
        assert d["system_id"] == "test_system"
        assert d["success"] is True
        assert d["runtime_seconds"] == 120.5
        assert d["num_samples_generated"] == 10


class TestRMSDComputation:
    """Tests for RMSD computation functions."""
    
    def test_compute_rmsd_simple_identical(self):
        """Test RMSD of identical coordinates is zero."""
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=float)
        
        rmsd = compute_rmsd_simple(coords, coords)
        
        assert rmsd < 1e-6
    
    def test_compute_rmsd_simple_translation(self):
        """Test RMSD is zero for translated coordinates (after alignment)."""
        coords1 = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=float)
        
        coords2 = coords1 + np.array([10, 20, 30])
        
        rmsd = compute_rmsd_simple(coords1, coords2)
        
        assert rmsd < 1e-6
    
    def test_compute_rmsd_simple_known_value(self):
        """Test RMSD computation with known result."""
        # Square in XY plane
        coords1 = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=float)
        
        # Distorted square - can't be aligned perfectly
        coords2 = np.array([
            [0, 0, 0],
            [1.5, 0, 0],  # Stretched
            [1.5, 1, 0],
            [0, 1, 0],
        ], dtype=float)
        
        rmsd = compute_rmsd_simple(coords1, coords2)
        
        # RMSD should be non-zero for distorted coordinates
        assert rmsd > 0.1


class TestPDBParsing:
    """Tests for PDB file parsing."""
    
    def test_extract_ca_coords(self, tmp_path):
        """Test CA coordinate extraction from PDB."""
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.326   1.548   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.992   2.826   0.000  1.00  0.00           C
END
"""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)
        
        coords = extract_ca_coords_from_pdb(pdb_path)
        
        assert coords is not None
        assert coords.shape == (2, 3)
        np.testing.assert_array_almost_equal(coords[0], [1.458, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords[1], [3.992, 2.826, 0.0])


class TestMetricsResult:
    """Tests for MetricsResult class."""
    
    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = MetricsResult(
            system_id="test",
            pdb1="1ABC_A",
            pdb2="2XYZ_A",
            ground_truth_rmsd=5.0,
            min_rmsd_to_alt=3.5,
            conformational_coverage=0.7,
        )
        
        d = metrics.to_dict()
        
        assert d["system_id"] == "test"
        assert d["min_rmsd_to_alt"] == 3.5
        assert d["conformational_coverage"] == 0.7


class TestReportGeneration:
    """Tests for report generation."""
    
    def test_generate_markdown_report(self, tmp_path):
        """Test Markdown report generation."""
        metrics = [
            {
                "system_id": "test1",
                "pdb1": "1ABC_A",
                "pdb2": "2XYZ_A",
                "ground_truth_rmsd": 5.0,
                "min_rmsd_to_alt": 3.5,
                "mean_rmsd_to_alt": 4.0,
                "conformational_coverage": 0.7,
                "num_samples": 10,
                "error_message": None,
            },
            {
                "system_id": "test2",
                "pdb1": "3DEF_A",
                "pdb2": "4GHI_A",
                "ground_truth_rmsd": 8.0,
                "min_rmsd_to_alt": None,
                "conformational_coverage": None,
                "num_samples": 0,
                "error_message": "Test error",
            },
        ]
        
        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        output_path = tmp_path / "report.md"
        report = generate_markdown_report(metrics_path, output_path)
        
        assert output_path.exists()
        assert "ConforMix Benchmark Report" in report
        assert "test1" in report
        assert "test2" in report
        assert "Error" in report
    
    def test_generate_html_report(self, tmp_path):
        """Test HTML report generation."""
        metrics = [
            {
                "system_id": "test1",
                "pdb1": "1ABC_A",
                "pdb2": "2XYZ_A",
                "ground_truth_rmsd": 5.0,
                "min_rmsd_to_alt": 3.5,
                "conformational_coverage": 0.7,
                "num_samples": 10,
                "error_message": None,
            },
        ]
        
        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        output_path = tmp_path / "report.html"
        report = generate_html_report(metrics_path, output_path)
        
        assert output_path.exists()
        assert "<!DOCTYPE html>" in report
        assert "test1" in report


class TestCountSamples:
    """Tests for sample counting."""
    
    def test_count_samples(self, tmp_path):
        """Test counting sample files."""
        (tmp_path / "sample1.cif").write_text("test")
        (tmp_path / "sample2.cif").write_text("test")
        (tmp_path / "sample3.pdb").write_text("test")
        (tmp_path / "other.txt").write_text("test")
        
        count = count_samples(tmp_path)
        
        assert count == 3


class TestIntegration:
    """Integration tests."""
    
    def test_dry_run_benchmark(self, tmp_path):
        """Test dry run of benchmark (no actual execution)."""
        csv_content = """system_id,pdb1,pdb2,RMSD,TM-score
test1,1ABC_A,2XYZ_A,5.0,0.8
test2,3DEF_A,4GHI_A,8.0,0.7
"""
        csv_path = tmp_path / "test_dataset.csv"
        csv_path.write_text(csv_content)
        
        config = BenchmarkConfig(
            dataset_name="test",
            csv_path=csv_path,
            output_dir=tmp_path / "output",
        )
        
        results = run_benchmark(config, dry_run=True)
        
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
