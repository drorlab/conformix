#!/usr/bin/env python3
import sys
import os
import argparse

def run_twisted(protein, beta_values_str, out_dir=None, pairs_str=None, exclude_residues=None,
                samples_per_beta=10, particles_per_run=5, ess_threshold=1/3, twist_full_sequence=False):
    """
    Generate and run twisted RMSD calculation commands.
    
    Args:
        protein: Protein name (e.g., "mor-cropped")
        pairs_str: String of comma-separated residue pairs (e.g., "1,50 50,100 100,150")
        beta_values_str: String of comma-separated beta values (e.g., "1,2,3,4,5,6,7,8,9,10")
        out_dir: Optional output directory
    """
    # Set default output directory if not provided
    if out_dir is None:
        out_dir = f"proteins/{protein}/boltz_outputs/rmsd_twisting/"
    
    samples_per_beta = int(samples_per_beta)
    particles_per_run = int(particles_per_run)
    repeats_per_beta = samples_per_beta / particles_per_run

    if pairs_str:
        # Parse pairs from string (format: "1,50 50,100 100,150")
        pairs = []
        for pair in pairs_str.split():
            res1, res2 = pair.split(',')
            pairs.append((res1, res2))
    
        # Run command for each residue pair
        for res1, res2 in pairs:
            cmd = (
                f"python src/boltz/run_twisted.py predict "
                f"proteins/{protein}/{protein}.fasta "
                f"--out_dir {out_dir} "
                f"--input_cif proteins/{protein}/boltz_outputs/boltz_results_{protein}/predictions/{protein}/{protein}_model_0.cif "
                f"--beta_values {beta_values_str} "
                f"{('--twist_rmsd_full_sequence' if twist_full_sequence else '--twist_rmsd')} "
                f"--twist_residue1 {res1} "
                f"--twist_residue2 {res2} "
                f"--exclude_residues {exclude_residues} "
                f"--ess_threshold {ess_threshold} "
                f"--override "
            )
            print(f"Running for residues {res1}-{res2}...")
            print(cmd)
            for i in range(int(repeats_per_beta)): 
                os.system(cmd)
    
    else:
        # Run command for all residues
        cmd = (
            f"python src/boltz/run_twisted.py predict "
            f"proteins/{protein}/{protein}.fasta "
            f"--out_dir {out_dir} "
            f"--input_cif proteins/{protein}/boltz_outputs/boltz_results_{protein}/predictions/{protein}/{protein}_model_0.cif "
            f"--beta_values {beta_values_str} "
            f"{('--twist_rmsd_full_sequence' if twist_full_sequence else '--twist_rmsd')} "
            f"--exclude_residues {exclude_residues} "
            f"--ess_threshold {ess_threshold} "
            f"--override "
        )
        print("Running for all residues...")
        print(cmd)
        for i in range(int(repeats_per_beta)):
            os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description='Run twisted RMSD calculations for protein residue pairs')
    parser.add_argument('protein', help='Protein name (e.g., "mor-cropped")')
    parser.add_argument('out_dir', help='Output directory for results')
    parser.add_argument('beta_values', help='Comma-separated beta values (e.g., "1,2,3,4,5,6,7,8,9,10")')
    parser.add_argument('--pairs', help='Space-separated residue pairs (e.g., "1,50 50,100 100,150")', default=None)
    parser.add_argument('--exclude_residues', help="Residues to exclude from the twisting region. Comma-separated list of residue numbers or ranges. \
        Examples: '100' (single residue), '50-55' (range), '100,50-55,105-110' (multiple specifications).", default=None)
    parser.add_argument('--samples_per_beta', help='Number of samples per beta value', default=10)
    parser.add_argument('--particles_per_run', help='Number of diffusion samples, i.e. particles in TDS', default=5)
    parser.add_argument('--ess_threshold', help='controls resampling', default=1/3)
    parser.add_argument('--twist_full_sequence', action='store_true', help='Run for the full sequence')

    
    args = parser.parse_args()
    
    run_twisted(args.protein, args.beta_values, args.out_dir, args.pairs, args.exclude_residues, args.samples_per_beta, args.particles_per_run, args.ess_threshold, args.twist_full_sequence)

if __name__ == "__main__":
    main()
