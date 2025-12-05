"""
run_conformixrmsd_boltz.py

This script provides a command-line interface to generate an initial conformation
and then run guided ("twisted") sampling based on user-defined settings for RMSD.

Example Usage:
1. Generate samples guided by RMSD to a generated reference (structured elements only):
   python run_conformixrmsd_boltz.py \
       --fasta_path your_protein.fasta \
       --out_dir ./boltz_output_structured

2. Generate samples guided by RMSD to a user-provided reference CIF, targeting specific residues only:
   python run_conformixrmsd_boltz.py \
       --fasta_path your_protein.fasta \
       --out_dir ./boltz_output_specified \
       --reference_cif ./my_reference.cif \
       --subset_residues "10-50,80-100" \
       --no-structured_regions_only

3. Generate more samples on the full sequence with a different target range and advanced settings:
   python run_conformixrmsd_boltz.py \
       --fasta_path your_protein.fasta \
       --out_dir ./boltz_output_advanced \
       --no-structured_regions_only \
       --twist_target_start 0 --twist_target_stop 5 --num_twist_targets 6 \
       --samples_per_target 5 \
       --twist_strength 20.0
"""
import argparse
import sys
from pathlib import Path
import numpy as np

from boltz import run_untwisted, run_twisted
from boltz.utils import cif_to_xtc

def get_sequence_length_from_fasta(fasta_path: Path) -> int:
    """Reads a FASTA file and returns the length of the first sequence."""
    seq = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return len(seq)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run guided sampling with Boltz using RMSD constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Inputs ---
    parser.add_argument("--fasta_path", type=Path, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to save all outputs.")

    # --- Guidance Settings ---
    parser.add_argument(
        "--structured_regions_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guide RMSD on structured regions (alpha-helices/beta-sheets) only. Use --no-structured_regions_only to guide on the full sequence backbone.",
    )
    parser.add_argument("--reference_cif", type=Path, default=None, help="Optional path to a reference CIF file for RMSD guidance. If not provided, one will be generated.")
    parser.add_argument(
        "--subset_residues",
        type=str,
        default=None,
        help="A comma-separated list of residue ranges (e.g., '10-20,45-55') to include in the RMSD calculation. If provided, this subset is used for the RMSD. Can be combined with --structured_regions_only.",
    )

    # --- Sampling Parameters ---
    parser.add_argument("--twist_target_start", type=float, default=0.0, help="Start value for the target RMSD in Angstroms.")
    parser.add_argument("--twist_target_stop", type=float, default=10.0, help="Stop value for the target RMSD.")
    parser.add_argument("--num_twist_targets", type=int, default=11, help="Number of target RMSD values to sample between start and stop (inclusive).")
    parser.add_argument("--samples_per_target", type=int, default=3, help="Number of structures to generate for each target value (sets 'diffusion_samples').")
    
    # --- Advanced Settings ---
    parser.add_argument("--twist_strength", type=float, default=15.0, help="Strength parameter for guidance.")
    parser.add_argument("--tstart", type=str, default="200", help="Timestep on which to start applying guidance.")
    parser.add_argument("--tstop", type=str, default="0", help="Timestep on which to stop applying guidance.")
    
    # --- Config ---
    parser.add_argument("--accelerator", type=str, default="gpu", help="Hardware accelerator to use ('gpu', 'cpu').")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--cache", type=str, default="~/.cache", help="Location for cache dir")

    args = parser.parse_args()

    # --- 1. Setup and Validation ---
    print("--- Settings ---")
    for key, value in vars(args).items():
        print(f"{key:<25}: {value}")
    print("----------------")
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.fasta_path.exists():
        print(f"Error: FASTA file not found at {args.fasta_path}")
        sys.exit(1)

    # --- 2. Load Model ---
    print("\nLoading Boltz model...")
    model_module = run_untwisted.load_model(cache=args.cache)
    print("Model loaded successfully.")

    # --- 3. Prepare Reference Structure ---
    reference_cif_path = args.reference_cif
    if not reference_cif_path:
        print("\nGenerating default Boltz prediction as reference...")
        ref_dir = args.out_dir / "default_reference"
        ref_dir.mkdir(exist_ok=True)
        
        run_untwisted.predict.callback(
            data=str(args.fasta_path),
            out_dir=str(ref_dir),
            model_module=model_module,
            diffusion_samples=1,
            output_format="mmcif",
            accelerator=args.accelerator,
            devices=args.devices,
            cache=args.cache,
        )
        
        # Find the generated file
        generated_files = list(ref_dir.rglob("*.cif"))
        if not generated_files:
            print("Error: Failed to generate default reference structure.")
            sys.exit(1)
        reference_cif_path = generated_files[0]
        print(f"Default reference saved to: {reference_cif_path}")
    else:
        if not reference_cif_path.exists():
            print(f"Error: Provided reference CIF not found at {reference_cif_path}")
            sys.exit(1)
        print(f"\nUsing user-provided reference: {reference_cif_path}")

    # --- 4. Prepare Parameters for Guided Sampling ---
    twist_target_values = np.linspace(args.twist_target_start, args.twist_target_stop, args.num_twist_targets).tolist()
    
    twisted_params = dict(
        data=str(args.fasta_path),
        out_dir=str(args.out_dir),
        model_module=model_module,
        diffusion_samples=args.samples_per_target,
        twist_target_values=twist_target_values,
        twist_strength_values=args.twist_strength,
        tstart_step=args.tstart,
        tstop_step=args.tstop,
        output_format="mmcif",
        accelerator=args.accelerator,
        devices=args.devices,
        cache=args.cache,
        input_cif=str(reference_cif_path),
        subset_residues=args.subset_residues,
        override=True,
    )

    # Control RMSD calculation scope
    if args.structured_regions_only:
        print("INFO: Guiding RMSD on structured regions (alpha-helices and beta-sheets).")
    else:
        print("INFO: Guiding RMSD on all backbone atoms in the full sequence.")
        twisted_params["twist_rmsd_full_sequence"] = True


    # --- 5. Run Guided Sampling ---
    print(f"\nStarting guided sampling with {len(twist_target_values)} target values...")
    print(f"Target RMSDs (Å): {[f'{v:.2f}' for v in twist_target_values]}")
    try:
        run_twisted.predict.callback(**twisted_params)
        print("\n✅ Guided sampling generation complete!")
        print(f"Results saved in: {args.out_dir}")
    except Exception as e:
        print(f"\n❌ An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 6. Filter generated samples ---
    print("f\nFiltering samples for physicality and sorting by Principal Component 1")
    filter_out = str(args.out_dir / "final_filtered")
    cif_to_xtc.process_all_cifs_to_single_output(
            parent_dir=args.out_dir,
            output_dir=filter_out)
    print("\n✅ Samples filtered and sorted!")
    print(f"Results saved in: {filter_out}")


if __name__ == "__main__":
    main()
