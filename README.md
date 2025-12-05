https://github.com/user-attachments/assets/f82138e6-15f0-4724-82a0-93794eaee38a
# ConforMix ([arXiv link](https://arxiv.org/pdf/2512.03312v1))

https://github.com/user-attachments/assets/8472ebf1-7913-44e5-81d3-1bb5dcfa37c9

https://github.com/user-attachments/assets/ca830613-af60-4d5b-bc81-5dc6361b91a4

https://github.com/user-attachments/assets/9e9606ba-04fe-4cd0-9159-bc685a50361e

The above shows protein conformation samples generated from ConforMix-RMSD-Boltz. ConforMix is an inference-time enhanced sampling algorithm for biomolecular diffusion models. It uses [Twisted Diffusion Sampler](https://github.com/blt2114/twisted_diffusion_sampler) to generate samples conditioned on features of the structure. In one instantiation, ConforMix-RMSD, we scan for protein flexibility/motion by generating samples that are distinct (by RMSD) from the default prediction. 

For initial exploration, we recommend ConforMix-Boltz. 

## ConforMix-Boltz

ConforMix is flexible and offers a number of tunable parameters. To get started, try

```
git clone https://github.com/drorlab/conformix.git 
pip install conformix/conformix_boltz
python -m boltz.run_conformixrmsd_boltz \
--fasta_path conformix/examples/P0205.fasta \
--out_dir outputs/conformix_P0205 \
--num_twist_targets 10 \
--samples_per_target 2 \
--structured_regions_only
```

This will produce (1) the default Boltz-1 prediction for that FASTA file, (2) ConforMix samples ranging from 0 to 20Å RMSD away from the default prediction, (3) a filtered, sorted trajectory made from those samples that can be viewed in tools like VMD, PyMOL, or ChimeraX. 

By default, RMSD is computed only on regions that have secondary structure (alpha helices or beta sheets) in the default prediction. This avoids trivial sampling in which only loops move. They are still free to move and often do. For in-depth investigation of a protein, consider using an argument like
```
--subset_residues 1-50,57
```

that limits the RMSD calculation to the residue ranges specified. This can be combined  with the `--structured_regions_only` option. 

## ConforMix-BioEmu

We also provide an implementation of ConforMix-BioEmu. This implementation is less thoroughly tested. We recommend using it for targeted sampling, i.e. when you can provide some input constraints (in the form of a reference structure) and set of residues for RMSD measurement. 

To get started, try
```
git clone https://github.com/drorlab/conformix.git
pip install conformix/conformix_bioemu
python -m bioemu.sample \
--sequence DAYAQWLKDGGPSSGRPPPS \
--num_samples 50  \
--output_dir outputs/conformix_bioemu_trp_cage  \
--resample_start 30 \
--twist_rmsd True \
--untwisted_input   \
--rmsd_all_residues True \
--denoiser_type first_order_tds \
--twist_k 10 \
--beta 0.5 \
--batch-size 10
```

For more details on the method and testing, please see [our paper](https://arxiv.org/pdf/2512.03312v1). 
