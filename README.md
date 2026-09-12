# Machine Learning-Based Identification of HSA Binding Poses

This repository implements a machine-learning workflow that improves binding-pose identification for human serum albumin (HSA) using dynamic-stability features obtained from short molecular dynamics (MD) simulations. Features such as contact persistence, distance fluctuations, and interaction-energy fluctuations in MD trajectories are quantified and combined with docking scores to distinguish crystal-like poses from false-positive poses.

## Related Publication

This repository is the public implementation of the HSA binding-pose identification workflow reported in the following article:

Kourogi Y, et al. *Dynamic stability–driven machine learning improves binding pose identification on human serum albumin*. Journal of Computer-Aided Molecular Design 40, 118 (2026). https://doi.org/10.1007/s10822-026-00824-3

## Archived Runtime Outputs

Complete runtime outputs from the verified v1.0.3 example execution are available from Zenodo: https://doi.org/10.5281/zenodo.22721650

This repository contains Python code for running the workflow. It was reconstructed as an easy-to-run command-line workflow based on a series of Jupyter Notebooks developed during the research. Codex (GPT-5.6 Sol) was used for code development and debugging, and the developers performed test runs.

## Support Policy

This repository is provided as is for reproducing the research results. Individual support and ongoing maintenance are not guaranteed.

## Prerequisites

The conda environment is named `hsa-md`. Create and activate it from the repository root as follows:

```bash
conda env create -f environment.yml
conda activate hsa-md
```

`environment.yml` specifies the Python packages and Amber command-line tools required by the workflow, including OpenMM, PDBFixer, AmberTools, Open Babel, Hydride, RDKit, MDAnalysis, ProLIF, and LightGBM.

The GNINA executable is not distributed through conda-forge, bioconda, or GNINA's official conda channel. This workflow therefore uses the official prebuilt GNINA v1.3.2 binary. The CUDA runtime libraries required to launch the binary are included in `environment.yml`. Download the binary from the official release, verify its SHA-256 checksum, and install it into the environment as follows:

```bash
curl -fL https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 \
  -o gnina.1.3.2
echo "5d33538324b40050a03aa262d51832837e0ea6cc100945abbd2d7b732589690e  gnina.1.3.2" \
  | sha256sum -c -
install -m 755 gnina.1.3.2 "$CONDA_PREFIX/bin/gnina"
patchelf --set-rpath '$ORIGIN/../lib' "$CONDA_PREFIX/bin/gnina"
"$CONDA_PREFIX/bin/gnina" --version
```

`patchelf` configures a path relative to the current environment so that the official GNINA binary can locate the CUDA runtime libraries installed in that conda environment. This setting applies only to GNINA and does not modify `LD_LIBRARY_PATH`.

The workflow gives priority to the GNINA executable installed in the active Python environment and automatically configures the Open Babel data directory from the same environment. Therefore, no additional environment-variable settings are normally required even if another GNINA executable is present on `PATH`.

GNINA always runs on the CPU with `--no_gpu`. In contrast, OpenMM uses `--platform CUDA` by default for MD and automatically falls back to the CPU when CUDA is unavailable. Thus, the standard configuration on a CUDA-capable system is GNINA on the CPU and MD on the GPU. The prebuilt GNINA binary still requires CUDA runtime libraries merely to launch, so these libraries are included in `environment.yml`.

The default number of GNINA CPU threads is a conservative, processor-vendor-independent value: half of the logical CPUs available to the process, capped at eight. In addition, because GNINA's CPU CNN evaluation may use all logical CPUs despite the `--cpu` setting, as described in [official Issue #314](https://github.com/gnina/gnina/issues/314), the workflow sets `OMP_NUM_THREADS` and `OMP_THREAD_LIMIT` to the same value as `--cpu`. On job schedulers and shared computing systems, explicitly specify the number of CPUs allocated to the job.

```bash
python hsa_docking_md.py --stage dock --cpus 4
```

Changing `--cpus` does not change the blind-docking search space, `--exhaustiveness 64`, the number of generated poses, or the random seed. Reducing the number of CPU threads increases the computation time.

> [!WARNING]
> GNINA v1.3.2 cannot currently run on the GPU with GeForce RTX 50-series (Blackwell) cards. [Official Issue #330](https://github.com/gnina/gnina/issues/330), which reports `no kernel image is available for execution on the device` on an RTX 5090, and [official Issue #340](https://github.com/gnina/gnina/issues/340), concerning source builds for Blackwell, both remained unresolved as of September 12, 2026. CPU execution is available, and this workflow fixes GNINA to CPU execution. This restriction and CPU setting apply only to GNINA; OpenMM MD runs on CUDA by default. A compatible NVIDIA driver must be installed separately to use CUDA with OpenMM.

After installation, verify the required commands and Python packages as follows:

```bash
for cmd in gnina obabel antechamber parmchk2 tleap cpptraj ante-MMPBSA.py MMPBSA.py; do command -v "$cmd" || exit 1; done
python -c "import openmm, pdbfixer, MDAnalysis, prolif, lightgbm, biotite, hydride, dimorphite_dl, rdkit; print('environment OK')"
```

Run all commands from the repository root. The default `final_model/` and `runs/` paths are relative to the repository root.

## Running the Complete Workflow

The default value of `--stage` is `all`. The following command runs `prepare`, `dock`, `md`, `features`, and `predict` sequentially:

```bash
python hsa_docking_md.py --stage all
```

The default input is chain A of apo HSA `1AO6` with ibuprofen:

```text
SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
```

## Using Custom Input

Use `--pdb-id`, `--chain`, `--smiles`, and `--ligand-name` to change the receptor and ligand. Because `--ligand-name` is also used in the output-directory name, specify a short name consisting of letters, numbers, periods, hyphens, and underscores.

For example, run aspirin against chain A of apo HSA `1AO6` as follows:

```bash
python hsa_docking_md.py \
  --pdb-id 1AO6 \
  --chain A \
  --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \
  --ligand-name aspirin
```

For other inputs, replace the PDB ID, chain ID, SMILES, and ligand name above.

## Main Processing Steps

1. Prepare the receptor at pH 7.4 with PDBFixer. Missing terminal residues are not modeled; only missing internal residues are modeled.
2. Determine the ligand species at pH 7.4 with Dimorphite-DL and generate a 3D structure with Open Babel.
3. Generate ten poses with GNINA and select one pose for MD according to the specified criterion.
4. Restore the pH 7.4 bond orders and charges onto the heavy-atom coordinates output by GNINA, then add hydrogens. Hydride is used automatically if validation of Open Babel's hydrogen-completion result fails.
5. Build the system with GAFF2, ff19SB, TIP3P, and 0.15 M NaCl, then run equilibration (0.5 ns) and production MD (5 ns) with OpenMM.
6. Predict `label_2p5` and `label_5p0` with both `final_model/lgb_ds`, which uses only three GNINA score columns, and `final_model/lgb_ds_md`, which uses the GNINA scores plus MD features, for 65 columns in total.

## Pose Selection

By default, the pose with the highest `cnn_pose_score` is selected. Pose selection is part of the `dock` stage, so it can be specified both during a complete run with `--stage all` and during a staged run with `--stage dock`.

`--pose-selection` accepts the following three values:

| Value | Selection method |
|---|---|
| `cnn_pose_score` | Select the pose with the highest `CNNscore` (default) |
| `vina_affinity` | Select the pose with the lowest `minimizedAffinity` |
| `number` | Select the pose number specified with `--pose-number` |

The number of poses generated by GNINA is specified with `--num-poses` and defaults to ten. When using `number`, `--pose-number` must be between 1 and `--num-poses`.

```bash
# Select the pose with the best (lowest) Vina affinity during a complete run
python hsa_docking_md.py --pose-selection vina_affinity

# Use prepared input and explicitly select pose 3 from poses 1--10 in the dock stage
python hsa_docking_md.py --stage dock \
  --pose-selection number --pose-number 3
```

The `md`, `features`, and `predict` stages do not perform pose selection. They use `docking/selected_pose_ph74.sdf`, saved by the preceding `dock` stage. To change the selection criterion, rerun from `dock` before running the subsequent stages.

## Running Individual Stages

```bash
python hsa_docking_md.py --stage prepare
python hsa_docking_md.py --stage dock
python hsa_docking_md.py --stage md
python hsa_docking_md.py --stage features
python hsa_docking_md.py --stage predict
```

Output is written to `runs/<PDB_ID>_<ligand_name>/`. The `runs/` directory is generated at runtime and is excluded from version control.
To reduce storage requirements, the original MD trajectories are saved as `equilibration.xtc` and `production.xtc` rather than DCD files. A water- and ion-stripped analysis trajectory, `production_nowat.nc`, is also generated.
Prediction results from both models are stored under `models` in `prediction/prediction.json` and as a model-wise table in `prediction/prediction.csv`.

## Third-Party Notices

Parts of the MD execution workflow were adapted from [making-it-rain](https://github.com/pablo-arantes/making-it-rain) and modified and extended from its Google Colab-oriented implementation for local execution. See `THIRD_PARTY_NOTICES.md` for license and copyright notices.

## License

The code in this repository is provided under the [MIT License](LICENSE). See `THIRD_PARTY_NOTICES.md` for third-party license notices.
