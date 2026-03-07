# HSA Binding Pose Identification with Machine Learning

Machine-learning framework that incorporates dynamic stability features extracted from short molecular dynamics (MD) simulations to improve binding pose identification on human serum albumin (HSA). Time-dependent interaction patterns in MD trajectories—such as contact persistence, distance fluctuations, and interaction energy variability—are systematically quantified as machine-learnable features and combined with docking scores and ligand descriptors to discriminate crystal-like poses from false positives.

## Repository layout

**Data:**

- `data/curated/predictions/`: raw external validation predictions (9UPD)
- `data/curated/tables/`: SI table CSVs (machine-readable versions with extended columns)
  - `Table_S1.csv`: HSA–ligand complex structures used for training and external validation
  - `Table_S4A.csv`: cross-validation performance metrics (mean ± SD)
  - `Table_S4B.csv`: external validation performance metrics
  - `Table_S4C.csv`: pose-wise predicted probabilities in external validation (9UPD)
- `data/example/`: representative GAFF2 parameter files (frcmod and mol2), solvated AMBER topology, and tleap input template (1E7A pose01)

**Notebooks** (`notebooks/`):

- `01_add_descriptors.ipynb` – `09_shap_local_explanation.ipynb`: analysis pipeline (see `notebooks/README.md`)
- `gnina_docking_workflow.ipynb`: GNINA docking workflow (10 poses/complex, whole-protein autobox)
- `md_simulation_workflow.ipynb`: MD simulation workflow (OpenMM + AMBER ff19SB/GAFF2/TIP3P, 5 ns NPT)

**Not included** (due to file size): full MD trajectories, raw docking intermediates, GAFF2 parameters for all complexes. A representative trajectory for 1E7A pose01 is deposited in Zenodo.

## Pipeline overview

```
1. Receptor preparation       -> protein PDB (OpenMM Setup; localhost GUI, not included)
2. GNINA docking              -> notebooks/gnina_docking_workflow.ipynb
3. GAFF2 parameterization     -> AMBERTools (antechamber, parmchk2, tleap)
4. MD simulation              -> notebooks/md_simulation_workflow.ipynb
5. LigF descriptor calc.      -> notebooks/01_add_descriptors.ipynb
6. MD feature extraction      -> notebooks/02_add_md_features.ipynb
7. RMSD labeling              -> notebooks/03_add_pose_rmsd.ipynb
8. Feature selection          -> notebooks/04_feature_selection_global.ipynb
9. ML model training (LOPOCV) -> notebooks/05_lopocv_lgb_optuna.ipynb
10. Feature importance        -> notebooks/06_feature_importance_shap.ipynb
                                 notebooks/07_feature_importance_permutation.ipynb
11. Final model + external val.-> notebooks/08_external_validation.ipynb
12. SHAP local explanation     -> notebooks/09_shap_local_explanation.ipynb
```

External tools required for the upstream workflow:

- [GNINA](https://github.com/gnina/gnina) (docking)
- [AMBERTools >= 23](https://ambermd.org/GetAmber.php) (antechamber, tleap, cpptraj)
- [OpenBabel >= 3.1](https://openbabel.org/) (ligand protonation)

## Column name mapping

Some CSV files and notebooks use abbreviated prefixes that correspond to feature set names in the paper:

| Prefix | Paper notation |
|---|---|
| `ds` | DockF |
| `ds_desc` | DockF+LigF |
| `ds_md` | DockF+MDF |
| `ds_desc_md` | DockF+LigF+MDF |

## Setup

```bash
conda env create -f environment.yml
conda activate hsa-binding-pose-ml
```

## Citation

Code and curated datasets are archived in Zenodo. DOI will be added after deposition.

## Third-party notice

Parts of the MD workflow were adapted from [making-it-rain](https://github.com/pablo-arantes/making-it-rain). See `THIRD_PARTY_NOTICES.md` for details.

## License

MIT License (see `LICENSE`). Third-party notices in `THIRD_PARTY_NOTICES.md`.


