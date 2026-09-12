#!/usr/bin/env python3
"""HSA--ligand docking, MD, feature extraction, and prediction workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS


DEFAULT_PDB_ID = "1AO6"
DEFAULT_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
WATER_AND_IONS = {"WAT", "HOH", "Na+", "Cl-", "K+", "MG", "Mg+"}


def run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Print and run an external command, stopping immediately on failure."""
    print("+", " ".join(map(str, command)))
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        tail = (result.stdout + "\n" + result.stderr)[-4000:]
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{tail}")
    if result.stdout.strip():
        print(result.stdout[-1500:])


def require_programs(names: list[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Required commands were not found: " + ", ".join(missing))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def case_name(pdb_id: str, ligand_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ligand_name).strip("_")
    return f"{pdb_id.upper()}_{safe or 'ligand'}"


def paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.output_dir).resolve() / case_name(args.pdb_id, args.ligand_name)
    return {
        "root": root,
        "input": root / "input",
        "prepared": root / "prepared",
        "docking": root / "docking",
        "md": root / "md",
        "features": root / "features",
        "prediction": root / "prediction",
        "manifest": root / "manifest.json",
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for key in ("root", "input", "prepared", "docking", "md", "features", "prediction"):
        p[key].mkdir(parents=True, exist_ok=True)


def fetch_pdb(pdb_id: str, destination: Path) -> None:
    if destination.exists():
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading PDB: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    except Exception as exc:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: {exc}") from exc


def filter_to_chain(raw_pdb: Path, chain_id: str, destination: Path) -> None:
    """Create a single-chain PDB while retaining SEQRES for PDBFixer gap detection."""
    kept: list[str] = []
    selected_seen = False
    for line in raw_pdb.read_text(errors="replace").splitlines(True):
        record = line[:6].strip()
        if record == "SEQRES":
            if line[11:12] == chain_id:
                kept.append(line)
            continue
        if record in {"ATOM", "HETATM", "ANISOU"}:
            if line[21:22] == chain_id:
                kept.append(line)
                selected_seen = True
            continue
        if record == "TER":
            if selected_seen:
                kept.append(line)
            continue
        if record == "SSBOND":
            if line[15:16] == chain_id and line[29:30] == chain_id:
                kept.append(line)
            continue
        if record in {"HEADER", "TITLE", "COMPND", "SOURCE", "EXPDTA", "REMARK", "CRYST1"}:
            kept.append(line)
    if not selected_seen:
        raise ValueError(f"No ATOM/HETATM records were found for chain {chain_id}.")
    destination.write_text("".join(kept) + "END\n")


def missing_residue_report(fixer) -> list[dict]:
    records = []
    chains = list(fixer.topology.chains())
    for (chain_index, index), residues in fixer.missingResidues.items():
        chain = chains[chain_index]
        records.append({
            "chain": chain.id,
            "insertion_index": int(index),
            "residues": list(residues),
            "is_terminal": index == 0 or index == len(list(chain.residues())),
        })
    return records


def residue_groups(pdb_path: Path) -> tuple[list[tuple[str, int, str]], dict[tuple[str, int, str], str], dict[tuple[str, int, str], set[str]], dict[tuple[str, int, str], np.ndarray]]:
    order: list[tuple[str, int, str]] = []
    resnames: dict[tuple[str, int, str], str] = {}
    atoms: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    sg: dict[tuple[str, int, str], np.ndarray] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key = (line[21:22], int(line[22:26]), line[26:27])
        if key not in atoms:
            order.append(key)
            resnames[key] = line[17:20].strip().upper()
        atom_name = line[12:16].strip()
        atoms[key].add(atom_name)
        if atom_name == "SG":
            sg[key] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return order, resnames, atoms, sg


def normalize_amber_residues(pdb_path: Path) -> list[list[int]]:
    """Assign Amber residue names from pH 7.4 His hydrogens and S--S distances."""
    order, resnames, atoms, sg = residue_groups(pdb_path)
    pairs: list[tuple[tuple[str, int, str], tuple[str, int, str]]] = []
    unused = set(sg)
    for first in order:
        if first not in unused:
            continue
        candidates = [second for second in unused if second != first and np.linalg.norm(sg[first] - sg[second]) < 2.3]
        if candidates:
            second = min(candidates, key=lambda x: np.linalg.norm(sg[first] - sg[x]))
            pairs.append((first, second))
            unused.discard(first)
            unused.discard(second)
    disulfide_residues = {res for pair in pairs for res in pair}
    rename: dict[tuple[str, int, str], str] = {}
    for key, names in atoms.items():
        if key in disulfide_residues and resnames.get(key) == "CYS":
            rename[key] = "CYX"
        elif resnames.get(key) != "HIS":
            continue
        elif "HD1" in names and "HE2" in names:
            rename[key] = "HIP"
        elif "HD1" in names:
            rename[key] = "HID"
        elif "HE2" in names:
            rename[key] = "HIE"
    rewritten = []
    for line in pdb_path.read_text().splitlines(True):
        if line.startswith(("ATOM  ", "HETATM")):
            key = (line[21:22], int(line[22:26]), line[26:27])
            if key in rename:
                line = line[:17] + f"{rename[key]:>3}" + line[20:]
        rewritten.append(line)
    pdb_path.write_text("".join(rewritten))
    return [[a[1], b[1]] for a, b in pairs]


def strip_pdb_hydrogens(source: Path, destination: Path) -> None:
    lines = []
    for line in source.read_text().splitlines(True):
        if line.startswith(("ATOM  ", "HETATM")) and line[76:78].strip().upper() == "H":
            continue
        lines.append(line)
    destination.write_text("".join(lines))


def prepare_receptor(args: argparse.Namespace, p: dict[str, Path]) -> None:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    raw = p["input"] / f"{args.pdb_id.upper()}.pdb"
    chain_pdb = p["input"] / f"{args.pdb_id.upper()}_chain_{args.chain}.pdb"
    receptor = p["prepared"] / "receptor_ph74.pdb"
    report_path = p["prepared"] / "receptor_preparation.json"
    fetch_pdb(args.pdb_id, raw)
    filter_to_chain(raw, args.chain, chain_pdb)
    fixer = PDBFixer(filename=str(chain_pdb))
    fixer.findMissingResidues()
    missing_before = missing_residue_report(fixer)
    for key in list(fixer.missingResidues):
        chain_index, index = key
        chain = list(fixer.topology.chains())[chain_index]
        if index == 0 or index == len(list(chain.residues())):
            del fixer.missingResidues[key]
    missing_internal = missing_residue_report(fixer)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    with receptor.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    disulfides = normalize_amber_residues(receptor)
    strip_pdb_hydrogens(receptor, p["prepared"] / "receptor_amber_noh.pdb")
    write_json(report_path, {
        "pdb_id": args.pdb_id.upper(), "chain": args.chain, "ph": 7.4,
        "missing_before": missing_before,
        "missing_internal_filled": missing_internal,
        "disulfide_pairs_pdb_residue_number": disulfides,
    })


def prepare_ligand(args: argparse.Namespace, p: dict[str, Path]) -> None:
    from dimorphite_dl import protonate_smiles

    ligand = p["prepared"] / "ligand_ph74.sdf"
    variants = protonate_smiles(args.smiles, ph_min=7.4, ph_max=7.4, precision=0.0, max_variants=1)
    if len(variants) != 1:
        raise RuntimeError("Could not determine a unique ligand species at pH 7.4.")
    protonated_smiles = variants[0]
    (p["prepared"] / "ligand_ph74.smi").write_text(protonated_smiles + "\n")
    run(["obabel", f"-:{protonated_smiles}", "-O", str(ligand), "--gen3d"])
    molecule = Chem.MolFromMolFile(str(ligand), removeHs=False)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise RuntimeError("Could not read the pH 7.4 ligand SDF.")
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    write_json(p["prepared"] / "ligand_preparation.json", {
        "input_smiles": args.smiles, "protonated_smiles": protonated_smiles, "ph": 7.4, "formal_charge": charge,
        "heavy_atoms": sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()),
    })


def validate_hydrogenated_sdf(path: Path, expected_charge: int) -> bool:
    molecule = Chem.MolFromMolFile(str(path), removeHs=False)
    return bool(
        molecule is not None
        and molecule.GetNumConformers() == 1
        and sum(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()) > 0
        and sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) == expected_charge
    )


def add_hydrogens_with_hydride(heavy_sdf: Path, output_sdf: Path) -> None:
    from biotite.structure.io.mol import SDFile, get_structure, set_structure
    from hydride.add import add_hydrogen

    sdf = SDFile.read(str(heavy_sdf))
    atoms = get_structure(sdf)
    hydrogenated, _ = add_hydrogen(atoms)
    set_structure(sdf, hydrogenated)
    sdf.write(str(output_sdf))


def restore_pose_protonation(raw_sdf: Path, prepared_sdf: Path, output_sdf: Path) -> str:
    """Restore pH 7.4 bonds, charges, and hydrogens onto GNINA heavy-atom coordinates."""
    template = Chem.MolFromMolFile(str(prepared_sdf), removeHs=False)
    raw = Chem.MolFromMolFile(str(raw_sdf), removeHs=False)
    if template is None or raw is None:
        raise RuntimeError("Could not read the SDF files required to protonate the pose.")
    template_heavy, raw_heavy = Chem.RemoveHs(template), Chem.RemoveHs(raw)
    if template_heavy.GetNumAtoms() != raw_heavy.GetNumAtoms():
        raise RuntimeError("The GNINA pose and input ligand have different heavy-atom counts.")
    mcs = rdFMCS.FindMCS([template_heavy, raw_heavy], atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact, timeout=30)
    query = Chem.MolFromSmarts(mcs.smartsString)
    template_match, raw_match = template_heavy.GetSubstructMatch(query), raw_heavy.GetSubstructMatch(query)
    if len(template_match) != template_heavy.GetNumAtoms() or len(raw_match) != raw_heavy.GetNumAtoms():
        raise RuntimeError("Could not obtain a complete atom mapping between the GNINA pose and input ligand.")
    template_xyz = np.array(template_heavy.GetConformer().GetPositions())[list(template_match)]
    raw_xyz = np.array(raw_heavy.GetConformer().GetPositions())[list(raw_match)]
    source_center, target_center = template_xyz.mean(0), raw_xyz.mean(0)
    left, _, right_t = np.linalg.svd((template_xyz - source_center).T @ (raw_xyz - target_center))
    rotation = left @ np.diag([1, 1, np.linalg.det(left @ right_t)]) @ right_t
    transformed = (np.array(template_heavy.GetConformer().GetPositions()) - source_center) @ rotation + target_center
    for template_index, raw_index in zip(template_match, raw_match):
        transformed[template_index] = np.array(raw_heavy.GetConformer().GetAtomPosition(raw_index))
    conformer = template_heavy.GetConformer()
    for index, xyz in enumerate(transformed):
        conformer.SetAtomPosition(index, xyz)
    heavy_sdf = output_sdf.with_name(output_sdf.stem + "_heavy.sdf")
    writer = Chem.SDWriter(str(heavy_sdf))
    writer.write(template_heavy)
    writer.close()
    expected_charge = int(sum(atom.GetFormalCharge() for atom in template.GetAtoms()))
    method = "openbabel"
    try:
        run(["obabel", str(heavy_sdf), "-O", str(output_sdf), "-h"])
        if not validate_hydrogenated_sdf(output_sdf, expected_charge):
            raise RuntimeError("The Open Babel output has an invalid hydrogen count or formal charge.")
    except Exception as openbabel_error:
        print(f"Open Babel hydrogen addition failed; falling back to Hydride: {openbabel_error}")
        add_hydrogens_with_hydride(heavy_sdf, output_sdf)
        method = "hydride"
        if not validate_hydrogenated_sdf(output_sdf, expected_charge):
            raise RuntimeError("SDF validation failed after adding hydrogens with Hydride.")
    heavy_sdf.unlink(missing_ok=True)
    return method


def pose_scores(docked_sdf: Path, docking_dir: Path) -> pd.DataFrame:
    molecules = [m for m in Chem.SDMolSupplier(str(docked_sdf), removeHs=False) if m is not None]
    if not molecules:
        raise RuntimeError("The GNINA output SDF contains no valid poses.")
    rows = []
    for index, molecule in enumerate(molecules, start=1):
        out = docking_dir / f"pose{index:02d}_raw.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(molecule)
        writer.close()
        props = molecule.GetPropsAsDict()
        rows.append({
            "pose_number": index,
            "pose_file": out.name,
            "vina_affinity": float(props.get("minimizedAffinity", np.nan)),
            "cnn_pose_score": float(props.get("CNNscore", np.nan)),
            "cnn_affinity": float(props.get("CNNaffinity", np.nan)),
        })
    return pd.DataFrame(rows)


def dock(args: argparse.Namespace, p: dict[str, Path]) -> None:
    receptor = p["prepared"] / "receptor_ph74.pdb"
    ligand = p["prepared"] / "ligand_ph74.sdf"
    if not receptor.exists() or not ligand.exists():
        raise FileNotFoundError("Run the prepare stage first.")
    docked = p["docking"] / "docked.sdf"
    cpu_threads = max(1, args.cpus)
    environment_gnina = Path(sys.prefix) / "bin" / "gnina"
    gnina_executable = str(environment_gnina) if environment_gnina.is_file() else "gnina"
    command = [
        gnina_executable, "-r", str(receptor), "-l", str(ligand),
        "--autobox_ligand", str(receptor), "--autobox_add", str(args.autobox_add),
        "--cpu", str(cpu_threads), "--num_modes", str(args.num_poses),
        "--exhaustiveness", str(args.exhaustiveness), "--seed", str(args.seed),
        "--no_gpu",
    ]
    command.extend(["-o", str(docked)])
    gnina_env = os.environ.copy()
    # GNINA's CPU CNN evaluation may use every logical CPU despite --cpu,
    # so apply the same limit to OpenMP.
    gnina_env["OMP_NUM_THREADS"] = str(cpu_threads)
    gnina_env["OMP_THREAD_LIMIT"] = str(cpu_threads)
    babel_root = Path(sys.prefix) / "share" / "openbabel"
    babel_data_dirs = sorted(
        path for path in babel_root.glob("*") if (path / "space-groups.txt").is_file()
    )
    if babel_data_dirs:
        gnina_env.setdefault("BABEL_DATADIR", str(babel_data_dirs[-1]))
    print(f"GNINA CPU threads: {cpu_threads} (OpenMP limited to the same value)")
    run(command, env=gnina_env)
    scores = pose_scores(docked, p["docking"])
    if args.pose_selection == "cnn_pose_score":
        selected = scores.loc[scores["cnn_pose_score"].idxmax()]
    elif args.pose_selection == "vina_affinity":
        selected = scores.loc[scores["vina_affinity"].idxmin()]
    else:
        selected_rows = scores[scores["pose_number"] == args.pose_number]
        if selected_rows.empty:
            raise ValueError(f"pose-number must be between 1 and {len(scores)}.")
        selected = selected_rows.iloc[0]
    raw_selected = p["docking"] / str(selected["pose_file"])
    selected_ph74 = p["docking"] / "selected_pose_ph74.sdf"
    hydrogen_method = restore_pose_protonation(raw_selected, p["prepared"] / "ligand_ph74.sdf", selected_ph74)
    scores["selected"] = scores["pose_number"].eq(int(selected["pose_number"]))
    scores.to_csv(p["docking"] / "gnina_scores.csv", index=False)
    write_json(p["docking"] / "selection.json", {
        "criterion": args.pose_selection, "pose_number": int(selected["pose_number"]),
        "selected_raw_sdf": raw_selected.name, "selected_ph74_sdf": selected_ph74.name,
        "hydrogen_completion": hydrogen_method,
    })


def ligand_charge(sdf: Path) -> int:
    molecule = Chem.MolFromMolFile(str(sdf), removeHs=False)
    if molecule is None:
        raise RuntimeError("RDKit could not read the selected pose.")
    return int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


def tleap_volume(log_path: Path) -> float:
    match = re.findall(r"Volume:\s*([0-9.Ee+-]+)", log_path.read_text(errors="replace"))
    if not match:
        raise RuntimeError("Could not obtain the solvent-box volume from the tleap output.")
    return float(match[-1])


def write_tleap_input(path: Path, receptor: Path, mol2: Path, frcmod: Path, disulfides: list[list[int]],
                      prmtop: Path, inpcrd: Path, padding: float, salt_pairs: int | None) -> None:
    lines = [
        "source leaprc.protein.ff19SB", "source leaprc.gaff2", "source leaprc.water.tip3p",
        f"loadamberparams {frcmod}", f"REC = loadpdb {receptor}",
    ]
    lines.extend([
        f"LIG = loadmol2 {mol2}", "SYS = combine { REC LIG }", "check SYS",
        "addIons2 SYS Na+ 0", "addIons2 SYS Cl- 0", f"solvatebox SYS TIP3PBOX {padding} 0.75",
    ])
    if salt_pairs is not None and salt_pairs > 0:
        lines.append(f"addIonsRand SYS Na+ {salt_pairs} Cl- {salt_pairs}")
    lines.extend([f"saveamberparm SYS {prmtop} {inpcrd}", "quit"])
    path.write_text("\n".join(lines) + "\n")


def validate_disulfides(prmtop: Path, expected_count: int) -> None:
    import parmed as pmd

    structure = pmd.load_file(str(prmtop))
    pairs = set()
    for bond in structure.bonds:
        first, second = bond.atom1, bond.atom2
        if first.name == "SG" and second.name == "SG" and first.residue.name == "CYX" and second.residue.name == "CYX":
            pairs.add(tuple(sorted((first.residue.idx, second.residue.idx))))
    if len(pairs) != expected_count:
        raise RuntimeError(
            f"Disulfide bond count mismatch: expected {expected_count}, found {len(pairs)}"
        )


def parameterize_and_solvate(args: argparse.Namespace, p: dict[str, Path]) -> tuple[Path, Path]:
    receptor = p["prepared"] / "receptor_amber_noh.pdb"
    selected = p["docking"] / "selected_pose_ph74.sdf"
    md = p["md"]
    mol2, frcmod = md / "ligand.mol2", md / "ligand.frcmod"
    charge = ligand_charge(selected)
    run(["antechamber", "-i", str(selected), "-fi", "sdf", "-o", str(mol2), "-fo", "mol2",
         "-c", "bcc", "-nc", str(charge), "-rn", "LIG", "-at", "gaff2"], cwd=md)
    run(["parmchk2", "-i", str(mol2), "-f", "mol2", "-o", str(frcmod), "-s", "gaff2"], cwd=md)
    report = load_json(p["prepared"] / "receptor_preparation.json")
    disulfides = report.get("disulfide_pairs_pdb_residue_number", [])
    prelim_top, prelim_crd = md / "prelim.prmtop", md / "prelim.inpcrd"
    prelim_in, prelim_log = md / "tleap_prelim.in", md / "tleap_prelim.log"
    write_tleap_input(prelim_in, receptor, mol2, frcmod, disulfides, prelim_top, prelim_crd, args.padding, None)
    with prelim_log.open("w") as handle:
        result = subprocess.run(["tleap", "-f", str(prelim_in)], cwd=md, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"The preliminary tleap run failed. Check {prelim_log}.")
    volume = tleap_volume(prelim_log)
    salt_pairs = max(0, round(volume * args.salt_molar * 6.02214076e-4))
    prmtop, inpcrd = md / "system.prmtop", md / "system.inpcrd"
    tleap_in = md / "tleap.in"
    write_tleap_input(tleap_in, receptor, mol2, frcmod, disulfides, prmtop, inpcrd, args.padding, salt_pairs)
    run(["tleap", "-f", str(tleap_in)], cwd=md)
    validate_disulfides(prmtop, len(disulfides))
    write_json(md / "system_setup.json", {"ligand_formal_charge": charge, "salt_molar": args.salt_molar,
                                            "preliminary_volume_A3": volume, "salt_pairs": salt_pairs,
                                            "disulfides": disulfides})
    return prmtop, inpcrd


def restrained_heavy_atoms(topology) -> list[int]:
    indices = []
    for atom in topology.atoms():
        symbol = atom.element.symbol if atom.element is not None else ""
        if atom.residue.name not in WATER_AND_IONS and symbol != "H":
            indices.append(atom.index)
    return indices


def add_restraints(system, positions, atom_indices: list[int], force_constant):
    import openmm as mm
    from openmm import unit

    force = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
    for name in ("k", "x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for index in atom_indices:
        xyz = positions[index].value_in_unit(unit.nanometer)
        force.addParticle(index, [force_constant, *xyz])
    system.addForce(force)


def platform_and_properties(requested: str):
    import openmm as mm

    try:
        platform = mm.Platform.getPlatformByName(requested)
        return platform, {"Precision": "mixed"} if requested in {"CUDA", "OpenCL"} else {}
    except Exception:
        print(f"{requested} is unavailable; using the CPU platform instead.")
        return mm.Platform.getPlatformByName("CPU"), {}


def simulation_from_system(topology, system, temperature, timestep, platform, properties):
    import openmm as mm
    from openmm import unit
    from openmm.app import Simulation

    integrator = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, timestep)
    integrator.setConstraintTolerance(1e-6)
    return Simulation(topology, system, integrator, platform, properties)


def run_md(args: argparse.Namespace, p: dict[str, Path]) -> None:
    import openmm as mm
    from openmm import unit
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile, HBonds, PME, PDBFile, StateDataReporter, XTCReporter

    prmtop_path, inpcrd_path = parameterize_and_solvate(args, p)
    md = p["md"]
    prmtop, inpcrd = AmberPrmtopFile(str(prmtop_path)), AmberInpcrdFile(str(inpcrd_path))
    temperature, pressure = args.temperature * unit.kelvin, args.pressure * unit.bar
    timestep = args.timestep_fs * unit.femtoseconds
    platform, properties = platform_and_properties(args.platform)
    base_kwargs = dict(nonbondedMethod=PME, nonbondedCutoff=1.0 * unit.nanometer,
                       constraints=HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)

    equil_system = prmtop.createSystem(**base_kwargs)
    equil_system.addForce(mm.MonteCarloBarostat(pressure, temperature))
    add_restraints(equil_system, inpcrd.positions, restrained_heavy_atoms(prmtop.topology),
                   args.restraint_kj_mol_nm2 * unit.kilojoule_per_mole / unit.nanometer**2)
    equil = simulation_from_system(prmtop.topology, equil_system, temperature, timestep, platform, properties)
    equil.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        equil.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    equil.minimizeEnergy(maxIterations=args.minimize_steps)
    equil.context.setVelocitiesToTemperature(temperature, args.seed)
    equil_steps = round(args.equil_ns * 1_000_000 / args.timestep_fs)
    report_steps = max(1, round(args.report_ps * 1000 / args.timestep_fs))
    equil.reporters.append(XTCReporter(str(md / "equilibration.xtc"), report_steps))
    equil.reporters.append(StateDataReporter(str(md / "equilibration.log"), report_steps,
                                             step=True, potentialEnergy=True, temperature=True, volume=True, separator="\t"))
    equil.step(equil_steps)
    state = equil.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    with (md / "equilibration_final.pdb").open("w") as handle:
        PDBFile.writeFile(prmtop.topology, state.getPositions(), handle)

    production_system = prmtop.createSystem(**base_kwargs)
    production_system.addForce(mm.MonteCarloBarostat(pressure, temperature))
    production = simulation_from_system(prmtop.topology, production_system, temperature, timestep, platform, properties)
    production.context.setPositions(state.getPositions())
    production.context.setVelocities(state.getVelocities())
    box = state.getPeriodicBoxVectors()
    production.context.setPeriodicBoxVectors(*box)
    production_steps = round(args.production_ns * 1_000_000 / args.timestep_fs)
    production.reporters.append(XTCReporter(str(md / "production.xtc"), report_steps))
    production.reporters.append(StateDataReporter(str(md / "production.log"), report_steps,
                                                  step=True, potentialEnergy=True, temperature=True, volume=True, separator="\t"))
    production.step(production_steps)
    final_state = production.context.getState(getPositions=True, enforcePeriodicBox=True)
    with (md / "production_final.pdb").open("w") as handle:
        PDBFile.writeFile(prmtop.topology, final_state.getPositions(), handle)
    write_json(md / "md_parameters.json", {"equil_ns": args.equil_ns, "production_ns": args.production_ns,
                                              "temperature_K": args.temperature, "pressure_bar": args.pressure,
                                              "timestep_fs": args.timestep_fs, "platform_requested": args.platform})
    strip_trajectory(p)


def strip_trajectory(p: dict[str, Path]) -> None:
    import parmed as pmd

    md = p["md"]
    topology = pmd.load_file(str(md / "system.prmtop"))
    topology.strip(":WAT,Na+,Cl-,K+,MG")
    topology.save(str(md / "system_nowat.prmtop"), overwrite=True)
    script = md / "strip.cpptraj"
    script.write_text(
        f"parm {md / 'system.prmtop'}\ntrajin {md / 'production.xtc'}\nautoimage\n"
        "strip :WAT,Na+,Cl-,K+,MG\ntrajout production_nowat.nc netcdf\nrun\n"
    )
    run(["cpptraj", "-i", str(script)], cwd=md)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    keys = ("mean", "std", "median", "min", "max", "p10", "p90", "last", "slope", "delta", "drift")
    if values.size == 0 or not np.isfinite(values).any():
        return {prefix + key: float("nan") for key in keys}
    finite = values[np.isfinite(values)]
    result = {
        prefix + "mean": float(np.mean(finite)), prefix + "std": float(np.std(finite)),
        prefix + "median": float(np.median(finite)), prefix + "min": float(np.min(finite)),
        prefix + "max": float(np.max(finite)), prefix + "p10": float(np.percentile(finite, 10)),
        prefix + "p90": float(np.percentile(finite, 90)), prefix + "last": float(finite[-1]),
    }
    result[prefix + "slope"] = float(np.polyfit(np.arange(finite.size), finite, 1)[0]) if finite.size > 1 else float("nan")
    half = finite.size // 2
    early, late = finite[:half], finite[half:]
    result[prefix + "delta"] = float(np.mean(late) - np.mean(early)) if early.size and late.size else float("nan")
    result[prefix + "drift"] = float(finite[-1] - np.mean(early)) if early.size else float("nan")
    return result


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_center, reference_center = mobile.mean(0), reference.mean(0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ np.diag([1, 1, np.linalg.det(left @ right_t)]) @ right_t
    return (mobile - mobile_center) @ rotation + reference_center


def ligand_selection(universe):
    groups = [res for res in universe.residues if res.resname.upper() == "LIG"]
    if len(groups) != 1:
        raise RuntimeError(f"Expected exactly one LIG residue, but found {len(groups)}.")
    return groups[0].atoms


def interaction_energy(universe, topology_path: Path, protein, ligand) -> tuple[np.ndarray, np.ndarray]:
    """Calculate pairwise LIG--protein interactions using Amber charges and LJ parameters."""
    import parmed as pmd
    from MDAnalysis.lib.distances import capped_distance

    topology = pmd.load_file(str(topology_path))
    charges = np.array([atom.charge for atom in topology.atoms])
    sigmas = np.array([atom.sigma for atom in topology.atoms])
    epsilons = np.array([atom.epsilon for atom in topology.atoms])
    p_idx, l_idx = protein.indices, ligand.indices
    eelec, evdw = [], []
    for _ in universe.trajectory:
        pairs, distances = capped_distance(protein.positions, ligand.positions, max_cutoff=12.0,
                                           box=universe.dimensions, return_distances=True)
        if not len(distances):
            eelec.append(0.0); evdw.append(0.0); continue
        pi, li = p_idx[pairs[:, 0]], l_idx[pairs[:, 1]]
        r = distances
        eelec.append(float(np.sum(332.06371 * charges[pi] * charges[li] / r)))
        sigma = (sigmas[pi] + sigmas[li]) / 2
        epsilon = np.sqrt(epsilons[pi] * epsilons[li])
        ratio6 = (sigma / r) ** 6
        evdw.append(float(np.sum(4 * epsilon * (ratio6**2 - ratio6))))
    return np.array(eelec), np.array(evdw)


def prolif_features(universe, ligand) -> dict[str, float]:
    from prolif import Fingerprint

    expected = {
        "md_prolif_interaction_entropy", "md_prolif_polar_ratio", "md_prolif_aromatic_ratio",
        "md_prolif_any_contact_count_mean", "md_prolif_any_contact_count_last", "md_prolif_hbond_freq",
        "md_prolif_ionic_freq", "md_prolif_hydrophobic_count_mean", "md_prolif_hbond_count_mean",
        "md_prolif_hbond_count_max", "md_prolif_hbond_count_last", "md_prolif_aromatic_count_median",
        "md_prolif_aromatic_count_max", "md_prolif_aromatic_count_p90", "md_prolif_aromatic_count_last",
        "md_prolif_ionic_count_mean", "md_prolif_ionic_count_max", "md_prolif_ionic_count_last",
        "md_prolif_vdw_count_mean", "md_prolif_vdw_count_last",
    }
    result = {key: 0.0 for key in expected}
    protein = universe.select_atoms("protein and byres around 14.0 group ligand", ligand=ligand)
    if not protein.n_atoms:
        return result
    fp = Fingerprint()
    fp.run(universe.trajectory[::10], ligand, protein)
    frame = fp.to_dataframe()
    if frame is None or frame.empty:
        raise RuntimeError("Could not create the ProLIF interaction table.")
    table = frame.astype(bool)
    any_count = table.sum(axis=1).to_numpy(float)
    result["md_prolif_any_contact_count_mean"] = float(any_count.mean())
    result["md_prolif_any_contact_count_last"] = float(any_count[-1])
    mapping = {"Hydrophobic": "hydrophobic", "HBAcceptor": "hbond", "HBDonor": "hbond",
               "PiStacking": "aromatic", "CationPi": "aromatic", "VdWContact": "vdw",
               "Cationic": "ionic", "Anionic": "ionic", "PiCation": "ionic"}
    counts = {name: np.zeros(len(table), dtype=float) for name in ("hydrophobic", "hbond", "aromatic", "ionic", "vdw")}
    if not isinstance(table.columns, pd.MultiIndex) or "interaction" not in table.columns.names:
        return result
    interaction_values = table.columns.get_level_values("interaction")
    presence = {}
    for interaction in interaction_values.unique():
        category = mapping.get(interaction)
        if category is None:
            continue
        values = table.loc[:, interaction_values == interaction].sum(axis=1).to_numpy(float)
        counts[category] += values
        presence[category] = max(presence.get(category, 0.0), float((values > 0).mean()))
    result["md_prolif_hbond_freq"] = presence.get("hbond", 0.0)
    result["md_prolif_ionic_freq"] = presence.get("ionic", 0.0)
    total = sum(float(value.mean()) for value in counts.values())
    if total:
        result["md_prolif_polar_ratio"] = (counts["hbond"].mean() + counts["ionic"].mean()) / total
        result["md_prolif_aromatic_ratio"] = counts["aromatic"].mean() / total
    freq = np.array(list(presence.values()), dtype=float)
    freq = freq[freq > 0]
    if freq.size > 1:
        probability = freq / freq.sum()
        result["md_prolif_interaction_entropy"] = float(-(probability * np.log(probability)).sum())
    for name, values in counts.items():
        if name == "hydrophobic":
            result["md_prolif_hydrophobic_count_mean"] = float(values.mean())
        elif name == "hbond":
            result.update({"md_prolif_hbond_count_mean": float(values.mean()), "md_prolif_hbond_count_max": float(values.max()), "md_prolif_hbond_count_last": float(values[-1])})
        elif name == "aromatic":
            result.update({"md_prolif_aromatic_count_median": float(np.median(values)), "md_prolif_aromatic_count_max": float(values.max()), "md_prolif_aromatic_count_p90": float(np.percentile(values, 90)), "md_prolif_aromatic_count_last": float(values[-1])})
        elif name == "ionic":
            result.update({"md_prolif_ionic_count_mean": float(values.mean()), "md_prolif_ionic_count_max": float(values.max()), "md_prolif_ionic_count_last": float(values[-1])})
        elif name == "vdw":
            result.update({"md_prolif_vdw_count_mean": float(values.mean()), "md_prolif_vdw_count_last": float(values[-1])})
    return result


def run_mmpbsa(p: dict[str, Path]) -> Path:
    md = p["md"]
    input_file = md / "mmpbsa.in"
    input_file.write_text("""&general
  interval=10, strip_mask=:WAT,Na+,Cl-,K+,MG,
/
&gb
  igb=2, saltcon=0.15,
/
&pb
  istrng=0.15, inp=2, radiopt=0, prbrad=1.4,
/
""")
    run(["ante-MMPBSA.py", "-p", str(md / "system.prmtop"), "-c", str(md / "complex.prmtop"),
         "-r", str(md / "receptor.prmtop"), "-l", str(md / "ligand.prmtop"),
         "-s", ":WAT,Na+,Cl-,K+,MG", "-n", ":LIG", "--radii", "mbondi2"], cwd=md)
    output = md / "FINAL_RESULTS_MMPBSA.dat"
    run(["MMPBSA.py", "-O", "-i", str(input_file), "-o", str(output), "-sp", str(md / "system.prmtop"),
         "-cp", str(md / "complex.prmtop"), "-rp", str(md / "receptor.prmtop"), "-lp", str(md / "ligand.prmtop"),
         "-y", str(md / "production.xtc")], cwd=md)
    return output


def parse_mmpbsa(result_path: Path) -> dict[str, float]:
    text = result_path.read_text(errors="replace")
    output: dict[str, float] = {}
    sections = (("GENERALIZED BORN:", "POISSON BOLTZMANN:", "md_mmgbsa_delta_total", "md_mmgbsa_std"),
                ("POISSON BOLTZMANN:", None, "md_mmpbsa_delta_total", "md_mmpbsa_std"))
    for start, end, mean_key, std_key in sections:
        section = text.split(start, 1)
        if len(section) == 1:
            continue
        body = section[1].split(end, 1)[0] if end else section[1]
        match = re.search(r"DELTA TOTAL\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", body)
        if match:
            output[mean_key], output[std_key] = map(float, match.groups())
    return output


def extract_features(args: argparse.Namespace, p: dict[str, Path]) -> pd.DataFrame:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array

    top, trajectory = p["md"] / "system_nowat.prmtop", p["md"] / "production_nowat.nc"
    if not top.exists() or not trajectory.exists():
        raise FileNotFoundError("Complete the md stage first.")
    universe = mda.Universe(str(top), str(trajectory))
    protein, ligand = universe.select_atoms("protein"), ligand_selection(universe)
    ca = universe.select_atoms("protein and name CA")
    if not protein.n_atoms or not ligand.n_atoms or not ca.n_atoms:
        raise RuntimeError("Could not obtain the protein, LIG, or CA atoms required for MD analysis.")
    universe.trajectory[0]
    ca_reference, ligand_reference = ca.positions.copy(), ligand.positions.copy()
    contact_resids = np.unique(protein.resids[(distance_array(protein.positions, ligand.positions).min(axis=1) < 5.0)])
    contact_atoms = universe.select_atoms("protein and resid " + " ".join(map(str, contact_resids))) if len(contact_resids) else protein
    rmsd_ca, rg, distance, rmsd_lig, contact_counts, contact_per_atom = [], [], [], [], [], []
    aligned_ca_frames = []
    unique_residues: set[int] = set()
    for _ in universe.trajectory:
        aligned_ca = kabsch(ca.positions, ca_reference)
        rmsd_ca.append(float(np.sqrt(np.mean((aligned_ca - ca_reference) ** 2))))
        mobile_center, ref_center = ca.positions.mean(0), ca_reference.mean(0)
        covariance = (ca.positions - mobile_center).T @ (ca_reference - ref_center)
        left, _, right_t = np.linalg.svd(covariance)
        rotation = left @ np.diag([1, 1, np.linalg.det(left @ right_t)]) @ right_t
        transformed_ligand = (ligand.positions - mobile_center) @ rotation + ref_center
        rmsd_lig.append(float(np.sqrt(np.mean((transformed_ligand - ligand_reference) ** 2))))
        aligned_ca_frames.append(aligned_ca)
        rg.append(float(protein.radius_of_gyration()))
        d = distance_array(contact_atoms.positions, ligand.positions, box=universe.dimensions)
        distance.append(float(d.min()))
        all_distances = distance_array(protein.positions, ligand.positions, box=universe.dimensions)
        count = int((all_distances < 4.5).sum())
        contact_counts.append(count)
        contact_per_atom.append(count / ligand.n_atoms)
        near = protein.resids[all_distances.min(axis=1) < 4.5]
        unique_residues.update(map(int, near))
    aligned_ca_array = np.asarray(aligned_ca_frames)
    rmsf = np.sqrt(np.mean(np.sum((aligned_ca_array - aligned_ca_array.mean(axis=0)) ** 2, axis=2), axis=0))
    eelec, evdw = interaction_energy(universe, top, protein, ligand)
    pd.DataFrame({"rmsd_ca_A": rmsd_ca, "radius_gyration_A": rg, "distance_A": distance,
                  "rmsd_lig_A": rmsd_lig, "eelec_kcal_mol": eelec, "evdw_kcal_mol": evdw}).to_csv(
                      p["features"] / "timeseries.csv", index=False)
    pd.DataFrame({"resid": ca.resids, "rmsf_ca_A": rmsf}).to_csv(p["features"] / "rmsf_ca.csv", index=False)
    values: dict[str, float] = {}
    values.update(summarize(np.array(rmsd_ca), "md_rmsd_ca_"))
    values.update(summarize(np.array(rg), "md_rg_"))
    values.update(summarize(np.array(distance), "md_distance_"))
    values.update(summarize(np.array(eelec), "md_eelec_"))
    values.update(summarize(np.array(evdw), "md_evdw_"))
    values.update(summarize(np.array(rmsd_lig), "md_rmsd_lig_"))
    values.update({"md_rmsf_protein_mean": float(rmsf.mean()), "md_rmsf_protein_max": float(rmsf.max()),
                   "md_rmsf_protein_std": float(rmsf.std())})
    values.update(summarize(np.array(contact_counts), "md_contact_"))
    values.update(summarize(np.array(contact_per_atom), "md_contact_per_atom_"))
    values["md_contact_unique_residues"] = float(len(unique_residues))
    values.update(prolif_features(universe, ligand))
    values.update(parse_mmpbsa(run_mmpbsa(p)))
    scores = pd.read_csv(p["docking"] / "gnina_scores.csv")
    selected = scores.loc[scores["selected"]].iloc[0]
    values.update({"vina_affinity": float(selected["vina_affinity"]), "cnn_pose_score": float(selected["cnn_pose_score"]),
                   "cnn_affinity": float(selected["cnn_affinity"])})
    feature_list = pd.read_csv(Path(args.md_model_dir) / "label_2p5_final" / "features.csv")["feature"].tolist()
    missing = [name for name in feature_list if name not in values]
    if missing:
        raise RuntimeError("Model features not implemented: " + ", ".join(missing))
    row = pd.DataFrame([{name: values[name] for name in feature_list}])
    nonfinite = row.columns[~np.isfinite(row.to_numpy(float)).all(axis=0)].tolist()
    if nonfinite:
        raise RuntimeError("Model features with no finite value: " + ", ".join(nonfinite))
    row.to_csv(p["features"] / "model_features.csv", index=False)
    return row


def predict(args: argparse.Namespace, p: dict[str, Path]) -> None:
    import lightgbm as lgb

    feature_path = p["features"] / "model_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Complete the features stage first.")
    features = pd.read_csv(feature_path)
    result = {"pdb_id": args.pdb_id.upper(), "ligand_name": args.ligand_name, "smiles": args.smiles,
              "models": {}}
    rows = []
    model_dirs = {"lgb_ds": Path(args.ds_model_dir), "lgb_ds_md": Path(args.md_model_dir)}
    for model_name, model_dir in model_dirs.items():
        model_result = {"model_dir": str(model_dir)}
        for target in ("label_2p5", "label_5p0"):
            directory = model_dir / f"{target}_final"
            order = pd.read_csv(directory / "features.csv")["feature"].tolist()
            missing = [name for name in order if name not in features.columns]
            if missing:
                raise RuntimeError(f"Features required by {model_name}/{target} are missing: " + ", ".join(missing))
            model = lgb.Booster(model_file=str(directory / "model.txt"))
            metrics = load_json(directory / "metrics.json")
            probability = float(model.predict(features[order], num_iteration=model.current_iteration())[0])
            threshold = float(metrics["bestF1_threshold"])
            prediction = int(probability >= threshold)
            model_result[target] = {
                "probability": probability, "threshold": threshold, "prediction": prediction,
            }
            rows.append({
                "pdb_id": args.pdb_id.upper(), "ligand_name": args.ligand_name, "smiles": args.smiles,
                "model": model_name, "target": target, "probability": probability,
                "threshold": threshold, "prediction": prediction,
            })
        result["models"][model_name] = model_result
    write_json(p["prediction"] / "prediction.json", result)
    pd.DataFrame(rows).to_csv(p["prediction"] / "prediction.csv", index=False)


def update_manifest(args: argparse.Namespace, p: dict[str, Path]) -> None:
    data = load_json(p["manifest"])
    data.pop("model_dir", None)
    data.update({"pdb_id": args.pdb_id.upper(), "chain": args.chain, "smiles": args.smiles,
                 "ligand_name": args.ligand_name,
                 "model_dirs": {"lgb_ds": str(args.ds_model_dir), "lgb_ds_md": str(args.md_model_dir)}})
    write_json(p["manifest"], data)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GNINA--MD--LightGBM workflow for HSA ligands")
    p.add_argument("--stage", choices=("prepare", "dock", "md", "features", "predict", "all"), default="all")
    p.add_argument("--pdb-id", default=DEFAULT_PDB_ID)
    p.add_argument("--chain", default="A")
    p.add_argument("--smiles", default=DEFAULT_SMILES)
    p.add_argument("--ligand-name", default="ibuprofen")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--md-model-dir", default="final_model/lgb_ds_md", help="Path to the lgb_ds_md model using MD features")
    p.add_argument("--ds-model-dir", default="final_model/lgb_ds", help="Path to the lgb_ds model using GNINA features only")
    p.add_argument("--pose-selection", choices=("cnn_pose_score", "vina_affinity", "number"),
                   default="cnn_pose_score", help="Criterion used to select the GNINA pose passed to MD")
    p.add_argument("--pose-number", type=int, default=1,
                   help="Pose number selected when --pose-selection is number")
    p.add_argument("--num-poses", type=int, default=10, help="Number of poses generated by GNINA")
    p.add_argument("--exhaustiveness", type=int, default=64)
    p.add_argument("--autobox-add", type=float, default=4.0)
    p.add_argument(
        "--cpus", type=int, default=default_cpu_count(),
        help="GNINA CPU threads (default: half the logical CPUs, capped at 8)",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--padding", type=float, default=12.0)
    p.add_argument("--salt-molar", type=float, default=0.15)
    p.add_argument("--equil-ns", type=float, default=0.5)
    p.add_argument("--production-ns", type=float, default=5.0)
    p.add_argument("--temperature", type=float, default=310.0)
    p.add_argument("--pressure", type=float, default=1.0)
    p.add_argument("--timestep-fs", type=float, default=2.0)
    p.add_argument("--report-ps", type=float, default=10.0)
    p.add_argument("--restraint-kj-mol-nm2", type=float, default=700.0)
    p.add_argument("--minimize-steps", type=int, default=20000)
    p.add_argument("--platform", choices=("CUDA", "OpenCL", "CPU"), default="CUDA")
    return p


def default_cpu_count() -> int:
    """Return a conservative GNINA default that avoids oversubscribing shared hosts."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return min(8, max(1, available // 2))


def main() -> None:
    args = parser().parse_args()
    args.pdb_id = args.pdb_id.upper()
    if len(args.pdb_id) != 4 or not re.fullmatch(r"[A-Z0-9]{4}", args.pdb_id):
        raise ValueError("--pdb-id must be a four-character PDB ID.")
    if args.pose_selection == "number" and not 1 <= args.pose_number <= args.num_poses:
        raise ValueError("--pose-number must be between 1 and --num-poses.")
    if args.cpus < 1:
        raise ValueError("--cpus must be at least 1.")
    require_programs(["gnina", "obabel", "antechamber", "parmchk2", "tleap", "cpptraj", "ante-MMPBSA.py", "MMPBSA.py"])
    p = paths(args)
    ensure_dirs(p)
    update_manifest(args, p)
    if args.stage in {"prepare", "all"}:
        prepare_receptor(args, p)
        prepare_ligand(args, p)
    if args.stage in {"dock", "all"}:
        dock(args, p)
    if args.stage in {"md", "all"}:
        run_md(args, p)
    if args.stage in {"features", "all"}:
        extract_features(args, p)
    if args.stage in {"predict", "all"}:
        predict(args, p)
    print(f"Completed: {p['root']}")


if __name__ == "__main__":
    main()
