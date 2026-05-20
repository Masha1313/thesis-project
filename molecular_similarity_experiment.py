from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ZMPY3D as zm
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdShapeAlign, rdShapeHelpers
from scipy.stats import spearmanr


@dataclass
class ZernikeCache:
    max_order: int
    binomial_cache: object
    gcache_pqr_linear: object
    gcache_complex: object
    gcache_complex_index: object
    clm_cache_3d: object
    clm_cache: object
    s_id: object
    n: object
    l: object
    m: object
    mu: object
    k: object
    is_nlm_value: object


def load_zernike_cache(max_order: int = 6) -> ZernikeCache:
    cache_dir = Path(zm.__file__).with_name("cache_data")

    with (cache_dir / "BinomialCache.pkl").open("rb") as file:
        binomial_cache_pkl = pickle.load(file)

    with (cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl").open("rb") as file:
        cache_pkl = pickle.load(file)

    rotation_index = cache_pkl["RotationIndex"]

    return ZernikeCache(
        max_order=max_order,
        binomial_cache=binomial_cache_pkl["BinomialCache"],
        gcache_pqr_linear=cache_pkl["GCache_pqr_linear"],
        gcache_complex=cache_pkl["GCache_complex"],
        gcache_complex_index=cache_pkl["GCache_complex_index"],
        clm_cache_3d=cache_pkl["CLMCache3D"],
        clm_cache=cache_pkl["CLMCache"],
        s_id=np.squeeze(rotation_index["s_id"][0, 0]) - 1,
        n=np.squeeze(rotation_index["n"][0, 0]),
        l=np.squeeze(rotation_index["l"][0, 0]),
        m=np.squeeze(rotation_index["m"][0, 0]),
        mu=np.squeeze(rotation_index["mu"][0, 0]),
        k=np.squeeze(rotation_index["k"][0, 0]),
        is_nlm_value=np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1,
    )


def molecule_shape_features(mol: Chem.Mol) -> dict[str, float | str]:
    conf = mol.GetConformer()
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ]
    )

    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    l1, l2, l3 = eigenvalues
    eps = 1e-8

    elongation = l1 / (l3 + eps)
    rod_ratio = l1 / (l2 + eps)
    flat_ratio = l2 / (l3 + eps)

    if rod_ratio > 3 and flat_ratio < 2:
        shape_type = "elongated"
    elif rod_ratio < 2 and flat_ratio > 3:
        shape_type = "flat"
    elif rod_ratio < 2 and flat_ratio < 2:
        shape_type = "compact"
    else:
        shape_type = "mixed"

    return {
        "lambda1": l1,
        "lambda2": l2,
        "lambda3": l3,
        "elongation": elongation,
        "rod_ratio": rod_ratio,
        "flat_ratio": flat_ratio,
        "shape_type": shape_type,
    }


def create_hard_voxel_from_sdf(
    mol: Chem.Mol,
    cube_size: int = 128,
    radius_scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    pt = Chem.GetPeriodicTable()
    conf = mol.GetConformer()
    xyz = conf.GetPositions()

    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    center_mass = np.average(xyz, axis=0, weights=masses)
    xyz_centered = xyz - center_mass

    radii = np.array([pt.GetRvdw(atom.GetSymbol()) * radius_scale for atom in mol.GetAtoms()])
    max_extent = np.max(np.linalg.norm(xyz_centered, axis=1) + radii)

    voxel_cube = np.zeros((cube_size, cube_size, cube_size), dtype=np.float64)
    grid_center = cube_size // 2
    target_radius_vox = grid_center - 2
    scale = target_radius_vox / max_extent
    corner_xyz = np.array([-grid_center, -grid_center, -grid_center], dtype=np.float64)

    for i, atom in enumerate(mol.GetAtoms()):
        radius = pt.GetRvdw(atom.GetSymbol()) * radius_scale
        atom_pos_scaled = xyz_centered[i] * scale
        radius_scaled = radius * scale
        atom_grid_pos = np.round(atom_pos_scaled).astype(int) + grid_center

        r_vox = int(np.ceil(radius_scaled))
        z, y, x = np.ogrid[-r_vox : r_vox + 1, -r_vox : r_vox + 1, -r_vox : r_vox + 1]
        mask = (x**2 + y**2 + z**2) <= (radius_scaled**2)

        z_s = max(0, atom_grid_pos[0] - r_vox)
        z_e = min(cube_size, atom_grid_pos[0] + r_vox + 1)
        y_s = max(0, atom_grid_pos[1] - r_vox)
        y_e = min(cube_size, atom_grid_pos[1] + r_vox + 1)
        x_s = max(0, atom_grid_pos[2] - r_vox)
        x_e = min(cube_size, atom_grid_pos[2] + r_vox + 1)

        mask_z_s = z_s - (atom_grid_pos[0] - r_vox)
        mask_z_e = mask_z_s + (z_e - z_s)
        mask_y_s = y_s - (atom_grid_pos[1] - r_vox)
        mask_y_e = mask_y_s + (y_e - y_s)
        mask_x_s = x_s - (atom_grid_pos[2] - r_vox)
        mask_x_e = mask_x_s + (x_e - x_s)

        voxel_cube[z_s:z_e, y_s:y_e, x_s:x_e][
            mask[mask_z_s:mask_z_e, mask_y_s:mask_y_e, mask_x_s:mask_x_e]
        ] = 1.0

    return voxel_cube, corner_xyz


def calculate_zernike_descriptor(
    voxel3d: np.ndarray,
    corner: np.ndarray,
    grid_width: float,
    cache: ZernikeCache,
    param: dict[str, float],
) -> np.ndarray:
    dimension_bbox_scaled = voxel3d.shape
    xyz_sample_struct = {
        "X_sample": np.arange(dimension_bbox_scaled[0] + 1),
        "Y_sample": np.arange(dimension_bbox_scaled[1] + 1),
        "Z_sample": np.arange(dimension_bbox_scaled[2] + 1),
    }

    volume_mass, center, _ = zm.calculate_bbox_moment(voxel3d, 1, xyz_sample_struct)
    average_voxel_dist_2_center, _ = zm.calculate_molecular_radius(
        voxel3d,
        center,
        volume_mass,
        param["default_radius_multiplier"],
    )

    sphere_xyz_sample_struct = zm.get_bbox_moment_xyz_sample(
        center,
        average_voxel_dist_2_center,
        dimension_bbox_scaled,
    )
    _, _, sphere_bbox_moment = zm.calculate_bbox_moment(
        voxel3d,
        cache.max_order,
        sphere_xyz_sample_struct,
    )
    _, z_moment_raw = zm.calculate_bbox_moment_2_zm(
        cache.max_order,
        cache.gcache_complex,
        cache.gcache_pqr_linear,
        cache.gcache_complex_index,
        cache.clm_cache_3d,
        sphere_bbox_moment,
    )

    all_abs = []
    for order in range(2, cache.max_order + 1):
        all_abs.extend(zm.calculate_ab_rotation_all(z_moment_raw, order))

    ab_list_all = np.vstack(all_abs)
    zm_list_all = zm.calculate_zm_by_ab_rotation(
        z_moment_raw,
        cache.binomial_cache,
        ab_list_all,
        cache.max_order,
        cache.clm_cache,
        cache.s_id,
        cache.n,
        cache.l,
        cache.m,
        cache.mu,
        cache.k,
        cache.is_nlm_value,
    )

    zm_list_all = np.stack(zm_list_all, axis=3)
    zm_list_all = np.transpose(zm_list_all, (2, 1, 0, 3))
    zm_list_all = zm_list_all[~np.isnan(zm_list_all)]
    return zm_list_all.reshape(-1, ab_list_all.shape[0])


def calculate_zernike_similarity(zernike_a: np.ndarray, zernike_b: np.ndarray) -> tuple[float, tuple[int, int]]:
    a = np.asarray(zernike_a)
    b = np.asarray(zernike_b)

    norm_a = np.linalg.norm(a, axis=0, keepdims=True)
    norm_b = np.linalg.norm(b, axis=0, keepdims=True)
    dot = np.abs(a.conj().T @ b)
    cosine = dot / (norm_a.T @ norm_b)

    max_cos = float(np.max(cosine))
    idx_a, idx_b = np.unravel_index(np.argmax(cosine), cosine.shape)
    return max_cos, (int(idx_a), int(idx_b))


def compare_rdkit_3d_euclidean(mol1: Chem.Mol, mol2: Chem.Mol) -> tuple[float, float]:
    fp1 = rdMolDescriptors.GetUSR(mol1)
    fp2 = rdMolDescriptors.GetUSR(mol2)
    dist = float(np.linalg.norm(np.array(fp1) - np.array(fp2)))
    scaled_distance = dist / (1.0 + dist)
    return scaled_distance, dist


def filter_by_heavy_atoms(df: pd.DataFrame, min_atoms: int, max_atoms: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None and min_atoms <= mol.GetNumHeavyAtoms() <= max_atoms:
            rows.append(row)
    return pd.DataFrame(rows)


def collect_valid_3d_molecules(
    df_group: pd.DataFrame,
    target_n: int = 500,
    seed: int = 20,
) -> tuple[list[Chem.Mol], list[str], list[str]]:
    shuffled = df_group.sample(frac=1, random_state=seed).reset_index(drop=True)
    molecules = []
    chembl_ids = []
    smiles_list = []

    for _, row in shuffled.iterrows():
        if len(molecules) >= target_n:
            break

        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue

        try:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDG()
            params.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params) == -1:
                continue
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("_Name", str(row["chembl_id"]))

            molecules.append(mol)
            chembl_ids.append(row["chembl_id"])
            smiles_list.append(row["smiles"])
        except Exception:
            continue

    if len(molecules) < target_n:
        raise ValueError(f"Only {len(molecules)} valid 3D molecules collected, need {target_n}")

    return molecules, chembl_ids, smiles_list


def compute_voxels_for_molecules(
    molecules: list[Chem.Mol],
    cube_size: int = 64,
    radius_scale: float = 0.8,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], list[float], list[str]]:
    voxelcubes = []
    corners = []
    elongations = []
    rod_ratios = []
    flat_ratios = []
    shape_types = []

    for mol in molecules:
        voxel_cube, corner_xyz = create_hard_voxel_from_sdf(
            mol,
            cube_size=cube_size,
            radius_scale=radius_scale,
        )
        shape = molecule_shape_features(mol)

        voxelcubes.append(voxel_cube)
        corners.append(corner_xyz)
        elongations.append(shape["elongation"])
        rod_ratios.append(shape["rod_ratio"])
        flat_ratios.append(shape["flat_ratio"])
        shape_types.append(shape["shape_type"])

    return voxelcubes, corners, elongations, rod_ratios, flat_ratios, shape_types


def compute_zernike_for_all(
    voxelcubes: list[np.ndarray],
    corners: list[np.ndarray],
    grid_width: float,
    cache: ZernikeCache,
    param: dict[str, float],
) -> list[np.ndarray]:
    return [
        calculate_zernike_descriptor(voxel, corner, grid_width, cache, param)
        for voxel, corner in zip(voxelcubes, corners)
    ]


def ensure_3d(molecules: list[Chem.Mol]) -> list[Chem.Mol]:
    mol_list = []
    for mol in molecules:
        m = Chem.Mol(mol)
        m = Chem.AddHs(m, addCoords=True)
        if m.GetNumConformers() == 0:
            raise ValueError("Molecule has zero conformers")
        mol_list.append(m)
    return mol_list


def compute_one_vs_all(
    target_idx: int,
    molecules: list[Chem.Mol],
    zernike_descriptors: list[np.ndarray],
    m_list: list[Chem.Mol],
) -> pd.DataFrame:
    query_mol = molecules[target_idx]
    query_zm = zernike_descriptors[target_idx]
    rows = []

    for j in range(len(molecules)):
        if j == target_idx:
            continue

        z_similarity, _ = calculate_zernike_similarity(query_zm, zernike_descriptors[j])
        z_distance = 1.0 - z_similarity

        rdkit_scaled_distance, rdkit_raw_distance = compare_rdkit_3d_euclidean(query_mol, molecules[j])

        m1 = Chem.Mol(m_list[target_idx])
        m2 = Chem.Mol(m_list[j])
        rdShapeAlign.AlignMol(m2, m1)
        tanimoto_similarity = 1.0 - rdShapeHelpers.ShapeTanimotoDist(m1, m2)

        rows.append(
            {
                "target_idx": target_idx,
                "other_idx": j,
                "combined": z_distance + 0.1 * rdkit_raw_distance,
                "zernike": z_distance,
                "rdkit": rdkit_scaled_distance,
                "tanimoto": tanimoto_similarity,
            }
        )

    return pd.DataFrame(rows)


def top_for_tanimoto_topk(
    df_one: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    ascending: bool,
    k_values: list[int] | tuple[int, ...] = (1, 2, 5, 10),
) -> pd.DataFrame:
    df_m = df_one.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
    df_t = df_one.sort_values("tanimoto", ascending=False).reset_index(drop=True)
    metric_rank = {idx: pos for pos, idx in enumerate(df_m["other_idx"], start=1)}

    results = []
    for k_value in k_values:
        top_t = df_t.head(k_value)["other_idx"].tolist()
        positions = [metric_rank[idx] for idx in top_t]
        results.append(
            {
                "metric": metric_name,
                "tanimoto_top_k": k_value,
                "needed_top_n": max(positions),
                "positions_of_tanimoto_topk": positions,
            }
        )

    return pd.DataFrame(results)


def analyze_query(
    molecules: list[Chem.Mol],
    zernike_descriptors: list[np.ndarray],
    target_idx: int = 0,
) -> dict[str, object]:
    m_list = ensure_3d(molecules)
    df_one = compute_one_vs_all(target_idx, molecules, zernike_descriptors, m_list)

    coverage_zernike = top_for_tanimoto_topk(df_one, "zernike", "zernike", ascending=True)
    coverage_rdkit = top_for_tanimoto_topk(df_one, "rdkit", "rdkit", ascending=True)
    coverage_combined = top_for_tanimoto_topk(df_one, "combined", "combined", ascending=True)

    return {
        "df_one": df_one,
        "spearman_zernike": spearmanr(df_one["tanimoto"], df_one["zernike"]).statistic,
        "spearman_rdkit": spearmanr(df_one["tanimoto"], df_one["rdkit"]).statistic,
        "spearman_combined": spearmanr(df_one["tanimoto"], df_one["combined"]).statistic,
        "coverage_zernike": coverage_zernike,
        "coverage_rdkit": coverage_rdkit,
        "coverage_combined": coverage_combined,
    }


def run_one_group_experiment_multiple_queries(
    df: pd.DataFrame,
    min_atoms: int,
    max_atoms: int,
    cache: ZernikeCache,
    param: dict[str, float],
    target_n: int = 500,
    seed: int = 42,
    grid_width: float = 0.2,
    n_runs: int = 3,
) -> dict[str, object]:
    print(f"\nGroup {min_atoms}-{max_atoms} heavy atoms")
    df_group = filter_by_heavy_atoms(df, min_atoms, max_atoms)
    print(f"Rows in group before 3D filtering: {len(df_group)}")

    molecules, chembl_ids, smiles_list = collect_valid_3d_molecules(df_group, target_n, seed)
    print(f"Valid 3D molecules collected: {len(molecules)}")

    voxelcubes, corners, elongations, rod_ratios, flat_ratios, shape_types = compute_voxels_for_molecules(
        molecules,
        cube_size=64,
        radius_scale=0.8,
    )
    print("Voxelization done.")

    zernike_descriptors = compute_zernike_for_all(voxelcubes, corners, grid_width, cache, param)
    print("Zernike descriptors computed.")

    random.seed(seed)
    target_indices = random.sample(range(len(molecules)), n_runs)
    runs = {}

    for run_id, target_idx in enumerate(target_indices):
        print(f"\n--- Run {run_id} | target_idx = {target_idx} ---")
        analysis = analyze_query(molecules, zernike_descriptors, target_idx)
        print(f"Spearman (Tanimoto vs Zernike): {analysis['spearman_zernike']:.4f}")
        print(f"Spearman (Tanimoto vs RDKit):   {analysis['spearman_rdkit']:.4f}")
        print(f"Spearman (Tanimoto vs Combined): {analysis['spearman_combined']:.4f}")
        runs[run_id] = {"target_idx": target_idx, "analysis": analysis}

    return {
        "group_range": f"{min_atoms}-{max_atoms}",
        "molecules": molecules,
        "chembl_ids": chembl_ids,
        "smiles_list": smiles_list,
        "elongations": elongations,
        "rod_ratios": rod_ratios,
        "flat_ratios": flat_ratios,
        "shape_types": shape_types,
        "voxelcubes": voxelcubes,
        "corners": corners,
        "zernike_descriptors": zernike_descriptors,
        "runs": runs,
    }


def run_full_experiment_multiple_queries(
    csv_file: str = "chembl_raw_dump.csv",
    seed: int = 42,
    target_n: int = 500,
    grid_width: float = 0.2,
    n_runs: int = 3,
    groups: list[tuple[int, int]] | None = None,
    max_order: int = 6,
) -> dict[str, object]:
    df = pd.read_csv(csv_file)
    cache = load_zernike_cache(max_order=max_order)
    param = {"default_radius_multiplier": 1.6}
    groups = groups or [(15, 20), (20, 25), (25, 30)]

    return {
        f"{min_atoms}-{max_atoms}": run_one_group_experiment_multiple_queries(
            df=df,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            cache=cache,
            param=param,
            target_n=target_n,
            seed=seed,
            grid_width=grid_width,
            n_runs=n_runs,
        )
        for min_atoms, max_atoms in groups
    }


def run_multiple_seeds(
    csv_file: str = "chembl_raw_dump.csv",
    seeds: list[int] | tuple[int, ...] = (10, 20, 30),
    target_n: int = 500,
    grid_width: float = 0.2,
    n_runs: int = 3,
    groups: list[tuple[int, int]] | None = None,
    max_order: int = 6,
) -> dict[int, dict[str, object]]:
    all_seed_results = {}
    for seed in seeds:
        print(f"\n\nSeed {seed}")
        all_seed_results[seed] = run_full_experiment_multiple_queries(
            csv_file=csv_file,
            seed=seed,
            target_n=target_n,
            grid_width=grid_width,
            n_runs=n_runs,
            groups=groups,
            max_order=max_order,
        )
    return all_seed_results


def build_all_coverage_summary(all_seed_results: dict[int, dict[str, object]]) -> pd.DataFrame:
    rows = []

    for seed, results in all_seed_results.items():
        for group_name, group_data in results.items():
            for run_id, run_data in group_data["runs"].items():
                target_idx = run_data["target_idx"]
                analysis = run_data["analysis"]

                for cov_name in ["coverage_zernike", "coverage_rdkit", "coverage_combined"]:
                    cov_df = analysis[cov_name].copy()
                    cov_df["seed"] = seed
                    cov_df["group"] = group_name
                    cov_df["run_id"] = run_id
                    cov_df["target_idx"] = target_idx
                    rows.append(cov_df)

    return pd.concat(rows, ignore_index=True)


def build_correlation_summary(all_seed_results: dict[int, dict[str, object]]) -> pd.DataFrame:
    rows = []

    for seed, seed_results in all_seed_results.items():
        for group_name, group_data in seed_results.items():
            for run_id, run_data in group_data["runs"].items():
                target_idx = run_data["target_idx"]
                analysis = run_data["analysis"]
                rows.append(
                    {
                        "seed": seed,
                        "group": group_name,
                        "run_id": run_id,
                        "target_idx": target_idx,
                        "target_elongation": group_data["elongations"][target_idx],
                        "spearman_tanimoto_vs_zernike": analysis["spearman_zernike"],
                        "spearman_tanimoto_vs_rdkit": analysis["spearman_rdkit"],
                        "spearman_tanimoto_vs_combined": analysis["spearman_combined"],
                        "rod_ratio": group_data["rod_ratios"][target_idx],
                        "flat_ratio": group_data["flat_ratios"][target_idx],
                        "shape_type": group_data["shape_types"][target_idx],
                    }
                )

    return pd.DataFrame(rows)


def summarize_coverage(df_all_cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df_all_cov.copy()
    df["needed_top_n_all"] = df["positions_of_tanimoto_topk"].apply(max)
    df["needed_top_n_at_least_one"] = df["positions_of_tanimoto_topk"].apply(min)

    all_mean = coverage_mean(df, "needed_top_n_all")
    at_least_one_mean = coverage_mean(df, "needed_top_n_at_least_one")
    return df, all_mean, at_least_one_mean


def coverage_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    result = (
        df.groupby(["group", "metric", "tanimoto_top_k"], as_index=False)[value_col]
        .agg(mean_needed_top_n="mean", std_needed_top_n="std")
    )
    result["mean_needed_top_n"] = result["mean_needed_top_n"].round(2)
    result["std_needed_top_n"] = result["std_needed_top_n"].round(2)
    return result


def coverage_pretty(mean_df: pd.DataFrame) -> pd.DataFrame:
    pretty = mean_df.pivot(
        index=["group", "tanimoto_top_k"],
        columns="metric",
        values="mean_needed_top_n",
    ).reset_index()
    pretty.columns.name = None
    return pretty[["group", "tanimoto_top_k", "rdkit", "zernike", "combined"]]


def atom_difference(mol1: Chem.Mol, mol2: Chem.Mol) -> int:
    return abs(mol1.GetNumHeavyAtoms() - mol2.GetNumHeavyAtoms())


def compute_pairwise_metrics(
    molecules: list[Chem.Mol],
    zernike_descriptors: list[np.ndarray],
    max_atom_diff: int | None = None,
) -> pd.DataFrame:
    m_list = ensure_3d(molecules)
    rows = []

    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            atom_diff = atom_difference(molecules[i], molecules[j])
            if max_atom_diff is not None and atom_diff > max_atom_diff:
                continue

            z_similarity, _ = calculate_zernike_similarity(zernike_descriptors[i], zernike_descriptors[j])
            z_distance = 1.0 - z_similarity
            rdkit_scaled_distance, rdkit_raw_distance = compare_rdkit_3d_euclidean(molecules[i], molecules[j])

            m1 = Chem.Mol(m_list[i])
            m2 = Chem.Mol(m_list[j])
            rdShapeAlign.AlignMol(m2, m1)
            tanimoto_similarity = 1.0 - rdShapeHelpers.ShapeTanimotoDist(m1, m2)

            rows.append(
                {
                    "pair_id": f"{i}_{j}",
                    "i": i,
                    "j": j,
                    "zernike": z_distance,
                    "rdkit": rdkit_scaled_distance,
                    "rdkit_raw_distance": rdkit_raw_distance,
                    "tanimoto": tanimoto_similarity,
                    "combined": z_distance + 0.1 * rdkit_raw_distance,
                    "atom_diff": atom_diff,
                    "pair_size": max(molecules[i].GetNumHeavyAtoms(), molecules[j].GetNumHeavyAtoms()),
                }
            )

    return pd.DataFrame(rows)


def pairwise_spearman_summary(df_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["zernike", "rdkit", "combined"]:
        corr = df_pairs[metric].corr(df_pairs["tanimoto"], method="spearman")
        rows.append({"metric": f"Tanimoto vs {metric}", "spearman": corr})
    return pd.DataFrame(rows)
