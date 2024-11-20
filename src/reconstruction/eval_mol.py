from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from analysis.smiles_main import convert_pyg_to_smiles
from analysis.rdkit_functions import check_valency, mol2smiles
import numpy as np
from fcd_torch import FCD


def calculate_tanimoto_similarity(smiles1, smiles2):
    # Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Generate fingerprints for the molecules
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

    # Calculate Tanimoto similarity
    tanimoto_similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return tanimoto_similarity



def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def calculate_similarity(fp1, fp2):
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def compare_datasets_tanimoto_similarity(train_smiles_list, recon_smiles_list, k):
    # Convert SMILES to fingerprints for both datasets
    train_fps = [smiles_to_fingerprint(smiles) for smiles in train_smiles_list]
    recon_fps = [smiles_to_fingerprint(smiles) for smiles in recon_smiles_list]

    top_k_similarities = []
    top_k_indices = []
    for recon_fp in recon_fps:
        similarities = [calculate_similarity(recon_fp, train_fp) for train_fp in train_fps]

        # Sort the similarities and select top-k smallest
        smallest_similarities = sorted(similarities, reverse=True)[:k]
        # Get the sorted indices of the similarities
        indices_of_largest = np.argsort(similarities)[::-1][:k]

        top_k_similarities.append(smallest_similarities)
        top_k_indices.append(indices_of_largest)
    return top_k_similarities, top_k_indices

def compare_datasets_fcd_score(smiles_list_1, smiles_list_2, device):
    fcd = FCD(device=device, n_jobs=8)
    return fcd(smiles_list_1, smiles_list_2)

def compute_validity(dataset, smiles_list):
    valid_molecules = 0
    total_molecules = len(dataset)
    valid_molecules = len(smiles_list)

    validity = valid_molecules / total_molecules
    return validity


def calculate_snn_average(dataset_g_smiles, dataset_e_smiles):
    dataset_g_fps = [smiles_to_fingerprint(smiles) for smiles in dataset_g_smiles]
    dataset_e_fps = [smiles_to_fingerprint(smiles) for smiles in dataset_e_smiles]

    max_similarities_g = []
    max_similarities_e = []

    # For each molecule in G, find the most similar molecule in E
    for g_fp in dataset_g_fps:
        similarities = [calculate_similarity(g_fp, e_fp) for e_fp in dataset_e_fps]
        max_similarities_g.append(max(similarities))

    # For each molecule in E, find the most similar molecule in G
    for e_fp in dataset_e_fps:
        similarities = [calculate_similarity(e_fp, g_fp) for g_fp in dataset_g_fps]
        max_similarities_e.append(max(similarities))

    # Calculate the average of these highest similarity scores
    average_snn_similarity = (np.mean(max_similarities_g) + np.mean(max_similarities_e)) / 2

    return average_snn_similarity