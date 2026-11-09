"""
atom_features_18.py -- the 18 node features of Table 1, computed correctly.

The existing builders compute features on `pyg_graph_to_mol(data)`, a molecule
rebuilt from atomic numbers with every bond set to SINGLE and no charges or
stereo. On that molecule aromaticity, hybridisation, H count, formal charge,
chirality, Gasteiger/Crippen/TPSA are all wrong.

Fix: compute every feature on the molecule parsed from the original SMILES.
`smiles_to_data()` builds the graph from `Chem.MolFromSmiles(smi)` in
`mol.GetAtoms()` order, so re-parsing the same SMILES gives the same atom order
and `data.edge_index` still lines up (checked by `check_alignment`).

Column order (18):
  0  atomic number          1  degree (heavy-atom bonds)   2  valence electrons
  3-5 hybridisation one-hot [sp, sp2, sp3]                 6  potential aromaticity (element-level, as Table 1 defines it)
  7  formal charge          8  H-bond donor                9  H-bond acceptor
  10-11 CIP chirality [R, S]                               12 total H count
  13 Gasteiger charge       14 Crippen logP contribution   15 TPSA contribution
  16 in aromatic ring       17 smallest ring size (0 = not in a ring)
"""
import os

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors

FEATURE_NAMES = [
    'atomic_num', 'degree', 'valence_e', 'hyb_sp', 'hyb_sp2', 'hyb_sp3',
    'potential_aromatic', 'formal_charge', 'hbd', 'hba', 'cip_R', 'cip_S',
    'num_h', 'gasteiger', 'crippen_logp', 'tpsa', 'in_aromatic_ring',
    'min_ring_size',
]

_PT = Chem.GetPeriodicTable()
_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
_HYB = {Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2}      # SP3D/SP3D2/S/unspecified -> all zero
_AROMATIC_LIKE = {6, 7, 8, 16, 34, 52}      # Table 1 dim 7: "potential aromaticity"


def _finite(x):
    x = float(x)
    return x if np.isfinite(x) else 0.0


def atom_features_18(mol):
    """[N, 18] float32 tensor for a sanitised RDKit mol (heavy atoms only)."""
    n = mol.GetNumAtoms()
    X = np.zeros((n, 18), dtype=np.float32)

    # molecule-level calculations, each on the real (sanitised) molecule
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    donors, acceptors = set(), set()
    for f in _FACTORY.GetFeaturesForMol(mol):
        if f.GetFamily() == 'Donor':
            donors.update(f.GetAtomIds())
        elif f.GetFamily() == 'Acceptor':
            acceptors.update(f.GetAtomIds())

    mg = Chem.Mol(mol)                       # copy: Gasteiger writes atom props
    AllChem.ComputeGasteigerCharges(mg, throwOnParamFailure=False)
    # Crippen assigns logP to hydrogens too. Compute on the explicit-H molecule
    # and fold each H's share into its heavy atom, so column 14 sums to
    # Crippen.MolLogP(mol). Heavy atoms keep their indices after AddHs.
    mh = Chem.AddHs(mol)
    crippen = np.zeros(n)
    for a, (logp, _) in zip(mh.GetAtoms(), rdMolDescriptors._CalcCrippenContribs(mh)):
        i = a.GetIdx()                       # atoms added by AddHs have idx >= n
        k = i if i < n else a.GetNeighbors()[0].GetIdx()
        crippen[k] += logp
    crippen = [(c, None) for c in crippen]
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    ring = mol.GetRingInfo()

    for a in mol.GetAtoms():
        i, z = a.GetIdx(), a.GetAtomicNum()
        X[i, 0] = z
        X[i, 1] = a.GetDegree()
        X[i, 2] = _PT.GetNOuterElecs(z)
        h = _HYB.get(a.GetHybridization())
        if h is not None:
            X[i, 3 + h] = 1.0
        X[i, 6] = 1.0 if z in _AROMATIC_LIKE else 0.0
        X[i, 7] = a.GetFormalCharge()
        X[i, 8] = 1.0 if i in donors else 0.0
        X[i, 9] = 1.0 if i in acceptors else 0.0
        cip = a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else ''
        X[i, 10] = 1.0 if cip == 'R' else 0.0
        X[i, 11] = 1.0 if cip == 'S' else 0.0
        X[i, 12] = a.GetTotalNumHs()
        X[i, 13] = _finite(mg.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))
        X[i, 14] = _finite(crippen[i][0])
        X[i, 15] = _finite(tpsa[i])
        X[i, 16] = 1.0 if a.GetIsAromatic() else 0.0
        X[i, 17] = ring.MinAtomRingSize(i)
    return torch.from_numpy(X)


def bond_features(mol, edge_index):
    """[E, 6] aligned with edge_index: one-hot [single, double, triple, aromatic],
    conjugated, in ring. Table 2 without bond length (no 3D embedding needed)."""
    bt = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
          Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    E = edge_index.shape[1]
    out = np.zeros((E, 6), dtype=np.float32)
    for e in range(E):
        b = mol.GetBondBetweenAtoms(int(edge_index[0, e]), int(edge_index[1, e]))
        if b is None:
            continue
        k = bt.get(b.GetBondType())
        if k is not None:
            out[e, k] = 1.0
        out[e, 4] = float(b.GetIsConjugated())
        out[e, 5] = float(b.IsInRing())
    return torch.from_numpy(out)


def check_alignment(mol, data):
    """Raise if the re-parsed SMILES does not match the PyG graph atom-for-atom."""
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if len(z) != data.x.shape[0] or z != data.x.view(-1).tolist():
        raise ValueError('SMILES atom order does not match data.x')
