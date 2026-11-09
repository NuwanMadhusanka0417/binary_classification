"""
create_graph_new.py -- graph construction with the 18 node features of Table 1.

Same graph structure and S2VGraph objects as src/create_graphs.py; only the
node features differ. Every feature is computed on the molecule parsed from
the original SMILES (see src/atom_features_18.py), not on a molecule rebuilt
from atomic numbers, which loses bond orders, charges and explicit hydrogens.

src/create_graphs.py (7 features) is left unchanged so the baseline still
reproduces.
"""
import torch
from rdkit import Chem
import networkx as nx

from src.atom_features_18 import atom_features_18, check_alignment


def expand_atomic_features(data, mol):
    """[N, 18] node features for one molecule.

    data: PyG Data from smiles_to_data (x = atomic numbers)
    mol:  RDKit Mol from Chem.MolFromSmiles on the same SMILES
    """
    check_alignment(mol, data)          # raises if atom order differs from data.x
    return atom_features_18(mol), mol


def create_nodetags(mol):
  # node tags for one mol

  # Create a NetworkX graph
  nx_graph = nx.Graph()

  # Add nodes (atoms)
  for i, atom in enumerate(mol.GetAtoms()):
      atom_symbol = atom.GetSymbol()
      nx_graph.add_node(i, label=atom_symbol)

  # Add edges (bonds)
  for bond in mol.GetBonds():
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      nx_graph.add_edge(start, end)

  node_tags = [atom.GetSymbol() for atom in mol.GetAtoms()]

  return nx_graph, node_tags


class S2VGraph(object):
  def __init__(self, g, label,mol, node_tags=None, node_features=None):
    '''
        g: a networkx graph
        label: an integer graph label
        node_tags: a list of integer node tags
        node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
        edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
        neighbors: list of neighbors (without self-loop)
    '''
    self.label = label
    self.g = g
    self.node_tags = node_tags
    self.neighbors = []
    self.node_features = node_features
    self.edge_mat = 0
    self.mol = mol
    self.max_neighbor = 0


def create_graph_list(dataset, smiles):
  """dataset and smiles must be aligned row for row, as load_full() returns
  them (dataset, df) -> create_graph_list(dataset, df['smiles'])."""
  smiles = list(smiles)
  if len(smiles) != len(dataset):
    raise ValueError(f'{len(dataset)} graphs but {len(smiles)} SMILES')
  g_list = []
  for data, smi in zip(dataset, smiles):
    mol = Chem.MolFromSmiles(smi)       # the real molecule, not a rebuild
    node_features, mol = expand_atomic_features(data, mol)
    g, node_tags = create_nodetags(mol)
    g_list.append(S2VGraph(g = g, label= data.y, mol= mol, node_tags= node_tags,
                           node_features=node_features.to(torch.float32)))
  return g_list
