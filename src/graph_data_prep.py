from rdkit import Chem

import rdkit
hybridization_list = [Chem.rdchem.HybridizationType.S,Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]

chiral_centers = [Chem.rdchem.ChiralType.CHI_OTHER,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED]

ring_dict = [False, True]



import torch 
device = 'cuda'
def get_atom_features(mol, ring_dict, chiral_centers, hybridization_list):
    atomic_number = []
    num_hs = []
    implicit_hs = []
    explicit_hs = []
    valence_electrons = []
    ring_info = []
    chiral_info = []
    degree_without_h = []
    total_degree = []
    formal_charge = []
    hybridization = []
    aromatic_info = []
    mass = []
    radical_electrons = []
    w2v_embedding = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum() + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        num_hs.append(atom.GetTotalNumHs(includeNeighbors = True)  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        implicit_hs.append(atom.GetNumImplicitHs()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        explicit_hs.append(atom.GetNumExplicitHs()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]) )
        valence_electrons.append(atom.GetExplicitValence()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        ring_info.append(ring_dict.index(atom.IsInRing())  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        chiral_info.append(chiral_centers.index(atom.GetChiralTag())  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        degree_without_h.append(atom.GetDegree()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        total_degree.append(atom.GetTotalDegree()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        formal_charge.append(atom.GetFormalCharge()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        hybridization.append(hybridization_list.index(atom.GetHybridization())  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        aromatic_info.append(ring_dict.index(atom.GetIsAromatic())  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        mass.append(atom.GetMass()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        radical_electrons.append(atom.GetNumRadicalElectrons()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))
        w2v_embedding.append(model[atom.GetSymbol()].tolist()  + np.random.choice([0.1, -0.1, 0, 0.2, -0.2]))

    atomic_features = torch.FloatTensor([atomic_number, num_hs, explicit_hs, implicit_hs, valence_electrons, ring_info, chiral_info,degree_without_h, total_degree , formal_charge, 
                         hybridization,aromatic_info, mass, radical_electrons])
    
    # final_features = torch.cat([atomic_features.permute(1, 0), torch.FloatTensor(np.array(w2v_embedding))], dim = 1)

    return atomic_features.to(device).permute(1, 0)

def get_edge_index(mol):
    row, col = [], []
    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        
    return torch.tensor([row, col], dtype=torch.long).to(device)

# from torch_geometric.data.dataloader import DataLoader

def prepare_dataloader(mol_list):
    data_list = []

    for i, mol in enumerate(mol_list):

        x = get_atom_features(mol)
        edge_index = get_edge_index(mol)

        data = torch_geometric.data.data.Data(x = x, edge_index = edge_index)
        data_list.append(data)

    return DataLoader(data_list, batch_size=3, shuffle=False), data_list


import os.path as osp
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from rdkit.Chem import AllChem

class MyOwnDataset(Dataset):
  def __init__(self, root, transform=None, pre_transform=None):
      super(MyOwnDataset, self).__init__(root, transform, pre_transform)
      # self.dataset = dataset

  @property
  def raw_file_names(self):
      return []

  @property
  def processed_file_names(self):
      return ['data_' + str(col) + ".pt" for col in range(len(c))]

  def download(self):
    return []

  def process(self):
    for idx in tqdm(range(complete_protein_data_2.shape[0]), position = 0, leave = True):
      smile_string = complete_protein_data_2.loc[idx]['smile_notation']
      test_mol_1 = AllChem.MolFromSmiles(smile_string)
      target = complete_protein_data_2.loc[idx]['target_label']

      node_features = get_atom_features(test_mol_1, ring_dict, chiral_centers, hybridization_list)
      edge_index =  get_edge_index(test_mol_1)

      # print(node_features.shape, edge_index.shape, target.shape)
      data = Data(x = node_features, edge_index = edge_index, y = target)

      torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

  def len(self):
      return len(self.processed_file_names)

  def get(self, idx):
      data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
      return data

from torch.utils.data import Dataset, DataLoader
from rdkit.Chem import AllChem

class ProtData(Dataset):
  def __init__(self, dataset,  ring_dict, chiral_centers, hybridization_list, smile_col, target_col, label_type = 'train'):
    self.label_type = label_type
    self.smile_col = smile_col
    self.target_col = target_col
    self.dataset = dataset 
    self.ring_dict = ring_dict
    self.chiral_centers= chiral_centers
    self.hybridization_list = hybridization_list

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):

    if self.label_type !='test':

       smile_string = self.dataset.loc[idx][self.smile_col]
       test_mol_1 = AllChem.MolFromSmiles(smile_string)
       target = self.dataset.loc[idx][self.target_col]
       
       node_features = get_atom_features(test_mol_1, self.ring_dict, self.chiral_centers, self.hybridization_list)
       edge_index =  get_edge_index(test_mol_1)
       
       return node_features, edge_index, target

    else:
       smile_string = self.dataset.loc[idx][self.smile_col]
       test_mol_1 = AllChem.MolFromSmiles(smile_string)
      #  target = self.dataset.loc[idx][self.target_col]
       
       node_features = get_atom_features(test_mol_1, self.ring_dict, self.chiral_centers, self.hybridization_list)
       edge_index =  get_edge_index(test_mol_1)
       
       return node_features, edge_index