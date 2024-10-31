from ase.calculators.calculator import Calculator
from .model import CGCNN
from .data import AtomsToGraphs, AseDBDataset
import torch
from torch_geometric.data import Batch
from ase import neighborlist
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso


def get_matrix(atoms, ratio=1.5*0.89):
    
    cutOff = neighborlist.natural_cutoffs(atoms)
    cutOff = np.array(cutOff) * ratio 
    neighborList = neighborlist.NeighborList(cutOff, skin=0,
                                             self_interaction=False, bothways=True)
    neighborList.update(atoms)
    matrix = neighborList.get_connectivity_matrix().toarray()

    return matrix

def gen_G(atoms):
    neigh_matrix = get_matrix(atoms)
    G = nx.from_numpy_array(neigh_matrix)
    for atom in atoms:
        G.add_node(atom.index, symbol=atom.symbol)
    return G

def gen_sample_G(symbols=['C', 'O1', 'O2', 'H'], edges=[('C', 'O1'), ('C', 'O2'), ('O2', 'H')]):

    g = nx.Graph()
    for i, symbol in enumerate(symbols):
        g.add_node(i, symbol=''.join([char for char in symbol if not char.isdigit()]))

    if edges:
        for sym1, sym2 in edges:
            g.add_edge(symbols.index(sym1), symbols.index(sym2))
        
    return g

samples_G = {'H': gen_sample_G(symbols=['H'], edges=None),
             'COOH': gen_sample_G(symbols=['C', 'O1', 'O2', 'H'], edges=[('C', 'O1'), ('C', 'O2'), ('O2', 'H')]),
             'OCHO': gen_sample_G(symbols=['C', 'O1', 'O2', 'H'], edges=[('C', 'O1'), ('C', 'O2'), ('C', 'H')]),
             'CO': gen_sample_G(symbols=['C', 'O'], edges=[('C', 'O')]),
             'COH': gen_sample_G(symbols=['C', 'O', 'H'], edges=[('C', 'O'), ('O', 'H')]),
             'CHO': gen_sample_G(symbols=['C', 'O', 'H'], edges=[('C', 'O'), ('C', 'H')]),
             'COCOH': gen_sample_G(symbols=['C1', 'O1', 'C2', 'O2', 'H'], edges=[('C1', 'O1'), ('C1', 'C2'), ('C2', 'O2'), ('O2', 'H')]),
             'COCHO': gen_sample_G(symbols=['C1', 'O1', 'C2', 'O2', 'H'], edges=[('C1', 'O1'), ('C1', 'C2'), ('C2', 'O2'), ('C2', 'H')]),
             'COCO': gen_sample_G(symbols=['C1', 'O1', 'C2', 'O2'], edges=[('C1', 'O1'), ('C1', 'C2'), ('C2', 'O2')]),
             }


class mlCalculator(Calculator):
    implemented_properties = ["energy"]
    def __init__(self, checkpoint_path):
        super().__init__()
        
        chk = torch.load(checkpoint_path)
        self.model = CGCNN(**chk["config"]['config_model'])
        self.model.load_state_dict(chk['state_dict'])
        
        self.model.eval()
        
        self.a2g = AtomsToGraphs(r_distances=True)
        
    def calculate(self, atoms, properties, system_changes) -> None:
        
        if 2 not in atoms.get_tags():
            tags = [2 if atom.symbol in ['C', 'H', 'O'] else 0 for atom in atoms]
            atoms.set_tags(tags)
        
        super().calculate(atoms, properties, system_changes)
        
        atoms_ads = atoms[[atom.index for atom in atoms if atom.symbol in ['C', 'H', 'O']]]
        G_ads = gen_G(atoms_ads)

        for name, sample_G in samples_G.items():
            nm = iso.categorical_node_match("symbol", "C")
            if nx.is_isomorphic(G_ads, sample_G, node_match=nm):
                ads_name = name
                break
        
        atoms_ = AseDBDataset.pre_convert(atoms, ads_name=ads_name, mode='com')
        data_object = self.a2g.convert(atoms_)
        batch = Batch.from_data_list([data_object])

        predictions = self.model(batch)
        self.results["energy"] = predictions.item()