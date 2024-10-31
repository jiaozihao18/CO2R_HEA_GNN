import numpy as np
from torch.utils.data import Dataset
from ase.db import connect
from torch_geometric.data import Data
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import ase
from ase.atoms import Atom
import pickle
import lmdb
from sklearn.model_selection import KFold
from torch.utils.data import Subset


class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.
    
    """

    def __init__(
        self,
        max_neigh: int = 200,
        radius: int = 6,
        r_energy: bool = False,
        r_forces: bool = False,
        r_distances: bool = False,
        r_edges: bool = True,
        r_fixed: bool = True,
        r_pbc: bool = True,
    ) -> None:
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges
        self.r_pbc = r_pbc

    def _get_neighbors_pymatgen(self, atoms: ase.Atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(r=self.radius, numerical_tol=0, exclude_self=True)

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def convert(self, atoms: ase.Atoms):
        """Convert a single atomic stucture to a graph."""
        
        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]
        tags = torch.Tensor(atoms.get_tags())

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags)

        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(*split_idx_dist)
            data.edge_index = edge_index
            # data.cell_offsets = cell_offsets
            if self.r_distances:
                data.distances = edge_distances
            
        if self.r_energy:
            energy = torch.Tensor(atoms.get_potential_energy(apply_constraint=False))
            data.y = energy
            
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.force = forces  
            
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms
                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx
            
        if self.r_pbc:
            data.pbc = atoms.pbc

        return data


pesu_dict = {"H":91, "COOH":92, "OCHO":93, "CO":94, "COH":95, 
              "CHO":96, "COCO":97, "COCOH":98, "COCHO":99}

class AseDBDataset(Dataset):
    """
    A dataset class for handling ASE DB databases.

    Args:
        db_path (str): Path to the ASE DB file.
        extra_db_keys (list, optional): List of additional keys to extract from the database.
        a2g_args (dict, optional): Keyword arguments for ocpmodels.preprocessing.AtomsToGraphs().
        apply_pre_convert ('com', 'cop', optional): Whether to apply a pre-conversion function.
    """

    def __init__(self, **kwargs) -> None:
        
        self.db = connect(kwargs["db_path"])
        self.a2g = AtomsToGraphs(**kwargs.get("a2g_args", {}))
        self.extra_db_keys = kwargs.get("extra_db_keys", [])
        self.apply_pre_convert = kwargs.get("apply_pre_convert", False)

    def __len__(self) -> int:
        return len(self.db)

    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self.__len__()))]
        
        if idx + 1 > len(self.db):
            # raise IndexError("Index out of range")
            raise StopIteration

        # Get row from database
        row = self.db.get(id=idx+1)  # ASE DB id is 1-based index
        atoms = row.toatoms()
        if self.apply_pre_convert:
            atoms = self.pre_convert(atoms, row.ads_name, self.apply_pre_convert)
            
        # Convert to data object
        data_object = self.a2g.convert(atoms)
        data_object.sid = idx

        # Add extra database keys if present
        for db_key in self.extra_db_keys:
            if isinstance(row[db_key], str):
                data_object[db_key] = row[db_key]
            else:
                data_object[db_key] = torch.tensor(row[db_key])
        return data_object
    
    @staticmethod
    def pre_convert(atoms, ads_name, mode):

        atoms_slab = atoms[atoms.get_tags() != 2]
        atoms_ads = atoms[atoms.get_tags() == 2]
        if mode == 'com':
            atoms_pseu = Atom(pesu_dict[ads_name], position=atoms_ads.get_center_of_mass(), tag=2)
        elif mode == 'cop':
            atoms_pseu = Atom(pesu_dict[ads_name], position=atoms.positions.mean(axis=0), tag=2)
        atoms_sp = atoms_slab + atoms_pseu
        
        return atoms_sp

class LmdbDataset(Dataset):

    def __init__(self, path: str):
        
        self.env = lmdb.open(path,
                             subdir=False,
                             readonly=True,
                             lock=False,
                             readahead=True,
                             meminit=False,
                             max_readers=1)
        
        self.num_samples = pickle.loads(self.env.begin().get("length".encode("ascii")))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self.__len__()))]

        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        if datapoint_pickled is None:
            raise IndexError(f"Index {idx} out of range")
        data = pickle.loads(datapoint_pickled)
        return data

    def close_db(self) -> None:
        self.env.close()
    
    def __del__(self):
        self.close_db()


def split_list(data, ratios):

    shuffled_data = random.sample(data, len(data))
    
    total = len(shuffled_data)
    indices = [int(r * total) for r in ratios]
    if sum(indices) != total:
        indices[-1] += total - sum(indices)

    split_lists = []
    current_index = 0
    for index in indices:
        split_lists.append(shuffled_data[current_index:current_index + index])
        current_index += index

    return split_lists


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def get_data_loader(dataset, idxs_dict, train_ratio=0.8, val_ratio=0.1, seed=42, batch_size=20):
    """
    Parameters:
    - idxs_dict (dict): A dictionary where keys are catagery names and values are lists of indexes.
    Returns:
    - tuple: Contains three DataLoader objects corresponding to training, validation, and test datasets.
    """
    
    set_seed(seed)

    train_idxs = []
    val_idxs = []
    test_idxs = []

    test_ratio = 1 - train_ratio - val_ratio

    for ads_name, idxs in idxs_dict.items():
        tmp1, tmp2, tmp3 = split_list(idxs, [train_ratio, val_ratio, test_ratio])
        train_idxs.extend(tmp1)
        val_idxs.extend(tmp2)
        test_idxs.extend(tmp3)

    train_sampler = SubsetRandomSampler(train_idxs)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_subset = Subset(dataset, val_idxs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    test_subset = Subset(dataset, test_idxs)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def split_dataset_k_fold(dataset, idxs_dict, test_ratio, k, seed=42, batch_size=20):
    """
    Split the dataset into K folds with a fixed test set and return DataLoaders.

    Args:
    - dataset: Dataset, the dataset to be split
    - idxs_dict: dict, a dictionary with ads_name as keys and lists of indices as values
    - test_ratio: float, the ratio of the test set
    - k: int, the number of folds for cross-validation
    - seed: int, random seed for shuffling
    - batch_size: int, the batch size for DataLoader

    Returns:
    - kfolds: list of tuples, each containing train, validation, and test DataLoader for one fold
    """
    set_seed(seed)
    
    # Copy idxs_dict to avoid modifying the original
    idxs_dict_copy = {ads_name: idxs[:] for ads_name, idxs in idxs_dict.items()}
    
    # Split the entire dataset into fixed test set and remaining data
    remaining_idxs_dict = {}
    test_idxs = []

    for ads_name, idxs in idxs_dict_copy.items():
        np.random.shuffle(idxs)
        test_size = int(len(idxs) * test_ratio)
        test_data = idxs[:test_size]
        remaining_data = idxs[test_size:]
        remaining_idxs_dict[ads_name] = remaining_data
        test_idxs.extend(test_data)

    # Perform K-Fold split for train and val sets on the remaining data
    kfolds = [([], [], test_idxs) for _ in range(k)]

    for ads_name, idxs in remaining_idxs_dict.items():
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []

        for train_index, val_index in kf.split(idxs):
            train_data = [idxs[i] for i in train_index]
            val_data = [idxs[i] for i in val_index]
            folds.append((train_data, val_data))

        for i in range(k):
            kfolds[i][0].extend(folds[i][0])  # Extend train indices
            kfolds[i][1].extend(folds[i][1])  # Extend val indices

    # Create DataLoaders
    dataloaders = []
    for train_idxs, val_idxs, test_idxs in kfolds:
        train_subset = Subset(dataset, train_idxs)
        val_subset = Subset(dataset, val_idxs)
        test_subset = Subset(dataset, test_idxs)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        dataloaders.append((train_loader, val_loader, test_loader))

    return dataloaders

