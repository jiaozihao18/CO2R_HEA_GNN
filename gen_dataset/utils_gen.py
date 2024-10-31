import ase.io
import pickle
from ase.visualize import view
from ase.geometry import get_layers
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
import random
import numpy as np
from ase.data import covalent_radii
from itertools import product
import os
from ase.constraints import FixAtoms

# Randomly rotate the adsorbate
def random_rotate_adsorbate(atoms, bind_idx):
    zrot = np.random.uniform(0, 360)
    atoms.rotate(zrot, "z", center=atoms.positions[bind_idx])

    z = np.random.uniform(np.cos(np.pi / 9), 1.0)
    phi = np.random.uniform(0, 2 * np.pi)
    sqrt_val = np.sqrt(1 - z * z)
    rotvec = np.array([sqrt_val * np.cos(phi), sqrt_val * np.sin(phi), z])

    atoms.rotate(a=(0, 0, 1), v=rotvec, center=atoms.positions[bind_idx])
    return atoms

# Set tags for surface atoms: 0 for bulk, 1 for surface
def get_tags(atoms):
    layer_index = get_layers(atoms, miller=(0, 0, 1), tolerance=0.5)[0]
    max_layer = max(layer_index)
    tags_l = [1 if layer_index[atom.index] == max_layer else 0 for atom in atoms]
    return tags_l

# Check for atomic overlap
def is_overlap(atoms):
    ads_index = [atom.index for atom in atoms if atom.tag == 2]
    surf_index = [atom.index for atom in atoms if atom.tag == 1]

    pairs = product(ads_index, surf_index)
    for pair in pairs:
        dist = atoms.get_distance(pair[0], pair[1], mic=True)
        threshold = sum(covalent_radii[atoms[idx].number] for idx in pair)
        if dist < threshold * 0.8:
            return True
    return False

def random_gen(slab, ads_path="adsorbates.pkl", ads_index=8, gen_num=1):
    
    # Read slab and adsorbate data only once
    # slab = ase.io.read(slab_path)
    adsorbate_dict = pickle.load(open(ads_path, "rb"))
    
    # Pre-compute layer index and tags for the slab
    layer_index = get_layers(slab, miller=(0, 0, 1), tolerance=0.2)[0]
    max_layer = max(layer_index)
    tags_slab = [1 if layer_index[atom.index] == max_layer else 0 for atom in slab]
    
    # Fix 1 layers
    fix_index = [m for m, n in enumerate(layer_index) if n in [0]]
    c = FixAtoms(indices=fix_index)
    slab.set_constraint(c)

    # Get adsorbate details
    adsorbate_l = adsorbate_dict[ads_index]
    adsorbate = adsorbate_l[0]
    bind_idxs = adsorbate_l[2]
    
    # Get slab sites
    struct = AseAtomsAdaptor.get_structure(slab)
    asf = AdsorbateSiteFinder(struct)
    all_sites = asf.find_adsorption_sites(distance=0)["all"]
    
    atoms_l = []
    while len(atoms_l) < gen_num:
        
        # Randomly choose sites on the slab
        site = random.choice(all_sites)
        adsorbate_c = adsorbate.copy()
        
        # Randomly select a binding index and rotate the adsorbate
        bind_idx = random.choice(bind_idxs)
        adsorbate_c = random_rotate_adsorbate(adsorbate_c, bind_idx)

        # Translate the adsorbate to the selected site
        trans_vector = site - adsorbate_c.positions[bind_idx] + [0, 0, 2]
        adsorbate_c.translate(trans_vector)

        # Combine slab and adsorbate and set tags
        combined_atoms = slab + adsorbate_c
        combined_atoms.set_tags(tags_slab + [2]*len(adsorbate_c))
        
        # Check for overlap and add to the list if no overlap
        if not is_overlap(combined_atoms):
            atoms_l.append(combined_atoms)

    return atoms_l
