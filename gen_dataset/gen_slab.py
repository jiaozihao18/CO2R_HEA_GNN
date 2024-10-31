import random
import json
import math
from ase.build import fcc111
from ase.visualize import view
from ase.calculators.vasp import Vasp
import numpy as np
from ase.db import connect
from ase.geometry import get_layers

def generate_element_list(n):
    elements = ['Cu', 'Ag', 'Au', 'Pt', 'Pd', 'Al']
    result = [random.choice(elements) for _ in range(n)]
    return result

def get_random_alloy(vol_data, num=27):

    elements_l = generate_element_list(num)
    
    vol = 0
    for ele in elements_l:
        vol += vol_data[ele]["Fm-3m"]
    vol = vol*4/num
    a = math.pow(vol, 1/3)

    atoms = fcc111('Cu', a=a, size=(3, 3, 3), vacuum=10.0)
    atoms.symbols = elements_l
    
    tags_l = [0]*18+[1]*9
    atoms.set_tags(tags_l)
    
    return atoms

db = connect("hea_slab.db")
run_cmd="srun vasp_std"

with open('comp_ref_vol.json', 'r', encoding='utf-8') as f:
    vol_data = json.load(f)

while True:
    atoms = get_random_alloy(vol_data)
    atoms.pbc=True
    calc = Vasp(system = "HEA_alloy", ncore=4, istart=1,
                icharg=1, lwave=False, lcharg=False,
                encut=450, ismear=1, sigma=0.2,
                ediff=1e-6, nelmin=5, nelm=60,
                gga="RP", pp="PBE", lreal="Auto",
                algo='Fast', isym=0,
                ediffg=-0.05, ibrion=2, potim=0.2,
                nsw=100, isif=3,
                gamma=True, kpts=[3, 3, 1],
                command=run_cmd)
    atoms.calc = calc
    calc.calculate(atoms)

    if np.sqrt(np.sum(atoms.get_forces()**2, axis=1)).max() < 0.05:
        #layer_index = get_layers(atoms, miller=(0, 0, 1), tolerance=0.2)[0]
        #if layer_index.max() == 2:
        #    if all(atoms.get_tags()[layer_index < 2] == 0) and all(atoms.get_tags()[layer_index == 2] == 1):
        db.write(atoms)

