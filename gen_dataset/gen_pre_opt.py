from ase.db import connect
from utils_gen import random_gen
import random
from ase.geometry import get_layers
import ase.io
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import ase.io
from ase.optimize import BFGS
from ocdata.utils import DetectTrajAnomaly

checkpoint = "eq2_153M_ec4_allmd.pt"
calc = OCPCalculator(checkpoint_path=checkpoint, cpu=False)   # if you have a GPU

ads_index = 104
ads_path = "adsorbates+extra.pkl"
slab_path = "sum_hea_slab.db"

slab_db = connect(slab_path)
ads_preopt_db = connect("ads_preopt_%s.db"%ads_index)

num=0

while num<1000:

    sel_slab = True
    while sel_slab:
        slab_id = random.randint(1, len(slab_db))
        slab = slab_db.get_atoms(id=slab_id)
        layer_index = get_layers(slab, miller=(0, 0, 1), tolerance=0.2)[0]
        cond1 = layer_index.max() == 2
        cond2 = all(slab.get_tags()[layer_index < 2] == 0)
        cond3 = all(slab.get_tags()[layer_index == 2] == 1)
        if cond1 and cond2 and cond3:
            sel_slab = False

    atoms = random_gen(slab, ads_path=ads_path, ads_index=ads_index, gen_num=1)[0]
    atoms.pbc=True
    
    initial_atoms = atoms.copy()
    
    atoms.calc = calc
    opt = BFGS(atoms, trajectory=None, logfile='-')
    opt.run(fmax=0.05, steps=200)
    
    atom_tags = initial_atoms.get_tags()
    detector = DetectTrajAnomaly(initial_atoms, atoms, atom_tags)
    anom = (
        detector.is_adsorbate_dissociated()
        or detector.is_adsorbate_desorbed()
        or detector.has_surface_changed()
        or detector.is_adsorbate_intercalated()
    )
    
    atoms.calc=None
    ads_preopt_db.write(atoms, slab_id=slab_id, anom=anom)
    
    num+=1
    

    
