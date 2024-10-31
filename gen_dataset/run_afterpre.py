from ase.db import connect
from ase.calculators.vasp import Vasp
import ase.io
import sys

pre_opt_db = connect(sys.argv[1])

run_cmd="mpirun vasp_std"

ads_opt_db = connect("ads_opt.db")
ads_traj_db = connect("ads_traj.db")

for row in pre_opt_db.select():

    if row.anom == False:
        
        if list(ads_opt_db.select(pre_opt_id=row.id)) == []:
            
            calc = Vasp(system = "HEA_alloy", ncore=4, istart=1,
                        icharg=1, lwave=False, lcharg=False,
                        encut=450, ismear=1, sigma=0.2,
                        ediff=1e-6, nelmin=5, nelm=60,
                        gga="RP", pp="PBE", lreal="Auto",
                        algo='Fast', isym=0,
                        ediffg=-0.05, ibrion=2, potim=0.2,
                        nsw=500, isif=2,
                        gamma=True, kpts=[3, 3, 1],
                        command=run_cmd)

            atoms.calc = calc
            calc.calculate(atoms)

            ads_opt_db.write(atoms, slab_id=row.slab_id, pre_opt_id=row.id)

            atoms_l = ase.io.read("OUTCAR", index=":")
            for atoms in atoms_l:
                ads_traj_db.write(atoms, slab_id=row.slab_id, pre_opt_id=row.id)
