import json
import math
import ase.io
import subprocess
from ase.geometry import get_layers
from ase.constraints import FixAtoms
from collections import Counter
import os
import csv

def modify_lines_in_file(file_in, data_dict, file_out=None):
    
        with open(file_in, 'r') as file_in:
            lines = file_in.readlines()
        
        for line_number, new_content in data_dict.items():
            lines[line_number - 1] = new_content + '\n'

        if file_out:
            with open(file_out, 'w') as file_out:
                file_out.writelines(lines)
        else:
            return lines

def get_ini_a(ratio_dict):
    
    with open(basic_file_path+'/comp_ref_vol.json', 'r', encoding='utf-8') as f:
        vol_data = json.load(f)

        vol = 0
        for ele, ratio in ratio_dict.items():
            vol += vol_data[ele]["Fm-3m"]*ratio
        a = math.pow(vol*4, 1/3)
    return a

def gen_ratio_cmd(ratio_dict, seed=123456):
    
    ratios = [ratio for ele, ratio in ratio_dict.items()]

    cmd = ""
    ratio1 = 1
    for i, ratio in enumerate(ratios[1:]):
        ratio_from1 = round(ratio/ratio1, 3)
        ratio1 -= ratio
        cmd += "set type 1 type/ratio %s %s %s\n"%(i+2, ratio_from1, seed)

    return cmd

def get_ele_ratio(atoms):
    counter = Counter(atoms.symbols)
    ele_ratio = {}
    for ele, count in counter.items():
        ele_ratio[ele] = round(count/len(atoms), 3)
    return ele_ratio

def get_top_ratio(atoms):
    layer_index = get_layers(atoms, miller=(0, 0, 1), tolerance=0.1)[0]
    top_index = [m for m, n in enumerate(layer_index) if n == layer_index.max()]
    atoms_top = atoms[top_index]
    return get_ele_ratio(atoms_top)

def run(ratio_dict, temp, surface='111'):
    
    name = ''.join(["%s%s"%(key, value) for key, value in ratio_dict.items()])+'_%s'%temp+'_%s'%surface
    
    if os.path.isfile("xyz/mc_%s.xyz"%name):
        print("skip %s"%name)
        return
    
    command = "srun -K lmp -in in.run" 

    # run npt
    ini_a = round(get_ini_a(ratio_dict), 3)
    ratio_cmd = gen_ratio_cmd(ratio_dict)
    ele_str = ' '.join(list(ratio_dict.keys()))

    data_dict = {9: "lattice fcc %s"%ini_a,
                11: "create_box %s box"%len(ratio_dict),
                15: ratio_cmd,
                19: "pair_coeff * * %s/CuAgAuNiPdPtAl_Zhou04.eam.alloy %s"%(basic_file_path, ele_str),
                39: "velocity all create %s 123456"%temp,
                44: "fix  1 all npt temp %s %s 0.1 iso 1 1 1"%(temp, temp),
                }

    modify_lines_in_file(basic_file_path+"/in.cell_npt", data_dict, "in.run")

    #command = "mpirun -np 10 lmp_intel_cpu_intelmpi -in in.run"
    with open('log', 'w') as f:
        subprocess.call(command, shell=True, stdout=f, cwd=None)
        
    
    with open('lattice.dat', 'r') as file:
        first_line = file.readline()
    values =  [round(float(val), 3) for val in first_line.split()[-3:]]
    a = values[0]
    
    # lattice信息写入文件
    with open('result_lattice.txt', 'a') as file:
        file.write("%s, ini_a:%s, opt_a:%s\n"%(ratio_dict, ini_a, a))

    # run mc
    type_str = ' '.join(map(str, list(range(1, len(ratio_dict)+1))))
    
    if surface == '111':
        data_dict = {9: "lattice fcc %s orient x 1 1 -2 orient y -1 1 0 orient z 1 1 1"%a,
                    11: "create_box %s box"%len(ratio_dict),
                    18: ratio_cmd,
                    22: "pair_coeff * * %s/CuAgAuNiPdPtAl_Zhou04.eam.alloy %s"%(basic_file_path, ele_str),
                    # 42: "variable var1 equal round(random(1,%s,123456))"%len(ratio_dict),
                    # 43: "variable var2 equal round(random(1,%s,123456))"%len(ratio_dict),
                    52: "fix 1 all atom/swap 1 25 123456 %s types ${type1} ${type2} region surface"%temp
                    }
    elif surface == '100':
        data_dict = {9: "lattice fcc %s orient x 0 1 0 orient y 0 0 1 orient z 1 0 0"%a,
                    11: "create_box %s box"%len(ratio_dict),
                    18: ratio_cmd,
                    22: "pair_coeff * * %s/CuAgAuNiPdPtAl_Zhou04.eam.alloy %s"%(basic_file_path, ele_str),
                    45: "fix 1 all atom/swap 1 20 123456 %s types %s region surface"%(temp, type_str),
                    }
    
    if len(ratio_dict) == 5:
        data_dict[45] = "variable type1 index 1 1 1 1 2 2 2 3 3 4"
        data_dict[46] = "variable type2 index 2 3 4 5 3 4 5 4 5 5"
    
    if len(ratio_dict) == 4:
        data_dict[45] = "variable type1 index 1 1 1 2 2 3"
        data_dict[46] = "variable type2 index 2 3 4 3 4 4"
    
    modify_lines_in_file(basic_file_path+"/in.%s_mcmin"%surface, data_dict, "in.run")

    #command = "mpirun -np 10 lmp_intel_cpu_intelmpi -in in.run"
    with open('log', 'w') as f:
        subprocess.call(command, shell=True, stdout=f, cwd=None)

    # analysis
    # atoms_npt = ase.io.read("npt.xyz", index=-1)
    # atoms_mc = ase.io.read("mcmd.xyz", index=-1)

    # with open('result.txt', 'a') as file:
    #     file.write("\nratio dict: %s, temp: %s\n"%(ratio_dict, temp))
    #     file.write("npt atoms ratio: %s\n"%get_ele_ratio(atoms_npt))
    #     file.write("mc atoms ratio: %s\n"%get_ele_ratio(atoms_mc))
    #     file.write("mc top ratio: %s\n"%get_top_ratio(atoms_mc))
    
    os.popen("mkdir -p xyz").read()
    # os.popen("mv npt.xyz xyz/npt_%s.xyz"%name).read()
    os.popen("mv mc.xyz xyz/mc_%s.xyz"%name).read()


basic_file_path = "basic_files"

with open(basic_file_path+'/stable_hea.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_list = []
    for row in reader:
        data = json.loads(row[0])
        data_list.append(data)

for ratio_dict in data_list:
    run(ratio_dict, 300, '111')
