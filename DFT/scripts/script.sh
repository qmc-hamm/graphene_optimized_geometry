python twist_bilayer_graphene.py
python POSCAR_generator.py
cp INCAR Simulation_folder
cp POSCAR Simulation_folder
cp POTCAR Simulation_folder
cp KPOINTS Simulation_folder
## copy your own jobscript and run VASP
## get the output CONTCAR, CHGCAR etc
## change KPOINTS accordingly
