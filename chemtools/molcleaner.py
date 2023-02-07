"""A simple 'throaway' script which removes the last few
lines of each xyz file (these are metadata that will cause
ase to error out when reading the file) and removes
problem characters"""
import os
import sys

os.chdir(sys.argv[1])
if "cleaned_qm9_mols" not in os.listdir():
    os.mkdir("cleaned_qm9_mols")
os.chdir("qm9_mols")
xyz_files = [os.path.abspath(f) for f in os.listdir() if f.endswith(".xyz")]
os.chdir(os.path.join("..", "cleaned_qm9_mols"))
for xyz in xyz_files:
    with open(xyz, "r") as fhandle:
        r = fhandle.readlines()
    xout = os.path.basename(xyz)
    with open(xout, "w+") as fhandle:
        for l in r[:-3]:
            _ = fhandle.write(l.replace("*^", "e"))
