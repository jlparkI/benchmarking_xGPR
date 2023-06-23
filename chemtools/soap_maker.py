"""xyz files, using the yvalues obtained from molecule_net as labels
(we do this for consistency, to be certain we are using the same
y-values as other publications).
The xyz files need to be 'cleaned' first (i.e. remove comment lines
at the end of the xyz file)."""
import os
import sys
from ase.io import read
import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP

BATCH_SIZE = 250
atom_key = {"C":0, "N":1, "O":2, "F":3}
DTYPES = ["zpve", "cv", "u298_atom", "g298_atom",
        "h298_atom", "u0_atom"]


def get_yvalue_dict():
    """Build a dictionary mapping molecule ids to all
    of the y-values of interest."""
    molnet_data = pd.read_csv("qm9_.csv")
    yvalues = molnet_data[DTYPES].values
    yvalue_dict = dict(zip([int(z.split("_")[1])
            for z in molnet_data["mol_id"].tolist()],
            list(yvalues) ))
    return yvalue_dict


def get_id(fname):
    """Extracts the id number for a molecule
    from the xyz file."""
    with open(fname, "r") as fhandle:
        _ = fhandle.readline()
        metadata = fhandle.readline().strip().split()
    return int(metadata[1])


def featurize_split(cv_split, yval_dict, output_fname):
    """Generates .npy files containing 'chunked' data
    for the input list of xyz files.
    Args:
        cv_split (list): A list of xyz files.
        yval_dict (dict): A dictionary mapping mol ids
            to y-values.
        output_fname (str): A string containing the
            prefix for all output file names.
    """
    structs, yvals, indices, atom_ids = [], [], [], []
    species = {"C", "H", "O", "N", "F"}

    conv_soaper = SOAP(species=species,
                periodic=False,
                sigma = 0.25,
                nmax=12, lmax=9,
                weighting={"function":"pow",
                    "r0":1.5, "m":9, "c":1, "d":1},
                r_cut = 3.25,
                sparse=False)
    batchnum = 0
    for xyz_file in cv_split:
        mol_id = get_id(xyz_file)
        structs.append(read(xyz_file))
        yvals.append(yval_dict[mol_id])
        atom_ids.append([atom_key[s] for s in structs[-1].get_chemical_symbols()
            if s != "H"])
        indices.append([s for s, i in
                enumerate(structs[-1].get_chemical_symbols()) if i != "H"])
        if len(structs) >= BATCH_SIZE:
            blend_soap(conv_soaper, structs, yvals,
                    indices, batchnum, output_fname, atom_ids)
            structs, yvals, indices, atom_ids = [], [], [], []
            batchnum += 1
    if len(structs) > 0:
        blend_soap(conv_soaper, structs, yvals,
                indices, batchnum, output_fname, atom_ids)


def blend_soap(conv_soaper, structs, yvals,
        indices, batchnum, output_fname, atom_ids):
    """Performs the actual work of converting a minibatch of molecules
    into soap descriptors and saving to file.
    Args:
        conv_soaper: The object that will generate the SOAP descriptors for
            use by convolution kernels.
        structs (list): A list of ase atoms objects containing the
            molecules to be processed.
        yvals (list): A list of shape M arrays where M is the number
            of properties we are predicting.
        indices (list): A list of lists of indices. Each list indicates
            the heavy atoms for the corresponding molecule, since we
            do not need to include hydrogens.
        batchnum (int): The number of this minibatch; used to generate
            the output filename.
        output_fname (str): Prefix for the output filename.
    """
    print("SOAPing another batch...")
    xmats = conv_soaper.create(structs, positions = indices, n_jobs=1)
    ydata = np.stack(yvals)
    zero_padded = [np.zeros((9, xmats[0].shape[1] + len(atom_key))) for xmat in xmats]
    for i, xmat in enumerate(xmats):
        xmat /= np.linalg.norm(xmat, axis=1)[:,None]
        zero_padded[i][:xmats[i].shape[0], :xmats[i].shape[1]] = xmat
        for j, idx in enumerate(atom_ids[i]):
            zero_padded[i][j, xmat.shape[1] + idx] = 1.0

    xmats = np.stack(zero_padded).astype(np.float32)
    print(f"{xmats.shape}")

    print("Saving another batch...", flush=True)
    np.save(f"{output_fname}_{batchnum}_xfolded.npy", xmats)
    for j, dtype in enumerate(DTYPES):
        np.save(f"{output_fname}_{batchnum}_{dtype}_yvalues.npy",
                        ydata[:,j])


def featurize_xyzfiles(target_dir, chemdata_path):
    """Obtains a list of xyz files for a target directory, splits them
    up into train - valid - test, and sets them up for feature generation."""
    
    bad_mols = set()
    with open("inconsistent_geom_mols.txt", "r") as fhandle:
        for line in fhandle:
            bad_mols.add(int(line.split()[0]))
    
    os.chdir(chemdata_path)
    yval_dict = get_yvalue_dict()
    os.chdir("cleaned_qm9_mols")
    raw_xyz_files = [os.path.abspath(f) for f in os.listdir() if f.endswith("xyz")]
    xyz_files = []
    for xyz in raw_xyz_files:
        mol_id = get_id(xyz)
        if mol_id not in bad_mols:
            xyz_files.append(xyz)
    xyz_files.sort()
    print(f"There are {len(xyz_files)} files.")
    rng = np.random.default_rng(123)

    idx = rng.permutation(len(xyz_files))
    cutoff_valid, cutoff_test = 110000, 120000
    cv_splits = [idx[:cutoff_valid], idx[cutoff_valid:cutoff_test],
                    idx[cutoff_test:]]
    cv_splits = [[xyz_files[i] for i in s.tolist()] for s in cv_splits]

    os.chdir(target_dir)
    if "train" not in os.listdir():
        os.mkdir("train")
        os.mkdir("valid")
        os.mkdir("test")
    batch_dirs = ["train", "valid", "test"]

    for batch_dir, cv_split in zip(batch_dirs, cv_splits):
        os.chdir(batch_dir)
        featurize_split(cv_split, yval_dict, "qm9")
        os.chdir("..")

    print("All done.")

if __name__ == "__main__":
    featurize_xyzfiles(sys.argv[1], sys.argv[2])    
