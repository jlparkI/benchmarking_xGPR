"""Preps the GB1 dataset for evaluation in the active learning
experiments."""
import os
import pandas as pd
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit

def prep_act_learn(start_dir):
    """Preps the GB1 test data for modeling in an active
    learning experiment."""
    os.chdir(os.path.join(start_dir, "benchmark_evals", "active_learn", "raw_data"))
    if "GB1.csv" not in os.listdir():
        raise ValueError("The active learning dataset has not been downloaded yet!")

    fpaths = {"GB1":os.path.abspath("GB1.csv")}

    os.chdir("..")
    if "encoded_data" not in os.listdir():
        os.mkdir("encoded_data")
    os.chdir("encoded_data")

    for dname, fpath in fpaths.items():
        os.chdir(os.path.join(start_dir, "benchmark_evals", "active_learn", "encoded_data"))
        generate_data_array(dname, fpath)

    os.chdir(start_dir)
    print("Active learning data encoding is complete.")


def generate_data_array(dname, fpath):
    """Encodes the data for GB1."""
    if dname not in os.listdir():
        os.mkdir(dname)

    encoder = MSAEncodingToolkit("PFASUM62_standardized")
    raw_data = pd.read_csv(fpath)
    seqs = raw_data["Variants"].tolist()
    yvals = raw_data["Fitness"].values
    #Standardize data so that 1 is max fitness.
    yvals = (yvals - yvals.min()) / (yvals.max() - yvals.min())

    encoder.encode_sequence_list(seqs, yvals, dname,
                blocksize=None, mode = "flat")
