"""Preps the fluorescence dataset (mutations in GFP) for modeling."""
import os
import numpy as np
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit
from Bio import SeqIO

def prep_fluorescence(start_dir):
    """Preps the fluorescence dataset for modeling using either a convolution
    or fixed vector kernel."""
    os.chdir(os.path.join("benchmark_evals", "fluorescence_eval", "raw_data"))
    if "FLUORESCENCE.fasta" not in os.listdir():
        raise ValueError("The fluorescence dataset has not been downloaded yet!")

    filepaths, yvalues = get_dataset_fps(start_dir)
    target_dirs = ["onehot", "onehot_conv"]
    for i, enc_type in enumerate(target_dirs):
        os.chdir(start_dir)
        os.chdir(os.path.join("benchmark_evals", "fluorescence_eval"))
        print(f"Now encoding fluorescence using {enc_type}")
        if target_dirs[i] not in os.listdir():
            os.mkdir(target_dirs[i])
        os.chdir(target_dirs[i])
        if "standard" not in os.listdir():
            os.mkdir("standard")
        os.chdir("standard")
        if i == 1:
            generate_output_arrays(enc_type, filepaths, yvalues)
        else:
            generate_output_arrays(enc_type, filepaths, yvalues)
    print("Fluorescence encoding is complete.")


def generate_output_arrays(enc_type, filepaths, yvalues):
    """Generates output arrays for a specific encoding type
    (either convolution or 'flat')."""
    if "conv" in enc_type:
        mode = "conv"
        encoding_name = enc_type.split("_conv")[0]
    else:
        mode = "flat"
        encoding_name = enc_type

    encoder = MSAEncodingToolkit(encoding_name)
    ofnames = ["train", "test"]
    for i, ofname in enumerate(ofnames):
        if ofname not in os.listdir():
            os.mkdir(ofname)
        os.chdir(ofname)
        encoder.encode_fasta_file(filepaths[i], np.asarray(yvalues[i]),
                os.getcwd(), blocksize=2000, mode=mode)
        os.chdir("..")


def get_dataset_fps(start_dir):
    """Organizes the input data into fasta files for training
    and testing that can be encoded by the helper function."""
    flist = os.listdir()
    if "test.fasta" not in flist or "train.fasta" not in flist:
        seqlists = [[], []]
        yvalues = [[], []]
        #We CAN separate out the validation set, but there's no
        #point in doing so, because we tune hyperparameters through
        #marginal likelihood rather than by testing on a validation set.
        seqtype_map = {"train":"train", "test":"test", "valid":"train"}
        seqtypes_idx = ["train", "test"]

        with open("FLUORESCENCE.fasta", "r") as fhandle:
            for seqrec in SeqIO.parse(fhandle, "fasta"):
                seqtype = seqrec.description.split("_")[-1].split(".json")[0]
                seqtype = seqtype_map[seqtype]
                seqindex = seqtypes_idx.index(seqtype)
                seqlists[seqindex].append(seqrec)
                yvalue = seqrec.description.split("_[")[1].split("]_")[0]
                yvalues[seqindex].append(float(yvalue))
        output_filenames = []

        for i, seqtype in enumerate(seqtypes_idx):
            with open(f"{seqtype}.fasta", "w+") as fhandle:
                SeqIO.write(seqlists[i], fhandle, "fasta")
                output_filenames.append(os.path.abspath(f"{seqtype}.fasta"))
        return output_filenames, yvalues

    output_filenames = ["train.fasta", "test.fasta"]
    yvalues = [[],[],[]]

    for i, output_filename in enumerate(output_filenames):
        with open(output_filename, "r") as fhandle:
            for seqrec in SeqIO.parse(fhandle, "fasta"):
                yvalue = seqrec.description.split("_[")[1].split("]_")[0]
                yvalues[i].append(float(yvalue))

    output_filenames = [os.path.abspath(fname) for fname in output_filenames]
    return output_filenames, yvalues
