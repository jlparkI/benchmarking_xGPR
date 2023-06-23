"""Preps the fluorescence dataset (mutations in GFP) for modeling."""
import os
import numpy as np
from Bio import SeqIO
import esm
from xGPR.static_layers.fast_conv import FastConv1d
from .build_esm_reps import generate_esm_embeddings, save_conv_reps
from .protein_tools.msa_encoding_toolkit import MSAEncodingToolkit


def prep_fluorescence(start_dir):
    """Preps the fluorescence dataset for modeling using either a convolution
    or fixed vector kernel."""
    os.chdir(os.path.join("benchmark_evals", "fluorescence_eval", "raw_data"))
    if "FLUORESCENCE.fasta" not in os.listdir():
        raise ValueError("The fluorescence dataset has not been downloaded yet!")

    esm_model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model.cuda()

    statlayer = FastConv1d(seq_width = 1280, device = "gpu",
                        conv_width = [9], num_features = 5000,
                        mode = "maxpool")

    filepaths, yvalues = get_dataset_fps(start_dir)
    target_dirs = ["onehot_conv", "esm_reps"]
    for i, enc_type in enumerate(target_dirs):
        os.chdir(start_dir)
        os.chdir(os.path.join("benchmark_evals", "fluorescence_eval"))
        print(f"Now encoding fluorescence using {enc_type}")
        if target_dirs[i] not in os.listdir():
            os.mkdir(target_dirs[i])
        os.chdir(target_dirs[i])
        if "esm" not in enc_type:
            generate_output_arrays(enc_type, filepaths, yvalues)
        else:
            generate_esm_arrays(filepaths, yvalues,
                        esm_model, alphabet, batch_converter,
                        statlayer)


    print("Fluorescence encoding is complete.")


def generate_esm_arrays(filepaths, yvalues,
        esm_model, alphabet, batch_converter, statlayer):
    """Generate ESM embeddings for a specific split."""
    rep_dir = os.getcwd()
    if "standard" not in os.listdir():
        os.mkdir("standard")
    if "conv" not in os.listdir():
        os.mkdir("conv")

    ofnames = ["train", "test"]

    for i, (filepath, ofname) in enumerate(zip(filepaths, ofnames)):
        os.chdir(os.path.join(rep_dir, "standard"))

        with open(filepath, "r") as fhandle:
            raw_seqs = [str(s.seq) for s in SeqIO.parse(filepath, "fasta")]

        conv_reps = generate_esm_embeddings(ofname, raw_seqs, yvalues[i],
                esm_model, alphabet, batch_converter, statlayer, 10)

        os.chdir(os.path.join(rep_dir, "conv"))
        save_conv_reps(ofname, conv_reps, yvalues[i])


def generate_output_arrays(enc_type, filepaths, yvalues):
    """Generates output arrays for a specific encoding type
    (either convolution or 'flat')."""
    if "standard" not in os.listdir():
        os.mkdir("standard")
    os.chdir("standard")
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
                os.getcwd(), blocksize=2000, verbose=True,
                mode=mode)
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
